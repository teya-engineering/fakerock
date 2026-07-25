# Contributing to fakerock

## Development Setup

```bash
git clone git@github.com:saltpay/fakerock.git
cd fakerock
go build -o fakerock ./cmd/fakerock
go test ./...
```

The binary on its own needs an OpenAI-compatible backend. Ollama is the easiest one to run locally:

```bash
ollama serve
ollama pull qwen3:1.7b
BACKEND_BASE_URL=http://localhost:11434/v1 BACKEND_MODEL=qwen3:1.7b ./fakerock
```

Then call it with the AWS CLI, which is the same path a real client takes:

```bash
AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_REGION=eu-west-1 \
aws bedrock-runtime converse \
  --endpoint-url http://localhost:8080 \
  --model-id anything \
  --messages '[{"role":"user","content":[{"text":"Reply with exactly: hello"}]}]'
```

To test the packaged image instead, build it. That downloads a ~1GB model, so it is slow the first
time:

```bash
docker build -t fakerock .
docker run -p 8099:8080 fakerock
```

## Architecture Overview

fakerock translates in one direction and back:

```
AWS SDK  →  internal/server  →  internal/translate  →  internal/backend  →  llama.cpp / Ollama
```

A `Converse` request is decoded into `bedrock.ConverseRequest`, turned into an
`openai.ChatRequest`, sent to the backend, and the reply is turned back into a
`bedrock.ConverseResponse`. `ConverseStream` does exactly the same, then replays the finished
response as event stream frames.

### Package Layout

| Package | Purpose |
|---|---|
| `cmd/fakerock/` | Entry point: load config, wire the server, listen |
| `internal/config/` | Environment variables, parsed and validated once at boot |
| `internal/server/` | HTTP routing and handlers, one file per operation |
| `internal/translate/` | Pure translation functions between the two wire shapes |
| `internal/backend/` | HTTP client for the OpenAI-compatible backend |
| `internal/bedrock/` | Bedrock wire types |
| `internal/openai/` | OpenAI wire types |

`internal/translate` holds most of the logic and does no I/O. Everything variable, including the
model name and the measured latency, is passed in. That keeps it testable without a server or a
model.

### Routing

`Server.ServeHTTP` matches decoded path segments by hand rather than using a router. Model ids are
usually inference profile ARNs, which contain slashes and reach us percent-encoded inside a single
path segment. A router that unescapes the path before matching splits the ARN across segments and
the route stops matching.

| Path | Handler |
|---|---|
| `GET /health` | `health.go`|
| `POST /model/{id}/converse` | `converse.go` |
| `POST /model/{id}/converse-stream` | `stream.go` |
| `POST /model/{id}/invoke` | `invoke.go` (Titan text embeddings only) |
| `POST /guardrail/{id}/version/{v}/apply` | `guardrail.go` (always passes content through) |
| `GET`/`POST /admin/model` | `admin.go` (read or swap the backend model at runtime) |

The model id in the path is logged and otherwise ignored. Every request goes to the configured
backend model.

### Error Responses

AWS SDKs pick the exception class off the `x-amzn-ErrorType` header. Without it every failure
reaches the caller as an opaque generic error, so all failures go through `writeError`, never
`http.Error`.

| Code | When |
|---|---|
| `ValidationException` | Malformed body, missing field, unsupported content block |
| `ResourceNotFoundException` | Unknown path |
| `ModelErrorException` | Backend call or response translation failed |

Streaming validates the request and calls the backend before writing anything, because once the
first frame is written the status is fixed at 200 and the client can no longer be told the request
was bad.

### Streaming

fakerock does not stream from the backend. It waits for the complete response, then replays it as
the frames a streaming client expects: `messageStart`, text deltas chunked at 40 runes, tool use
blocks, `contentBlockStop`, `messageStop`, `metadata`. Time to first token is that of the whole
generation. Every other observable detail matches a real stream.

Frames are encoded with the AWS `eventstream` package, which is the one direct dependency.

### Tool Calling

This is the fragile part, and the reason the project exists for agent workloads.

- `toolConfig.tools` become OpenAI `tools` with the JSON schema passed through untouched
- `toolResult` blocks become their own `role: tool` messages, so they land directly after the
  assistant message carrying the matching `tool_call_id`, which is the order OpenAI requires
- `stopReason` is `tool_use` whenever tool calls are present, even if the backend reports
  `finish_reason: stop`. Some backends do exactly that, and a client that trusted `end_turn` there
  would drop out of the agent loop.

Tool calling only works if llama.cpp runs with `--jinja`. Without it the tool definitions are
ignored silently and the loop ends after one turn with no error.

### Embeddings

`InvokeModel` serves the Titan text embedding shape and nothing else. The vector is trimmed or
zero-padded to the requested width so it fits a fixed-width store even when the backend's native
width differs. The width is passed to the backend too, so a Matryoshka model can return it directly.

This makes the shape right for testing, not the vector semantically equivalent to Titan. It is not
re-normalised to unit length.

## The Image

`Dockerfile` builds the wrapper, downloads the model in a separate stage, and lands both on a
llama.cpp base. `entrypoint.sh` starts each bundled llama-server only when the matching
`BACKEND_*_BASE_URL` still points at it, waits for those that did start, runs a one-token warmup on
the chat server when it is local, then execs fakerock. The Bedrock port only opens after all of
that, so a TCP readiness probe (or `GET /health` on the listen address) waits for a model that
can actually answer when the bundled path is in use.

Every source is a build argument (`GO_IMAGE`, `CURL_IMAGE`, `BASE_IMAGE`, `MODEL_URL`) so a build
with no internet access can point each one at a mirror.

Changing a default usually means touching `Dockerfile`, `entrypoint.sh` and the README table
together. The README documents the defaults as set by the image, which are not the same as the
binary's own defaults.

## Code Style

- Standard Go formatting (`gofmt`), tabs for indentation
- Wrap errors with context: `fmt.Errorf("decoding backend response: %w", err)`
- Structured logging with `log/slog`: one `Info` line per request, full bodies at `Debug`
- Comments explain why, not what
- Keep dependencies minimal, prefer the standard library

## Debugging

`LOG_LEVEL=debug` logs the exact OpenAI-format JSON sent to the backend and the raw JSON it
returned. That is the first thing to look at when the model answers in text where a tool call was
expected, because it shows whether the tools reached the model at all.

For llama.cpp's own logs (slot activity, prompt processing, timings) set
`LLAMA_ARG_LOG_VERBOSITY=5`. llama-server reads it from the environment.

## Tests

```bash
go test ./...          # all tests
./entrypoint_test.sh   # bundled-backend URL matching in entrypoint_lib.sh
go vet ./...           # static analysis
go test -race ./...    # the server holds mutable state, so races matter
```

Server tests use a fake `Backend`, so nothing external is needed. New translation logic needs a test
in the matching `_test.go` file.

CI runs `go vet`, `go test` and a Docker build on every pull request. The image is the product, so a
broken Dockerfile fails the build.

## Commit Messages

Plain summaries of what changed. No conventional-commit prefixes are required; there is no
version tagging in this repo.
