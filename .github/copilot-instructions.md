# fakerock - AI Coding Assistant Instructions

## Project Overview

**fakerock** is a Go service that speaks the Amazon Bedrock Runtime API and answers from a local
OpenAI-compatible model instead of AWS. Point an AWS SDK at it and the Bedrock code path runs
unchanged, with no AWS account, no credentials and no spend.

The Docker image is the product: it ships llama.cpp and a baked-in GGUF model, so `docker run` is
the whole setup.

### Business Intent

- **Remove AWS from the local loop** for teams building on Bedrock
- **Keep the code path real**: the app still builds Bedrock requests and parses Bedrock responses
- **Support agent loops**, so tool calling has to work in both directions
- **Stay cheap to run**: one small model, CPU by default, no external services

### Non-goals

- Not a Bedrock emulator. Only the operations listed in the README are served.
- Not a security boundary. Request signatures are accepted without being checked. Local use only.
- Not an evaluation tool. It checks that the Bedrock integration is correct, not that prompts are
  good. Answers are slower, weaker, and non-deterministic.

## Architecture

Bedrock request in, OpenAI chat request out, backend answers, translate back:

```
AWS SDK  →  internal/server  →  internal/translate  →  internal/backend  →  llama.cpp / Ollama
                  ↑                     ↓
            internal/bedrock      internal/openai        (wire types, no logic)
```

- `cmd/fakerock/` - loads config, wires the server, listens
- `internal/config/` - environment variables, validated once at boot
- `internal/server/` - HTTP routing and handlers, one file per operation
- `internal/translate/` - pure functions between the two wire shapes. No I/O.
- `internal/backend/` - HTTP client for the OpenAI-compatible backend
- `internal/bedrock/`, `internal/openai/` - wire types only

### Invariants

1. **`internal/translate` stays pure.** No HTTP, no clock, no globals. Everything variable is a
   parameter. This is what makes it testable in isolation, and it holds most of the logic.
2. **Routing is hand-written.** Model ids are often inference-profile ARNs, which arrive
   percent-encoded inside one path segment. Do not swap in a router that decodes the path first.
3. **Errors carry `x-amzn-ErrorType`.** Without that header AWS SDKs surface every failure as an
   opaque generic error. Always use `writeError`, never `http.Error`.
4. **Streaming validates before the first frame.** Once a frame is written the status is fixed at
   200 and the client can no longer be told the request was bad.
5. **Unsupported input fails loudly.** Content blocks that cannot be translated return a `400`
   naming the block. Silently dropping input leaves the model answering confidently about something
   it never saw.
6. **Backend failures pass through.** A backend `400` or `403` reaches the caller as-is. Do not
   swallow, retry, or paper over it.

## Code Style

- Standard Go formatting (`gofmt`), tabs for indentation
- Errors wrapped with context: `fmt.Errorf("decoding backend response: %w", err)`
- Structured logging via `log/slog`. `Debug` for the full request and response bodies, `Info` for
  one line per request, `Error` for failures.
- Comments explain *why*, especially non-obvious protocol behaviour. Do not narrate what the code
  already says.
- Keep dependencies minimal. The only direct dependency is the AWS event stream encoder; prefer the
  standard library for everything else.

## Development Guidelines

### When Adding an Operation

1. Add the wire types to `internal/bedrock` (and `internal/openai` if the backend shape is new)
2. Add the translation to `internal/translate` as pure functions, with table tests
3. Add a handler file in `internal/server` and a route in `ServeHTTP`
4. Validate the request body before calling the backend
5. Update the README table of supported operations

### When Changing Translation

Round-trip behaviour is the thing under test. Check both directions and both transports: `Converse`
and `ConverseStream` share the same translation, so a change to one changes both.

Tool calling is the fragile part. `toolUse` ids must survive the round trip, tool results must land
directly after the assistant message holding their `tool_call_id`, and `stopReason` must be
`tool_use` whenever tool calls are present, even when the backend reports `finish_reason: stop`.

### Testing

- `go test ./...` and `go vet ./...` both run in CI, and so does a Docker build
- Server tests use a fake `Backend`, so no model is needed
- New translation logic needs a test in the matching `_test.go` file

## AI Assistant Guidelines

1. **Keep it a stand-in, not a re-implementation.** Fidelity to the API shape matters; fidelity to
   AWS internals does not.
2. **Do not add configuration knobs speculatively.** Every environment variable is documented in the
   README and has to stay that way.
3. **Do not silently degrade.** A wrong answer is worse than a clear `400`.
4. **Remember the image.** Changes to defaults usually touch `Dockerfile`, `entrypoint.sh` and the
   README table together.
5. **Keep the model small.** The bundled model is 1.7B on a CPU. Anything that adds prompt overhead
   costs real latency and accuracy.
