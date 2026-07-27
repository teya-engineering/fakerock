# fakerock

**A stand-in for Amazon Bedrock that runs on your laptop.**

It speaks the Bedrock Runtime API, and answers using a local model instead of AWS. Point your app's
AWS SDK at it and the Bedrock code path runs unchanged, with no AWS account, no credentials and no
spend.

The image ships llama.cpp and a model inside it, so nothing else is needed to run it.

## Quick start

```bash
docker run -p 8099:8080 ghcr.io/saltpay/fakerock:cpu
```

On a GPU host use the `cuda` tag and tell llama.cpp how many layers to offload:

```bash
docker run --gpus all -e LLAMA_NGL=99 -p 8099:8080 ghcr.io/saltpay/fakerock:cuda
```

First start takes about 30 seconds while the model loads and a one-token warmup completion runs, so
the first real request does not pay the first-inference setup cost. The API port only opens after
both, so anything that waits for the port (a TCP readiness probe, `docker run` port checks) waits
for a model that can actually answer. After that:

```bash
AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_REGION=eu-west-1 \
aws bedrock-runtime converse \
  --endpoint-url http://localhost:8099 \
  --model-id anything \
  --messages '[{"role":"user","content":[{"text":"Reply with exactly: hello"}]}]'
```

The model id is ignored. Send a short name or a full inference profile ARN, whatever your app
already sends. Every request goes to the one model in the image.

## Pointing an app at it

Any AWS SDK works. Give it the endpoint and any non-empty credentials:

```
AWS_ENDPOINT_URL_BEDROCK_RUNTIME=http://localhost:8099
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local
```

Use fake account ids in your ARNs, for example `arn:aws:bedrock:eu-west-1:000000000000:...`. If the
endpoint override ever fails to apply, the call then fails loudly instead of quietly reaching real
Bedrock and charging you.

## Spring AI

`BedrockProxyChatModel` takes clients you build yourself, so set the endpoint on the builders. That
is more reliable than the environment variable, which has to survive whatever launches the JVM.

Give it a property that is empty everywhere except locally:

```yaml
# application.yml, no value: the SDK resolves the real regional endpoint
app:
  ai:
    bedrock:
      endpoint:

---
# application-local.yml
app:
  ai:
    bedrock:
      endpoint: http://localhost:8099
```

A null override means "use the default", so no branching is needed at the call site:

```java
public static URI overrideOrNull(String endpoint) {
    return endpoint == null || endpoint.isBlank() ? null : URI.create(endpoint);
}

var client = BedrockRuntimeClient.builder()
    .region(Region.of(region))
    .endpointOverride(overrideOrNull(bedrockEndpoint))
    .credentialsProvider(DefaultCredentialsProvider.builder().build())
    .build();
```

Build the async client the same way. Spring AI uses the sync client for `Converse` and the async one
for `ConverseStream`, so overriding only one sends half your traffic to AWS.

```java
BedrockProxyChatModel.builder()
    .bedrockRuntimeClient(client)
    .bedrockRuntimeAsyncClient(asyncClient)
    .options(BedrockChatOptions.builder().model(inferenceProfileArn).build())
    .build();
```

Credentials still have to resolve, so set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to any
value. Dummy values are better than none: they stop `DefaultCredentialsProvider` picking up your real
SSO profile if the endpoint override ever fails to apply.

Watch out for clients built somewhere other than your own configuration. Spring AI's
`TitanEmbeddingBedrockApi` builds its own internally and accepts no endpoint override, so embeddings
cannot be pointed here this way.

## What it supports

| Operation | Behaviour |
|---|---|
| `Converse` | Full translation, including tool calling |
| `ConverseStream` | Same, returned as AWS event stream frames |
| `ApplyGuardrail` | Always returns `action: NONE`, so content passes through |
| `InvokeModel` (Titan text embeddings) | Translated to the backend's `/v1/embeddings`, resized to the requested width |

Tool calling works in both directions: `toolConfig` reaches the model, `toolUse` blocks come back,
and `toolResult` blocks are fed in on the next turn. `stopReason` is `tool_use` when the model calls
a tool, so agent loops run to completion.

`cachePoint` and `guardrailConfig` are accepted and ignored. Token counts are real numbers from the
backend. Cache token counts are always zero.

`image` blocks (`png`, `jpeg`, `gif`, `webp`) are translated to OpenAI `image_url` parts, but only
produce a real answer when `BACKEND_MODEL` points at a vision-capable model (e.g. `llama3.2-vision`
or an ollama `*-vl`). The bundled Qwen3 1.7B is text-only. fakerock only translates; whatever the
backend says comes straight back, so a text-only model returns its own `400` ("does not support
image input") and a subscription-gated one returns `403`: surfaced loud, never swallowed.

Embeddings go to a backend OpenAI `/v1/embeddings` endpoint. They are off by default: set
`LLAMA_EMBEDDINGS=on` and the image runs a second llama-server in embedding mode on the bundled model
(port 8082), since one server serves chat or embeddings, not both. Titan's `dimensions` (or
`BACKEND_EMBEDDING_DIMENSIONS` when the request omits it) sets the output width, and the vector is
trimmed or zero-padded to match, so it fits a fixed-width store even when the backend's native width
differs. This makes the shape right for testing, not the vector semantically equivalent to Titan.
The bundled model is a chat model, so it embeds but not well; point `BACKEND_EMBEDDING_BASE_URL` and
`BACKEND_EMBEDDING_MODEL` at a real embedding backend (for example Ollama with an embedding model)
when retrieval quality matters.

The bundled chat llama-server starts when `LLAMA_CHAT=on` (the default). Set `LLAMA_CHAT=off` to
skip it when `BACKEND_BASE_URL` points elsewhere.

`GET /health` on the Bedrock listen address returns `200` and the current model when the backend
answers, and `503` with the reason when it does not. It calls the backend's `/v1/models` on every
request, which runs no inference, so a backend that dies an hour after startup flips the check.
Probe this rather than a llama-server port, which may never open.

## Environment variables

Defaults as set by the image. The binary on its own defaults to `http://localhost:11434/v1` and
`qwen3:1.7b`, which suit a local Ollama.

| Variable | Default | Description |
|---|---|---|
| `LISTEN_ADDR` | `:8080` | Address the Bedrock API listens on |
| `LLAMA_CTX` | `32768` | Context size. A prompt with 19 tools can reach 19k tokens |
| `LLAMA_NGL` | `0` | Layers offloaded to the GPU. Needs a CUDA `BASE_IMAGE` |
| `LLAMA_REASONING` | `off` | Thinking. Off because Converse has nowhere to carry it |
| `LLAMA_CHAT` | `on` | Set `off` to skip the bundled chat llama-server |
| `LLAMA_EMBEDDINGS` | `off` | Set `on` to run a second llama-server for embeddings on the bundled model |
| `LLAMA_WARMUP` | `on` | Run a one-token completion at startup, before the Bedrock API opens. Set `off` to skip |
| `LOG_LEVEL` | `info` | Wrapper log level. `debug` logs the full request and response JSON exchanged with the backend |
| `BACKEND_BASE_URL` | `http://127.0.0.1:8081/v1` | Where the model is served for chat |
| `BACKEND_EMBEDDING_BASE_URL` | `http://127.0.0.1:8082/v1` | Where the model is served for embeddings |
| `BACKEND_MODEL` | `local` | Model name sent to the backend for chat |
| `BACKEND_EMBEDDING_MODEL` | `BACKEND_MODEL` | Model name sent to the backend for embeddings |
| `BACKEND_EMBEDDING_DIMENSIONS` | unset | Output width when a request omits `dimensions`. Must be positive |
| `BACKEND_TIMEOUT` | `5m` | How long to wait for a completion |

If the context is too small you get a clear `400` naming the token counts, not a silent truncation.

## Seeing what the model was asked

Set `LOG_LEVEL=debug` and every call logs the exact OpenAI-format JSON sent to the backend and the
raw JSON it returned: the tool list, every message, and the model's reply. That is the first thing
to look at when the model answers in text where a tool call was expected, since it shows whether the
tools reached the model and what it actually generated. For llama.cpp's own logs (slot activity,
prompt processing, timings) set `LLAMA_ARG_LOG_VERBOSITY=5`; llama-server reads it from the
environment.

## Using a different model

The image bakes in Qwen3 1.7B (Q4_K_M). It handles a handful of tools well. With a long tool list it
starts choosing the wrong tool or malforming arguments, which looks like a bug but is the model
being small.

Build with a bigger one:

```bash
docker build --build-arg MODEL_URL=<url to a .gguf> -t fakerock .
```

Any GGUF works, but it must have a chat template that supports tools, otherwise tool calls never
come back.

## Switching model at runtime

`BACKEND_MODEL` sets the boot default. To swap the target model without recreating the container,
POST to `/admin/model`:

```bash
curl -s localhost:8080/admin/model -d '{"model":"llama3.2-vision"}'
curl -s localhost:8080/admin/model   # read the current value
```

The next Converse request uses the new model. The swap holds until the next swap or a restart. An
empty or missing `model` returns a `400`.

## Building behind a restricted network

Every image and the model are build arguments, so a build with no internet access can point each one
at a mirror:

| Argument | Default |
|---|---|
| `GO_IMAGE` | `golang:1.25-alpine` |
| `CURL_IMAGE` | `curlimages/curl:latest` |
| `BASE_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server` |
| `MODEL_URL` | `https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf` |

Go modules are still fetched normally.

## Running on a GPU

The `cuda` tag is built against llama.cpp's CUDA base. Build your own with:

```bash
docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda -t fakerock .
```

`LLAMA_NGL` defaults to `0`. Without it the CUDA image holds a GPU reservation and runs on the CPU
anyway, which looks configured and performs like it isn't.

## Using your own model server

To use an existing Ollama or llama.cpp instead of the bundled one, point `BACKEND_BASE_URL` at any
OpenAI-compatible `/v1/chat/completions` endpoint and set `LLAMA_CHAT=off`:

```bash
docker run -p 8099:8080 \
  -e BACKEND_BASE_URL=http://host.docker.internal:11434/v1 \
  -e BACKEND_MODEL=qwen3:1.7b \
  -e LLAMA_CHAT=off \
  ghcr.io/saltpay/fakerock:cpu
```

Or run the binary on its own:

```bash
go build -o fakerock ./cmd/fakerock
BACKEND_BASE_URL=http://localhost:11434/v1 BACKEND_MODEL=qwen3:1.7b ./fakerock
```

## What it does not do

- **`InvokeModel` beyond Titan text embeddings.** Only Titan embeddings are translated; other
  `InvokeModel` bodies are not implemented.
- **Documents and video.** Content blocks other than `text`, `image`, `toolUse`, `toolResult` and
  `cachePoint` are rejected with a `400` that names the block. This is deliberate. Dropping an
  uploaded file silently would leave the model answering confidently about something it never saw.
- **Guardrails.** The endpoint exists so guardrail-wrapped clients keep working. It enforces nothing.
- **Request signing.** Signatures are accepted without being checked. Run it locally only.

## Good to know

Answers come from a small model running on your CPU, so they are slower and weaker than the real
thing, and they are not deterministic. This is a development and testing tool. It checks that your
Bedrock code path is correct, not that your prompts are good.
