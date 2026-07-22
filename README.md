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

First start takes about 30 seconds while the model loads. After that:

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

Tool calling works in both directions: `toolConfig` reaches the model, `toolUse` blocks come back,
and `toolResult` blocks are fed in on the next turn. `stopReason` is `tool_use` when the model calls
a tool, so agent loops run to completion.

`cachePoint` and `guardrailConfig` are accepted and ignored. Token counts are real numbers from the
backend. Cache token counts are always zero.

## Environment variables

Defaults as set by the image. The binary on its own defaults to `http://localhost:11434/v1` and
`qwen3:1.7b`, which suit a local Ollama.

| Variable | Default | Description |
|---|---|---|
| `LISTEN_ADDR` | `:8080` | Address the Bedrock API listens on |
| `LLAMA_CTX` | `32768` | Context size. A prompt with 19 tools can reach 19k tokens |
| `LLAMA_NGL` | `0` | Layers offloaded to the GPU. Needs a CUDA `BASE_IMAGE` |
| `LLAMA_REASONING` | `off` | Thinking. Off because Converse has nowhere to carry it |
| `BACKEND_BASE_URL` | `http://127.0.0.1:8081/v1` | Where the model is served |
| `BACKEND_MODEL` | `local` | Model name sent to the backend |
| `BACKEND_TIMEOUT` | `5m` | How long to wait for a completion |

If the context is too small you get a clear `400` naming the token counts, not a silent truncation.

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

To use an existing Ollama or llama.cpp instead of the bundled one, run the binary on its own and
point it at any OpenAI-compatible `/v1/chat/completions` endpoint:

```bash
go build -o fakerock ./cmd/fakerock
BACKEND_BASE_URL=http://localhost:11434/v1 BACKEND_MODEL=qwen3:1.7b ./fakerock
```

## What it does not do

- **`InvokeModel` and embeddings.** Only the Converse API is implemented. Embedding-backed features
  such as RAG will not work against this.
- **Images and documents.** Content blocks other than `text`, `toolUse`, `toolResult` and
  `cachePoint` are rejected with a `400` that names the block. This is deliberate. Dropping an
  uploaded file silently would leave the model answering confidently about something it never saw.
- **Guardrails.** The endpoint exists so guardrail-wrapped clients keep working. It enforces nothing.
- **Request signing.** Signatures are accepted without being checked. Run it locally only.

## Good to know

Answers come from a small model running on your CPU, so they are slower and weaker than the real
thing, and they are not deterministic. This is a development and testing tool. It checks that your
Bedrock code path is correct, not that your prompts are good.
