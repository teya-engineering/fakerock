package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/saltpay/fakerock/internal/backend"
	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

type stubBackend struct {
	got       openai.ChatRequest
	resp      openai.ChatResponse
	chunks    []openai.ChatChunk
	err       error
	gotEmbed  openai.EmbeddingRequest
	embedResp openai.EmbeddingResponse
	embedErr  error
	pingErr   error
}

func (s *stubBackend) Ping(context.Context) error {
	return s.pingErr
}

func (s *stubBackend) Chat(_ context.Context, req openai.ChatRequest) (openai.ChatResponse, error) {
	s.got = req
	return s.resp, s.err
}

// ChatStream delivers the stub's chunks, then returns err, so err with no chunks fails before
// anything is streamed and err with chunks fails part way through.
func (s *stubBackend) ChatStream(_ context.Context, req openai.ChatRequest, onChunk func(openai.ChatChunk) error) error {
	s.got = req
	for _, chunk := range s.chunks {
		if err := onChunk(chunk); err != nil {
			return err
		}
	}
	return s.err
}

func (s *stubBackend) Embeddings(_ context.Context, req openai.EmbeddingRequest) (openai.EmbeddingResponse, error) {
	s.gotEmbed = req
	return s.embedResp, s.embedErr
}

func newTestServer(t *testing.T, backend Backend) *Server {
	t.Helper()
	return New(backend, "test-model", "test-embedding-model", 0)
}

func post(t *testing.T, srv *Server, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	return rec
}

func TestConverseHappyPath(t *testing.T) {
	backend := &stubBackend{resp: openai.ChatResponse{
		Choices: []openai.Choice{{
			Message:      openai.Message{Role: "assistant", Content: "hi there"},
			FinishReason: openai.FinishReasonStop,
		}},
		Usage: openai.Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
	}}
	srv := newTestServer(t, backend)

	rec := post(t, srv, "/model/sonnet/converse", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if backend.got.Model != "test-model" {
		t.Errorf("backend model = %q, want test-model", backend.got.Model)
	}

	var resp bedrock.ConverseResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if resp.StopReason != bedrock.StopReasonEndTurn {
		t.Errorf("stopReason = %q", resp.StopReason)
	}
	if len(resp.Output.Message.Content) != 1 || *resp.Output.Message.Content[0].Text != "hi there" {
		t.Errorf("content = %+v", resp.Output.Message.Content)
	}
	if resp.Usage.InputTokens != 3 || resp.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestConverseResolvesEncodedArn(t *testing.T) {
	backend := &stubBackend{resp: openai.ChatResponse{
		Choices: []openai.Choice{{Message: openai.Message{Content: "ok"}, FinishReason: openai.FinishReasonStop}},
	}}
	srv := newTestServer(t, backend)

	arn := "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.anthropic.sonnet-v1"
	path := "/model/" + url.PathEscape(arn) + "/converse"

	rec := post(t, srv, path, `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if backend.got.Model != "test-model" {
		t.Errorf("backend model = %q, want test-model", backend.got.Model)
	}
}

func TestConverseRejectsEmptyMessages(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	rec := post(t, srv, "/model/sonnet/converse", `{"messages":[]}`)

	assertAWSError(t, rec, http.StatusBadRequest, errValidation)
}

func TestConverseRejectsMalformedBody(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	rec := post(t, srv, "/model/sonnet/converse", `{`)

	assertAWSError(t, rec, http.StatusBadRequest, errValidation)
}

func TestConverseBackendFailure(t *testing.T) {
	srv := newTestServer(t, &stubBackend{err: errors.New("connection refused")})

	rec := post(t, srv, "/model/sonnet/converse", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	assertAWSError(t, rec, http.StatusBadGateway, errModel)
}

func TestApplyGuardrailPassesThrough(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	rec := post(t, srv, "/guardrail/gr-123/version/DRAFT/apply", `{"source":"OUTPUT","content":[]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	var resp bedrock.ApplyGuardrailResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if resp.Action != bedrock.GuardrailActionNone {
		t.Errorf("action = %q, want NONE", resp.Action)
	}
}

func TestInvokeEmbeddingHappyPath(t *testing.T) {
	backend := &stubBackend{embedResp: openai.EmbeddingResponse{
		Data:  []openai.EmbeddingData{{Embedding: []float32{0.1, 0.2, 0.3, 0.4}}},
		Usage: openai.Usage{PromptTokens: 7},
	}}
	srv := newTestServer(t, backend)

	rec := post(t, srv, "/model/amazon.titan-embed-text-v2:0/invoke", `{"inputText":"hello","dimensions":2}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200, body = %s", rec.Code, rec.Body.String())
	}
	if backend.gotEmbed.Input != "hello" {
		t.Errorf("backend input = %q, want hello", backend.gotEmbed.Input)
	}
	if backend.gotEmbed.Model != "test-embedding-model" {
		t.Errorf("backend model = %q, want test-embedding-model", backend.gotEmbed.Model)
	}
	if backend.gotEmbed.Dimensions == nil || *backend.gotEmbed.Dimensions != 2 {
		t.Errorf("backend dimensions = %v, want 2 forwarded", backend.gotEmbed.Dimensions)
	}

	var resp bedrock.TitanEmbeddingResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if len(resp.Embedding) != 2 {
		t.Errorf("embedding len = %d, want 2 (trimmed to requested dimensions)", len(resp.Embedding))
	}
	if resp.InputTextTokenCount != 7 {
		t.Errorf("inputTextTokenCount = %d, want 7", resp.InputTextTokenCount)
	}
}

func TestInvokeEmbeddingRejectsNonPositiveDimensions(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	rec := post(t, srv, "/model/amazon.titan-embed-text-v2:0/invoke", `{"inputText":"hi","dimensions":-1}`)

	assertAWSError(t, rec, http.StatusBadRequest, errValidation)
}

func TestUnknownOperation(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	rec := post(t, srv, "/model/sonnet/unknown-op", `{}`)

	assertAWSError(t, rec, http.StatusNotFound, errNotFound)
}

func TestGetIsRejected(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	req := httptest.NewRequest(http.MethodGet, "/model/sonnet/converse", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	assertAWSError(t, rec, http.StatusMethodNotAllowed, errValidation)
}

func assertAWSError(t *testing.T, rec *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	if rec.Code != status {
		t.Fatalf("status = %d, want %d, body = %s", rec.Code, status, rec.Body)
	}
	if got := rec.Header().Get("x-amzn-ErrorType"); got != code {
		t.Errorf("x-amzn-ErrorType = %q, want %q", got, code)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decoding error body: %v", err)
	}
	if body["__type"] != code || body["message"] == "" {
		t.Errorf("error body = %+v", body)
	}
}

// End-to-end coverage that outputConfig.textFormat.jsonSchema survives HTTP parsing
// and reaches the backend as OpenAI response_format. Without this the schema is
// silently dropped at the Amazon SDK boundary and the model answers with unconstrained text.
func TestConverseForwardsOutputConfigAsResponseFormat(t *testing.T) {
	backend := &stubBackend{resp: openai.ChatResponse{
		Choices: []openai.Choice{{
			Message:      openai.Message{Role: "assistant", Content: `{"answer":42}`},
			FinishReason: openai.FinishReasonStop,
		}},
	}}
	srv := newTestServer(t, backend)

	schema := `{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"]}`
	body := `{
      "messages":[{"role":"user","content":[{"text":"hi"}]}],
      "outputConfig":{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{
        "name":"Answer",
        "schema":` + strconv.Quote(schema) + `
      }}}}
    }`

	rec := post(t, srv, "/model/sonnet/converse", body)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	rf := backend.got.ResponseFormat
	if rf == nil {
		t.Fatal("backend request had no response_format — outputConfig was dropped")
	}
	if rf.Type != "json_schema" {
		t.Errorf("response_format.type = %q, want json_schema", rf.Type)
	}
	if rf.JSONSchema == nil {
		t.Fatal("response_format.json_schema is nil")
	}
	if rf.JSONSchema.Name != "Answer" {
		t.Errorf("json_schema.name = %q, want Answer", rf.JSONSchema.Name)
	}
	if !rf.JSONSchema.Strict {
		t.Error("json_schema.strict = false, want true so the backend enforces the schema")
	}
	// The schema arrived as an escaped JSON string on the wire; it must be inlined as
	// a JSON object in the OpenAI request, not re-quoted. Compare parsed trees so
	// whitespace differences don't matter.
	var got, want any
	if err := json.Unmarshal(rf.JSONSchema.Schema, &got); err != nil {
		t.Fatalf("json_schema.schema is not valid JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(schema), &want); err != nil {
		t.Fatalf("expected schema is not valid JSON: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("json_schema.schema = %v, want %v", got, want)
	}
}

// Effort and other model-specific settings travel in additionalModelRequestFields. A gateway in
// front of Bedrock only sees them if they reach the backend request.
func TestConverseForwardsAdditionalModelRequestFields(t *testing.T) {
	backend := &stubBackend{resp: openai.ChatResponse{
		Choices: []openai.Choice{{Message: openai.Message{Content: "ok"}, FinishReason: openai.FinishReasonStop}},
	}}
	srv := newTestServer(t, backend)

	body := `{"messages":[{"role":"user","content":[{"text":"hi"}]}],` +
		`"additionalModelRequestFields":{"output_config":{"effort":"low"}}}`
	rec := post(t, srv, "/model/sonnet/converse", body)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if got := string(backend.got.Extra["output_config"]); got != `{"effort":"low"}` {
		t.Errorf("output_config = %s, want {\"effort\":\"low\"}", got)
	}
}

func TestConverseRejectsConflictingAdditionalModelRequestFields(t *testing.T) {
	backend := &stubBackend{}
	srv := newTestServer(t, backend)

	body := `{"messages":[{"role":"user","content":[{"text":"hi"}]}],` +
		`"inferenceConfig":{"maxTokens":512},` +
		`"additionalModelRequestFields":{"max_tokens":5}}`
	rec := post(t, srv, "/model/sonnet/converse", body)

	assertAWSError(t, rec, http.StatusBadRequest, errValidation)
	if backend.got.Model != "" {
		t.Error("backend was called for a request that should have been rejected")
	}
}

// Bedrock's default outputConfig.textFormat.type is "text"; forwarding a synthetic
// response_format for that case would silently force JSON on callers who never asked
// for it. Absent outputConfig should behave the same as an explicit "text" type.
func TestConverseTextOutputConfigDoesNotSetResponseFormat(t *testing.T) {
	for name, body := range map[string]string{
		"absent": `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`,
		"explicit_text": `{"messages":[{"role":"user","content":[{"text":"hi"}]}],` +
			`"outputConfig":{"textFormat":{"type":"text"}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			backend := &stubBackend{resp: openai.ChatResponse{
				Choices: []openai.Choice{{Message: openai.Message{Content: "ok"}, FinishReason: openai.FinishReasonStop}},
			}}
			srv := newTestServer(t, backend)

			rec := post(t, srv, "/model/sonnet/converse", body)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
			}
			if backend.got.ResponseFormat != nil {
				t.Errorf("response_format = %+v, want nil for text output", backend.got.ResponseFormat)
			}
		})
	}
}

// An untranslatable block (document, video) would leave the model answering confidently
// about content it never received, so the request must fail instead of dropping the block.
func TestConverseRejectsUnsupportedContentBlocks(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	for _, block := range []string{
		`{"document":{"format":"pdf","name":"invoice","source":{"bytes":"aGk="}}}`,
		`{"video":{"format":"mp4","source":{"bytes":"aGk="}}}`,
	} {
		body := `{"messages":[{"role":"user","content":[{"text":"what is this?"},` + block + `]}]}`

		rec := post(t, srv, "/model/sonnet/converse", body)

		assertAWSError(t, rec, http.StatusBadRequest, errValidation)
	}
}

func TestCallerBearerTokenReachesTheBackend(t *testing.T) {
	operations := []struct{ name, path, body string }{
		{"converse", "/model/m/converse", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`},
		{"converse-stream", "/model/m/converse-stream", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`},
		{"invoke", "/model/amazon.titan-embed-text-v2:0/invoke", `{"inputText":"hi"}`},
	}
	headers := []struct{ name, sent, want string }{
		{"bearer", "Bearer abc", "Bearer abc"},
		{"aws signature", "AWS4-HMAC-SHA256 Credential=x/20261009/eu-west-1/bedrock/aws4_request", ""},
		{"none", "", ""},
	}

	for _, op := range operations {
		for _, header := range headers {
			t.Run(op.name+" "+header.name, func(t *testing.T) {
				var gotAuth string
				fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					gotAuth = r.Header.Get("Authorization")
					raw, _ := io.ReadAll(r.Body)
					switch {
					case r.URL.Path == "/embeddings":
						_, _ = io.WriteString(w, `{"data":[{"embedding":[0.1]}]}`)
					case strings.Contains(string(raw), `"stream":true`):
						_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`+"\n\n"+"data: [DONE]\n\n")
					default:
						_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
					}
				}))
				t.Cleanup(fake.Close)
				srv := New(backend.New(fake.URL, fake.URL, 5*time.Second), "m", "e", 0)

				req := httptest.NewRequest(http.MethodPost, op.path, strings.NewReader(op.body))
				if header.sent != "" {
					req.Header.Set("Authorization", header.sent)
				}
				rec := httptest.NewRecorder()
				srv.ServeHTTP(rec, req)

				if rec.Code != http.StatusOK {
					t.Fatalf("status = %d: %s", rec.Code, rec.Body)
				}
				if gotAuth != header.want {
					t.Errorf("backend Authorization = %q, want %q", gotAuth, header.want)
				}
			})
		}
	}
}
