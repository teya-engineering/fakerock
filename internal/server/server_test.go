package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

type stubBackend struct {
	got       openai.ChatRequest
	resp      openai.ChatResponse
	err       error
	gotEmbed  openai.EmbeddingRequest
	embedResp openai.EmbeddingResponse
	embedErr  error
}

func (s *stubBackend) Chat(_ context.Context, req openai.ChatRequest) (openai.ChatResponse, error) {
	s.got = req
	return s.resp, s.err
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
