package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/saltpay/fakerock/internal/openai"
)

func TestAdminModelSwap(t *testing.T) {
	backend := &stubBackend{resp: openai.ChatResponse{
		Choices: []openai.Choice{{Message: openai.Message{Role: "assistant", Content: "ok"}, FinishReason: openai.FinishReasonStop}},
	}}
	srv := newTestServer(t, backend)

	if rec := post(t, srv, "/admin/model", `{"model":"llama3.2-vision"}`); rec.Code != http.StatusOK {
		t.Fatalf("swap status = %d, body = %s", rec.Code, rec.Body)
	}

	getReq := httptest.NewRequest(http.MethodGet, "/admin/model", nil)
	getRec := httptest.NewRecorder()
	srv.ServeHTTP(getRec, getReq)
	var got map[string]string
	if err := json.Unmarshal(getRec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if got["model"] != "llama3.2-vision" {
		t.Errorf("model = %q, want llama3.2-vision", got["model"])
	}

	post(t, srv, "/model/sonnet/converse", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)
	if backend.got.Model != "llama3.2-vision" {
		t.Errorf("backend model = %q, want llama3.2-vision", backend.got.Model)
	}
}

func TestAdminModelRejectsEmpty(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})
	if rec := post(t, srv, "/admin/model", `{"model":""}`); rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}
