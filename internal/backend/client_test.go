package backend

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/saltpay/fakerock/internal/openai"
)

func sseServer(t *testing.T, status int, body string, gotBody *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if gotBody != nil {
			raw, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(raw, gotBody); err != nil {
				t.Errorf("decoding request: %v", err)
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func collect(t *testing.T, srv *httptest.Server) ([]openai.ChatChunk, error) {
	t.Helper()
	client := New(srv.URL, srv.URL, 5*time.Second)
	var chunks []openai.ChatChunk
	err := client.ChatStream(context.Background(), openai.ChatRequest{Model: "m"}, func(chunk openai.ChatChunk) error {
		chunks = append(chunks, chunk)
		return nil
	})
	return chunks, err
}

func TestChatStreamParsesChunksUntilDone(t *testing.T) {
	body := ": keep-alive\n\n" +
		`data: {"choices":[{"delta":{"role":"assistant","content":"he"}}]}` + "\n\n" +
		`data:{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
		`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}` + "\n\n" +
		"data: [DONE]\n\n"
	var request map[string]any
	srv := sseServer(t, http.StatusOK, body, &request)

	chunks, err := collect(t, srv)
	if err != nil {
		t.Fatal(err)
	}

	if request["stream"] != true {
		t.Errorf("stream = %v, want true", request["stream"])
	}
	if opts, _ := request["stream_options"].(map[string]any); opts["include_usage"] != true {
		t.Errorf("stream_options = %v, want include_usage true", request["stream_options"])
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Choices[0].Delta.Content != "he" {
		t.Errorf("first chunk = %+v", chunks[0])
	}
	if call := chunks[1].Choices[0].Delta.ToolCalls[0]; call.ID != "call_1" || call.Function.Arguments != "{}" {
		t.Errorf("tool call = %+v", call)
	}
	if chunks[2].Usage == nil || chunks[2].Usage.TotalTokens != 3 {
		t.Errorf("usage chunk = %+v", chunks[2])
	}
}

func TestChatStreamBackendErrorStatus(t *testing.T) {
	srv := sseServer(t, http.StatusBadRequest, `{"error":{"message":"bad thinking"}}`, nil)

	chunks, err := collect(t, srv)

	if err == nil || !strings.Contains(err.Error(), "400") || !strings.Contains(err.Error(), "bad thinking") {
		t.Errorf("err = %v, want the backend status and body", err)
	}
	if len(chunks) != 0 {
		t.Errorf("chunks = %+v, want none", chunks)
	}
}

func TestChatStreamWithoutDoneIsAnError(t *testing.T) {
	srv := sseServer(t, http.StatusOK, `data: {"choices":[{"delta":{"content":"he"}}]}`+"\n\n", nil)

	_, err := collect(t, srv)

	if err == nil {
		t.Fatal("expected an error for a stream that ended before [DONE]")
	}
}

func TestChatStreamStopsOnCallbackError(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"a"}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"content":"b"}}]}` + "\n\n" +
		"data: [DONE]\n\n"
	srv := sseServer(t, http.StatusOK, body, nil)
	client := New(srv.URL, srv.URL, 5*time.Second)
	stop := errors.New("client went away")

	calls := 0
	err := client.ChatStream(context.Background(), openai.ChatRequest{Model: "m"}, func(openai.ChatChunk) error {
		calls++
		return stop
	})

	if !errors.Is(err, stop) {
		t.Errorf("err = %v, want the callback error", err)
	}
	if calls != 1 {
		t.Errorf("callback calls = %d, want 1", calls)
	}
}

func TestChatStreamReadsEventsOfAnySize(t *testing.T) {
	arguments := `{"text":"` + strings.Repeat("a", 2<<20) + `"}`
	encoded, _ := json.Marshal(arguments)
	body := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":` +
		string(encoded) + `}}]}}]}` + "\n\n" + "data: [DONE]\n\n"
	srv := sseServer(t, http.StatusOK, body, nil)

	chunks, err := collect(t, srv)
	if err != nil {
		t.Fatal(err)
	}
	if got := chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments; got != arguments {
		t.Errorf("arguments length = %d, want %d", len(got), len(arguments))
	}
}

func TestAuthRefusalReachesTheCaller(t *testing.T) {
	refusal := `{"error":{"message":"invalid api key"}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, refusal)
	}))
	t.Cleanup(srv.Close)
	client := New(srv.URL, srv.URL, 5*time.Second)

	_, err := client.Chat(context.Background(), openai.ChatRequest{Model: "m"})

	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("err = %v, want the backend status and message", err)
	}
}
