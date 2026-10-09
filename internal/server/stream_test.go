package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

type decodedEvent struct {
	messageType   string
	eventType     string
	exceptionType string
	payload       map[string]any
}

func decodeFrame(t *testing.T, decoder *eventstream.Decoder, reader io.Reader) (decodedEvent, error) {
	t.Helper()
	message, err := decoder.Decode(reader, nil)
	if err != nil {
		return decodedEvent{}, err
	}
	var payload map[string]any
	if err := json.Unmarshal(message.Payload, &payload); err != nil {
		t.Fatalf("decoding payload: %v", err)
	}
	header := func(name string) string {
		if value := message.Headers.Get(name); value != nil {
			return value.String()
		}
		return ""
	}
	return decodedEvent{
		messageType:   header(":message-type"),
		eventType:     header(":event-type"),
		exceptionType: header(":exception-type"),
		payload:       payload,
	}, nil
}

func decodeEventStream(t *testing.T, body []byte) []decodedEvent {
	t.Helper()
	decoder := eventstream.NewDecoder()
	reader := bytes.NewReader(body)

	var events []decodedEvent
	for {
		event, err := decodeFrame(t, decoder, reader)
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("decoding frame %d: %v", len(events), err)
		}
		events = append(events, event)
	}
}

func textChunk(text string) openai.ChatChunk {
	return openai.ChatChunk{Choices: []openai.ChunkChoice{{Delta: openai.ChunkDelta{Content: text}}}}
}

func TestConverseStreamEmitsDecodableFrames(t *testing.T) {
	backend := &stubBackend{chunks: []openai.ChatChunk{
		textChunk("hi "),
		textChunk("there"),
		{Choices: []openai.ChunkChoice{{FinishReason: openai.FinishReasonStop}}},
		{Usage: &openai.Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}},
	}}
	srv := newTestServer(t, backend)

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("Content-Type"); got != eventStreamContentType {
		t.Errorf("Content-Type = %q, want %q", got, eventStreamContentType)
	}
	if backend.got.Model != "test-model" {
		t.Errorf("backend model = %q, want test-model", backend.got.Model)
	}

	events := decodeEventStream(t, rec.Body.Bytes())
	var types []string
	for _, e := range events {
		if e.messageType != "event" {
			t.Errorf("%s :message-type = %q", e.eventType, e.messageType)
		}
		types = append(types, e.eventType)
	}
	want := []string{
		bedrock.EventMessageStart,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockStop,
		bedrock.EventMessageStop,
		bedrock.EventMetadata,
	}
	if strings.Join(types, ",") != strings.Join(want, ",") {
		t.Fatalf("events = %v, want %v", types, want)
	}
	if delta := events[2].payload["delta"].(map[string]any); delta["text"] != "there" {
		t.Errorf("delta = %+v", delta)
	}
	if events[4].payload["stopReason"] != bedrock.StopReasonEndTurn {
		t.Errorf("messageStop = %+v", events[4])
	}
	if usage := events[5].payload["usage"].(map[string]any); usage["totalTokens"] != float64(5) {
		t.Errorf("usage = %+v", usage)
	}
}

func TestConverseStreamToolUseFrames(t *testing.T) {
	call := func(id, name, arguments string) openai.ChatChunk {
		return openai.ChatChunk{Choices: []openai.ChunkChoice{{Delta: openai.ChunkDelta{ToolCalls: []openai.ToolCallDelta{{
			ID: id, Function: openai.FunctionCallDelta{Name: name, Arguments: arguments},
		}}}}}}
	}
	backend := &stubBackend{chunks: []openai.ChatChunk{
		call("call_1", "get_weather", ""),
		call("", "", `{"city":`),
		call("", "", `"Lisbon"}`),
		{Choices: []openai.ChunkChoice{{FinishReason: openai.FinishReasonToolCalls}}},
	}}
	srv := newTestServer(t, backend)

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"weather?"}]}]}`)

	events := decodeEventStream(t, rec.Body.Bytes())
	if len(events) != 7 {
		t.Fatalf("events = %d: %+v", len(events), events)
	}
	start := events[1].payload["start"].(map[string]any)["toolUse"].(map[string]any)
	if start["toolUseId"] != "call_1" || start["name"] != "get_weather" {
		t.Errorf("contentBlockStart = %+v", start)
	}
	for i, want := range []string{`{"city":`, `"Lisbon"}`} {
		delta := events[2+i].payload["delta"].(map[string]any)["toolUse"].(map[string]any)
		if delta["input"] != want {
			t.Errorf("delta %d input = %v, want %s", i, delta["input"], want)
		}
	}
	if events[5].payload["stopReason"] != bedrock.StopReasonToolUse {
		t.Errorf("messageStop = %+v", events[5])
	}
}

// gatedBackend sends one chunk, then waits for release before finishing. A handler that
// buffered the stream would deliver nothing until release.
type gatedBackend struct {
	stubBackend
	release chan struct{}
}

func (g *gatedBackend) ChatStream(ctx context.Context, _ openai.ChatRequest, onChunk func(openai.ChatChunk) error) error {
	if err := onChunk(textChunk("first")); err != nil {
		return err
	}
	select {
	case <-g.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	return onChunk(textChunk("second"))
}

func TestConverseStreamFlushesEachChunkAsItArrives(t *testing.T) {
	backend := &gatedBackend{release: make(chan struct{})}
	httpSrv := httptest.NewServer(newTestServer(t, backend))
	t.Cleanup(httpSrv.Close)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(backend.release) }) }
	// Cleanups run last-in first-out, so a failed test unblocks the handler before Close waits on it.
	t.Cleanup(release)

	// The request runs in the goroutine because without a flush even the response headers
	// would not arrive, and the test must time out rather than hang.
	type early struct {
		resp   *http.Response
		events []decodedEvent
	}
	arrived := make(chan early, 1)
	go func() {
		resp, err := http.Post(httpSrv.URL+"/model/sonnet/converse-stream", "application/json",
			strings.NewReader(`{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`))
		if err != nil {
			t.Errorf("posting: %v", err)
			arrived <- early{}
			return
		}
		decoder := eventstream.NewDecoder()
		var events []decodedEvent
		for range 2 {
			event, err := decodeFrame(t, decoder, resp.Body)
			if err != nil {
				t.Errorf("decoding frame: %v", err)
				break
			}
			events = append(events, event)
		}
		arrived <- early{resp: resp, events: events}
	}()

	var resp *http.Response
	select {
	case got := <-arrived:
		resp = got.resp
		if resp == nil {
			t.FailNow()
		}
		t.Cleanup(func() { _ = resp.Body.Close() })
		if len(got.events) != 2 || got.events[1].payload["delta"].(map[string]any)["text"] != "first" {
			t.Fatalf("early frames = %+v", got.events)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no frames arrived while the backend was still generating")
	}
	release()

	rest, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if events := decodeEventStream(t, rest); len(events) != 4 {
		t.Errorf("remaining events = %d: %+v", len(events), events)
	}
}

func TestConverseStreamBackendFailureIsAnAWSError(t *testing.T) {
	srv := newTestServer(t, &stubBackend{err: errors.New("backend returned 400: bad thinking")})

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	assertAWSError(t, rec, http.StatusBadGateway, errModel)
}

func TestConverseStreamWithoutChoicesIsAnAWSError(t *testing.T) {
	srv := newTestServer(t, &stubBackend{chunks: []openai.ChatChunk{
		{Usage: &openai.Usage{PromptTokens: 3, TotalTokens: 3}},
	}})

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	assertAWSError(t, rec, http.StatusBadGateway, errModel)
}

func TestConverseStreamFailureMidStreamSendsException(t *testing.T) {
	srv := newTestServer(t, &stubBackend{
		chunks: []openai.ChatChunk{textChunk("partial")},
		err:    errors.New("backend stream ended without [DONE]"),
	})

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	events := decodeEventStream(t, rec.Body.Bytes())
	last := events[len(events)-1]
	if last.messageType != "exception" || last.exceptionType != errModelStream {
		t.Fatalf("last frame = %+v, want a %s exception", last, errModelStream)
	}
	if !strings.Contains(last.payload["message"].(string), "[DONE]") {
		t.Errorf("exception message = %v", last.payload["message"])
	}
	for _, e := range events {
		if e.eventType == bedrock.EventMessageStop {
			t.Error("a failed stream must not end with messageStop")
		}
	}
}

func TestConverseStreamRejectsEmptyMessages(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[]}`)

	assertAWSError(t, rec, http.StatusBadRequest, errValidation)
}

func TestConverseStreamForwardsAdditionalModelRequestFields(t *testing.T) {
	backend := &stubBackend{chunks: []openai.ChatChunk{textChunk("ok")}}
	srv := newTestServer(t, backend)

	body := `{"messages":[{"role":"user","content":[{"text":"hi"}]}],` +
		`"additionalModelRequestFields":{"output_config":{"effort":"low"}}}`
	rec := post(t, srv, "/model/sonnet/converse-stream", body)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if got := string(backend.got.Extra["output_config"]); got != `{"effort":"low"}` {
		t.Errorf("output_config = %s, want {\"effort\":\"low\"}", got)
	}
}

func TestConverseStreamRejectsConflictingAdditionalModelRequestFields(t *testing.T) {
	backend := &stubBackend{}
	srv := newTestServer(t, backend)

	body := `{"messages":[{"role":"user","content":[{"text":"hi"}]}],` +
		`"additionalModelRequestFields":{"model":"other"}}`
	rec := post(t, srv, "/model/sonnet/converse-stream", body)

	assertAWSError(t, rec, http.StatusBadRequest, errValidation)
	if backend.got.Model != "" {
		t.Error("backend was called for a request that should have been rejected")
	}
}
