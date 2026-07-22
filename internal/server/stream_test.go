package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

type decodedEvent struct {
	eventType string
	payload   map[string]any
}

func decodeEventStream(t *testing.T, body []byte) []decodedEvent {
	t.Helper()
	decoder := eventstream.NewDecoder()
	reader := bytes.NewReader(body)

	var events []decodedEvent
	for {
		message, err := decoder.Decode(reader, nil)
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("decoding frame %d: %v", len(events), err)
		}
		if got := message.Headers.Get(":message-type").String(); got != "event" {
			t.Errorf("frame %d :message-type = %q", len(events), got)
		}
		var payload map[string]any
		if err := json.Unmarshal(message.Payload, &payload); err != nil {
			t.Fatalf("decoding frame %d payload: %v", len(events), err)
		}
		events = append(events, decodedEvent{
			eventType: message.Headers.Get(":event-type").String(),
			payload:   payload,
		})
	}
}

func TestConverseStreamEmitsDecodableFrames(t *testing.T) {
	backend := &stubBackend{resp: openai.ChatResponse{
		Choices: []openai.Choice{{
			Message:      openai.Message{Role: "assistant", Content: "hi there"},
			FinishReason: openai.FinishReasonStop,
		}},
		Usage: openai.Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
	}}
	srv := newTestServer(t, backend)

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("Content-Type"); got != eventStreamContentType {
		t.Errorf("Content-Type = %q, want %q", got, eventStreamContentType)
	}

	events := decodeEventStream(t, rec.Body.Bytes())
	if len(events) != 5 {
		t.Fatalf("events = %d: %+v", len(events), events)
	}
	if events[0].eventType != bedrock.EventMessageStart || events[0].payload["role"] != "assistant" {
		t.Errorf("first event = %+v", events[0])
	}
	if events[1].eventType != bedrock.EventContentBlockDelta {
		t.Errorf("second event = %+v", events[1])
	}
	if delta := events[1].payload["delta"].(map[string]any); delta["text"] != "hi there" {
		t.Errorf("delta = %+v", delta)
	}
	if events[3].payload["stopReason"] != bedrock.StopReasonEndTurn {
		t.Errorf("messageStop = %+v", events[3])
	}
	if events[4].eventType != bedrock.EventMetadata {
		t.Errorf("last event = %+v", events[4])
	}
}

func TestConverseStreamToolUseFrames(t *testing.T) {
	backend := &stubBackend{resp: openai.ChatResponse{
		Choices: []openai.Choice{{
			Message: openai.Message{Role: "assistant", ToolCalls: []openai.ToolCall{{
				ID:       "call_1",
				Type:     "function",
				Function: openai.FunctionCall{Name: "get_weather", Arguments: `{"city":"Lisbon"}`},
			}}},
			FinishReason: openai.FinishReasonToolCalls,
		}},
	}}
	srv := newTestServer(t, backend)

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"weather?"}]}]}`)

	events := decodeEventStream(t, rec.Body.Bytes())
	if len(events) != 6 {
		t.Fatalf("events = %d: %+v", len(events), events)
	}
	start := events[1].payload["start"].(map[string]any)["toolUse"].(map[string]any)
	if start["toolUseId"] != "call_1" || start["name"] != "get_weather" {
		t.Errorf("contentBlockStart = %+v", start)
	}
	delta := events[2].payload["delta"].(map[string]any)["toolUse"].(map[string]any)
	if delta["input"] != `{"city":"Lisbon"}` {
		t.Errorf("input = %v", delta["input"])
	}
	if events[4].payload["stopReason"] != bedrock.StopReasonToolUse {
		t.Errorf("messageStop = %+v", events[4])
	}
}

func TestConverseStreamBackendFailureIsAnAWSError(t *testing.T) {
	srv := newTestServer(t, &stubBackend{err: errors.New("connection refused")})

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)

	assertAWSError(t, rec, http.StatusBadGateway, errModel)
}

func TestConverseStreamRejectsEmptyMessages(t *testing.T) {
	srv := newTestServer(t, &stubBackend{})

	rec := post(t, srv, "/model/sonnet/converse-stream", `{"messages":[]}`)

	assertAWSError(t, rec, http.StatusBadRequest, errValidation)
}
