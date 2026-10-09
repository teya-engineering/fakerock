package translate

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

func eventTypes(events []bedrock.Event) []string {
	types := make([]string, len(events))
	for i, e := range events {
		types[i] = e.Type
	}
	return types
}

func textChunk(text string) openai.ChatChunk {
	return openai.ChatChunk{Choices: []openai.ChunkChoice{{Delta: openai.ChunkDelta{Content: text}}}}
}

func toolChunk(index int, id, name, arguments string) openai.ChatChunk {
	return openai.ChatChunk{Choices: []openai.ChunkChoice{{Delta: openai.ChunkDelta{ToolCalls: []openai.ToolCallDelta{{
		Index:    index,
		ID:       id,
		Function: openai.FunctionCallDelta{Name: name, Arguments: arguments},
	}}}}}}
}

func finishChunk(reason string) openai.ChatChunk {
	return openai.ChatChunk{Choices: []openai.ChunkChoice{{FinishReason: reason}}}
}

// feed runs every chunk through a fresh Stream and returns all events, including Finish.
func feed(t *testing.T, chunks ...openai.ChatChunk) []bedrock.Event {
	t.Helper()
	var stream Stream
	var events []bedrock.Event
	for i, chunk := range chunks {
		got, err := stream.Chunk(chunk)
		if err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
		events = append(events, got...)
	}
	finished, err := stream.Finish(1500 * time.Millisecond)
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	return append(events, finished...)
}

func TestStreamTextThenTwoToolCalls(t *testing.T) {
	events := feed(t,
		openai.ChatChunk{Choices: []openai.ChunkChoice{{Delta: openai.ChunkDelta{Role: "assistant"}}}},
		textChunk("Let me "),
		textChunk("check."),
		toolChunk(0, "call_a", "get_weather", ""),
		toolChunk(0, "", "", `{"city":`),
		toolChunk(0, "", "", `"Lisbon"}`),
		toolChunk(1, "call_b", "get_time", `{"zone":`),
		toolChunk(1, "", "", `"Europe/`),
		toolChunk(1, "", "", `Lisbon"}`),
		finishChunk(openai.FinishReasonToolCalls),
		openai.ChatChunk{Usage: &openai.Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30}},
	)

	want := []bedrock.Event{
		{Type: bedrock.EventMessageStart, Payload: bedrock.MessageStart{Role: "assistant"}},
		textDelta(0, "Let me "),
		textDelta(0, "check."),
		{Type: bedrock.EventContentBlockStop, Payload: bedrock.ContentBlockStop{ContentBlockIndex: 0}},
		toolStart(1, "call_a", "get_weather"),
		toolDelta(1, `{"city":`),
		toolDelta(1, `"Lisbon"}`),
		{Type: bedrock.EventContentBlockStop, Payload: bedrock.ContentBlockStop{ContentBlockIndex: 1}},
		toolStart(2, "call_b", "get_time"),
		toolDelta(2, `{"zone":`),
		toolDelta(2, `"Europe/`),
		toolDelta(2, `Lisbon"}`),
		{Type: bedrock.EventContentBlockStop, Payload: bedrock.ContentBlockStop{ContentBlockIndex: 2}},
		{Type: bedrock.EventMessageStop, Payload: bedrock.MessageStop{StopReason: bedrock.StopReasonToolUse}},
		{Type: bedrock.EventMetadata, Payload: bedrock.Metadata{
			Usage:   bedrock.Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
			Metrics: bedrock.Metrics{LatencyMs: 1500},
		}},
	}
	assertEvents(t, events, want)
}

func TestStreamTextOnly(t *testing.T) {
	events := feed(t, textChunk("hi "), textChunk("there"), finishChunk(openai.FinishReasonStop))

	want := []string{
		bedrock.EventMessageStart,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockStop,
		bedrock.EventMessageStop,
		bedrock.EventMetadata,
	}
	if got := eventTypes(events); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("events = %v, want %v", got, want)
	}
	if stop := events[4].Payload.(bedrock.MessageStop); stop.StopReason != bedrock.StopReasonEndTurn {
		t.Errorf("stopReason = %q", stop.StopReason)
	}
}

func TestStreamTextAfterToolCallOpensNewBlock(t *testing.T) {
	events := feed(t,
		toolChunk(0, "call_a", "f", `{}`),
		textChunk("done"),
	)

	var stops []int
	for _, e := range events {
		if stop, ok := e.Payload.(bedrock.ContentBlockStop); ok {
			stops = append(stops, stop.ContentBlockIndex)
		}
	}
	if len(stops) != 2 || stops[0] != 0 || stops[1] != 1 {
		t.Errorf("contentBlockStop indexes = %v, want [0 1]", stops)
	}
}

// A backend that reports finish_reason "stop" alongside tool calls must still end the turn
// with tool_use, or the client drops out of its agent loop.
func TestStreamToolCallsWinOverFinishReasonStop(t *testing.T) {
	events := feed(t, toolChunk(0, "call_a", "f", `{}`), finishChunk(openai.FinishReasonStop))

	stop := events[len(events)-2].Payload.(bedrock.MessageStop)
	if stop.StopReason != bedrock.StopReasonToolUse {
		t.Errorf("stopReason = %q, want tool_use", stop.StopReason)
	}
}

func TestStreamToolCallWithoutArgumentsGetsEmptyObject(t *testing.T) {
	events := feed(t, toolChunk(0, "call_a", "list_stores", ""))

	want := []bedrock.Event{
		{Type: bedrock.EventMessageStart, Payload: bedrock.MessageStart{Role: "assistant"}},
		toolStart(0, "call_a", "list_stores"),
		toolDelta(0, `{}`),
		{Type: bedrock.EventContentBlockStop, Payload: bedrock.ContentBlockStop{ContentBlockIndex: 0}},
	}
	assertEvents(t, events[:len(want)], want)
}

// Some backends repeat the id on every fragment of the same call.
func TestStreamRepeatedToolCallIDContinuesTheBlock(t *testing.T) {
	events := feed(t,
		toolChunk(0, "call_a", "f", `{"a":`),
		toolChunk(0, "call_a", "", `1}`),
	)

	starts := 0
	for _, e := range events {
		if e.Type == bedrock.EventContentBlockStart {
			starts++
		}
	}
	if starts != 1 {
		t.Errorf("contentBlockStart events = %d, want 1", starts)
	}
}

func TestStreamRejectsArgumentsForAClosedToolCall(t *testing.T) {
	var stream Stream
	for _, chunk := range []openai.ChatChunk{
		toolChunk(0, "call_a", "f", `{"a":1}`),
		toolChunk(1, "call_b", "g", `{}`),
	} {
		if _, err := stream.Chunk(chunk); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if _, err := stream.Chunk(toolChunk(0, "", "", `,"b":2}`)); err == nil {
		t.Fatal("expected an error for arguments sent to a tool call that is already closed")
	}
}

// Converse rejects tool arguments that are not JSON, so the stream must too.
func TestStreamRejectsInvalidToolArguments(t *testing.T) {
	var stream Stream
	if _, err := stream.Chunk(toolChunk(0, "call_a", "get_weather", `{"city":`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := stream.Finish(time.Second); err == nil || !strings.Contains(err.Error(), "get_weather") {
		t.Errorf("err = %v, want an error naming the tool", err)
	}
}

func TestStreamRejectsInvalidToolArgumentsWhenTheNextBlockOpens(t *testing.T) {
	var stream Stream
	if _, err := stream.Chunk(toolChunk(0, "call_a", "get_weather", `{"city":`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := stream.Chunk(textChunk("done")); err == nil {
		t.Error("expected an error for the unfinished arguments of the closed tool call")
	}
}

func TestStreamGatewayErrorChunk(t *testing.T) {
	var stream Stream

	_, err := stream.Chunk(openai.ChatChunk{Error: &openai.StreamError{Message: "model overloaded"}})

	if err == nil || !strings.Contains(err.Error(), "model overloaded") {
		t.Errorf("err = %v, want the gateway's message", err)
	}
}

// Converse fails with ErrNoChoices when the backend answers without choices, so the stream must
// too, and before any event so the caller still gets a normal error.
func TestStreamWithoutChoicesFails(t *testing.T) {
	var stream Stream
	events, err := stream.Chunk(openai.ChatChunk{Usage: &openai.Usage{PromptTokens: 3, TotalTokens: 3}})
	if err != nil || len(events) != 0 {
		t.Fatalf("usage-only chunk gave events %v, err %v; want none", eventTypes(events), err)
	}

	if _, err := stream.Finish(time.Second); !errors.Is(err, ErrNoChoices) {
		t.Errorf("err = %v, want ErrNoChoices", err)
	}
}

func TestStreamMapsLengthToMaxTokens(t *testing.T) {
	events := feed(t, textChunk("cut"), finishChunk(openai.FinishReasonLength))

	if stop := events[len(events)-2].Payload.(bedrock.MessageStop); stop.StopReason != bedrock.StopReasonMaxTokens {
		t.Errorf("stopReason = %q, want max_tokens", stop.StopReason)
	}
}

func textDelta(index int, text string) bedrock.Event {
	return bedrock.Event{Type: bedrock.EventContentBlockDelta, Payload: bedrock.ContentBlockDelta{
		ContentBlockIndex: index,
		Delta:             bedrock.Delta{Text: text},
	}}
}

func toolStart(index int, id, name string) bedrock.Event {
	return bedrock.Event{Type: bedrock.EventContentBlockStart, Payload: bedrock.ContentBlockStart{
		ContentBlockIndex: index,
		Start:             bedrock.BlockStart{ToolUse: &bedrock.ToolUseStart{ToolUseID: id, Name: name}},
	}}
}

func toolDelta(index int, input string) bedrock.Event {
	return bedrock.Event{Type: bedrock.EventContentBlockDelta, Payload: bedrock.ContentBlockDelta{
		ContentBlockIndex: index,
		Delta:             bedrock.Delta{ToolUse: &bedrock.ToolUseDelta{Input: input}},
	}}
}

func assertEvents(t *testing.T, got, want []bedrock.Event) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d events %v, want %d %v", len(got), eventTypes(got), len(want), eventTypes(want))
	}
	for i := range want {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Errorf("event %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}
