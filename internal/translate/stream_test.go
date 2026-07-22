package translate

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/saltpay/fakerock/internal/bedrock"
)

func eventTypes(events []bedrock.Event) []string {
	types := make([]string, len(events))
	for i, e := range events {
		types[i] = e.Type
	}
	return types
}

func TestStreamEventsText(t *testing.T) {
	body := strings.Repeat("a", 95)
	resp := bedrock.ConverseResponse{
		Output:     bedrock.Output{Message: bedrock.Message{Role: "assistant", Content: []bedrock.ContentBlock{{Text: &body}}}},
		StopReason: bedrock.StopReasonEndTurn,
		Usage:      bedrock.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
	}

	events := StreamEvents(resp)

	want := []string{
		bedrock.EventMessageStart,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockStop,
		bedrock.EventMessageStop,
		bedrock.EventMetadata,
	}
	if got := eventTypes(events); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("events = %v, want %v", got, want)
	}

	var rebuilt strings.Builder
	for _, e := range events {
		if delta, ok := e.Payload.(bedrock.ContentBlockDelta); ok {
			rebuilt.WriteString(delta.Delta.Text)
		}
	}
	if rebuilt.String() != body {
		t.Errorf("reassembled text = %q", rebuilt.String())
	}

	if stop := events[len(events)-2].Payload.(bedrock.MessageStop); stop.StopReason != bedrock.StopReasonEndTurn {
		t.Errorf("stopReason = %q", stop.StopReason)
	}
	if meta := events[len(events)-1].Payload.(bedrock.Metadata); meta.Usage.TotalTokens != 3 {
		t.Errorf("metadata usage = %+v", meta.Usage)
	}
}

func TestStreamEventsToolUse(t *testing.T) {
	resp := bedrock.ConverseResponse{
		Output: bedrock.Output{Message: bedrock.Message{Role: "assistant", Content: []bedrock.ContentBlock{
			{ToolUse: &bedrock.ToolUse{
				ToolUseID: "call_1",
				Name:      "get_weather",
				Input:     json.RawMessage(`{"city":"Lisbon"}`),
			}},
		}}},
		StopReason: bedrock.StopReasonToolUse,
	}

	events := StreamEvents(resp)

	want := []string{
		bedrock.EventMessageStart,
		bedrock.EventContentBlockStart,
		bedrock.EventContentBlockDelta,
		bedrock.EventContentBlockStop,
		bedrock.EventMessageStop,
		bedrock.EventMetadata,
	}
	if got := eventTypes(events); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("events = %v, want %v", got, want)
	}

	start := events[1].Payload.(bedrock.ContentBlockStart)
	if start.Start.ToolUse == nil || start.Start.ToolUse.ToolUseID != "call_1" || start.Start.ToolUse.Name != "get_weather" {
		t.Errorf("start = %+v", start)
	}
	delta := events[2].Payload.(bedrock.ContentBlockDelta)
	if delta.Delta.ToolUse == nil || delta.Delta.ToolUse.Input != `{"city":"Lisbon"}` {
		t.Errorf("delta = %+v", delta)
	}
}

func TestStreamEventsIndexesBlocksSeparately(t *testing.T) {
	body := "here"
	resp := bedrock.ConverseResponse{
		Output: bedrock.Output{Message: bedrock.Message{Role: "assistant", Content: []bedrock.ContentBlock{
			{Text: &body},
			{ToolUse: &bedrock.ToolUse{ToolUseID: "call_1", Name: "f", Input: json.RawMessage(`{}`)}},
		}}},
		StopReason: bedrock.StopReasonToolUse,
	}

	events := StreamEvents(resp)

	stops := []int{}
	for _, e := range events {
		if stop, ok := e.Payload.(bedrock.ContentBlockStop); ok {
			stops = append(stops, stop.ContentBlockIndex)
		}
	}
	if len(stops) != 2 || stops[0] != 0 || stops[1] != 1 {
		t.Errorf("contentBlockStop indexes = %v, want [0 1]", stops)
	}
}

func TestStreamEventsEmptyContent(t *testing.T) {
	resp := bedrock.ConverseResponse{
		Output:     bedrock.Output{Message: bedrock.Message{Role: "assistant", Content: []bedrock.ContentBlock{}}},
		StopReason: bedrock.StopReasonEndTurn,
	}

	if got := eventTypes(StreamEvents(resp)); len(got) != 3 {
		t.Errorf("events = %v, want messageStart, messageStop, metadata", got)
	}
}
