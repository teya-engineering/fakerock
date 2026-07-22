package translate

import (
	"errors"
	"testing"
	"time"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

func TestFromOpenAIText(t *testing.T) {
	resp := openai.ChatResponse{
		Choices: []openai.Choice{{
			Message:      openai.Message{Role: "assistant", Content: "hello"},
			FinishReason: openai.FinishReasonStop,
		}},
		Usage: openai.Usage{PromptTokens: 131, CompletionTokens: 49, TotalTokens: 180},
	}

	got, err := FromOpenAI(resp, 250*time.Millisecond)
	if err != nil {
		t.Fatalf("FromOpenAI: %v", err)
	}

	if got.StopReason != bedrock.StopReasonEndTurn {
		t.Errorf("stopReason = %q", got.StopReason)
	}
	if len(got.Output.Message.Content) != 1 || got.Output.Message.Content[0].Text == nil ||
		*got.Output.Message.Content[0].Text != "hello" {
		t.Fatalf("content = %+v", got.Output.Message.Content)
	}
	if got.Usage != (bedrock.Usage{InputTokens: 131, OutputTokens: 49, TotalTokens: 180}) {
		t.Errorf("usage = %+v", got.Usage)
	}
	if got.Metrics.LatencyMs != 250 {
		t.Errorf("latencyMs = %d", got.Metrics.LatencyMs)
	}
}

func TestFromOpenAIToolUse(t *testing.T) {
	resp := openai.ChatResponse{
		Choices: []openai.Choice{{
			Message: openai.Message{
				Role: "assistant",
				ToolCalls: []openai.ToolCall{{
					ID:       "call_e4ulx3h0",
					Type:     "function",
					Function: openai.FunctionCall{Name: "get_weather", Arguments: `{"city":"Lisbon"}`},
				}},
			},
			FinishReason: openai.FinishReasonToolCalls,
		}},
	}

	got, err := FromOpenAI(resp, time.Second)
	if err != nil {
		t.Fatalf("FromOpenAI: %v", err)
	}

	if got.StopReason != bedrock.StopReasonToolUse {
		t.Errorf("stopReason = %q", got.StopReason)
	}
	if len(got.Output.Message.Content) != 1 {
		t.Fatalf("content = %+v", got.Output.Message.Content)
	}
	use := got.Output.Message.Content[0].ToolUse
	if use == nil || use.ToolUseID != "call_e4ulx3h0" || use.Name != "get_weather" {
		t.Fatalf("toolUse = %+v", use)
	}
	if string(use.Input) != `{"city":"Lisbon"}` {
		t.Errorf("input = %s", use.Input)
	}
}

func TestFromOpenAIToolCallsOverrideFinishReason(t *testing.T) {
	resp := openai.ChatResponse{
		Choices: []openai.Choice{{
			Message: openai.Message{
				ToolCalls: []openai.ToolCall{{ID: "call_1", Function: openai.FunctionCall{Name: "f", Arguments: ""}}},
			},
			FinishReason: openai.FinishReasonStop,
		}},
	}

	got, err := FromOpenAI(resp, 0)
	if err != nil {
		t.Fatalf("FromOpenAI: %v", err)
	}
	if got.StopReason != bedrock.StopReasonToolUse {
		t.Errorf("stopReason = %q, want tool_use", got.StopReason)
	}
	if string(got.Output.Message.Content[0].ToolUse.Input) != "{}" {
		t.Errorf("input = %s, want {}", got.Output.Message.Content[0].ToolUse.Input)
	}
}

func TestFromOpenAIStopReasons(t *testing.T) {
	cases := map[string]string{
		openai.FinishReasonLength:        bedrock.StopReasonMaxTokens,
		openai.FinishReasonContentFilter: bedrock.StopReasonFiltered,
		"":                               bedrock.StopReasonEndTurn,
	}

	for finish, want := range cases {
		resp := openai.ChatResponse{Choices: []openai.Choice{{
			Message:      openai.Message{Content: "x"},
			FinishReason: finish,
		}}}
		got, err := FromOpenAI(resp, 0)
		if err != nil {
			t.Fatalf("FromOpenAI(%q): %v", finish, err)
		}
		if got.StopReason != want {
			t.Errorf("finish_reason %q -> %q, want %q", finish, got.StopReason, want)
		}
	}
}

func TestFromOpenAIRejectsInvalidArguments(t *testing.T) {
	resp := openai.ChatResponse{Choices: []openai.Choice{{
		Message: openai.Message{ToolCalls: []openai.ToolCall{{
			ID:       "call_1",
			Function: openai.FunctionCall{Name: "f", Arguments: "not json"},
		}}},
	}}}

	if _, err := FromOpenAI(resp, 0); err == nil {
		t.Fatal("expected an error for non-json tool arguments")
	}
}

func TestFromOpenAINoChoices(t *testing.T) {
	if _, err := FromOpenAI(openai.ChatResponse{}, 0); !errors.Is(err, ErrNoChoices) {
		t.Fatalf("err = %v, want ErrNoChoices", err)
	}
}
