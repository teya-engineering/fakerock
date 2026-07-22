package translate

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

var ErrNoChoices = errors.New("backend returned no choices")

func FromOpenAI(resp openai.ChatResponse, latency time.Duration) (bedrock.ConverseResponse, error) {
	if len(resp.Choices) == 0 {
		return bedrock.ConverseResponse{}, ErrNoChoices
	}
	choice := resp.Choices[0]

	content := []bedrock.ContentBlock{}
	if choice.Message.Content != "" {
		text := choice.Message.Content
		content = append(content, bedrock.ContentBlock{Text: &text})
	}

	for i, call := range choice.Message.ToolCalls {
		input, err := toolInput(call.Function.Arguments)
		if err != nil {
			return bedrock.ConverseResponse{}, fmt.Errorf("tool_calls[%d] (%s): %w", i, call.Function.Name, err)
		}
		content = append(content, bedrock.ContentBlock{ToolUse: &bedrock.ToolUse{
			ToolUseID: call.ID,
			Name:      call.Function.Name,
			Input:     input,
		}})
	}

	return bedrock.ConverseResponse{
		Output:     bedrock.Output{Message: bedrock.Message{Role: "assistant", Content: content}},
		StopReason: stopReason(choice),
		Usage: bedrock.Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		},
		Metrics: bedrock.Metrics{LatencyMs: latency.Milliseconds()},
	}, nil
}

// Tool calls win over finish_reason: some backends report "stop" while still emitting
// tool calls, and a client that trusts end_turn there would drop out of the agent loop.
func stopReason(choice openai.Choice) string {
	if len(choice.Message.ToolCalls) > 0 {
		return bedrock.StopReasonToolUse
	}
	switch choice.FinishReason {
	case openai.FinishReasonToolCalls:
		return bedrock.StopReasonToolUse
	case openai.FinishReasonLength:
		return bedrock.StopReasonMaxTokens
	case openai.FinishReasonContentFilter:
		return bedrock.StopReasonFiltered
	default:
		return bedrock.StopReasonEndTurn
	}
}

func toolInput(arguments string) (json.RawMessage, error) {
	if arguments == "" {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid([]byte(arguments)) {
		return nil, fmt.Errorf("arguments are not valid json: %q", arguments)
	}
	return json.RawMessage(arguments), nil
}
