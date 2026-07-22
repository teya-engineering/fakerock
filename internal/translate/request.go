package translate

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

func ToOpenAI(model string, req bedrock.ConverseRequest) (openai.ChatRequest, error) {
	out := openai.ChatRequest{Model: model}

	for _, s := range req.System {
		if s.Text != nil {
			out.Messages = append(out.Messages, openai.Message{Role: "system", Content: *s.Text})
		}
	}

	for i, m := range req.Messages {
		converted, err := convertMessage(m)
		if err != nil {
			return openai.ChatRequest{}, fmt.Errorf("messages[%d]: %w", i, err)
		}
		out.Messages = append(out.Messages, converted...)
	}

	if req.ToolConfig != nil {
		for i, t := range req.ToolConfig.Tools {
			if t.ToolSpec == nil {
				continue
			}
			if t.ToolSpec.Name == "" {
				return openai.ChatRequest{}, fmt.Errorf("toolConfig.tools[%d].toolSpec.name is required", i)
			}
			out.Tools = append(out.Tools, openai.Tool{
				Type: "function",
				Function: openai.Function{
					Name:        t.ToolSpec.Name,
					Description: t.ToolSpec.Description,
					Parameters:  t.ToolSpec.InputSchema.JSON,
				},
			})
		}
	}

	if c := req.InferenceConfig; c != nil {
		out.MaxTokens = c.MaxTokens
		out.Temperature = c.Temperature
		out.TopP = c.TopP
		out.Stop = c.StopSequences
	}

	return out, nil
}

// Tool results become their own messages so they land directly after the assistant
// message holding the matching tool_call_id, which is the order OpenAI requires.
func convertMessage(m bedrock.Message) ([]openai.Message, error) {
	var out []openai.Message
	var texts []string
	var calls []openai.ToolCall

	for _, c := range m.Content {
		switch {
		case c.ToolResult != nil:
			content, err := toolResultContent(*c.ToolResult)
			if err != nil {
				return nil, err
			}
			out = append(out, openai.Message{
				Role:       "tool",
				ToolCallID: c.ToolResult.ToolUseID,
				Content:    content,
			})
		case c.ToolUse != nil:
			args := string(c.ToolUse.Input)
			if args == "" {
				args = "{}"
			}
			calls = append(calls, openai.ToolCall{
				ID:       c.ToolUse.ToolUseID,
				Type:     "function",
				Function: openai.FunctionCall{Name: c.ToolUse.Name, Arguments: args},
			})
		case c.Text != nil:
			texts = append(texts, *c.Text)
		}
	}

	if len(texts) > 0 || len(calls) > 0 {
		out = append(out, openai.Message{
			Role:      m.Role,
			Content:   strings.Join(texts, "\n"),
			ToolCalls: calls,
		})
	}

	return out, nil
}

func toolResultContent(r bedrock.ToolResult) (string, error) {
	var parts []string
	for _, b := range r.Content {
		switch {
		case b.Text != nil:
			parts = append(parts, *b.Text)
		case b.JSON != nil:
			compact, err := compactJSON(b.JSON)
			if err != nil {
				return "", fmt.Errorf("toolResult %q: %w", r.ToolUseID, err)
			}
			parts = append(parts, compact)
		}
	}
	return strings.Join(parts, "\n"), nil
}

func compactJSON(raw json.RawMessage) (string, error) {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return "", fmt.Errorf("invalid json content: %w", err)
	}
	return buf.String(), nil
}
