package translate

import (
	"bytes"
	"encoding/base64"
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

	if oc := req.OutputConfig; oc != nil && oc.TextFormat != nil {
		rf, err := translateOutputFormat(oc.TextFormat)
		if err != nil {
			return openai.ChatRequest{}, err
		}
		out.ResponseFormat = rf
	}

	return out, nil
}

// translateOutputFormat converts Bedrock's outputConfig.textFormat into OpenAI's
// response_format. type=text (or absent) is Bedrock's default and stays a no-op so
// callers who never asked for structured output are not silently forced onto it.
// Anything else fails loudly rather than passing an unenforced schema to the model.
func translateOutputFormat(tf *bedrock.TextFormat) (*openai.ResponseFormat, error) {
	switch tf.Type {
	case "", "text":
		return nil, nil
	case "json_schema":
		if tf.Structure == nil || tf.Structure.JSONSchema == nil || tf.Structure.JSONSchema.Schema == "" {
			return nil, fmt.Errorf("outputConfig.textFormat.structure.jsonSchema.schema is required when type is json_schema")
		}
		// Bedrock delivers the schema as an escaped JSON string; OpenAI wants the tree
		// inline as an object. json.RawMessage lets it pass through unquoted, but we
		// validate first so a malformed schema surfaces as a 400 (Invariant #5) instead
		// of a downstream backend parse error.
		schema := tf.Structure.JSONSchema.Schema
		if !json.Valid([]byte(schema)) {
			return nil, fmt.Errorf("outputConfig.textFormat.structure.jsonSchema.schema is not valid JSON")
		}
		name := tf.Structure.JSONSchema.Name
		if name == "" {
			name = "response"
		}
		return &openai.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &openai.ResponseFormatJSONSchema{
				Name:   name,
				Schema: json.RawMessage(schema),
				Strict: true,
			},
		}, nil
	default:
		return nil, fmt.Errorf("outputConfig.textFormat.type %q not supported", tf.Type)
	}
}

// Tool results become their own messages so they land directly after the assistant
// message holding the matching tool_call_id, which is the order OpenAI requires.
func convertMessage(m bedrock.Message) ([]openai.Message, error) {
	var out []openai.Message
	var texts []string
	var images []bedrock.ImageBlock
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
		case c.Image != nil:
			images = append(images, *c.Image)
		}
	}

	if len(texts) > 0 || len(images) > 0 || len(calls) > 0 {
		out = append(out, openai.Message{
			Role:      m.Role,
			Content:   messageContent(texts, images),
			ToolCalls: calls,
		})
	}

	return out, nil
}

// messageContent returns a plain string when there are no images, the multimodal
// []ContentPart array when images are present, and nil when neither exists so json
// omits the field on tool-call-only messages.
func messageContent(texts []string, images []bedrock.ImageBlock) any {
	if len(images) > 0 {
		parts := make([]openai.ContentPart, 0, len(texts)+len(images))
		for _, t := range texts {
			parts = append(parts, openai.ContentPart{Type: "text", Text: t})
		}
		for _, img := range images {
			url := fmt.Sprintf("data:image/%s;base64,%s", img.Format, base64.StdEncoding.EncodeToString(img.Source.Bytes))
			parts = append(parts, openai.ContentPart{Type: "image_url", ImageURL: &openai.ImageURL{URL: url}})
		}
		return parts
	}
	if len(texts) > 0 {
		return strings.Join(texts, "\n")
	}
	return nil
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
