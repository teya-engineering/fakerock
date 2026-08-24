package translate

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

func text(s string) *string { return &s }

func TestToOpenAISystemAndText(t *testing.T) {
	req := bedrock.ConverseRequest{
		System: []bedrock.SystemBlock{
			{Text: text("be terse")},
			{CachePoint: &bedrock.CachePoint{Type: "default"}},
		},
		Messages: []bedrock.Message{
			{Role: "user", Content: []bedrock.ContentBlock{
				{Text: text("hello")},
				{Text: text("world")},
				{CachePoint: &bedrock.CachePoint{Type: "default"}},
			}},
		},
	}

	got, err := ToOpenAI("qwen3", req)
	if err != nil {
		t.Fatalf("ToOpenAI: %v", err)
	}

	want := []openai.Message{
		{Role: "system", Content: "be terse"},
		{Role: "user", Content: "hello\nworld"},
	}
	if len(got.Messages) != len(want) {
		t.Fatalf("messages = %d, want %d: %+v", len(got.Messages), len(want), got.Messages)
	}
	for i := range want {
		if got.Messages[i].Role != want[i].Role || got.Messages[i].Content != want[i].Content {
			t.Errorf("messages[%d] = %+v, want %+v", i, got.Messages[i], want[i])
		}
	}
	if got.Model != "qwen3" {
		t.Errorf("model = %q, want qwen3", got.Model)
	}
}

func TestToOpenAIImageBlock(t *testing.T) {
	// base64 of "hello" is "aGVsbG8=". Every Bedrock format maps to image/<format>.
	for _, format := range []string{"png", "jpeg", "gif", "webp"} {
		req := bedrock.ConverseRequest{
			Messages: []bedrock.Message{
				{Role: "user", Content: []bedrock.ContentBlock{
					{Text: text("what is in this image?")},
					{Image: &bedrock.ImageBlock{Format: format, Source: bedrock.ImageSource{Bytes: []byte("hello")}}},
				}},
			},
		}

		got, err := ToOpenAI("qwen3", req)
		if err != nil {
			t.Fatalf("ToOpenAI(%s): %v", format, err)
		}
		if len(got.Messages) != 1 {
			t.Fatalf("%s: messages = %d, want 1: %+v", format, len(got.Messages), got.Messages)
		}

		parts, ok := got.Messages[0].Content.([]openai.ContentPart)
		if !ok {
			t.Fatalf("%s: content = %T, want []openai.ContentPart", format, got.Messages[0].Content)
		}
		if len(parts) != 2 {
			t.Fatalf("%s: parts = %d, want 2: %+v", format, len(parts), parts)
		}
		if parts[0].Type != "text" || parts[0].Text != "what is in this image?" {
			t.Errorf("%s: parts[0] = %+v", format, parts[0])
		}
		want := "data:image/" + format + ";base64,aGVsbG8="
		if parts[1].Type != "image_url" || parts[1].ImageURL == nil || parts[1].ImageURL.URL != want {
			t.Errorf("%s: parts[1] = %+v, want url %q", format, parts[1], want)
		}
	}
}

func TestToOpenAIToolRoundTrip(t *testing.T) {
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{
			{Role: "user", Content: []bedrock.ContentBlock{{Text: text("weather in Lisbon?")}}},
			{Role: "assistant", Content: []bedrock.ContentBlock{
				{Text: text("checking")},
				{ToolUse: &bedrock.ToolUse{ToolUseID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"Lisbon"}`)}},
			}},
			{Role: "user", Content: []bedrock.ContentBlock{
				{ToolResult: &bedrock.ToolResult{ToolUseID: "call_1", Content: []bedrock.ToolResultBlock{{Text: text("21C")}}}},
				{Text: text("and tomorrow?")},
			}},
		},
		ToolConfig: &bedrock.ToolConfig{Tools: []bedrock.Tool{
			{ToolSpec: &bedrock.ToolSpec{
				Name:        "get_weather",
				Description: "weather for a city",
				InputSchema: bedrock.InputSchema{JSON: json.RawMessage(`{"type":"object"}`)},
			}},
		}},
	}

	got, err := ToOpenAI("qwen3", req)
	if err != nil {
		t.Fatalf("ToOpenAI: %v", err)
	}

	if len(got.Messages) != 4 {
		t.Fatalf("messages = %d, want 4: %+v", len(got.Messages), got.Messages)
	}

	assistant := got.Messages[1]
	if assistant.Role != "assistant" || assistant.Content != "checking" || len(assistant.ToolCalls) != 1 {
		t.Fatalf("assistant message = %+v", assistant)
	}
	call := assistant.ToolCalls[0]
	if call.ID != "call_1" || call.Type != "function" || call.Function.Name != "get_weather" {
		t.Errorf("tool call = %+v", call)
	}
	if call.Function.Arguments != `{"city":"Lisbon"}` {
		t.Errorf("arguments = %q", call.Function.Arguments)
	}

	// The tool result must precede the follow-up user text.
	if toolMsg := got.Messages[2]; toolMsg.Role != "tool" || toolMsg.ToolCallID != "call_1" || toolMsg.Content != "21C" {
		t.Errorf("tool message = %+v", toolMsg)
	}
	if userMsg := got.Messages[3]; userMsg.Role != "user" || userMsg.Content != "and tomorrow?" {
		t.Errorf("user message = %+v", userMsg)
	}

	if len(got.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(got.Tools))
	}
	if got.Tools[0].Function.Name != "get_weather" || string(got.Tools[0].Function.Parameters) != `{"type":"object"}` {
		t.Errorf("tool = %+v", got.Tools[0])
	}
}

func TestToOpenAIToolCallOnlyOmitsContent(t *testing.T) {
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{
			{Role: "assistant", Content: []bedrock.ContentBlock{
				{ToolUse: &bedrock.ToolUse{ToolUseID: "call_1", Name: "ping", Input: json.RawMessage(`{}`)}},
			}},
		},
	}

	got, err := ToOpenAI("qwen3", req)
	if err != nil {
		t.Fatalf("ToOpenAI: %v", err)
	}
	if got.Messages[0].Content != nil {
		t.Fatalf("content = %#v, want nil", got.Messages[0].Content)
	}

	// With Content set to any, only a nil interface is dropped by omitempty.
	raw, err := json.Marshal(got.Messages[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(raw, []byte(`"content"`)) {
		t.Errorf("content must be omitted on the wire: %s", raw)
	}
}

func TestToOpenAIToolResultJSONContent(t *testing.T) {
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{
			{Role: "user", Content: []bedrock.ContentBlock{
				{ToolResult: &bedrock.ToolResult{
					ToolUseID: "call_1",
					Content:   []bedrock.ToolResultBlock{{JSON: json.RawMessage("{\n  \"tempC\": 21\n}")}},
				}},
			}},
		},
	}

	got, err := ToOpenAI("qwen3", req)
	if err != nil {
		t.Fatalf("ToOpenAI: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != `{"tempC":21}` {
		t.Fatalf("messages = %+v", got.Messages)
	}
}

func TestToOpenAIInferenceConfig(t *testing.T) {
	maxTokens, temperature := 512, 0.0
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{{Text: text("hi")}}}},
		InferenceConfig: &bedrock.InferenceConfig{
			MaxTokens:     &maxTokens,
			Temperature:   &temperature,
			StopSequences: []string{"STOP"},
		},
	}

	got, err := ToOpenAI("qwen3", req)
	if err != nil {
		t.Fatalf("ToOpenAI: %v", err)
	}
	if got.MaxTokens == nil || *got.MaxTokens != 512 {
		t.Errorf("max_tokens = %v", got.MaxTokens)
	}
	if got.Temperature == nil || *got.Temperature != 0 {
		t.Errorf("temperature = %v", got.Temperature)
	}
	if len(got.Stop) != 1 || got.Stop[0] != "STOP" {
		t.Errorf("stop = %v", got.Stop)
	}
}

func TestToOpenAIRejectsUnnamedTool(t *testing.T) {
	req := bedrock.ConverseRequest{
		Messages:   []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{{Text: text("hi")}}}},
		ToolConfig: &bedrock.ToolConfig{Tools: []bedrock.Tool{{ToolSpec: &bedrock.ToolSpec{}}}},
	}

	if _, err := ToOpenAI("qwen3", req); err == nil {
		t.Fatal("expected an error for a tool spec without a name")
	}
}

func TestToOpenAIOutputConfigJSONSchema(t *testing.T) {
	// Bedrock's JsonSchemaDefinition.schema is a String — the SDK escapes the actual
	// JSON schema tree into a JSON string. OpenAI's response_format.json_schema.schema
	// wants the tree inline as an object, so translation has to unwrap the string.
	const schema = `{"type":"object","properties":{"verdict":{"type":"string","enum":["TP","FP"]}},"required":["verdict"],"additionalProperties":false}`
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{{Text: text("hi")}}}},
		OutputConfig: &bedrock.OutputConfig{
			TextFormat: &bedrock.TextFormat{
				Type: "json_schema",
				Structure: &bedrock.TextFormatStructure{
					JSONSchema: &bedrock.JSONSchemaDefinition{
						Schema: schema,
						Name:   "MatchVerdict",
					},
				},
			},
		},
	}

	got, err := ToOpenAI("qwen3", req)
	if err != nil {
		t.Fatalf("ToOpenAI: %v", err)
	}
	if got.ResponseFormat == nil {
		t.Fatal("response_format is nil, want json_schema")
	}
	if got.ResponseFormat.Type != "json_schema" {
		t.Errorf("response_format.type = %q, want json_schema", got.ResponseFormat.Type)
	}
	if got.ResponseFormat.JSONSchema == nil {
		t.Fatal("response_format.json_schema is nil")
	}
	if got.ResponseFormat.JSONSchema.Name != "MatchVerdict" {
		t.Errorf("json_schema.name = %q, want MatchVerdict", got.ResponseFormat.JSONSchema.Name)
	}
	if !got.ResponseFormat.JSONSchema.Strict {
		t.Error("json_schema.strict = false, want true (structured output must be enforced)")
	}
	// Schema must be the parsed tree, not the escaped string. Round-trip through a compact
	// buffer so the assertion is insensitive to key ordering / whitespace differences.
	var gotSchema, wantSchema bytes.Buffer
	if err := json.Compact(&gotSchema, got.ResponseFormat.JSONSchema.Schema); err != nil {
		t.Fatalf("json_schema.schema is not valid JSON: %v (%s)", err, got.ResponseFormat.JSONSchema.Schema)
	}
	if err := json.Compact(&wantSchema, []byte(schema)); err != nil {
		t.Fatal(err)
	}
	if gotSchema.String() != wantSchema.String() {
		t.Errorf("json_schema.schema = %s, want %s", gotSchema.String(), wantSchema.String())
	}
}

func TestToOpenAIOutputConfigTextIsNoOp(t *testing.T) {
	// type=text (or absent) is Bedrock's default — the model runs unconstrained. It must
	// not synthesise a response_format, otherwise callers who never asked for structured
	// output would suddenly get JSON-only replies.
	for _, typ := range []string{"", "text"} {
		req := bedrock.ConverseRequest{
			Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{{Text: text("hi")}}}},
			OutputConfig: &bedrock.OutputConfig{
				TextFormat: &bedrock.TextFormat{Type: typ},
			},
		}
		got, err := ToOpenAI("qwen3", req)
		if err != nil {
			t.Fatalf("type=%q: ToOpenAI: %v", typ, err)
		}
		if got.ResponseFormat != nil {
			t.Errorf("type=%q: response_format = %+v, want nil", typ, got.ResponseFormat)
		}
	}
}

func TestToOpenAIOutputConfigRejectsMissingSchema(t *testing.T) {
	// Silent drop would leave the model answering unconstrained while the caller
	// believes structured output is enforced (Invariant #5).
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{{Text: text("hi")}}}},
		OutputConfig: &bedrock.OutputConfig{
			TextFormat: &bedrock.TextFormat{Type: "json_schema"},
		},
	}
	if _, err := ToOpenAI("qwen3", req); err == nil {
		t.Fatal("expected an error when json_schema type has no schema")
	}
}

func TestToOpenAIOutputConfigRejectsInvalidSchemaJSON(t *testing.T) {
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{{Text: text("hi")}}}},
		OutputConfig: &bedrock.OutputConfig{
			TextFormat: &bedrock.TextFormat{
				Type: "json_schema",
				Structure: &bedrock.TextFormatStructure{
					JSONSchema: &bedrock.JSONSchemaDefinition{Schema: "{not valid"},
				},
			},
		},
	}
	if _, err := ToOpenAI("qwen3", req); err == nil {
		t.Fatal("expected an error for malformed schema JSON")
	}
}

func TestToOpenAIOutputConfigRejectsUnknownType(t *testing.T) {
	req := bedrock.ConverseRequest{
		Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{{Text: text("hi")}}}},
		OutputConfig: &bedrock.OutputConfig{
			TextFormat: &bedrock.TextFormat{Type: "yaml_schema"},
		},
	}
	if _, err := ToOpenAI("qwen3", req); err == nil {
		t.Fatal("expected an error for an unknown textFormat type")
	}
}
