package openai

import "encoding/json"

type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Tools          []Tool          `json:"tools,omitempty"`
	Stream         bool            `json:"stream"`
	StreamOptions  *StreamOptions  `json:"stream_options,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	// Extra holds model-specific fields sent as top-level keys of the request body. A gateway
	// in front of Bedrock, such as LiteLLM, puts them back into additionalModelRequestFields.
	Extra map[string]json.RawMessage `json:"-"`
}

// MarshalJSON appends Extra to the encoded struct. ToOpenAI rejects Extra keys the struct already
// encodes, so the result never holds the same key twice.
func (r ChatRequest) MarshalJSON() ([]byte, error) {
	type chatRequest ChatRequest
	body, err := json.Marshal(chatRequest(r))
	if err != nil || len(r.Extra) == 0 {
		return body, err
	}
	extra, err := json.Marshal(r.Extra)
	if err != nil {
		return nil, err
	}
	// Both are JSON objects: drop the struct's closing brace and Extra's opening one, join with a comma.
	merged := append(body[:len(body)-1], ',')
	return append(merged, extra[1:]...), nil
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ResponseFormat struct {
	Type       string                    `json:"type"`
	JSONSchema *ResponseFormatJSONSchema `json:"json_schema,omitempty"`
}

// ResponseFormatJSONSchema matches OpenAI's structured-output shape: the schema is
// inlined as an object (not a string) and strict=true asks the backend to enforce it.
type ResponseFormatJSONSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict,omitempty"`
}

// Content is a plain string or a []ContentPart for multimodal messages. With any,
// json omits it only when nil, so callers set nil (not "") to drop an empty content.
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ChatResponse struct {
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatChunk is one server-sent event of a streamed chat completion. With include_usage the
// last chunk before [DONE] has no choices and carries Usage. A gateway that fails after the
// stream has started sends Error in place of choices.
type ChatChunk struct {
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
	Error   *StreamError  `json:"error,omitempty"`
}

type ChunkChoice struct {
	Delta        ChunkDelta `json:"delta"`
	FinishReason string     `json:"finish_reason"`
}

type ChunkDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	ToolCalls []ToolCallDelta `json:"tool_calls,omitempty"`
}

// ToolCallDelta carries ID and Function.Name on the first fragment of a call, then
// Function.Arguments in pieces. Index ties the pieces together.
type ToolCallDelta struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Function FunctionCallDelta `json:"function"`
}

type FunctionCallDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type StreamError struct {
	Message string `json:"message"`
}

const (
	FinishReasonStop          = "stop"
	FinishReasonToolCalls     = "tool_calls"
	FinishReasonLength        = "length"
	FinishReasonContentFilter = "content_filter"
)

type EmbeddingRequest struct {
	Model      string `json:"model"`
	Input      string `json:"input"`
	Dimensions *int   `json:"dimensions,omitempty"`
}

type EmbeddingResponse struct {
	Data  []EmbeddingData `json:"data"`
	Usage Usage           `json:"usage"`
}

type EmbeddingData struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}
