package bedrock

import (
	"encoding/json"
	"fmt"
	"slices"
)

type ConverseRequest struct {
	Messages        []Message        `json:"messages"`
	System          []SystemBlock    `json:"system"`
	InferenceConfig *InferenceConfig `json:"inferenceConfig"`
	ToolConfig      *ToolConfig      `json:"toolConfig"`
}

type Message struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

type ContentBlock struct {
	Text       *string     `json:"text,omitempty"`
	Image      *ImageBlock `json:"image,omitempty"`
	ToolUse    *ToolUse    `json:"toolUse,omitempty"`
	ToolResult *ToolResult `json:"toolResult,omitempty"`
	CachePoint *CachePoint `json:"cachePoint,omitempty"`
}

type ImageBlock struct {
	Format string      `json:"format"`
	Source ImageSource `json:"source"`
}

type ImageSource struct {
	Bytes []byte `json:"bytes"` // encoding/json base64-decodes a JSON string into []byte
}

// Converse keeps growing block kinds (document, video, reasoningContent). Anything we
// cannot translate is rejected rather than dropped, so an unsupported block surfaces as a failed
// request instead of a confident answer about content the model never received.
func (b *ContentBlock) UnmarshalJSON(data []byte) error {
	type contentBlock ContentBlock
	if err := json.Unmarshal(data, (*contentBlock)(b)); err != nil {
		return err
	}
	return rejectUnknownKeys(data, "content block", "text", "image", "toolUse", "toolResult", "cachePoint")
}

type SystemBlock struct {
	Text       *string     `json:"text,omitempty"`
	CachePoint *CachePoint `json:"cachePoint,omitempty"`
}

func (b *SystemBlock) UnmarshalJSON(data []byte) error {
	type systemBlock SystemBlock
	if err := json.Unmarshal(data, (*systemBlock)(b)); err != nil {
		return err
	}
	return rejectUnknownKeys(data, "system block", "text", "cachePoint")
}

func rejectUnknownKeys(data []byte, what string, supported ...string) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	for key := range raw {
		if !slices.Contains(supported, key) {
			return fmt.Errorf("unsupported %s %q", what, key)
		}
	}
	return nil
}

type CachePoint struct {
	Type string `json:"type"`
}

type ToolUse struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type ToolResult struct {
	ToolUseID string            `json:"toolUseId"`
	Content   []ToolResultBlock `json:"content"`
	Status    string            `json:"status,omitempty"`
}

type ToolResultBlock struct {
	Text *string         `json:"text,omitempty"`
	JSON json.RawMessage `json:"json,omitempty"`
}

type InferenceConfig struct {
	MaxTokens     *int     `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type ToolConfig struct {
	Tools []Tool `json:"tools"`
}

type Tool struct {
	ToolSpec *ToolSpec `json:"toolSpec,omitempty"`
}

type ToolSpec struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema InputSchema `json:"inputSchema"`
}

type InputSchema struct {
	JSON json.RawMessage `json:"json"`
}

type ConverseResponse struct {
	Output     Output  `json:"output"`
	StopReason string  `json:"stopReason"`
	Usage      Usage   `json:"usage"`
	Metrics    Metrics `json:"metrics"`
}

type Output struct {
	Message Message `json:"message"`
}

type Usage struct {
	InputTokens           int `json:"inputTokens"`
	OutputTokens          int `json:"outputTokens"`
	TotalTokens           int `json:"totalTokens"`
	CacheReadInputTokens  int `json:"cacheReadInputTokens"`
	CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
}

type Metrics struct {
	LatencyMs int64 `json:"latencyMs"`
}

const (
	StopReasonEndTurn   = "end_turn"
	StopReasonToolUse   = "tool_use"
	StopReasonMaxTokens = "max_tokens"
	StopReasonFiltered  = "content_filtered"
)

type ApplyGuardrailResponse struct {
	Action      string            `json:"action"`
	Outputs     []GuardrailOutput `json:"outputs"`
	Assessments []json.RawMessage `json:"assessments"`
	Usage       GuardrailUsage    `json:"usage"`
}

type GuardrailOutput struct {
	Text string `json:"text"`
}

type GuardrailUsage struct {
	TopicPolicyUnits                    int `json:"topicPolicyUnits"`
	ContentPolicyUnits                  int `json:"contentPolicyUnits"`
	WordPolicyUnits                     int `json:"wordPolicyUnits"`
	SensitiveInformationPolicyUnits     int `json:"sensitiveInformationPolicyUnits"`
	SensitiveInformationPolicyFreeUnits int `json:"sensitiveInformationPolicyFreeUnits"`
	ContextualGroundingPolicyUnits      int `json:"contextualGroundingPolicyUnits"`
}

const GuardrailActionNone = "NONE"

// Titan text embeddings ride the InvokeModel API, a different shape from Converse.
type TitanEmbeddingRequest struct {
	InputText  string `json:"inputText"`
	Dimensions *int   `json:"dimensions,omitempty"`
}

type TitanEmbeddingResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}
