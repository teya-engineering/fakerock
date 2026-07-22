package bedrock

const (
	EventMessageStart      = "messageStart"
	EventContentBlockStart = "contentBlockStart"
	EventContentBlockDelta = "contentBlockDelta"
	EventContentBlockStop  = "contentBlockStop"
	EventMessageStop       = "messageStop"
	EventMetadata          = "metadata"
)

type Event struct {
	Type    string
	Payload any
}

type MessageStart struct {
	Role string `json:"role"`
}

type ContentBlockStart struct {
	ContentBlockIndex int        `json:"contentBlockIndex"`
	Start             BlockStart `json:"start"`
}

type BlockStart struct {
	ToolUse *ToolUseStart `json:"toolUse,omitempty"`
}

type ToolUseStart struct {
	ToolUseID string `json:"toolUseId"`
	Name      string `json:"name"`
}

type ContentBlockDelta struct {
	ContentBlockIndex int   `json:"contentBlockIndex"`
	Delta             Delta `json:"delta"`
}

type Delta struct {
	Text    string        `json:"text,omitempty"`
	ToolUse *ToolUseDelta `json:"toolUse,omitempty"`
}

// Input arrives as a JSON string, matching how Bedrock streams tool arguments.
type ToolUseDelta struct {
	Input string `json:"input"`
}

type ContentBlockStop struct {
	ContentBlockIndex int `json:"contentBlockIndex"`
}

type MessageStop struct {
	StopReason string `json:"stopReason"`
}

type Metadata struct {
	Usage   Usage   `json:"usage"`
	Metrics Metrics `json:"metrics"`
}
