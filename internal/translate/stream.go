package translate

import (
	"fmt"
	"strings"
	"time"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

// Stream turns streamed OpenAI chunks into Bedrock ConverseStream events as they arrive.
// Bedrock content blocks are strictly sequential, so at most one block is open: a text delta
// after a tool call opens a new text block, and a new tool call id closes whatever was open.
// Text blocks get no contentBlockStart, matching Bedrock.
//
// The zero value is ready to use. Call Chunk for every chunk, then Finish once.
type Stream struct {
	started      bool
	blocks       int // content blocks opened so far; the next block gets this index
	open         blockKind
	toolCallID   string
	toolName     string
	toolIndex    int             // OpenAI tool_calls index of the open tool block
	toolInput    strings.Builder // arguments streamed so far, checked when the block closes
	sawToolCall  bool
	finishReason string
	usage        openai.Usage
}

type blockKind int

const (
	noBlock blockKind = iota
	textBlock
	toolBlock
)

func (s *Stream) Chunk(chunk openai.ChatChunk) ([]bedrock.Event, error) {
	if chunk.Error != nil {
		return nil, fmt.Errorf("backend stream failed: %s", chunk.Error.Message)
	}

	if chunk.Usage != nil {
		s.usage = *chunk.Usage
	}
	// A chunk without choices, such as the usage chunk, starts nothing. Until a choice arrives no
	// event is sent, so a backend that never answers still fails as a normal error.
	if len(chunk.Choices) == 0 {
		return nil, nil
	}
	choice := chunk.Choices[0]
	events := s.start()

	if choice.Delta.Content != "" {
		if s.open != textBlock {
			closing, err := s.closeBlock()
			if err != nil {
				return nil, err
			}
			events = append(events, closing...)
			s.open = textBlock
			s.blocks++
		}
		events = append(events, bedrock.Event{Type: bedrock.EventContentBlockDelta, Payload: bedrock.ContentBlockDelta{
			ContentBlockIndex: s.index(),
			Delta:             bedrock.Delta{Text: choice.Delta.Content},
		}})
	}

	for _, call := range choice.Delta.ToolCalls {
		toolEvents, err := s.toolCall(call)
		if err != nil {
			return nil, err
		}
		events = append(events, toolEvents...)
	}

	if choice.FinishReason != "" {
		s.finishReason = choice.FinishReason
	}
	return events, nil
}

// Finish closes the open block and ends the message. latency is the whole generation. A stream
// that never carried a choice fails with ErrNoChoices, as Converse does.
func (s *Stream) Finish(latency time.Duration) ([]bedrock.Event, error) {
	if !s.started {
		return nil, ErrNoChoices
	}
	events, err := s.closeBlock()
	if err != nil {
		return nil, err
	}
	return append(events,
		bedrock.Event{Type: bedrock.EventMessageStop, Payload: bedrock.MessageStop{
			StopReason: stopReason(s.finishReason, s.sawToolCall),
		}},
		bedrock.Event{Type: bedrock.EventMetadata, Payload: bedrock.Metadata{
			Usage: bedrock.Usage{
				InputTokens:  s.usage.PromptTokens,
				OutputTokens: s.usage.CompletionTokens,
				TotalTokens:  s.usage.TotalTokens,
			},
			Metrics: bedrock.Metrics{LatencyMs: latency.Milliseconds()},
		}},
	), nil
}

// toolCall opens a new block when the delta carries an id the open block does not have.
// Everything else is an arguments fragment for the open tool block, and each fragment
// becomes its own delta so the client sees the input grow.
func (s *Stream) toolCall(call openai.ToolCallDelta) ([]bedrock.Event, error) {
	var events []bedrock.Event

	startsNewCall := call.ID != "" && (s.open != toolBlock || call.ID != s.toolCallID)
	if startsNewCall {
		closing, err := s.closeBlock()
		if err != nil {
			return nil, err
		}
		events = append(events, closing...)
		s.open = toolBlock
		s.blocks++
		s.toolCallID = call.ID
		s.toolName = call.Function.Name
		s.toolIndex = call.Index
		s.toolInput.Reset()
		s.sawToolCall = true
		events = append(events, bedrock.Event{Type: bedrock.EventContentBlockStart, Payload: bedrock.ContentBlockStart{
			ContentBlockIndex: s.index(),
			Start:             bedrock.BlockStart{ToolUse: &bedrock.ToolUseStart{ToolUseID: call.ID, Name: call.Function.Name}},
		}})
	} else if s.open != toolBlock || call.Index != s.toolIndex {
		return nil, fmt.Errorf("backend sent arguments for tool_calls[%d] while it is not the open tool call", call.Index)
	}

	if call.Function.Arguments != "" {
		s.toolInput.WriteString(call.Function.Arguments)
		events = append(events, toolInputDelta(s.index(), call.Function.Arguments))
	}
	return events, nil
}

func (s *Stream) start() []bedrock.Event {
	if s.started {
		return nil
	}
	s.started = true
	return []bedrock.Event{{Type: bedrock.EventMessageStart, Payload: bedrock.MessageStart{Role: "assistant"}}}
}

// closeBlock ends the open block. A tool call's arguments must add up to valid JSON, as
// Converse requires. One that streamed no arguments gets {}, the input Converse would return.
func (s *Stream) closeBlock() ([]bedrock.Event, error) {
	if s.open == noBlock {
		return nil, nil
	}
	var events []bedrock.Event
	if s.open == toolBlock {
		if _, err := toolInput(s.toolInput.String()); err != nil {
			return nil, fmt.Errorf("tool call %s (%s): %w", s.toolCallID, s.toolName, err)
		}
		if s.toolInput.Len() == 0 {
			events = append(events, toolInputDelta(s.index(), "{}"))
		}
	}
	events = append(events, bedrock.Event{Type: bedrock.EventContentBlockStop, Payload: bedrock.ContentBlockStop{ContentBlockIndex: s.index()}})
	s.open = noBlock
	return events, nil
}

func (s *Stream) index() int {
	return s.blocks - 1
}

func toolInputDelta(index int, input string) bedrock.Event {
	return bedrock.Event{Type: bedrock.EventContentBlockDelta, Payload: bedrock.ContentBlockDelta{
		ContentBlockIndex: index,
		Delta:             bedrock.Delta{ToolUse: &bedrock.ToolUseDelta{Input: input}},
	}}
}
