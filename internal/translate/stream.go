package translate

import (
	"github.com/saltpay/fakerock/internal/bedrock"
)

const textChunkRunes = 40

// StreamEvents replays an already-complete response as the event sequence a streaming
// client expects. Time-to-first-token is that of the whole generation; every other
// observable detail matches a real stream.
func StreamEvents(resp bedrock.ConverseResponse) []bedrock.Event {
	events := []bedrock.Event{
		{Type: bedrock.EventMessageStart, Payload: bedrock.MessageStart{Role: resp.Output.Message.Role}},
	}

	for index, block := range resp.Output.Message.Content {
		switch {
		case block.Text != nil:
			for _, chunk := range chunkText(*block.Text, textChunkRunes) {
				events = append(events, bedrock.Event{
					Type: bedrock.EventContentBlockDelta,
					Payload: bedrock.ContentBlockDelta{
						ContentBlockIndex: index,
						Delta:             bedrock.Delta{Text: chunk},
					},
				})
			}
		case block.ToolUse != nil:
			events = append(events,
				bedrock.Event{
					Type: bedrock.EventContentBlockStart,
					Payload: bedrock.ContentBlockStart{
						ContentBlockIndex: index,
						Start: bedrock.BlockStart{ToolUse: &bedrock.ToolUseStart{
							ToolUseID: block.ToolUse.ToolUseID,
							Name:      block.ToolUse.Name,
						}},
					},
				},
				bedrock.Event{
					Type: bedrock.EventContentBlockDelta,
					Payload: bedrock.ContentBlockDelta{
						ContentBlockIndex: index,
						Delta: bedrock.Delta{ToolUse: &bedrock.ToolUseDelta{
							Input: string(block.ToolUse.Input),
						}},
					},
				},
			)
		default:
			continue
		}

		events = append(events, bedrock.Event{
			Type:    bedrock.EventContentBlockStop,
			Payload: bedrock.ContentBlockStop{ContentBlockIndex: index},
		})
	}

	return append(events,
		bedrock.Event{Type: bedrock.EventMessageStop, Payload: bedrock.MessageStop{StopReason: resp.StopReason}},
		bedrock.Event{Type: bedrock.EventMetadata, Payload: bedrock.Metadata{Usage: resp.Usage, Metrics: resp.Metrics}},
	)
}

func chunkText(text string, size int) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	var chunks []string
	for start := 0; start < len(runes); start += size {
		end := min(start+size, len(runes))
		chunks = append(chunks, string(runes[start:end]))
	}
	return chunks
}
