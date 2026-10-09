package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
	"github.com/saltpay/fakerock/internal/translate"
)

const eventStreamContentType = "application/vnd.amazon.eventstream"

func (s *Server) handleConverseStream(w http.ResponseWriter, r *http.Request, modelID string) {
	var req bedrock.ConverseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errValidation, "malformed request body: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, errValidation, "messages is required")
		return
	}

	model := s.currentModel()
	chatReq, err := translate.ToOpenAI(model, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, errValidation, err.Error())
		return
	}

	slog.Info("converse-stream", "modelId", modelID, "model", model,
		"messages", len(chatReq.Messages), "tools", len(chatReq.Tools),
		"additionalFields", slices.Sorted(maps.Keys(chatReq.Extra)))

	var stream translate.Stream
	out := &eventWriter{w: w, encoder: eventstream.NewEncoder()}
	start := time.Now()
	err = s.backend.ChatStream(r.Context(), chatReq, func(chunk openai.ChatChunk) error {
		events, err := stream.Chunk(chunk)
		if err != nil {
			return err
		}
		return out.write(events)
	})
	if err == nil {
		var events []bedrock.Event
		events, err = stream.Finish(time.Since(start))
		if err == nil {
			err = out.write(events)
		}
	}
	if err == nil {
		return
	}

	slog.Error("streaming from backend failed", "model", model, "err", err)
	// After the first frame the status is fixed at 200, so only an exception frame can fail the call.
	if !out.started {
		writeError(w, http.StatusBadGateway, errModel, err.Error())
		return
	}
	if err := out.writeException(errModelStream, err.Error()); err != nil {
		slog.Error("writing stream exception failed", "model", model, "err", err)
	}
}

// eventWriter sends the 200 and the event stream content type with the first frame, so a
// failure before any frame can still go out as a normal AWS error.
type eventWriter struct {
	w       http.ResponseWriter
	encoder *eventstream.Encoder
	started bool
}

// write sends events and flushes, so each backend chunk reaches the client as it arrives.
func (e *eventWriter) write(events []bedrock.Event) error {
	if len(events) == 0 {
		return nil
	}
	e.begin()
	for _, event := range events {
		if err := writeEvent(e.w, e.encoder, event); err != nil {
			return err
		}
	}
	e.flush()
	return nil
}

func (e *eventWriter) writeException(exceptionType, message string) error {
	payload, err := json.Marshal(map[string]string{"message": message})
	if err != nil {
		return fmt.Errorf("encoding %s payload: %w", exceptionType, err)
	}
	e.begin()
	err = e.encoder.Encode(e.w, eventstream.Message{
		Headers: eventstream.Headers{
			{Name: ":message-type", Value: eventstream.StringValue("exception")},
			{Name: ":exception-type", Value: eventstream.StringValue(exceptionType)},
			{Name: ":content-type", Value: eventstream.StringValue("application/json")},
		},
		Payload: payload,
	})
	if err != nil {
		return fmt.Errorf("encoding %s frame: %w", exceptionType, err)
	}
	e.flush()
	return nil
}

func (e *eventWriter) begin() {
	if e.started {
		return
	}
	e.started = true
	e.w.Header().Set("Content-Type", eventStreamContentType)
	e.w.WriteHeader(http.StatusOK)
}

func (e *eventWriter) flush() {
	if flusher, ok := e.w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func writeEvent(w io.Writer, encoder *eventstream.Encoder, event bedrock.Event) error {
	payload, err := json.Marshal(event.Payload)
	if err != nil {
		return fmt.Errorf("encoding %s payload: %w", event.Type, err)
	}

	message := eventstream.Message{
		Headers: eventstream.Headers{
			{Name: ":message-type", Value: eventstream.StringValue("event")},
			{Name: ":event-type", Value: eventstream.StringValue(event.Type)},
			{Name: ":content-type", Value: eventstream.StringValue("application/json")},
		},
		Payload: payload,
	}

	if err := encoder.Encode(w, message); err != nil {
		return fmt.Errorf("encoding %s frame: %w", event.Type, err)
	}
	return nil
}
