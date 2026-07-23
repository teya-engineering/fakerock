package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/saltpay/fakerock/internal/bedrock"
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
		"messages", len(chatReq.Messages), "tools", len(chatReq.Tools))

	start := time.Now()
	chatResp, err := s.backend.Chat(r.Context(), chatReq)
	if err != nil {
		slog.Error("backend call failed", "model", model, "err", err)
		writeError(w, http.StatusBadGateway, errModel, err.Error())
		return
	}

	resp, err := translate.FromOpenAI(chatResp, time.Since(start))
	if err != nil {
		slog.Error("translating backend response failed", "model", model, "err", err)
		writeError(w, http.StatusBadGateway, errModel, err.Error())
		return
	}

	// Nothing has been written yet, so failures up to here can still be reported as a
	// normal AWS error. Once the first frame goes out the status is fixed at 200.
	w.Header().Set("Content-Type", eventStreamContentType)
	w.WriteHeader(http.StatusOK)

	if err := writeEvents(w, translate.StreamEvents(resp)); err != nil {
		slog.Error("writing event stream failed", "model", s.model, "err", err)
	}
}

func writeEvents(w http.ResponseWriter, events []bedrock.Event) error {
	encoder := eventstream.NewEncoder()
	flusher, _ := w.(http.Flusher)

	for _, event := range events {
		if err := writeEvent(w, encoder, event); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	return nil
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
