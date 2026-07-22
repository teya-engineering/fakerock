package server

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/translate"
)

func (s *Server) handleConverse(w http.ResponseWriter, r *http.Request, modelID string) {
	var req bedrock.ConverseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errValidation, "malformed request body: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, errValidation, "messages is required")
		return
	}

	chatReq, err := translate.ToOpenAI(s.model, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, errValidation, err.Error())
		return
	}

	slog.Info("converse", "modelId", modelID, "model", s.model,
		"messages", len(chatReq.Messages), "tools", len(chatReq.Tools))

	start := time.Now()
	chatResp, err := s.backend.Chat(r.Context(), chatReq)
	if err != nil {
		slog.Error("backend call failed", "model", s.model, "err", err)
		writeError(w, http.StatusBadGateway, errModel, err.Error())
		return
	}

	resp, err := translate.FromOpenAI(chatResp, time.Since(start))
	if err != nil {
		slog.Error("translating backend response failed", "model", s.model, "err", err)
		writeError(w, http.StatusBadGateway, errModel, err.Error())
		return
	}

	writeJSON(w, resp)
}
