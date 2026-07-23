package server

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/translate"
)

// InvokeModel carries Titan text embeddings. Only that shape is served; anything else is a request
// for a model this stand-in does not implement.
func (s *Server) handleInvoke(w http.ResponseWriter, r *http.Request, modelID string) {
	var req bedrock.TitanEmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errValidation, "malformed request body: "+err.Error())
		return
	}
	if req.InputText == "" {
		writeError(w, http.StatusBadRequest, errValidation, "inputText is required")
		return
	}
	if req.Dimensions != nil && *req.Dimensions <= 0 {
		writeError(w, http.StatusBadRequest, errValidation, "dimensions must be positive")
		return
	}

	// The store expects a fixed width. Titan defaults to 1024 when the request omits dimensions;
	// mirror that with a configured fallback so the backend's native width does not leak through.
	dimensions := s.embeddingDimensions
	if req.Dimensions != nil {
		dimensions = *req.Dimensions
	}

	embedReq := translate.ToOpenAIEmbedding(s.embeddingModel, req.InputText, dimensions)

	slog.Info("invoke", "modelId", modelID, "model", s.embeddingModel, "dimensions", dimensions)

	embedResp, err := s.backend.Embeddings(r.Context(), embedReq)
	if err != nil {
		slog.Error("backend call failed", "model", s.embeddingModel, "err", err)
		writeError(w, http.StatusBadGateway, errModel, err.Error())
		return
	}

	resp, err := translate.FromOpenAIEmbedding(embedResp, dimensions)
	if err != nil {
		slog.Error("translating backend response failed", "model", s.embeddingModel, "err", err)
		writeError(w, http.StatusBadGateway, errModel, err.Error())
		return
	}

	writeJSON(w, resp)
}
