package server

import (
	"encoding/json"
	"log/slog"
	"net/http"
)

// handleAdminModel reads (GET) or swaps (POST) the backend model at runtime, so the
// target can change without recreating the container. The boot default comes from
// BACKEND_MODEL; a swap holds until the next swap or restart.
func (s *Server) handleAdminModel(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, map[string]string{"model": s.currentModel()})
	case http.MethodPost:
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, errValidation, "malformed request body: "+err.Error())
			return
		}
		if body.Model == "" {
			writeError(w, http.StatusBadRequest, errValidation, "model is required")
			return
		}
		s.setModel(body.Model)
		slog.Info("model changed", "model", body.Model)
		writeJSON(w, map[string]string{"model": body.Model})
	default:
		writeError(w, http.StatusMethodNotAllowed, errValidation, "method not allowed: "+r.Method)
	}
}
