package server

import (
	"context"
	"log/slog"
	"net/http"
	"time"
)

// A probe should fail while the backend is merely slow, not hang with it. Kept well under a
// typical probe interval so a stuck backend cannot pile up checks.
const healthTimeout = 2 * time.Second

// handleHealth pings the backend on every call. Callers probe this rather than a llama-server port:
// bundled chat runs only when LLAMA_CHAT=on, so :8081 may be absent, and a backend that was up at
// boot can be gone by the time anyone asks.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errValidation, "method not allowed: "+r.Method)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), healthTimeout)
	defer cancel()

	if err := s.backend.Ping(ctx); err != nil {
		slog.Error("health check failed", "err", err)
		writeJSONStatus(w, http.StatusServiceUnavailable, map[string]string{
			"status": "unavailable",
			"model":  s.currentModel(),
			"error":  err.Error(),
		})
		return
	}
	writeJSON(w, map[string]string{"status": "ok", "model": s.currentModel()})
}
