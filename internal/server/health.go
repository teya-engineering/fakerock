package server

import "net/http"

// handleHealth reports that the wrapper is serving. Callers probe this rather than a llama-server
// port: bundled chat runs only when LLAMA_CHAT=on, so :8081 may be absent. fakerock starts
// listening only after any bundled server is ready and warmed, so a response here also means the
// backend path is wired.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errValidation, "method not allowed: "+r.Method)
		return
	}
	writeJSON(w, map[string]string{"status": "ok", "model": s.currentModel()})
}
