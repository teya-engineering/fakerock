package server

import "net/http"

// handleHealth reports that the wrapper is serving. Callers probe this rather than a llama-server
// port: the bundled chat server only runs when BACKEND_BASE_URL points at it, so :8081 is absent
// for external chat backends. fakerock starts listening only after any bundled server is ready
// and warmed, so a response here also means the backend path is wired.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errValidation, "method not allowed: "+r.Method)
		return
	}
	writeJSON(w, map[string]string{"status": "ok", "model": s.currentModel()})
}
