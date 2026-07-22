package server

import (
	"encoding/json"
	"log/slog"
	"net/http"
)

const (
	errValidation = "ValidationException"
	errNotFound   = "ResourceNotFoundException"
	errModel      = "ModelErrorException"
	errInternal   = "InternalServerException"
)

// AWS SDKs pick the exception class off x-amzn-ErrorType; without it every failure
// surfaces to the caller as an opaque generic error.
func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("x-amzn-ErrorType", code)
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(map[string]string{"__type": code, "message": message}); err != nil {
		slog.Error("writing error response", "err", err)
	}
}

func writeJSON(w http.ResponseWriter, body any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(body); err != nil {
		slog.Error("writing response", "err", err)
	}
}
