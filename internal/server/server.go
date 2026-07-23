package server

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/saltpay/fakerock/internal/openai"
)

type Backend interface {
	Chat(ctx context.Context, req openai.ChatRequest) (openai.ChatResponse, error)
	Embeddings(ctx context.Context, req openai.EmbeddingRequest) (openai.EmbeddingResponse, error)
}

type Server struct {
	backend             Backend
	model               string
	embeddingModel      string
	embeddingDimensions int
}

func New(backend Backend, model, embeddingModel string, embeddingDimensions int) *Server {
	return &Server{
		backend:             backend,
		model:               model,
		embeddingModel:      embeddingModel,
		embeddingDimensions: embeddingDimensions,
	}
}

// Routing is done by hand because model ids are often inference-profile ARNs, which
// carry slashes and reach us percent-encoded inside a single path segment.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	segments, err := pathSegments(r.URL.EscapedPath())
	if err != nil {
		writeError(w, http.StatusBadRequest, errValidation, err.Error())
		return
	}

	switch {
	case len(segments) == 3 && segments[0] == "model" && segments[2] == "converse":
		s.requirePost(w, r, func() { s.handleConverse(w, r, segments[1]) })
	case len(segments) == 3 && segments[0] == "model" && segments[2] == "converse-stream":
		s.requirePost(w, r, func() { s.handleConverseStream(w, r, segments[1]) })
	case len(segments) == 3 && segments[0] == "model" && segments[2] == "invoke":
		s.requirePost(w, r, func() { s.handleInvoke(w, r, segments[1]) })
	case len(segments) == 5 && segments[0] == "guardrail" && segments[2] == "version" && segments[4] == "apply":
		s.requirePost(w, r, func() { handleApplyGuardrail(w) })
	default:
		writeError(w, http.StatusNotFound, errNotFound, "unknown operation: "+r.URL.Path)
	}
}

func (s *Server) requirePost(w http.ResponseWriter, r *http.Request, handle func()) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errValidation, "method not allowed: "+r.Method)
		return
	}
	handle()
}

func pathSegments(escapedPath string) ([]string, error) {
	trimmed := strings.Trim(escapedPath, "/")
	if trimmed == "" {
		return nil, nil
	}
	segments := strings.Split(trimmed, "/")
	for i, segment := range segments {
		decoded, err := url.PathUnescape(segment)
		if err != nil {
			return nil, err
		}
		segments[i] = decoded
	}
	return segments, nil
}
