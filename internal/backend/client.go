package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/saltpay/fakerock/internal/openai"
)

type Client struct {
	baseURL          string
	embeddingBaseURL string
	http             *http.Client
}

func New(baseURL, embeddingBaseURL string, timeout time.Duration) *Client {
	return &Client{
		baseURL:          baseURL,
		embeddingBaseURL: embeddingBaseURL,
		http:             &http.Client{Timeout: timeout},
	}
}

func (c *Client) Chat(ctx context.Context, req openai.ChatRequest) (openai.ChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return openai.ChatResponse{}, fmt.Errorf("encoding backend request: %w", err)
	}
	// The full JSON in both directions is the ground truth when a model misbehaves, for
	// example answering in text where a tool call was expected. Debug level keeps it out
	// of normal runs; the bodies are large.
	slog.Debug("backend request", "url", c.baseURL+"/chat/completions", "body", string(body))

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return openai.ChatResponse{}, fmt.Errorf("building backend request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return openai.ChatResponse{}, fmt.Errorf("calling backend: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return openai.ChatResponse{}, fmt.Errorf("reading backend response: %w", err)
	}
	slog.Debug("backend response", "status", resp.StatusCode, "body", string(payload))

	if resp.StatusCode != http.StatusOK {
		return openai.ChatResponse{}, fmt.Errorf("backend returned %d: %s", resp.StatusCode, payload)
	}

	var chat openai.ChatResponse
	if err := json.Unmarshal(payload, &chat); err != nil {
		return openai.ChatResponse{}, fmt.Errorf("decoding backend response: %w", err)
	}
	return chat, nil
}

// Ping asks the backend for its model list. /v1/models is the only path both llama.cpp and Ollama
// serve, and it runs no inference, so a health check can call it per probe without competing with
// real requests for a slot. It proves the backend is reachable and answering, not that BACKEND_MODEL
// is usable: Ollama lists every installed model, and llama.cpp reports its own model path.
func (c *Client) Ping(ctx context.Context) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return fmt.Errorf("building backend request: %w", err)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return fmt.Errorf("calling backend: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("backend returned %d: %s", resp.StatusCode, payload)
	}
	return nil
}

func (c *Client) Embeddings(ctx context.Context, req openai.EmbeddingRequest) (openai.EmbeddingResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return openai.EmbeddingResponse{}, fmt.Errorf("encoding backend request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.embeddingBaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return openai.EmbeddingResponse{}, fmt.Errorf("building backend request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return openai.EmbeddingResponse{}, fmt.Errorf("calling backend: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return openai.EmbeddingResponse{}, fmt.Errorf("reading backend response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return openai.EmbeddingResponse{}, fmt.Errorf("backend returned %d: %s", resp.StatusCode, payload)
	}

	var embed openai.EmbeddingResponse
	if err := json.Unmarshal(payload, &embed); err != nil {
		return openai.EmbeddingResponse{}, fmt.Errorf("decoding backend response: %w", err)
	}
	return embed, nil
}
