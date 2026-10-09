package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
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
	resp, err := c.postChat(ctx, req)
	if err != nil {
		return openai.ChatResponse{}, err
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

// ChatStream asks the backend to stream and calls onChunk for every server-sent event as it
// arrives, so the caller can forward each one before the generation finishes. A non-200 status
// fails before onChunk is ever called. A stream that ends without [DONE] was cut off, and is an
// error rather than a short answer.
func (c *Client) ChatStream(ctx context.Context, req openai.ChatRequest, onChunk func(openai.ChatChunk) error) error {
	req.Stream = true
	req.StreamOptions = &openai.StreamOptions{IncludeUsage: true}

	resp, err := c.postChat(ctx, req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		payload, _ := io.ReadAll(resp.Body)
		slog.Debug("backend response", "status", resp.StatusCode, "body", string(payload))
		return fmt.Errorf("backend returned %d: %s", resp.StatusCode, payload)
	}

	// One event can hold a whole tool call's arguments, so lines are read with no length limit.
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		// Blank lines separate events and lines starting with ":" are keep-alive comments.
		if data, ok := strings.CutPrefix(line, "data:"); ok {
			data = strings.TrimSpace(data)
			if data == "[DONE]" {
				return nil
			}

			var chunk openai.ChatChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				return fmt.Errorf("decoding backend chunk: %w", err)
			}
			if err := onChunk(chunk); err != nil {
				return err
			}
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("reading backend stream: %w", err)
		}
	}
	return errors.New("backend stream ended without [DONE]")
}

func (c *Client) postChat(ctx context.Context, req openai.ChatRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encoding backend request: %w", err)
	}
	// The full JSON in both directions is the ground truth when a model misbehaves, for
	// example answering in text where a tool call was expected. Debug level keeps it out
	// of normal runs; the bodies are large.
	slog.Debug("backend request", "url", c.baseURL+"/chat/completions", "body", string(body))

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("building backend request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	authorize(httpReq)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling backend: %w", err)
	}
	return resp, nil
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
	authorize(httpReq)

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
	authorize(httpReq)

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

type bearerTokenKey struct{}

// WithBearerToken returns a context whose backend calls carry "Authorization: Bearer <token>".
func WithBearerToken(ctx context.Context, token string) context.Context {
	return context.WithValue(ctx, bearerTokenKey{}, token)
}

func authorize(req *http.Request) {
	if token, _ := req.Context().Value(bearerTokenKey{}).(string); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}
