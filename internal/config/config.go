package config

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultAddr           = ":8080"
	defaultBackendBaseURL = "http://localhost:11434/v1"
	defaultModel          = "qwen3:1.7b"
	defaultTimeout        = 5 * time.Minute
)

type Config struct {
	Addr                string
	BackendBaseURL      string
	EmbeddingBaseURL    string
	BackendTimeout      time.Duration
	Model               string
	EmbeddingModel      string
	EmbeddingDimensions int
	LogLevel            slog.Level
}

func Load() (Config, error) {
	timeout := defaultTimeout
	if raw := os.Getenv("BACKEND_TIMEOUT"); raw != "" {
		parsed, err := time.ParseDuration(raw)
		if err != nil {
			return Config{}, fmt.Errorf("BACKEND_TIMEOUT: %w", err)
		}
		timeout = parsed
	}

	model := valueOr(os.Getenv("BACKEND_MODEL"), defaultModel)

	dimensions := 0
	if raw := os.Getenv("BACKEND_EMBEDDING_DIMENSIONS"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			return Config{}, fmt.Errorf("BACKEND_EMBEDDING_DIMENSIONS: %w", err)
		}
		if parsed <= 0 {
			return Config{}, fmt.Errorf("BACKEND_EMBEDDING_DIMENSIONS must be positive, got %d", parsed)
		}
		dimensions = parsed
	}

	logLevel := slog.LevelInfo
	if raw := os.Getenv("LOG_LEVEL"); raw != "" {
		if err := logLevel.UnmarshalText([]byte(raw)); err != nil {
			return Config{}, fmt.Errorf("LOG_LEVEL: %w", err)
		}
	}

	backendBaseURL := strings.TrimSuffix(valueOr(os.Getenv("BACKEND_BASE_URL"), defaultBackendBaseURL), "/")

	return Config{
		Addr:                valueOr(os.Getenv("LISTEN_ADDR"), defaultAddr),
		BackendBaseURL:      backendBaseURL,
		EmbeddingBaseURL:    strings.TrimSuffix(valueOr(os.Getenv("BACKEND_EMBEDDING_BASE_URL"), backendBaseURL), "/"),
		BackendTimeout:      timeout,
		Model:               model,
		EmbeddingModel:      valueOr(os.Getenv("BACKEND_EMBEDDING_MODEL"), model),
		EmbeddingDimensions: dimensions,
		LogLevel:            logLevel,
	}, nil
}

func valueOr(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
