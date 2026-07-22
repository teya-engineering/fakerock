package config

import (
	"fmt"
	"os"
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
	Addr           string
	BackendBaseURL string
	BackendTimeout time.Duration
	Model          string
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

	return Config{
		Addr:           valueOr(os.Getenv("LISTEN_ADDR"), defaultAddr),
		BackendBaseURL: strings.TrimSuffix(valueOr(os.Getenv("BACKEND_BASE_URL"), defaultBackendBaseURL), "/"),
		BackendTimeout: timeout,
		Model:          valueOr(os.Getenv("BACKEND_MODEL"), defaultModel),
	}, nil
}

func valueOr(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
