package config

import (
	"log/slog"
	"testing"
)

func TestLoadEmbeddingDimensions(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		want    int
		wantErr bool
	}{
		{"unset defaults to zero", "", 0, false},
		{"positive value", "1024", 1024, false},
		{"zero is rejected", "0", 0, true},
		{"negative is rejected", "-1", 0, true},
		{"non-numeric is rejected", "wide", 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("BACKEND_EMBEDDING_DIMENSIONS", tt.value)

			cfg, err := Load()
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected an error for value %q", tt.value)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cfg.EmbeddingDimensions != tt.want {
				t.Errorf("EmbeddingDimensions = %d, want %d", cfg.EmbeddingDimensions, tt.want)
			}
		})
	}
}

func TestLoadLogLevel(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		want    slog.Level
		wantErr bool
	}{
		{"unset defaults to info", "", slog.LevelInfo, false},
		{"debug", "debug", slog.LevelDebug, false},
		{"uppercase", "WARN", slog.LevelWarn, false},
		{"garbage is rejected", "loud", 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("LOG_LEVEL", tt.value)

			cfg, err := Load()
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected an error for value %q", tt.value)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cfg.LogLevel != tt.want {
				t.Errorf("LogLevel = %v, want %v", cfg.LogLevel, tt.want)
			}
		})
	}
}

func TestLoadEmbeddingModelFallsBackToModel(t *testing.T) {
	t.Setenv("BACKEND_MODEL", "chat-model")
	t.Setenv("BACKEND_EMBEDDING_MODEL", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.EmbeddingModel != "chat-model" {
		t.Errorf("EmbeddingModel = %q, want chat-model (the chat model as fallback)", cfg.EmbeddingModel)
	}
}
