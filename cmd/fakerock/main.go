package main

import (
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/saltpay/fakerock/internal/backend"
	"github.com/saltpay/fakerock/internal/config"
	"github.com/saltpay/fakerock/internal/server"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		slog.Error("invalid configuration", "err", err)
		os.Exit(1)
	}

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           server.New(backend.New(cfg.BackendBaseURL, cfg.EmbeddingBaseURL, cfg.BackendTimeout), cfg.Model, cfg.EmbeddingModel, cfg.EmbeddingDimensions),
		ReadHeaderTimeout: 10 * time.Second,
	}

	slog.Info("listening", "addr", cfg.Addr, "backend", cfg.BackendBaseURL, "model", cfg.Model, "embeddingModel", cfg.EmbeddingModel)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
