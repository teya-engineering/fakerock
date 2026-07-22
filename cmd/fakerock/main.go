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
		Handler:           server.New(backend.New(cfg.BackendBaseURL, cfg.BackendTimeout), cfg.Model),
		ReadHeaderTimeout: 10 * time.Second,
	}

	slog.Info("listening", "addr", cfg.Addr, "backend", cfg.BackendBaseURL, "model", cfg.Model)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
