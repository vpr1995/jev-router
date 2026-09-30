// This file implements the serve command: configuration loading, telemetry
// setup, graceful shutdown and the OpenTelemetry HTTP middleware.
package cli

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	noopmetric "go.opentelemetry.io/otel/metric/noop"

	"github.com/vprprudhvi/jev-router/internal/telemetry"
	"github.com/vprprudhvi/jev-router/server"
)

// telemetryShutdownTimeout bounds the final telemetry flush when serve
// returns; a dead collector must never delay process exit for long.
const telemetryShutdownTimeout = 5 * time.Second

// MapTelemetryEnabled translates the YAML telemetry.enabled value
// ("auto"/"on"/"off") into the vocabulary internal/telemetry.Setup accepts
// ("auto"/"true"/"false"); any other value is an error (exit 1).
func MapTelemetryEnabled(value string) (string, error) {
	switch value {
	case "", "auto":
		return "auto", nil
	case "on", "true":
		return "true", nil
	case "off", "false":
		return "false", nil
	default:
		return "", fmt.Errorf("telemetry.enabled %q: must be %q, %q/%q or %q/%q",
			value, "auto", "on", "true", "off", "false")
	}
}

// telemetryActive reports whether Setup produced a real (exporting) pipeline
// rather than OTel's no-op providers, by checking the meter provider's
// concrete type.
func telemetryActive(provider *telemetry.Provider) bool {
	if provider == nil {
		return false
	}
	_, isNoop := provider.MeterProvider.(noopmetric.MeterProvider)
	return !isNoop
}

// runServe implements `jev-router serve [flags]`: load + assemble the
// configuration, configure telemetry, then serve the HTTP API until SIGINT
// or SIGTERM, shutting down gracefully and returning 0 on a clean stop.
func runServe(args []string, stdout, stderr io.Writer) error {
	fs := newFlagSet("serve")
	global := registerGlobalFlags(fs)
	configPath := fs.String("config", "", "configuration file (default: $JEV_ROUTER_CONFIG, else environment-only defaults)")
	addr := fs.String("addr", "", "listen address (default: server.addr from the configuration)")

	positional, err := parseCommandFlags(fs, args)
	if err != nil {
		return err
	}
	if len(positional) > 0 {
		return usageErrorf("serve: unexpected argument %q", positional[0])
	}
	logger, err := global.logger(stderr)
	if err != nil {
		return err
	}

	app, err := buildApp(*configPath, logger)
	if err != nil {
		return err
	}
	defer closeApp(logger, app)

	mode, err := MapTelemetryEnabled(app.File.Telemetry.Enabled)
	if err != nil {
		return err
	}
	provider, err := telemetry.Setup(context.Background(), telemetry.Config{
		Enabled: mode,
		Version: version,
	})
	if err != nil {
		return err
	}
	// Runs before closeApp (defers are LIFO): spans recorded while serving
	// are flushed before the components that produced them are released.
	defer shutdownTelemetry(logger, provider)

	// Instrument the HTTP server only when telemetry is actually exporting;
	// with the no-op default the server stays exactly as P8 built it.
	var middleware []func(http.Handler) http.Handler
	if telemetryActive(provider) {
		middleware = append(middleware, func(next http.Handler) http.Handler {
			return otelhttp.NewHandler(next, "jev-router")
		})
	}

	listenAddr := *addr
	if listenAddr == "" {
		listenAddr = app.File.Server.Addr
	}
	srv, err := server.New(server.Options{
		Router:      app.Router,
		Logger:      logger,
		Addr:        listenAddr,
		ReadTimeout: app.File.Server.ReadTimeout.Value(),
		Middleware:  middleware,
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	logger.Info("listening on " + listenAddr)
	if err := srv.Serve(ctx); err != nil {
		return err
	}
	return nil
}

// shutdownTelemetry flushes and shuts the telemetry pipeline down within
// telemetryShutdownTimeout. Failures (for example a dead collector) are
// logged, never returned: telemetry is best-effort and never changes the
// exit code.
func shutdownTelemetry(logger *slog.Logger, provider *telemetry.Provider) {
	ctx, cancel := context.WithTimeout(context.Background(), telemetryShutdownTimeout)
	defer cancel()
	if err := provider.Shutdown(ctx); err != nil {
		logger.Warn("telemetry shutdown", "error", err)
	}
}
