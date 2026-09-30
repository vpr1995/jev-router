// Black-box tests for internal/telemetry: the zero-config no-op contract,
// the OTLP-enabled path (pointed at a dead local port — a refused TCP dial,
// no collector and no real network traffic) and the shutdown contract.
//
// Export errors are expected and tolerated here: what these tests pin down is
// that Setup constructs SDK providers and that Shutdown returns within its
// deadline instead of hanging on an unreachable collector.
package telemetry_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel"

	"github.com/vprprudhvi/jev-router/internal/telemetry"
)

// endpointEnv is the standard OpenTelemetry variable auto mode keys off.
const endpointEnv = "OTEL_EXPORTER_OTLP_ENDPOINT"

// deadEndpoint is a closed local port: OTLP exports fail fast with a refused
// connection instead of hanging or touching the network.
const deadEndpoint = "http://127.0.0.1:9"

// TestSetupNoopModes pins the zero-config contract: disabled, or auto without
// an endpoint, must yield OTel's no-op providers, must not touch the global
// providers and must shut down cleanly (including the nil receiver).
func TestSetupNoopModes(t *testing.T) {
	cases := []struct {
		name    string
		enabled string
	}{
		{name: "explicitly disabled", enabled: "false"},
		{name: "empty defaults to auto", enabled: ""},
		{name: "auto without endpoint", enabled: "auto"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Force the auto-detection gate closed: an empty value means
			// "not configured" regardless of the developer machine.
			t.Setenv(endpointEnv, "")

			beforeTracer := otel.GetTracerProvider()
			beforeMeter := otel.GetMeterProvider()

			provider, err := telemetry.Setup(context.Background(), telemetry.Config{Enabled: tc.enabled})
			if err != nil {
				t.Fatalf("Setup() error = %v, want nil", err)
			}
			if provider == nil {
				t.Fatal("Setup() provider = nil")
			}
			if provider.TracerProvider == nil {
				t.Error("Setup() TracerProvider = nil, want a no-op provider")
			}
			if provider.MeterProvider == nil {
				t.Error("Setup() MeterProvider = nil, want a no-op provider")
			}

			// OTel's no-op tracer never records; an SDK provider always does.
			_, span := provider.TracerProvider.Tracer("test").Start(context.Background(), "noop-check")
			if span.IsRecording() {
				t.Error("no-op provider recorded a span; want a non-recording no-op span")
			}
			span.End()

			counter, err := provider.MeterProvider.Meter("test").Int64Counter("noop-check")
			if err != nil {
				t.Fatalf("Int64Counter() on the no-op meter error = %v", err)
			}
			counter.Add(context.Background(), 1) // must be a silent no-op

			if err := provider.Shutdown(context.Background()); err != nil {
				t.Errorf("Shutdown() error = %v, want nil for a no-op provider", err)
			}

			var nilProvider *telemetry.Provider
			if err := nilProvider.Shutdown(context.Background()); err != nil {
				t.Errorf("nil *Provider Shutdown() error = %v, want nil", err)
			}

			if after := otel.GetTracerProvider(); after != beforeTracer {
				t.Error("Setup() in no-op mode replaced the global tracer provider; want globals untouched")
			}
			if after := otel.GetMeterProvider(); after != beforeMeter {
				t.Error("Setup() in no-op mode replaced the global meter provider; want globals untouched")
			}
		})
	}
}

// TestSetupWithEndpointBuildsSDKProviders covers the enabled path: auto mode
// with an endpoint set builds SDK providers, records spans, and shuts down
// within its deadline against a dead collector (export errors tolerated,
// hangs not).
func TestSetupWithEndpointBuildsSDKProviders(t *testing.T) {
	t.Setenv(endpointEnv, deadEndpoint)

	provider, err := telemetry.Setup(context.Background(), telemetry.Config{
		Enabled:     "auto",
		ServiceName: "jev-router-test",
		Version:     "0.0.1-test",
	})
	if err != nil {
		t.Fatalf("Setup() error = %v, want nil", err)
	}
	if provider.TracerProvider == nil || provider.MeterProvider == nil {
		t.Fatal("Setup() returned nil providers; want SDK providers")
	}

	_, span := provider.TracerProvider.Tracer("test").Start(context.Background(), "sdk-check")
	if !span.IsRecording() {
		t.Error("enabled Setup() returned a non-recording provider; want the SDK tracer provider")
	}
	span.End()

	// Shutdown must return within the 3s deadline: the OTLP gRPC exporter
	// retries the refused connection until the context expires, so the
	// tolerated context.DeadlineExceeded error arrives at ~deadline — a
	// result *after* the deadline, or a hang, is a failure.
	const deadline = 3 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()

	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- provider.Shutdown(ctx) }()
	select {
	case err := <-done:
		if elapsed := time.Since(start); elapsed > deadline+250*time.Millisecond {
			t.Errorf("Shutdown() took %v, want at most %v", elapsed, deadline+250*time.Millisecond)
		}
		t.Logf("Shutdown() error (tolerated export failure) = %v", err)
	case <-time.After(deadline + time.Second):
		t.Fatalf("Shutdown() did not return within %v", deadline+time.Second)
	}

	// Idempotent: the second call returns without touching the providers
	// again (no error, no hang).
	second := make(chan error, 1)
	go func() { second <- provider.Shutdown(context.Background()) }()
	select {
	case err := <-second:
		t.Logf("second Shutdown() error (same tolerated result) = %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("second Shutdown() did not return; want an immediate idempotent return")
	}
}

// TestSetupIsRepeatable pins that a second Setup replaces the globals without
// panicking and that each generated provider shuts down independently. It
// bounds every shutdown (deadline plus a one-second scheduling guard) because
// the dead endpoint makes each flush run out its full deadline.
func TestSetupIsRepeatable(t *testing.T) {
	t.Setenv(endpointEnv, deadEndpoint)

	const deadline = time.Second

	for call := 1; call <= 2; call++ {
		provider, err := telemetry.Setup(context.Background(), telemetry.Config{Enabled: "auto"})
		if err != nil {
			t.Fatalf("Setup() call %d error = %v, want nil", call, err)
		}

		shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), deadline)
		shutDown := make(chan error, 1)
		go func() { shutDown <- provider.Shutdown(shutdownCtx) }()
		select {
		case err := <-shutDown:
			t.Logf("call %d: Shutdown() error (tolerated) = %v", call, err)
		case <-time.After(deadline + time.Second):
			cancelShutdown()
			t.Fatalf("Shutdown() after Setup() call %d did not return within %v", call, deadline+time.Second)
		}
		cancelShutdown()
	}
}

// TestSetupRejectsUnknownEnabled documents the strict validation: the YAML
// file's "on"/"off" spellings are rejected here on purpose, so wiring must
// map them to "true"/"false" instead of silently disabling telemetry.
func TestSetupRejectsUnknownEnabled(t *testing.T) {
	for _, value := range []string{"on", "off", "yes", "TRUE", "1"} {
		t.Run(value, func(t *testing.T) {
			provider, err := telemetry.Setup(context.Background(), telemetry.Config{Enabled: value})
			if err == nil {
				t.Fatalf("Setup(Enabled=%q) error = nil, want an error", value)
			}
			if provider != nil {
				t.Errorf("Setup(Enabled=%q) provider = non-nil, want nil", value)
			}
			if !strings.Contains(err.Error(), "Enabled") {
				t.Errorf("error = %q, want it to name the Enabled field", err)
			}
		})
	}
}
