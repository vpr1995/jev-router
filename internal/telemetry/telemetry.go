// Package telemetry wires OpenTelemetry traces and metrics for jev-router:
// OTLP exporters configured through the standard OpenTelemetry environment
// variables, with a guaranteed no-op default — zero configuration must not
// change behaviour. When telemetry is disabled, or auto-detection finds no
// OTLP endpoint, Setup returns OTel's no-op providers untouched.
//
// Setup is the only place in the repository that touches the OpenTelemetry
// SDK and the global providers: the library packages (the root modelrouter
// package, server, ...) depend on the OTel API only, so importing the router
// never pulls the SDK in or opens a network connection.
//
// Enabled semantics (Config.Enabled):
//
//   - "" or "auto": enabled exactly when OTEL_EXPORTER_OTLP_ENDPOINT holds a
//     non-empty value.
//   - "true": always enabled; the OTLP gRPC exporters then apply their own
//     defaults (for example localhost:4317) when no endpoint variable is set.
//   - "false": always disabled.
//
// Transport behaviour is deliberately tolerant: Setup fails only when a
// provider cannot be constructed at all. Export failures (unreachable
// collector, refused connection, ...) surface through Shutdown's error and
// are otherwise ignored — telemetry is best-effort and routing never depends
// on it.
package telemetry

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/metric"
	noopmetric "go.opentelemetry.io/otel/metric/noop"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	tracenoop "go.opentelemetry.io/otel/trace/noop"
)

const (
	// enabledAuto, enabledTrue and enabledFalse are the accepted values of
	// Config.Enabled.
	enabledAuto  = "auto"
	enabledTrue  = "true"
	enabledFalse = "false"

	// endpointEnv is the standard OpenTelemetry variable auto mode watches:
	// telemetry switches itself on when it holds a non-empty value.
	endpointEnv = "OTEL_EXPORTER_OTLP_ENDPOINT"

	// defaultServiceName is the service.name attached when Config.ServiceName
	// is empty.
	defaultServiceName = "jev-router"

	// attrServiceName and attrServiceVersion are the resource attribute keys
	// of the service identity.
	attrServiceName    = "service.name"
	attrServiceVersion = "service.version"

	// cleanupTimeout bounds the best-effort cleanup of a half-built pipeline
	// when a later step of Setup fails.
	cleanupTimeout = 5 * time.Second
)

// Config configures Setup.
type Config struct {
	// Enabled selects the telemetry mode: "auto" (the default when empty:
	// on exactly when OTEL_EXPORTER_OTLP_ENDPOINT is set), "true" (always
	// on) or "false" (always off). Any other value is an error — note that
	// the YAML file's telemetry.enabled uses "on"/"off", which its wiring
	// must map to "true"/"false" before calling Setup.
	Enabled string
	// ServiceName is the service.name resource attribute. Empty means
	// "jev-router".
	ServiceName string
	// Version, when set, is attached as the service.version resource
	// attribute.
	Version string
}

// Provider bundles the tracer and meter providers configured by Setup.
//
// Every field is safe to use with a no-op Provider. A nil *Provider is also
// valid: Shutdown is nil-safe, so callers can hold the zero result of a
// failed Setup without special cases.
type Provider struct {
	// TracerProvider supplies the application's tracer.
	TracerProvider trace.TracerProvider
	// MeterProvider supplies the application's meter.
	MeterProvider metric.MeterProvider

	// once makes Shutdown idempotent.
	once sync.Once
	// shutdown holds the provider shutdown functions, in order.
	shutdown []func(context.Context) error
	// err is the joined result of the first Shutdown call.
	err error
}

// Setup configures OpenTelemetry according to cfg.
//
// Disabled or auto-without-endpoint calls return OTel's no-op providers and
// do not touch the globals. Otherwise Setup constructs OTLP gRPC trace and
// metric exporters (batched spans, periodic metric export), builds SDK
// providers sharing one resource (service.name plus service.version when
// set), and installs them with otel.SetTracerProvider/SetMeterProvider so
// API-only code and third-party instrumentation follow them. A later Setup
// call simply installs a new pipeline; the previous one keeps working until
// its own Shutdown is called.
func Setup(ctx context.Context, cfg Config) (*Provider, error) {
	mode := cfg.Enabled
	if mode == "" {
		mode = enabledAuto
	}
	switch mode {
	case enabledAuto, enabledTrue, enabledFalse:
	default:
		return nil, fmt.Errorf("telemetry: invalid Enabled %q (want %q, %q or %q)",
			cfg.Enabled, enabledAuto, enabledTrue, enabledFalse)
	}

	if mode == enabledFalse || (mode == enabledAuto && os.Getenv(endpointEnv) == "") {
		// Zero-config (or explicitly disabled): no-op providers, no global
		// mutation.
		return &Provider{
			TracerProvider: tracenoop.NewTracerProvider(),
			MeterProvider:  noopmetric.NewMeterProvider(),
		}, nil
	}

	res, err := buildResource(cfg)
	if err != nil {
		return nil, err
	}

	traceExporter, err := otlptracegrpc.New(ctx)
	if err != nil {
		return nil, fmt.Errorf("telemetry: creating OTLP trace exporter: %w", err)
	}
	tracerProvider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(traceExporter),
		sdktrace.WithResource(res),
	)

	metricExporter, err := otlpmetricgrpc.New(ctx)
	if err != nil {
		// Do not leak the trace pipeline that was already built.
		cleanupCtx, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
		defer cancel()
		_ = tracerProvider.Shutdown(cleanupCtx)
		return nil, fmt.Errorf("telemetry: creating OTLP metric exporter: %w", err)
	}
	meterProvider := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExporter)),
		sdkmetric.WithResource(res),
	)

	// Install both providers as globals: API-only code and third-party
	// instrumentation resolve through otel.GetTracerProvider/GetMeterProvider.
	// This is the only global mutation in the repository.
	otel.SetTracerProvider(tracerProvider)
	otel.SetMeterProvider(meterProvider)

	return &Provider{
		TracerProvider: tracerProvider,
		MeterProvider:  meterProvider,
		shutdown: []func(context.Context) error{
			tracerProvider.Shutdown,
			meterProvider.Shutdown,
		},
	}, nil
}

// buildResource returns the resource shared by both providers: the SDK
// default resource merged with the configured service.name and, when set,
// service.version. The configured name wins the merge, so service.name is
// never left at the SDK's "unknown_service" default.
func buildResource(cfg Config) (*resource.Resource, error) {
	serviceName := cfg.ServiceName
	if serviceName == "" {
		serviceName = defaultServiceName
	}
	attrs := []attribute.KeyValue{attribute.String(attrServiceName, serviceName)}
	if cfg.Version != "" {
		attrs = append(attrs, attribute.String(attrServiceVersion, cfg.Version))
	}

	res, err := resource.Merge(resource.Default(), resource.NewSchemaless(attrs...))
	if err != nil {
		return nil, fmt.Errorf("telemetry: building resource: %w", err)
	}
	return res, nil
}

// Shutdown flushes and shuts down both providers, joining their errors.
//
// It is safe on a nil *Provider (returns nil) and idempotent: only the first
// call performs work, later calls return the same error without touching the
// providers again. Export failures land in the returned error; callers may
// log it, but nothing in the router reacts to it, so a dead collector never
// affects serving. The caller's ctx bounds the whole shutdown; pass a
// deadline when returning to a process exit path.
func (p *Provider) Shutdown(ctx context.Context) error {
	if p == nil {
		return nil
	}
	p.once.Do(func() {
		errs := make([]error, 0, len(p.shutdown))
		for _, shutdown := range p.shutdown {
			if err := shutdown(ctx); err != nil {
				errs = append(errs, err)
			}
		}
		p.err = errors.Join(errs...)
	})
	return p.err
}
