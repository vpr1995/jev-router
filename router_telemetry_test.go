// Black-box telemetry tests for the Router: spans and metrics emitted with a
// recording OpenTelemetry SDK, entirely offline.
//
// Nothing here installs global providers: TracerProvider/MeterProvider are
// injected through RouterOptions, so these tests are independent of
// internal/telemetry, of each other and of every other test in the package.
// Metric readers are ManualReaders collected synchronously — no asynchronous
// export and no timing flakiness.
//
// v2 mapped routing: the modelrouter.score span is gone; the route span
// carries model_id/provider/source/candidates/domain/fail_open, the complete
// span carries modelrouter.attempts, and catalog.fetch spans appear only when
// a metadata constraint actually needs the catalogs.
package modelrouter_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/internal/testutil"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// --- environment ------------------------------------------------------------

// telemetryEnv bundles the recording SDK providers a test needs: a span
// recorder for traces and a ManualReader for metrics. Each test builds its
// own, so no test can observe another's data.
type telemetryEnv struct {
	tracerProvider *sdktrace.TracerProvider
	meterProvider  *sdkmetric.MeterProvider
	spans          *tracetest.SpanRecorder
	reader         *sdkmetric.ManualReader
}

func newTelemetryEnv(t *testing.T) *telemetryEnv {
	t.Helper()
	spans := tracetest.NewSpanRecorder()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spans))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = tracerProvider.Shutdown(ctx)
		_ = meterProvider.Shutdown(ctx)
	})
	return &telemetryEnv{
		tracerProvider: tracerProvider,
		meterProvider:  meterProvider,
		spans:          spans,
		reader:         reader,
	}
}

// newRouterWithTelemetryOptions builds a Router over opts with the recording
// providers injected and fails the test on validation errors.
func newRouterWithTelemetryOptions(t *testing.T, env *telemetryEnv, opts modelrouter.RouterOptions) *modelrouter.Router {
	t.Helper()
	opts.TracerProvider = env.tracerProvider
	opts.MeterProvider = env.meterProvider
	router, err := modelrouter.New(opts)
	if err != nil {
		t.Fatalf("modelrouter.New() error = %v", err)
	}
	return router
}

// newRouterWithTelemetry builds a Router over the shared fakes with the
// recording providers injected. The completion may be nil for tests that
// never complete.
func newRouterWithTelemetry(t *testing.T, env *telemetryEnv, catalog *testutil.FakeCatalog, classifier *testutil.FakeClassifier, completion *testutil.FakeCompletion) *modelrouter.Router {
	t.Helper()
	return newRouterWithTelemetryOptions(t, env, testutil.Options(catalog, classifier, completion))
}

// --- span helpers -----------------------------------------------------------

func endedSpanNames(env *telemetryEnv) []string {
	ended := env.spans.Ended()
	names := make([]string, 0, len(ended))
	for _, span := range ended {
		names = append(names, span.Name())
	}
	return names
}

// requireSpan returns the single ended span with the given name, failing the
// test when there is none or more than one.
func requireSpan(t *testing.T, env *telemetryEnv, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	var found []sdktrace.ReadOnlySpan
	for _, span := range env.spans.Ended() {
		if span.Name() == name {
			found = append(found, span)
		}
	}
	if len(found) != 1 {
		t.Fatalf("ended spans named %q = %d, want exactly 1; all ended spans: %v", name, len(found), endedSpanNames(env))
	}
	return found[0]
}

func spanHasAttr(span sdktrace.ReadOnlySpan, key string) bool {
	for _, kv := range span.Attributes() {
		if string(kv.Key) == key {
			return true
		}
	}
	return false
}

func spanAttr(t *testing.T, span sdktrace.ReadOnlySpan, key string) attribute.Value {
	t.Helper()
	for _, kv := range span.Attributes() {
		if string(kv.Key) == key {
			return kv.Value
		}
	}
	t.Fatalf("span %q: attribute %q missing; have %v", span.Name(), key, span.Attributes())
	return attribute.Value{}
}

// --- metric helpers ---------------------------------------------------------

func collectMetrics(t *testing.T, reader *sdkmetric.ManualReader) metricdata.ResourceMetrics {
	t.Helper()
	var collected metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &collected); err != nil {
		t.Fatalf("ManualReader.Collect() error = %v", err)
	}
	return collected
}

func collectedMetricNames(collected metricdata.ResourceMetrics) []string {
	var names []string
	for _, scope := range collected.ScopeMetrics {
		for _, metric := range scope.Metrics {
			names = append(names, metric.Name)
		}
	}
	return names
}

func metricByName(t *testing.T, collected metricdata.ResourceMetrics, name string) metricdata.Metrics {
	t.Helper()
	for _, scope := range collected.ScopeMetrics {
		for _, metric := range scope.Metrics {
			if metric.Name == name {
				return metric
			}
		}
	}
	t.Fatalf("metric %q not collected; collected metrics: %v", name, collectedMetricNames(collected))
	return metricdata.Metrics{}
}

// requireSum returns the single data point of an int64 sum instrument.
func requireSum(t *testing.T, collected metricdata.ResourceMetrics, name string) metricdata.DataPoint[int64] {
	t.Helper()
	metric := metricByName(t, collected, name)
	sum, ok := metric.Data.(metricdata.Sum[int64])
	if !ok {
		t.Fatalf("metric %q data type = %T, want metricdata.Sum[int64]", name, metric.Data)
	}
	if len(sum.DataPoints) != 1 {
		t.Fatalf("metric %q data points = %d, want exactly 1", name, len(sum.DataPoints))
	}
	return sum.DataPoints[0]
}

func metricAttr(t *testing.T, set attribute.Set, key string) attribute.Value {
	t.Helper()
	value, ok := set.Value(attribute.Key(key))
	if !ok {
		t.Fatalf("data point: attribute %q missing; have %v", key, set.ToSlice())
	}
	return value
}

// --- tests ------------------------------------------------------------------

// TestRouteEmitsSpansAndDecisionMetric covers the happy path: exactly the
// route and classify spans as one tree — no catalog.fetch, because no
// metadata constraint forced a fetch — with the documented route attributes,
// plus one modelrouter.decisions data point.
func TestRouteEmitsSpansAndDecisionMetric(t *testing.T) {
	env := newTelemetryEnv(t)
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{
		Profile: testutil.CONFIDENT_TRIVIAL_PROFILE,
	}}
	opts := testutil.Options(catalog, classifier, nil)
	opts.Models = map[modelrouter.Domain][]modelrouter.ModelChoice{
		modelrouter.DomainOther: {choiceOf(testutil.CHEAP), choiceOf(testutil.EXPENSIVE)},
	}
	router := newRouterWithTelemetryOptions(t, env, opts)

	decision, err := router.Route(context.Background(), "hello", nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if decision.ModelID != testutil.CHEAP.ID || decision.Provider != testutil.CHEAP.Provider {
		t.Fatalf("Route() decision = %s/%s, want %s/%s",
			decision.Provider, decision.ModelID, testutil.CHEAP.Provider, testutil.CHEAP.ID)
	}

	// Trace tree: exactly one route and one classify span, classify a child
	// of route.
	routeSpan := requireSpan(t, env, "modelrouter.route")
	classifySpan := requireSpan(t, env, "modelrouter.classify")

	names := endedSpanNames(env)
	slices.Sort(names)
	wantNames := []string{
		"modelrouter.classify",
		"modelrouter.route",
	}
	if !slices.Equal(names, wantNames) {
		t.Errorf("ended spans = %v, want exactly %v", names, wantNames)
	}
	if classifySpan.Parent().SpanID() != routeSpan.SpanContext().SpanID() {
		t.Errorf("span %q parent span = %s, want the route span %s",
			classifySpan.Name(), classifySpan.Parent().SpanID(), routeSpan.SpanContext().SpanID())
	}
	if routeSpan.Status().Code != codes.Unset {
		t.Errorf("route span status = %v, want unset on success", routeSpan.Status().Code)
	}

	// The route span carries the v2 decision: winner, source and chain size.
	if got := spanAttr(t, routeSpan, "modelrouter.model_id").AsString(); got != testutil.CHEAP.ID {
		t.Errorf("route modelrouter.model_id = %q, want %q", got, testutil.CHEAP.ID)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.provider").AsString(); got != testutil.CHEAP.Provider {
		t.Errorf("route modelrouter.provider = %q, want %q", got, testutil.CHEAP.Provider)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.source").AsString(); got != modelrouter.SourceClassification {
		t.Errorf("route modelrouter.source = %q, want %q", got, modelrouter.SourceClassification)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.candidates").AsInt64(); got != 2 {
		t.Errorf("route modelrouter.candidates = %d, want 2 (the filtered chain)", got)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.domain").AsString(); got != string(modelrouter.DomainOther) {
		t.Errorf("route modelrouter.domain = %q, want %q", got, modelrouter.DomainOther)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.fail_open").AsBool(); got {
		t.Error("route modelrouter.fail_open = true, want false for a successful classification")
	}

	// classify carries the profile and the fail-open marker.
	if got := spanAttr(t, classifySpan, "modelrouter.domain").AsString(); got != string(modelrouter.DomainOther) {
		t.Errorf("classify modelrouter.domain = %q, want %q", got, modelrouter.DomainOther)
	}
	if got := spanAttr(t, classifySpan, "modelrouter.complexity_score").AsFloat64(); got != 0 {
		t.Errorf("classify modelrouter.complexity_score = %v, want 0", got)
	}
	if got := spanAttr(t, classifySpan, "modelrouter.complexity_confidence").AsFloat64(); got != 0.9 {
		t.Errorf("classify modelrouter.complexity_confidence = %v, want 0.9", got)
	}
	if got := spanAttr(t, classifySpan, "modelrouter.fail_open").AsBool(); got {
		t.Error("classify modelrouter.fail_open = true, want false for a successful classification")
	}
	if classifySpan.Status().Code != codes.Unset {
		t.Errorf("classify span status = %v, want unset on success", classifySpan.Status().Code)
	}

	// One decisions data point with the winning model, provider, domain,
	// source and fallback attributes.
	collected := collectMetrics(t, env.reader)
	point := requireSum(t, collected, "modelrouter.decisions")
	if point.Value != 1 {
		t.Errorf("modelrouter.decisions value = %d, want 1", point.Value)
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.model_id").AsString(); got != testutil.CHEAP.ID {
		t.Errorf("decisions modelrouter.model_id = %q, want %q", got, testutil.CHEAP.ID)
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.provider").AsString(); got != testutil.CHEAP.Provider {
		t.Errorf("decisions modelrouter.provider = %q, want %q", got, testutil.CHEAP.Provider)
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.domain").AsString(); got != string(modelrouter.DomainOther) {
		t.Errorf("decisions modelrouter.domain = %q, want %q", got, modelrouter.DomainOther)
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.fallback").AsBool(); got {
		t.Error("decisions modelrouter.fallback = true, want false")
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.source").AsString(); got != modelrouter.SourceClassification {
		t.Errorf("decisions modelrouter.source = %q, want %q", got, modelrouter.SourceClassification)
	}

	// The classify duration histogram records the one call, in seconds.
	histogramMetric := metricByName(t, collected, "modelrouter.classify.duration")
	if histogramMetric.Unit != "s" {
		t.Errorf("classify.duration unit = %q, want %q", histogramMetric.Unit, "s")
	}
	histogram, ok := histogramMetric.Data.(metricdata.Histogram[float64])
	if !ok {
		t.Fatalf("classify.duration data type = %T, want metricdata.Histogram[float64]", histogramMetric.Data)
	}
	if len(histogram.DataPoints) != 1 {
		t.Fatalf("classify.duration data points = %d, want 1", len(histogram.DataPoints))
	}
	if got := histogram.DataPoints[0].Count; got != 1 {
		t.Errorf("classify.duration count = %d, want 1", got)
	}
	if got := histogram.DataPoints[0].Sum; got < 0 {
		t.Errorf("classify.duration sum = %v, want a non-negative duration", got)
	}
}

// TestClassifierFailOpenSpansMarkFallback covers the handled-failure path:
// the classify span carries modelrouter.fail_open=true without an error
// status, the route span carries modelrouter.fail_open=true plus the default
// decision, and the decision has a nil Profile.
func TestClassifierFailOpenSpansMarkFallback(t *testing.T) {
	env := newTelemetryEnv(t)
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	classifier := &testutil.FakeClassifier{Err: &modelrouter.ClassifierUnavailableError{
		Err: errors.New("jev down"),
	}}
	// Empty OnClassifierError means the fail-open default; Options supplies
	// the default model.
	router := newRouterWithTelemetry(t, env, catalog, classifier, nil)

	decision, err := router.Route(context.Background(), "hello", nil)
	if err != nil {
		t.Fatalf("Route() error = %v, want the fail-open fallback to succeed", err)
	}
	if decision.Profile != nil {
		t.Errorf("decision.Profile = %+v, want nil after a fail-open fallback", decision.Profile)
	}
	if decision.ModelID != testutil.CHEAP.ID {
		t.Errorf("decision.ModelID = %q, want the default model %q", decision.ModelID, testutil.CHEAP.ID)
	}
	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("decision.Source = %q, want %q", decision.Source, modelrouter.SourceDefault)
	}

	classifySpan := requireSpan(t, env, "modelrouter.classify")
	if got := spanAttr(t, classifySpan, "modelrouter.fail_open").AsBool(); !got {
		t.Error("classify modelrouter.fail_open = false, want true")
	}
	if spanHasAttr(classifySpan, "modelrouter.domain") {
		t.Error("classify span has modelrouter.domain, want no profile attributes on the fail-open path")
	}
	if classifySpan.Status().Code != codes.Unset {
		t.Errorf("classify span status = %v, want unset (the failure was handled by fail-open)", classifySpan.Status().Code)
	}
	if len(classifySpan.Events()) == 0 {
		t.Error("classify span recorded no error event, want the failure recorded")
	}

	routeSpan := requireSpan(t, env, "modelrouter.route")
	if got := spanAttr(t, routeSpan, "modelrouter.fail_open").AsBool(); !got {
		t.Error("route modelrouter.fail_open = false, want true")
	}
	if got := spanAttr(t, routeSpan, "modelrouter.model_id").AsString(); got != testutil.CHEAP.ID {
		t.Errorf("route modelrouter.model_id = %q, want the default %q", got, testutil.CHEAP.ID)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.source").AsString(); got != modelrouter.SourceDefault {
		t.Errorf("route modelrouter.source = %q, want %q", got, modelrouter.SourceDefault)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.candidates").AsInt64(); got != 1 {
		t.Errorf("route modelrouter.candidates = %d, want 1 (the default model alone)", got)
	}
	if got := spanAttr(t, routeSpan, "modelrouter.domain").AsString(); got != "" {
		t.Errorf("route modelrouter.domain = %q, want empty when no profile exists", got)
	}
	if routeSpan.Status().Code != codes.Unset {
		t.Errorf("route span status = %v, want unset (fail-open routing succeeded)", routeSpan.Status().Code)
	}

	// The fallback still counts as a decision, marked as such.
	point := requireSum(t, collectMetrics(t, env.reader), "modelrouter.decisions")
	if point.Value != 1 {
		t.Errorf("modelrouter.decisions value = %d, want 1", point.Value)
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.fallback").AsBool(); !got {
		t.Error("decisions modelrouter.fallback = false, want true on the fallback path")
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.domain").AsString(); got != "" {
		t.Errorf("decisions modelrouter.domain = %q, want empty when no profile exists", got)
	}
	if got := metricAttr(t, point.Attributes, "modelrouter.source").AsString(); got != modelrouter.SourceDefault {
		t.Errorf("decisions modelrouter.source = %q, want %q", got, modelrouter.SourceDefault)
	}
}

// TestRouteCatalogFetchSpansOnlyWhenNeeded pins the on-demand catalog
// contract: a plain route with a broken catalog succeeds and emits no
// catalog.fetch span, while a metadata constraint fetches (and fails loudly,
// with the fetch span marked failed).
func TestRouteCatalogFetchSpansOnlyWhenNeeded(t *testing.T) {
	t.Run("plain route never fetches a broken catalog", func(t *testing.T) {
		env := newTelemetryEnv(t)
		catalog := &testutil.FakeCatalog{Err: errors.New("catalog down")}
		classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
		router := newRouterWithTelemetry(t, env, catalog, classifier, nil)

		decision, err := router.Route(context.Background(), "hello", nil)
		if err != nil {
			t.Fatalf("Route() error = %v, want success (no catalog metadata needed)", err)
		}
		if decision.ModelID != testutil.CHEAP.ID {
			t.Errorf("decision.ModelID = %q, want %q", decision.ModelID, testutil.CHEAP.ID)
		}
		if got := catalog.Calls(); got != 0 {
			t.Errorf("catalog.Calls() = %d, want 0", got)
		}
		names := endedSpanNames(env)
		slices.Sort(names)
		if want := []string{"modelrouter.classify", "modelrouter.route"}; !slices.Equal(names, want) {
			t.Errorf("ended spans = %v, want exactly %v (no catalog.fetch)", names, want)
		}
	})

	t.Run("metadata constraints fetch and fail loudly", func(t *testing.T) {
		env := newTelemetryEnv(t)
		catalog := &testutil.FakeCatalog{Err: errors.New("catalog down")}
		classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
		router := newRouterWithTelemetry(t, env, catalog, classifier, nil)

		_, err := router.Route(context.Background(), "hello", &modelrouter.RoutingConstraints{NeedsVision: true})
		if err == nil {
			t.Fatal("Route() error = nil, want the catalog failure")
		}
		if got := catalog.Calls(); got != 1 {
			t.Errorf("catalog.Calls() = %d, want 1", got)
		}
		if got := classifier.Calls(); got != 0 {
			t.Errorf("classifier.Calls() = %d, want 0 (the fetch fails before classification)", got)
		}
		fetchSpan := requireSpan(t, env, "modelrouter.catalog.fetch")
		if fetchSpan.Status().Code != codes.Error {
			t.Errorf("catalog.fetch status = %v, want error", fetchSpan.Status().Code)
		}
		routeSpan := requireSpan(t, env, "modelrouter.route")
		if routeSpan.Status().Code != codes.Error {
			t.Errorf("route status = %v, want error", routeSpan.Status().Code)
		}
	})
}

// TestCompleteSpanAttempts pins modelrouter.attempts on the complete span: 1
// when the primary candidate succeeds, 2 when the fallback recovered, with
// the span staying unset on a recovered fallback.
func TestCompleteSpanAttempts(t *testing.T) {
	t.Run("single attempt without fallback", func(t *testing.T) {
		env := newTelemetryEnv(t)
		catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
		classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
		completion := &testutil.FakeCompletion{Result: modelrouter.CompletionResult{Content: "ok"}}
		router := newRouterWithTelemetry(t, env, catalog, classifier, completion)

		result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
		if err != nil {
			t.Fatalf("Complete() error = %v", err)
		}
		if result.RoutingDecision.ModelID != testutil.CHEAP.ID {
			t.Errorf("RoutingDecision.ModelID = %q, want %q", result.RoutingDecision.ModelID, testutil.CHEAP.ID)
		}

		completeSpan := requireSpan(t, env, "modelrouter.complete")
		if got := spanAttr(t, completeSpan, "modelrouter.attempts").AsInt64(); got != 1 {
			t.Errorf("complete modelrouter.attempts = %d, want 1", got)
		}
		if got := spanAttr(t, completeSpan, "modelrouter.model_id").AsString(); got != testutil.CHEAP.ID {
			t.Errorf("complete modelrouter.model_id = %q, want %q", got, testutil.CHEAP.ID)
		}
		if got := spanAttr(t, completeSpan, "modelrouter.provider").AsString(); got != testutil.CHEAP.Provider {
			t.Errorf("complete modelrouter.provider = %q, want %q", got, testutil.CHEAP.Provider)
		}
		if completeSpan.Status().Code != codes.Unset {
			t.Errorf("complete span status = %v, want unset on success", completeSpan.Status().Code)
		}
	})

	t.Run("two attempts with a recovered fallback", func(t *testing.T) {
		env := newTelemetryEnv(t)
		catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
		classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
		completion := &testutil.FakeCompletion{Result: modelrouter.CompletionResult{Content: "ok"}}
		completion.Errs = []error{
			&modelrouter.ProviderError{Provider: testutil.CHEAP.Provider, Op: "complete", Err: errors.New("first down")},
			nil,
		}
		opts := testutil.Options(catalog, classifier, completion)
		opts.Models = map[modelrouter.Domain][]modelrouter.ModelChoice{
			modelrouter.DomainOther: {choiceOf(testutil.CHEAP), choiceOf(testutil.EXPENSIVE)},
		}
		router := newRouterWithTelemetryOptions(t, env, opts)

		result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
		if err != nil {
			t.Fatalf("Complete() error = %v, want the fallback to recover", err)
		}
		if result.RoutingDecision.ModelID != testutil.EXPENSIVE.ID {
			t.Errorf("RoutingDecision.ModelID = %q, want the fallback winner %q",
				result.RoutingDecision.ModelID, testutil.EXPENSIVE.ID)
		}

		completeSpan := requireSpan(t, env, "modelrouter.complete")
		if got := spanAttr(t, completeSpan, "modelrouter.attempts").AsInt64(); got != 2 {
			t.Errorf("complete modelrouter.attempts = %d, want 2", got)
		}
		if got := spanAttr(t, completeSpan, "modelrouter.model_id").AsString(); got != testutil.EXPENSIVE.ID {
			t.Errorf("complete modelrouter.model_id = %q, want %q", got, testutil.EXPENSIVE.ID)
		}
		if completeSpan.Status().Code != codes.Unset {
			t.Errorf("complete span status = %v, want unset (the fallback recovered)", completeSpan.Status().Code)
		}
		if len(completeSpan.Events()) == 0 {
			t.Error("complete span recorded no event for the failed attempt")
		}

		collected := collectMetrics(t, env.reader)
		errorPoint := requireSum(t, collected, "modelrouter.upstream.errors")
		if errorPoint.Value != 1 {
			t.Errorf("modelrouter.upstream.errors value = %d, want 1 (one failed attempt)", errorPoint.Value)
		}
		// The route-time decision counted once, before the completion ran.
		decisionPoint := requireSum(t, collected, "modelrouter.decisions")
		if decisionPoint.Value != 1 {
			t.Errorf("modelrouter.decisions value = %d, want 1", decisionPoint.Value)
		}
		if got := metricAttr(t, decisionPoint.Attributes, "modelrouter.source").AsString(); got != modelrouter.SourceClassification {
			t.Errorf("decisions modelrouter.source = %q, want %q", got, modelrouter.SourceClassification)
		}
	})
}

// TestCompleteProviderErrorSetsSpanStatusAndCounter covers the all-attempts-
// failed path: the complete span ends with an error status and one attempt,
// with no winning model attributes, and modelrouter.upstream.errors counts
// the failure with provider and op attributes.
func TestCompleteProviderErrorSetsSpanStatusAndCounter(t *testing.T) {
	env := newTelemetryEnv(t)
	providerErr := &modelrouter.ProviderError{
		Provider:   testutil.CHEAP.Provider,
		Op:         "complete",
		StatusCode: 502,
		Err:        errors.New("upstream down"),
	}
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{
		Profile: testutil.CONFIDENT_TRIVIAL_PROFILE,
	}}
	completion := &testutil.FakeCompletion{Err: providerErr}
	router := newRouterWithTelemetry(t, env, catalog, classifier, completion)

	_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
	var gotErr *modelrouter.ProviderError
	if !errors.As(err, &gotErr) || gotErr != providerErr {
		t.Fatalf("Complete() error = %v, want the fake *ProviderError unchanged", err)
	}

	completeSpan := requireSpan(t, env, "modelrouter.complete")
	if completeSpan.Status().Code != codes.Error {
		t.Errorf("complete span status = %v, want error", completeSpan.Status().Code)
	}
	if !strings.Contains(completeSpan.Status().Description, "upstream down") {
		t.Errorf("complete span status description = %q, want it to carry the provider failure", completeSpan.Status().Description)
	}
	if len(completeSpan.Events()) == 0 {
		t.Error("complete span recorded no error event")
	}
	if got := spanAttr(t, completeSpan, "modelrouter.attempts").AsInt64(); got != 1 {
		t.Errorf("complete modelrouter.attempts = %d, want 1 (single-candidate chain)", got)
	}
	// No candidate ever succeeded, so the span names no winner.
	if spanHasAttr(completeSpan, "modelrouter.model_id") {
		t.Error("complete span has modelrouter.model_id after total failure, want none")
	}

	collected := collectMetrics(t, env.reader)
	errorPoint := requireSum(t, collected, "modelrouter.upstream.errors")
	if errorPoint.Value != 1 {
		t.Errorf("modelrouter.upstream.errors value = %d, want 1", errorPoint.Value)
	}
	if got := metricAttr(t, errorPoint.Attributes, "modelrouter.provider").AsString(); got != testutil.CHEAP.Provider {
		t.Errorf("upstream.errors modelrouter.provider = %q, want %q", got, testutil.CHEAP.Provider)
	}
	if got := metricAttr(t, errorPoint.Attributes, "modelrouter.op").AsString(); got != "complete" {
		t.Errorf("upstream.errors modelrouter.op = %q, want %q", got, "complete")
	}

	// Routing succeeded before the completion failed, so the decision still
	// counted.
	decisionPoint := requireSum(t, collected, "modelrouter.decisions")
	if decisionPoint.Value != 1 {
		t.Errorf("modelrouter.decisions value = %d, want 1", decisionPoint.Value)
	}
}

// TestUpstreamErrorsCountsEveryFailedAttempt pins the per-attempt counter:
// two failed candidates mean +2, aggregated into one data point because both
// attempts share the provider/op attributes.
func TestUpstreamErrorsCountsEveryFailedAttempt(t *testing.T) {
	env := newTelemetryEnv(t)
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
	completion := &testutil.FakeCompletion{Errs: []error{
		&modelrouter.ProviderError{Provider: testutil.CHEAP.Provider, Op: "complete", Err: errors.New("first down")},
		&modelrouter.ProviderError{Provider: testutil.CHEAP.Provider, Op: "complete", Err: errors.New("second down")},
	}}
	opts := testutil.Options(catalog, classifier, completion)
	opts.Models = map[modelrouter.Domain][]modelrouter.ModelChoice{
		modelrouter.DomainOther: {choiceOf(testutil.CHEAP), choiceOf(testutil.EXPENSIVE)},
	}
	router := newRouterWithTelemetryOptions(t, env, opts)

	_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
	if err == nil {
		t.Fatal("Complete() error = nil, want the joined attempt failures")
	}
	if !strings.HasPrefix(err.Error(), "modelrouter: all 2 candidate models failed") {
		t.Errorf("error = %q, want the all-candidates-failed wrapper", err)
	}

	completeSpan := requireSpan(t, env, "modelrouter.complete")
	if got := spanAttr(t, completeSpan, "modelrouter.attempts").AsInt64(); got != 2 {
		t.Errorf("complete modelrouter.attempts = %d, want 2", got)
	}
	if completeSpan.Status().Code != codes.Error {
		t.Errorf("complete span status = %v, want error", completeSpan.Status().Code)
	}
	if got := len(completeSpan.Events()); got < 2 {
		t.Errorf("complete span events = %d, want at least 2 (one per failed attempt)", got)
	}

	errorPoint := requireSum(t, collectMetrics(t, env.reader), "modelrouter.upstream.errors")
	if errorPoint.Value != 2 {
		t.Errorf("modelrouter.upstream.errors value = %d, want 2 (one per failed attempt)", errorPoint.Value)
	}
}
