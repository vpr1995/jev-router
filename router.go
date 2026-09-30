// This file orchestrates one request: classify, select the configured
// candidate chain for the resulting domain, filter it against hard
// constraints, and dispatch completions — trying the chain in order and
// falling back to the default model.
//
// A few things worth knowing that aren't obvious from the code:
//
//   - Catalogs are fetched only when a constraint needs their metadata
//     (needs_vision, min_context_tokens, max_price_per_1k_tokens); provider
//     allow/exclude and chain selection need no catalog. When catalogs are
//     fetched, a failing one is never silently skipped — Route fails with
//     every catalog error joined.
//   - The Router constructs nothing and closes nothing: catalogs, the
//     classifier and completions are injected and owned by assembly code, so
//     there is deliberately no Close method.
//   - Classification failure never fabricates a profile: fail-open routes on
//     the default model with a nil Profile, fail-closed returns the
//     *ClassifierUnavailableError unchanged. A nil Classifier is treated as
//     unavailable rather than constructed here — the root package must not
//     import classify/.
//   - Route/Complete/ListModels emit OpenTelemetry spans and metrics through
//     the OTel API only; the tracer/meter providers default to the otel
//     globals (no-ops unless internal/telemetry installed SDK providers), so
//     the root package never imports the OTel SDK.
package modelrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

// OnClassifierError selects what the Router does when the Jev classification
// call cannot produce a profile.
type OnClassifierError string

const (
	// OnClassifierErrorFailOpen routes unclassified requests to the
	// configured default model (nil Profile) and logs a warning. It is the
	// default when the option is empty.
	OnClassifierErrorFailOpen OnClassifierError = "fail_open"
	// OnClassifierErrorFailClosed returns the classification error to the
	// caller unchanged.
	OnClassifierErrorFailClosed OnClassifierError = "fail_closed"
)

// Instrumentation identity: span names, metric names and attribute keys are
// part of the observable contract, so they stay as stable dotted strings.
const (
	// tracerName is shared by the router's tracer and meter.
	tracerName = "modelrouter"

	// Span names.
	spanRoute        = "modelrouter.route"
	spanCatalogFetch = "modelrouter.catalog.fetch"
	spanClassify     = "modelrouter.classify"
	spanModels       = "modelrouter.models"
	spanComplete     = "modelrouter.complete"

	// Span attribute keys.
	attrCatalogIndex         = "modelrouter.catalog.index"
	attrDomain               = "modelrouter.domain"
	attrComplexityScore      = "modelrouter.complexity_score"
	attrComplexityConfidence = "modelrouter.complexity_confidence"
	attrFailOpen             = "modelrouter.fail_open"
	attrCandidates           = "modelrouter.candidates"
	attrAttempts             = "modelrouter.attempts"
	attrSource               = "modelrouter.source"
	attrFallback             = "modelrouter.fallback"
	attrModelsCount          = "modelrouter.models.count"
	attrModelID              = "modelrouter.model_id"
	attrProvider             = "modelrouter.provider"
	attrOp                   = "modelrouter.op"

	// Metric instrument names.
	metricDecisions        = "modelrouter.decisions"
	metricClassifyDuration = "modelrouter.classify.duration"
	metricUpstreamErrors   = "modelrouter.upstream.errors"
)

// RouterOptions configures New.
//
// The zero value is not usable: at least one Catalog is required and at least
// one of Models or DefaultModel must be configured. Callers own every
// injected instance; the Router never closes them.
type RouterOptions struct {
	// Catalogs are the model sources consulted by ListModels and by Route
	// when a constraint needs model metadata (vision, context, price). At
	// least one is required. Any catalog failure on fetch makes Route return
	// an error joining every failure instead of skipping it.
	Catalogs []Catalog
	// Classifier is the request classifier (normally the Jev client).
	// Optional: nil is treated as unavailable and handled according to
	// OnClassifierError.
	Classifier Classifier
	// Completions maps provider names to completion implementations. A model
	// can be selected without a registered provider; the gap surfaces from
	// Complete as ErrProviderNotConfigured and — when the chain has further
	// candidates — advances the fallback.
	Completions map[string]Completion
	// Logger receives operational warnings (currently the fail-open
	// classifier fallback). Nil means slog.Default().
	Logger *slog.Logger
	// OnClassifierError selects fail-open (default when empty) or
	// fail-closed classification-error behavior.
	OnClassifierError OnClassifierError
	// Models maps each classified domain to an ordered chain of provider
	// model candidates. Route selects Models[domain] when the classifier
	// returned that domain; Complete then tries the chain in order,
	// advancing on any provider failure. Every chain must be non-empty with
	// non-empty Provider/Model fields; New rejects unknown domains and empty
	// chains.
	Models map[Domain][]ModelChoice
	// DefaultModel is used when there is no classification, or the
	// classified domain has no configured chain, and is appended to the
	// attempt chain (when not already present) as the last-resort fallback
	// for exhausted chains. Both Provider and Model must be non-empty.
	DefaultModel *ModelChoice
	// TracerProvider supplies the tracer behind the router's spans. Nil means
	// otel.GetTracerProvider(), a no-op unless the application installed an
	// SDK provider (for example via internal/telemetry.Setup).
	TracerProvider trace.TracerProvider
	// MeterProvider supplies the meter behind the router's metrics
	// (modelrouter.decisions, modelrouter.classify.duration and
	// modelrouter.upstream.errors). Nil means otel.GetMeterProvider().
	MeterProvider metric.MeterProvider
}

// Router orchestrates the routing pipeline and dispatches completions.
//
// A Router is immutable after New, so a single instance may be shared by any
// number of goroutines; Route, Complete and ListModels are safe for
// concurrent use provided the injected Catalogs, Classifier and Completions
// are themselves safe for concurrent use.
//
// The Router constructs nothing and closes nothing: injected catalogs,
// classifier and completions remain the caller's property for their whole
// lifetime, so there is deliberately no Close method. Assembly code owns
// construction and cleanup.
type Router struct {
	catalogs    []Catalog
	classifier  Classifier
	completions map[string]Completion

	logger            *slog.Logger
	onClassifierError OnClassifierError

	models       map[Domain][]ModelChoice
	defaultModel *ModelChoice

	// Telemetry handles, safe to use with the default no-op providers.
	tracer           trace.Tracer
	decisions        metric.Int64Counter
	classifyDuration metric.Float64Histogram
	upstreamErrors   metric.Int64Counter
}

// New validates opts and builds a Router.
//
// Empty OnClassifierError becomes OnClassifierErrorFailOpen and a nil Logger
// becomes slog.Default(). Models, DefaultModel, Catalogs and Completions are
// all copied, so later caller mutations cannot affect the Router.
func New(opts RouterOptions) (*Router, error) {
	if len(opts.Catalogs) == 0 {
		return nil, errors.New("modelrouter: at least one catalog is required")
	}

	onClassifierError := opts.OnClassifierError
	if onClassifierError == "" {
		onClassifierError = OnClassifierErrorFailOpen
	}
	if onClassifierError != OnClassifierErrorFailOpen && onClassifierError != OnClassifierErrorFailClosed {
		return nil, fmt.Errorf("modelrouter: invalid OnClassifierError %q (want %q or %q)",
			onClassifierError, OnClassifierErrorFailOpen, OnClassifierErrorFailClosed)
	}

	if len(opts.Models) == 0 && opts.DefaultModel == nil {
		return nil, errors.New("modelrouter: at least one of Models or DefaultModel is required")
	}
	models := make(map[Domain][]ModelChoice, len(opts.Models))
	for domain, chain := range opts.Models {
		if !IsValidDomain(domain) {
			return nil, fmt.Errorf("modelrouter: invalid domain %q in Models (want %q, %q, %q, %q or %q)",
				domain, DomainCode, DomainMathReasoning, DomainCreative, DomainFactualLookup, DomainOther)
		}
		if len(chain) == 0 {
			return nil, fmt.Errorf("modelrouter: Models[%q] must contain at least one model", domain)
		}
		for i, choice := range chain {
			if choice.Provider == "" {
				return nil, fmt.Errorf("modelrouter: Models[%q][%d].Provider must not be empty", domain, i)
			}
			if choice.Model == "" {
				return nil, fmt.Errorf("modelrouter: Models[%q][%d].Model must not be empty", domain, i)
			}
		}
		models[domain] = append([]ModelChoice(nil), chain...)
	}
	var defaultModel *ModelChoice
	if opts.DefaultModel != nil {
		if opts.DefaultModel.Provider == "" {
			return nil, errors.New("modelrouter: DefaultModel.Provider must not be empty")
		}
		if opts.DefaultModel.Model == "" {
			return nil, errors.New("modelrouter: DefaultModel.Model must not be empty")
		}
		copied := *opts.DefaultModel
		defaultModel = &copied
	}

	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	tracerProvider := opts.TracerProvider
	if tracerProvider == nil {
		tracerProvider = otel.GetTracerProvider()
	}
	meterProvider := opts.MeterProvider
	if meterProvider == nil {
		meterProvider = otel.GetMeterProvider()
	}
	tracer := tracerProvider.Tracer(tracerName)
	meter := meterProvider.Meter(tracerName)
	// Instrument names are fixed and valid, so construction cannot fail in
	// practice; ignoring the error just degrades to telemetry-less routing
	// instead of blocking it.
	decisions, _ := meter.Int64Counter(metricDecisions,
		metric.WithDescription("Routing decisions produced by successful Route calls."))
	classifyDuration, _ := meter.Float64Histogram(metricClassifyDuration,
		metric.WithUnit("s"),
		metric.WithDescription("Latency of the single classifier call, in seconds."))
	upstreamErrors, _ := meter.Int64Counter(metricUpstreamErrors,
		metric.WithDescription("Provider operation failures observed by Complete."))

	catalogs := append([]Catalog(nil), opts.Catalogs...)
	var completions map[string]Completion
	if opts.Completions != nil {
		completions = make(map[string]Completion, len(opts.Completions))
		for name, completion := range opts.Completions {
			completions[name] = completion
		}
	}

	return &Router{
		catalogs:          catalogs,
		classifier:        opts.Classifier,
		completions:       completions,
		logger:            logger,
		onClassifierError: onClassifierError,
		models:            models,
		defaultModel:      defaultModel,
		tracer:            tracer,
		decisions:         decisions,
		classifyDuration:  classifyDuration,
		upstreamErrors:    upstreamErrors,
	}, nil
}

// Route classifies prompt with one classifier call and selects the configured
// candidate chain for the outcome.
//
// The selected chain is Models[classified domain] (DefaultModel appended when
// set and not already present) when the classifier returned a domain with a
// configured chain, and just DefaultModel otherwise — no classification under
// fail_open, or a domain without a chain. DefaultModel must be configured for
// those cases; without one, Route returns a plain configuration error.
//
// Constraints apply to the chain, not to a catalog pool: allowed/excluded
// providers always, and needs_vision/min_context_tokens/max_price_per_1k_tokens
// only when set — which fetches the catalogs once (all-must-succeed; a
// candidate missing from the catalogs fails the metadata checks like an
// incompatible model). When every candidate is filtered out, Route returns a
// *NoCandidateModelError naming the configured candidates (plus catalog
// diagnostics when catalogs were fetched).
//
// When classification is unavailable, fail_closed returns the
// *ClassifierUnavailableError unchanged; fail_open proceeds with a nil
// Profile on the default model and logs a warning.
//
// Route is instrumented: one modelrouter.route span wraps the call, with a
// modelrouter.catalog.fetch child only when a fetch happened and a
// modelrouter.classify child always; a successful call also increments the
// modelrouter.decisions counter.
func (r *Router) Route(ctx context.Context, prompt string, constraints *RoutingConstraints) (RoutingDecision, error) {
	ctx, span := r.tracer.Start(ctx, spanRoute)
	defer span.End()

	c := RoutingConstraints{}
	if constraints != nil {
		c = *constraints
	}

	// Catalogs are consulted only when a constraint needs model metadata.
	var all []ModelInfo
	if constraintsNeedCatalog(c) {
		var err error
		all, err = r.fetchAllModels(ctx)
		if err != nil {
			recordSpanError(span, err)
			return RoutingDecision{}, err
		}
	}

	// The classify span's duration is recorded whether or not it succeeds.
	started := time.Now()
	_, classifySpan := r.tracer.Start(ctx, spanClassify)
	classification, classifyErr := r.classify(ctx, prompt)
	classifySeconds := time.Since(started).Seconds()

	var profile *RequestProfile
	var unavailable *ClassifierUnavailableError
	failOpen := classifyErr != nil &&
		errors.As(classifyErr, &unavailable) &&
		r.onClassifierError != OnClassifierErrorFailClosed
	if classifyErr == nil {
		profile = &classification.Profile
		classifySpan.SetAttributes(
			attribute.String(attrDomain, string(classification.Profile.Domain)),
			attribute.Float64(attrComplexityScore, classification.Profile.ComplexityScore),
			attribute.Float64(attrComplexityConfidence, classification.Profile.ComplexityConfidence),
			attribute.Bool(attrFailOpen, false),
		)
	} else {
		// Recorded as a span event either way; only an unhandled failure
		// switches the span to error status (the fail-open fallback is
		// visible through modelrouter.fail_open instead).
		classifySpan.RecordError(classifyErr)
		classifySpan.SetAttributes(attribute.Bool(attrFailOpen, failOpen))
		if !failOpen {
			classifySpan.SetStatus(codes.Error, classifyErr.Error())
		}
	}
	r.classifyDuration.Record(ctx, classifySeconds)
	classifySpan.End()

	if classifyErr != nil {
		if !failOpen {
			recordSpanError(span, classifyErr)
			return RoutingDecision{}, classifyErr
		}
		r.logger.Warn("classifier unavailable; using the default model",
			"error", classifyErr)
		classification = Classification{}
	}

	chain, source, err := r.selectCandidates(profile)
	if err != nil {
		recordSpanError(span, err)
		return RoutingDecision{}, err
	}
	candidates := filteredChoices(chain, all, c)
	if len(candidates) == 0 {
		err := &NoCandidateModelError{Message: noCandidateMessage(chain, all, c)}
		recordSpanError(span, err)
		return RoutingDecision{}, err
	}

	raw := classification.Raw
	if len(raw) == 0 {
		raw = nil
	}
	decision := RoutingDecision{
		ModelID:         candidates[0].Model,
		Provider:        candidates[0].Provider,
		Source:          source,
		Candidates:      candidates,
		Profile:         profile,
		RawJevResponse:  raw,
		ClassifierUsage: classification.Usage,
	}
	// domain is empty and modelrouter.fallback true when selection fell back
	// to the default model without a profile.
	domain := ""
	if profile != nil {
		domain = string(profile.Domain)
	}
	span.SetAttributes(
		attribute.String(attrModelID, decision.ModelID),
		attribute.String(attrProvider, decision.Provider),
		attribute.String(attrSource, source),
		attribute.Int(attrCandidates, len(candidates)),
		attribute.String(attrDomain, domain),
		attribute.Bool(attrFailOpen, profile == nil),
	)
	r.decisions.Add(ctx, 1, metric.WithAttributes(
		attribute.String(attrModelID, decision.ModelID),
		attribute.String(attrProvider, decision.Provider),
		attribute.Bool(attrFallback, profile == nil),
		attribute.String(attrDomain, domain),
		attribute.String(attrSource, source),
	))
	return decision, nil
}

// selectCandidates builds the ordered attempt chain for a classification
// outcome: the domain's configured chain (SourceClassification), with the
// default appended as a last-resort fallback when not already present, or the
// default model alone (SourceDefault).
func (r *Router) selectCandidates(profile *RequestProfile) ([]ModelChoice, string, error) {
	if profile != nil {
		if chain := r.models[profile.Domain]; len(chain) > 0 {
			return appendDefault(chain, r.defaultModel), SourceClassification, nil
		}
		if r.defaultModel != nil {
			return []ModelChoice{*r.defaultModel}, SourceDefault, nil
		}
		return nil, "", fmt.Errorf("modelrouter: no model mapping for domain %q and no default model", profile.Domain)
	}
	if r.defaultModel != nil {
		return []ModelChoice{*r.defaultModel}, SourceDefault, nil
	}
	return nil, "", errors.New("modelrouter: no default model configured for unclassified request")
}

// appendDefault returns a fresh chain with def appended, unless def is nil or
// already present. The input slice is never modified.
func appendDefault(chain []ModelChoice, def *ModelChoice) []ModelChoice {
	out := append([]ModelChoice(nil), chain...)
	if def == nil {
		return out
	}
	for _, choice := range out {
		if choice == *def {
			return out
		}
	}
	return append(out, *def)
}

// constraintsNeedCatalog reports whether Route must fetch the catalogs.
func constraintsNeedCatalog(c RoutingConstraints) bool {
	return c.NeedsVision || c.MinContextTokens > 0 || c.MaxPricePer1KTokens != nil
}

// filteredChoices applies constraints to the configured chain. Provider
// allow/exclude filters always apply; metadata checks apply only when the
// catalogs were fetched (all != nil), and a candidate missing from them fails
// those checks — its capabilities and price cannot be verified.
func filteredChoices(chain []ModelChoice, all []ModelInfo, c RoutingConstraints) []ModelChoice {
	out := make([]ModelChoice, 0, len(chain))
	for _, choice := range chain {
		if !c.AllowsProvider(choice.Provider) {
			continue
		}
		if all != nil && !choiceSatisfies(choice, all, c) {
			continue
		}
		out = append(out, choice)
	}
	return out
}

// choiceSatisfies reports whether one candidate passes the catalog-backed
// constraints (vision, context, price). A candidate the catalogs do not
// contain fails.
func choiceSatisfies(choice ModelChoice, all []ModelInfo, c RoutingConstraints) bool {
	for _, model := range all {
		if model.Provider != choice.Provider || model.ID != choice.Model {
			continue
		}
		if c.NeedsVision && !model.SupportsVision {
			return false
		}
		if c.MinContextTokens > 0 && model.ContextLength < c.MinContextTokens {
			return false
		}
		if c.MaxPricePer1KTokens != nil && model.BlendedPricePer1KTokens() > *c.MaxPricePer1KTokens {
			return false
		}
		return true
	}
	return false
}

// noCandidateMessage explains that every configured candidate was filtered
// out, reusing the catalog-level diagnostics when the catalogs were fetched.
func noCandidateMessage(chain []ModelChoice, all []ModelInfo, c RoutingConstraints) string {
	message := fmt.Sprintf("no configured candidate model satisfies the constraints (candidates: %s)",
		describeChoices(chain))
	if all != nil {
		message += "; " + DescribeEliminatingConstraints(all, c)
	}
	return message
}

// describeChoices renders an ordered chain as "provider/model" items.
func describeChoices(chain []ModelChoice) string {
	parts := make([]string, len(chain))
	for i, choice := range chain {
		parts[i] = choice.Provider + "/" + choice.Model
	}
	return strings.Join(parts, ", ")
}

// ListModels returns every model the configured catalogs serve after applying
// constraints, preserving catalog order and each catalog's own order.
//
// It mirrors Route's catalog stage: nil constraints means the zero
// RoutingConstraints, and a failure anywhere returns an error joining every
// failure (no partial pool). Unlike Route, an empty result is not an error —
// "nothing matches these filters" is a valid answer to "what can I route to".
//
// ListModels is instrumented with a modelrouter.models span whose
// modelrouter.models.count attribute is the size of the returned pool.
func (r *Router) ListModels(ctx context.Context, constraints *RoutingConstraints) ([]ModelInfo, error) {
	ctx, span := r.tracer.Start(ctx, spanModels)
	defer span.End()

	c := RoutingConstraints{}
	if constraints != nil {
		c = *constraints
	}

	all, err := r.fetchAllModels(ctx)
	if err != nil {
		recordSpanError(span, err)
		return nil, err
	}
	models := FilterModels(all, c)
	span.SetAttributes(attribute.Int(attrModelsCount, len(models)))
	return models, nil
}

// fetchAllModels calls every catalog's Models in order and concatenates the
// results. When any catalog fails, no partial pool is returned: the error
// joins every failure, shared by Route and ListModels.
func (r *Router) fetchAllModels(ctx context.Context) ([]ModelInfo, error) {
	all := make([]ModelInfo, 0)
	var failures []error
	for index, catalog := range r.catalogs {
		models, err := r.fetchCatalog(ctx, catalog, index)
		if err != nil {
			failures = append(failures, err)
			continue
		}
		all = append(all, models...)
	}
	if len(failures) > 0 {
		return nil, errors.Join(failures...)
	}
	return all, nil
}

// fetchCatalog calls one catalog's Models inside its own
// modelrouter.catalog.fetch span.
func (r *Router) fetchCatalog(ctx context.Context, catalog Catalog, index int) ([]ModelInfo, error) {
	ctx, span := r.tracer.Start(ctx, spanCatalogFetch)
	defer span.End()
	span.SetAttributes(attribute.Int(attrCatalogIndex, index))

	models, err := catalog.Models(ctx)
	if err != nil {
		recordSpanError(span, err)
		return nil, err
	}
	return models, nil
}

// Complete routes req (unless Model is set explicitly) and executes it,
// falling back through the decision's candidate chain.
//
// Either Prompt or Messages must be set; the Router does not materialize one
// from the other — providers map whichever field the caller set onto their
// native wire format.
//
// When req.Model is empty, Complete runs Route, then tries
// decision.Candidates in order. Any failure on an attempt — provider error,
// transport failure, or a provider with no registered Completion — advances
// to the next candidate; a success carries the decision updated to the
// candidate that actually served the request. When every attempt fails, a
// single-candidate chain returns that error unchanged, and a longer chain
// returns them wrapped as "all N candidate models failed" (errors.Join, so
// errors.Is/As still reach the inner errors). The loop stops early when ctx
// is done.
//
// When req.Model is set explicitly, req.Provider is required and neither
// routing nor fallback happens: exactly one provider attempt is made.
//
// Complete is instrumented with a modelrouter.complete span covering routing
// and the provider attempts; it is marked failed only when the whole
// operation fails, so a recovered fallback stays visible as span events
// without an error status.
func (r *Router) Complete(ctx context.Context, req CompletionRequest) (CompletionResult, error) {
	ctx, span := r.tracer.Start(ctx, spanComplete)
	defer span.End()

	if err := validateCompletionRequest(req); err != nil {
		r.recordFailure(ctx, span, err, "complete")
		return CompletionResult{}, err
	}
	if req.Model != "" {
		return r.completeExplicit(ctx, span, req)
	}

	decision, err := r.Route(ctx, routingPrompt(req), req.Constraints)
	if err != nil {
		r.recordFailure(ctx, span, err, "complete")
		return CompletionResult{}, err
	}
	return r.completeChain(ctx, span, req, decision)
}

// routingPrompt is the text classified for req: Prompt when set, otherwise the
// last user message (or the last message when none has the user role). It only
// feeds classification; providers still receive req unchanged.
func routingPrompt(req CompletionRequest) string {
	if req.Prompt != "" || len(req.Messages) == 0 {
		return req.Prompt
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			return req.Messages[i].Content
		}
	}
	return req.Messages[len(req.Messages)-1].Content
}

// completeExplicit executes an explicit-model request: no classifier, no
// fallback.
func (r *Router) completeExplicit(ctx context.Context, span trace.Span, req CompletionRequest) (CompletionResult, error) {
	if req.Provider == "" {
		err := errors.New("modelrouter: provider is required when model is set explicitly")
		r.recordFailure(ctx, span, err, "complete")
		return CompletionResult{}, err
	}
	completion, err := r.completionFor(req.Provider, "complete")
	if err != nil {
		r.recordFailure(ctx, span, err, "complete")
		return CompletionResult{}, err
	}
	span.SetAttributes(
		attribute.String(attrModelID, req.Model),
		attribute.String(attrProvider, req.Provider),
	)
	result, err := completion.Complete(ctx, req)
	if err != nil {
		r.recordFailure(ctx, span, err, "complete")
		return CompletionResult{}, err
	}
	// An explicit-model call was never routed, so the decision stays zero.
	result.RoutingDecision = RoutingDecision{}
	return result, nil
}

// completeChain tries every candidate in order until one succeeds.
func (r *Router) completeChain(ctx context.Context, span trace.Span, req CompletionRequest, decision RoutingDecision) (CompletionResult, error) {
	failures := make([]error, 0, len(decision.Candidates))
	for _, candidate := range decision.Candidates {
		result, err := r.attemptComplete(ctx, req, candidate)
		if err == nil {
			span.SetAttributes(
				attribute.Int(attrAttempts, len(failures)+1),
				attribute.String(attrModelID, candidate.Model),
				attribute.String(attrProvider, candidate.Provider),
			)
			decision.ModelID, decision.Provider = candidate.Model, candidate.Provider
			result.RoutingDecision = decision
			return result, nil
		}
		failures = append(failures, err)
		r.recordAttemptFailure(ctx, span, err, "complete")
		if ctx.Err() != nil {
			break
		}
	}
	span.SetAttributes(attribute.Int(attrAttempts, len(failures)))
	final := allCandidatesFailedError(failures)
	recordSpanError(span, final)
	return CompletionResult{}, final
}

// attemptComplete performs one provider attempt for candidate.
func (r *Router) attemptComplete(ctx context.Context, req CompletionRequest, candidate ModelChoice) (CompletionResult, error) {
	completion, err := r.completionFor(candidate.Provider, "complete")
	if err != nil {
		return CompletionResult{}, err
	}
	attempt := req
	attempt.Model = candidate.Model
	attempt.Provider = candidate.Provider
	return completion.Complete(ctx, attempt)
}

// classify runs the injected classifier, or reports unavailability when none
// is configured.
func (r *Router) classify(ctx context.Context, prompt string) (Classification, error) {
	if r.classifier == nil {
		return Classification{}, &ClassifierUnavailableError{
			Err: errors.New("no classifier configured"),
		}
	}
	return r.classifier.Classify(ctx, prompt)
}

// validateCompletionRequest enforces the prompt-or-messages rule used by
// Complete.
func validateCompletionRequest(req CompletionRequest) error {
	if req.Prompt == "" && len(req.Messages) == 0 {
		return errors.New("modelrouter: prompt or messages required")
	}
	return nil
}

// completionFor looks up the provider's completion, reporting a registry gap
// as a *ProviderError wrapping ErrProviderNotConfigured.
func (r *Router) completionFor(provider, op string) (Completion, error) {
	completion, ok := r.completions[provider]
	if !ok || completion == nil {
		return nil, &ProviderError{Provider: provider, Op: op, Err: ErrProviderNotConfigured}
	}
	return completion, nil
}

// allCandidatesFailedError aggregates attempt failures: a single attempt
// returns the original error unchanged, several are joined behind one
// message so errors.Is/As still reach every inner error.
func allCandidatesFailedError(failures []error) error {
	switch len(failures) {
	case 0:
		// Unreachable: Route never returns an empty candidate chain.
		return errors.New("modelrouter: no candidate models to try")
	case 1:
		return failures[0]
	default:
		return fmt.Errorf("modelrouter: all %d candidate models failed: %w", len(failures), errors.Join(failures...))
	}
}

// recordSpanError records err as a span event and marks the span failed; it
// is used for errors the instrumented operation returns to its caller.
func recordSpanError(span trace.Span, err error) {
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

// recordFailure records err on span, marks the span failed and, when err is a
// *ProviderError, increments modelrouter.upstream.errors with the failing
// provider and operation. op is the fallback operation name for
// *ProviderError values that carry none of their own.
func (r *Router) recordFailure(ctx context.Context, span trace.Span, err error, op string) {
	recordSpanError(span, err)
	r.recordUpstreamError(ctx, err, op)
}

// recordAttemptFailure records one failed provider attempt: the error is
// always added to the span as an event, but the span's error status is only
// set when the whole chain fails (by recordFailure/recordSpanError), so a
// recovered fallback does not mark the operation failed.
func (r *Router) recordAttemptFailure(ctx context.Context, span trace.Span, err error, op string) {
	span.RecordError(err)
	r.recordUpstreamError(ctx, err, op)
}

// recordUpstreamError increments modelrouter.upstream.errors when err is a
// *ProviderError.
func (r *Router) recordUpstreamError(ctx context.Context, err error, op string) {
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		return
	}
	if providerErr.Op != "" {
		op = providerErr.Op
	}
	r.upstreamErrors.Add(ctx, 1, metric.WithAttributes(
		attribute.String(attrProvider, providerErr.Provider),
		attribute.String(attrOp, op),
	))
}
