// Package testutil provides shared test doubles for Router tests: fake
// catalogs, classifiers and completions, plus fixture models/profiles the
// test suite reuses. It is an internal package so root tests can import it
// without internal test-file tricks.
package testutil

import (
	"context"
	"sync"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// CHEAP is a cheap, small-context, text-only model served by the "test"
// provider.
var CHEAP = modelrouter.ModelInfo{
	Provider:                   "test",
	ID:                         "vendor/cheap",
	PricePer1KPromptTokens:     0.1,
	PricePer1KCompletionTokens: 0.1,
	ContextLength:              8_000,
	SupportsVision:             false,
}

// EXPENSIVE is a pricier model with a large context window that supports
// vision.
var EXPENSIVE = modelrouter.ModelInfo{
	Provider:                   "test",
	ID:                         "vendor/expensive",
	PricePer1KPromptTokens:     5.0,
	PricePer1KCompletionTokens: 5.0,
	ContextLength:              128_000,
	SupportsVision:             true,
}

// CONFIDENT_TRIVIAL_PROFILE is a confidently classified trivial request —
// domain "other" (the domain Options maps to a CHEAP candidate) and
// complexity score 0 with high confidence.
var CONFIDENT_TRIVIAL_PROFILE = modelrouter.RequestProfile{
	Domain:               modelrouter.DomainOther,
	DomainConfidence:     0.9,
	ComplexityScore:      0.0,
	ComplexityConfidence: 0.9,
	NeedsLongContext:     0.0,
	NeedsVision:          0.0,
	LatencySensitive:     0.0,
}

// Compile-time interface checks.
var (
	_ modelrouter.Catalog    = (*FakeCatalog)(nil)
	_ modelrouter.Classifier = (*FakeClassifier)(nil)
	_ modelrouter.Completion = (*FakeCompletion)(nil)
)

// FakeCatalog is a Catalog test double. It returns ModelsList (or Err, when
// set) and counts calls; all state is mutex-protected so a single fake can
// back a Router exercised concurrently.
type FakeCatalog struct {
	// ModelsList is returned by Models. See the package comment for why it
	// is not named Models.
	ModelsList []modelrouter.ModelInfo
	// Err, when non-nil, is returned by every Models call.
	Err error

	mu    sync.Mutex
	calls int
}

// Models implements modelrouter.Catalog.
func (c *FakeCatalog) Models(_ context.Context) ([]modelrouter.ModelInfo, error) {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
	if c.Err != nil {
		return nil, c.Err
	}
	return c.ModelsList, nil
}

// Calls reports how many times Models was called.
func (c *FakeCatalog) Calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// FakeClassifier is a Classifier test double returning Classification (or
// Err, when set) and counting calls.
type FakeClassifier struct {
	// Classification is returned when Err is nil.
	Classification modelrouter.Classification
	// Err, when non-nil, is returned by every Classify call. Router tests
	// use a *modelrouter.ClassifierUnavailableError here to exercise
	// fail-open/fail-closed handling.
	Err error

	mu      sync.Mutex
	calls   int
	prompts []string
}

// Classify implements modelrouter.Classifier.
func (c *FakeClassifier) Classify(_ context.Context, prompt string) (modelrouter.Classification, error) {
	c.mu.Lock()
	c.calls++
	c.prompts = append(c.prompts, prompt)
	c.mu.Unlock()
	if c.Err != nil {
		return modelrouter.Classification{}, c.Err
	}
	return c.Classification, nil
}

// Prompts returns every prompt passed to Classify, in call order.
func (c *FakeClassifier) Prompts() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.prompts...)
}

// Calls reports how many times Classify was called.
func (c *FakeClassifier) Calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// FakeCompletion is a Completion test double. Complete replays Result/Errs
// and records every request so tests can assert what the Router filled in,
// including the per-attempt requests of a fallback chain.
type FakeCompletion struct {
	// Result is returned by Complete when the call has no scripted error.
	Result modelrouter.CompletionResult
	// Err, when non-nil, is returned by Complete calls that ran out of
	// scripted errors (or when no script is configured at all).
	Err error
	// Errs are consumed one per Complete call: attempt N of a fallback chain
	// sees Errs[N]. A nil entry makes that attempt succeed; once the slice is
	// exhausted Err applies.
	Errs []error

	mu            sync.Mutex
	requests      []modelrouter.CompletionRequest
	completeCalls int
}

// Complete implements modelrouter.Completion.
func (c *FakeCompletion) Complete(_ context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	c.mu.Lock()
	c.requests = append(c.requests, req)
	err := scriptedError(c.Errs, c.completeCalls, c.Err)
	c.completeCalls++
	c.mu.Unlock()
	if err != nil {
		return modelrouter.CompletionResult{}, err
	}
	return c.Result, nil
}

// scriptedError returns the error for call number calls: the scripted entry
// while the script lasts, the fallback Err afterwards.
func scriptedError(scripted []error, calls int, fallback error) error {
	if calls < len(scripted) {
		return scripted[calls]
	}
	return fallback
}

// Requests returns a copy of every request received so far, in call order.
func (c *FakeCompletion) Requests() []modelrouter.CompletionRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]modelrouter.CompletionRequest(nil), c.requests...)
}

// LastRequest returns the most recent request this fake received.
func (c *FakeCompletion) LastRequest() (modelrouter.CompletionRequest, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.requests) == 0 {
		return modelrouter.CompletionRequest{}, false
	}
	return c.requests[len(c.requests)-1], true
}

// Options builds RouterOptions with the given fakes wired: catalog becomes
// the single catalog, classifier the classifier (nil keeps the no-classifier
// behavior) and completion the Completion registered under CHEAP.Provider
// ("test").
//
// It also installs the single-provider routing config most tests need: the
// fixture domain ("other") maps to CHEAP, and DefaultModel is the same model,
// so a classified request and an unclassified fail-open request both select
// CHEAP as a one-candidate chain. Tests that need other chains overwrite
// opts.Models (and opts.DefaultModel) before passing the result to New.
func Options(catalog *FakeCatalog, classifier *FakeClassifier, completion *FakeCompletion) modelrouter.RouterOptions {
	cheap := modelrouter.ModelChoice{Provider: CHEAP.Provider, Model: CHEAP.ID}
	opts := modelrouter.RouterOptions{
		Models:       map[modelrouter.Domain][]modelrouter.ModelChoice{modelrouter.DomainOther: {cheap}},
		DefaultModel: &cheap,
	}
	if catalog != nil {
		opts.Catalogs = []modelrouter.Catalog{catalog}
	}
	if classifier != nil {
		opts.Classifier = classifier
	}
	if completion != nil {
		opts.Completions = map[string]modelrouter.Completion{CHEAP.Provider: completion}
	}
	return opts
}
