package modelrouter_test

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync/atomic"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/internal/testutil"
)

// These tests pin the Router's ownership contract: it constructs nothing and
// closes nothing. Assembly code injects every dependency and owns its
// lifecycle, so injected components must be used as-is, never wrapped, never
// closed, and never required to be closable.

// closableCatalog is a FakeCatalog with a Close method, used to prove the
// Router never performs lifecycle transitions on injected components.
type closableCatalog struct {
	*testutil.FakeCatalog
	closed atomic.Bool
}

// Close records that the component was closed.
func (c *closableCatalog) Close() { c.closed.Store(true) }

// Closed reports whether Close was called.
func (c *closableCatalog) Closed() bool { return c.closed.Load() }

// closableClassifier is a FakeClassifier with a Close method.
type closableClassifier struct {
	*testutil.FakeClassifier
	closed atomic.Bool
}

// Close records that the component was closed.
func (c *closableClassifier) Close() { c.closed.Store(true) }

// Closed reports whether Close was called.
func (c *closableClassifier) Closed() bool { return c.closed.Load() }

// TestRouterWithNoInjectedClassifierTreatsItAsUnavailable pins that the
// Router never constructs a classifier itself (the root package must not
// import classify/; assembly code does the wiring): no classifier is
// substituted, and an unclassifiable route is reported (under fail_closed)
// rather than silently acquiring a hidden one.
func TestRouterWithNoInjectedClassifierTreatsItAsUnavailable(t *testing.T) {
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	opts := modelrouter.RouterOptions{
		Catalogs:     []modelrouter.Catalog{catalog},
		DefaultModel: defaultChoicePtr(),
	}
	opts.OnClassifierError = modelrouter.OnClassifierErrorFailClosed
	router := mustNewRouter(t, opts)

	_, err := router.Route(context.Background(), "anything", nil)
	if err == nil {
		t.Fatal("Route() error = nil, want *ClassifierUnavailableError (no classifier was injected)")
	}
	var unavailable *modelrouter.ClassifierUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("Route() error = %v (%T), want *ClassifierUnavailableError", err, err)
	}
	if !strings.Contains(err.Error(), "no classifier configured") {
		t.Errorf("error = %v, want it to name the missing classifier", err)
	}
}

// TestRouterUsesInjectedComponentsDirectly pins that the Router uses the
// exact injected instances — observable through their own call counters —
// for routing, listing and completion dispatch. A plain route needs no
// catalog metadata, so the catalog counter is exercised through ListModels.
func TestRouterUsesInjectedComponentsDirectly(t *testing.T) {
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
	completion := &testutil.FakeCompletion{Result: modelrouter.CompletionResult{Content: "ok"}}
	router := mustNewRouter(t, testutil.Options(catalog, classifier, completion))

	if _, err := router.Route(context.Background(), "anything", nil); err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if classifier.Calls() != 1 {
		t.Errorf("classifier.Calls() = %d, want 1 (injected instance used directly)", classifier.Calls())
	}
	if catalog.Calls() != 0 {
		t.Errorf("catalog.Calls() = %d, want 0 after an unconstrained Route", catalog.Calls())
	}
	if _, err := router.ListModels(context.Background(), nil); err != nil {
		t.Fatalf("ListModels() error = %v", err)
	}
	if catalog.Calls() != 1 {
		t.Errorf("catalog.Calls() = %d, want 1 after ListModels", catalog.Calls())
	}

	if _, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "anything"}); err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if classifier.Calls() != 2 {
		t.Errorf("classifier.Calls() = %d, want 2 after Complete", classifier.Calls())
	}
	if _, ok := completion.LastRequest(); !ok {
		t.Error("injected completion never received a request")
	}
}

// TestRouterDoesNotCloseInjectedCatalogOrClassifier pins that components
// which happen to be closable are never closed by the Router (see also
// TestRouterOwnsNothingAndHasNoClose).
func TestRouterDoesNotCloseInjectedCatalogOrClassifier(t *testing.T) {
	catalog := &closableCatalog{FakeCatalog: &testutil.FakeCatalog{
		ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
	}}
	classifier := &closableClassifier{FakeClassifier: &testutil.FakeClassifier{
		Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE},
	}}
	router := mustNewRouter(t, modelrouter.RouterOptions{
		Catalogs:     []modelrouter.Catalog{catalog},
		Classifier:   classifier,
		DefaultModel: defaultChoicePtr(),
	})

	if _, err := router.Route(context.Background(), "anything", nil); err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if catalog.Closed() {
		t.Error("Router closed the injected catalog; injected components are caller-owned")
	}
	if classifier.Closed() {
		t.Error("Router closed the injected classifier; injected components are caller-owned")
	}
}

// TestRouterOwnsNothingAndHasNoClose pins that the Router exposes no close
// capability that could hide lifecycle decisions from the assembling code:
// it constructs nothing, so there is deliberately no Close method.
func TestRouterOwnsNothingAndHasNoClose(t *testing.T) {
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP}}
	router := mustNewRouter(t, modelrouter.RouterOptions{
		Catalogs:     []modelrouter.Catalog{catalog},
		DefaultModel: defaultChoicePtr(),
	})

	if _, ok := any(router).(io.Closer); ok {
		t.Error("Router implements io.Closer; it constructs nothing, so it owns nothing to close")
	}
}

// TestInjectedComponentsRemainUsableAcrossRouterLifetime pins that injected
// components stay usable for as many calls as the caller makes, and their
// lifecycle never transitions.
func TestInjectedComponentsRemainUsableAcrossRouterLifetime(t *testing.T) {
	catalog := &closableCatalog{FakeCatalog: &testutil.FakeCatalog{
		ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
	}}
	classifier := &closableClassifier{FakeClassifier: &testutil.FakeClassifier{
		Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE},
	}}
	completion := &testutil.FakeCompletion{Result: modelrouter.CompletionResult{Content: "ok"}}
	router := mustNewRouter(t, modelrouter.RouterOptions{
		Catalogs:     []modelrouter.Catalog{catalog},
		Classifier:   classifier,
		Completions:  map[string]modelrouter.Completion{testutil.CHEAP.Provider: completion},
		DefaultModel: defaultChoicePtr(),
	})

	for range 3 {
		if _, err := router.Route(context.Background(), "anything", nil); err != nil {
			t.Fatalf("Route() error = %v", err)
		}
	}
	// v2: ListModels is what exercises the injected catalog here; plain routes
	// do not consult it without metadata constraints.
	for range 3 {
		if _, err := router.ListModels(context.Background(), nil); err != nil {
			t.Fatalf("ListModels() error = %v", err)
		}
	}
	for range 2 {
		if _, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "anything"}); err != nil {
			t.Fatalf("Complete() error = %v", err)
		}
	}

	if classifier.Calls() != 5 {
		t.Errorf("classifier.Calls() = %d, want 5 (still usable after many calls)", classifier.Calls())
	}
	if catalog.Calls() != 3 {
		t.Errorf("catalog.Calls() = %d, want 3 (one per ListModels call)", catalog.Calls())
	}
	if catalog.Closed() || classifier.Closed() {
		t.Error("components were closed during Router use; the Router must never close injected components")
	}
}
