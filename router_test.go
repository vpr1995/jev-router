package modelrouter_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/internal/testutil"
)

// mustNewRouter builds a Router from opts, failing the test on validation
// errors.
func mustNewRouter(t *testing.T, opts modelrouter.RouterOptions) *modelrouter.Router {
	t.Helper()
	router, err := modelrouter.New(opts)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return router
}

// defaultChoice returns the routing config entry most tests share: the CHEAP
// fixture model on the "test" provider.
func defaultChoice() modelrouter.ModelChoice {
	return modelrouter.ModelChoice{Provider: testutil.CHEAP.Provider, Model: testutil.CHEAP.ID}
}

// defaultChoicePtr returns defaultChoice as the *ModelChoice DefaultModel
// wants it.
func defaultChoicePtr() *modelrouter.ModelChoice {
	choice := defaultChoice()
	return &choice
}

// choiceOf turns a catalog model into the ModelChoice a test configures.
func choiceOf(model modelrouter.ModelInfo) modelrouter.ModelChoice {
	return modelrouter.ModelChoice{Provider: model.Provider, Model: model.ID}
}

// assertCandidates pins an ordered attempt chain exactly.
func assertCandidates(t *testing.T, got, want []modelrouter.ModelChoice) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Errorf("Candidates = %v, want %v", got, want)
	}
}

// routerFixture bundles the shared test doubles: CHEAP + EXPENSIVE behind the
// "test" provider, a confidently trivial classification, and a
// capture-completion.
type routerFixture struct {
	catalog    *testutil.FakeCatalog
	classifier *testutil.FakeClassifier
	completion *testutil.FakeCompletion
}

func newRouterFixture() *routerFixture {
	return &routerFixture{
		catalog: &testutil.FakeCatalog{
			ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
		},
		classifier: &testutil.FakeClassifier{
			Classification: modelrouter.Classification{
				Profile: testutil.CONFIDENT_TRIVIAL_PROFILE,
				Raw:     json.RawMessage(`{"fake": true}`),
			},
		},
		completion: &testutil.FakeCompletion{
			Result: modelrouter.CompletionResult{
				Content:  "4",
				Model:    testutil.CHEAP.ID,
				Provider: testutil.CHEAP.Provider,
			},
		},
	}
}

// options returns the fixture's RouterOptions: the fakes plus the
// single-provider routing config testutil.Options installs (DomainOther →
// CHEAP, DefaultModel CHEAP).
func (f *routerFixture) options() modelrouter.RouterOptions {
	return testutil.Options(f.catalog, f.classifier, f.completion)
}

// optionsWithChain overrides the fixture's routing config for the fixture
// profile's domain ("other"): Models[DomainOther] = chain, DefaultModel = def
// (nil for no default).
func (f *routerFixture) optionsWithChain(chain []modelrouter.ModelChoice, def *modelrouter.ModelChoice) modelrouter.RouterOptions {
	opts := f.options()
	opts.Models = map[modelrouter.Domain][]modelrouter.ModelChoice{modelrouter.DomainOther: chain}
	opts.DefaultModel = def
	return opts
}

func (f *routerFixture) newRouter(t *testing.T) *modelrouter.Router {
	t.Helper()
	return mustNewRouter(t, f.options())
}

// failingClassifier returns a fixed non-availability error, to pin the
// contract that only *ClassifierUnavailableError failures are handled by
// OnClassifierError.
type failingClassifier struct{ err error }

func (f failingClassifier) Classify(context.Context, string) (modelrouter.Classification, error) {
	return modelrouter.Classification{}, f.err
}

// --- Route: selection -------------------------------------------------------

// TestRoutePicksFirstCandidateOfTheClassifiedChain pins mapped selection: the
// classified domain selects the configured chain, its first entry is the
// decision's primary model, and the whole chain travels as the attempt list.
func TestRoutePicksFirstCandidateOfTheClassifiedChain(t *testing.T) {
	f := newRouterFixture()
	expensive := choiceOf(testutil.EXPENSIVE)
	router := mustNewRouter(t, f.optionsWithChain(
		[]modelrouter.ModelChoice{defaultChoice(), expensive}, nil))

	decision, err := router.Route(context.Background(), "what is 2+2?", nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if decision.ModelID != "vendor/cheap" {
		t.Errorf("ModelID = %q, want %q", decision.ModelID, "vendor/cheap")
	}
	if decision.Provider != "test" {
		t.Errorf("Provider = %q, want %q", decision.Provider, "test")
	}
	if decision.Source != modelrouter.SourceClassification {
		t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceClassification)
	}
	assertCandidates(t, decision.Candidates, []modelrouter.ModelChoice{defaultChoice(), expensive})
	if decision.Profile == nil || *decision.Profile != testutil.CONFIDENT_TRIVIAL_PROFILE {
		t.Errorf("Profile = %+v, want %+v", decision.Profile, testutil.CONFIDENT_TRIVIAL_PROFILE)
	}
}

// TestRouteAppendsDefaultModelToChain pins the chain composition rule: a
// configured DefaultModel is appended to the classified domain's chain as the
// last-resort fallback, and never duplicated when it is already an entry.
func TestRouteAppendsDefaultModelToChain(t *testing.T) {
	f := newRouterFixture()
	expensive := choiceOf(testutil.EXPENSIVE)

	t.Run("appended when absent", func(t *testing.T) {
		def := modelrouter.ModelChoice{Provider: "test", Model: "vendor/default"}
		router := mustNewRouter(t, f.optionsWithChain(
			[]modelrouter.ModelChoice{defaultChoice(), expensive}, &def))

		decision, err := router.Route(context.Background(), "anything", nil)
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if decision.Source != modelrouter.SourceClassification {
			t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceClassification)
		}
		assertCandidates(t, decision.Candidates,
			[]modelrouter.ModelChoice{defaultChoice(), expensive, def})
		if decision.ModelID != defaultChoice().Model || decision.Provider != defaultChoice().Provider {
			t.Errorf("winner = %s/%s, want the first candidate %s/%s",
				decision.Provider, decision.ModelID, defaultChoice().Provider, defaultChoice().Model)
		}
	})

	t.Run("not duplicated when already present", func(t *testing.T) {
		def := expensive
		router := mustNewRouter(t, f.optionsWithChain(
			[]modelrouter.ModelChoice{defaultChoice(), expensive}, &def))

		decision, err := router.Route(context.Background(), "anything", nil)
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		assertCandidates(t, decision.Candidates, []modelrouter.ModelChoice{defaultChoice(), expensive})
	})
}

// TestRouteUnmappedDomainUsesDefaultModel pins default selection with a real
// classification in hand: the profile is kept, but the chain and the source
// are the default ones.
func TestRouteUnmappedDomainUsesDefaultModel(t *testing.T) {
	f := newRouterFixture()
	f.classifier.Classification.Profile = modelrouter.RequestProfile{
		Domain:               modelrouter.DomainCode,
		DomainConfidence:     0.9,
		ComplexityScore:      1.0,
		ComplexityConfidence: 0.9,
	}
	router := f.newRouter(t) // Options maps DomainOther only

	decision, err := router.Route(context.Background(), "write a function", nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if decision.ModelID != testutil.CHEAP.ID || decision.Provider != testutil.CHEAP.Provider {
		t.Errorf("winner = %s/%s, want the default %s/%s",
			decision.Provider, decision.ModelID, testutil.CHEAP.Provider, testutil.CHEAP.ID)
	}
	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceDefault)
	}
	assertCandidates(t, decision.Candidates, []modelrouter.ModelChoice{defaultChoice()})
	if decision.Profile == nil || decision.Profile.Domain != modelrouter.DomainCode {
		t.Errorf("Profile = %+v, want the classified code profile (unmapped domains keep their profile)", decision.Profile)
	}
}

// TestRouteWithoutMappingOrDefaultNamesTheDomain pins the configuration gap:
// a classified domain with no chain needs a default model, and the error
// names the domain (DomainOther stays mapped, so New accepts the config).
func TestRouteWithoutMappingOrDefaultNamesTheDomain(t *testing.T) {
	f := newRouterFixture()
	f.classifier.Classification.Profile = modelrouter.RequestProfile{
		Domain:           modelrouter.DomainMathReasoning,
		DomainConfidence: 0.9,
	}
	opts := f.options()
	opts.DefaultModel = nil
	router := mustNewRouter(t, opts)

	_, err := router.Route(context.Background(), "prove it", nil)
	if err == nil {
		t.Fatal("Route() error = nil, want a no-mapping error")
	}
	message := err.Error()
	for _, want := range []string{"no model mapping for domain", `"math_reasoning"`, "no default model"} {
		if !strings.Contains(message, want) {
			t.Errorf("error = %q, want it to contain %q", message, want)
		}
	}
	if got := f.classifier.Calls(); got != 1 {
		t.Errorf("classifier.Calls() = %d, want 1 (selection happens after classification)", got)
	}
}

// TestRouteProviderFiltersWithoutCatalog covers the provider allow/exclude
// filters on a configured chain: they need no catalog metadata, so a catalog
// that always fails is never consulted and routing succeeds.
func TestRouteProviderFiltersWithoutCatalog(t *testing.T) {
	f := newRouterFixture()
	f.catalog = &testutil.FakeCatalog{Err: errors.New("catalog down")} // must never be called
	ghost := modelrouter.ModelChoice{Provider: "ghost", Model: "vendor/ghost"}
	chain := []modelrouter.ModelChoice{defaultChoice(), ghost}

	t.Run("allowed providers keeps only listed providers", func(t *testing.T) {
		router := mustNewRouter(t, f.optionsWithChain(chain, nil))
		decision, err := router.Route(context.Background(), "anything",
			&modelrouter.RoutingConstraints{AllowedProviders: []string{"ghost"}})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if decision.Provider != "ghost" {
			t.Errorf("Provider = %q, want %q", decision.Provider, "ghost")
		}
		assertCandidates(t, decision.Candidates, []modelrouter.ModelChoice{ghost})
	})

	t.Run("excluded providers removes them", func(t *testing.T) {
		router := mustNewRouter(t, f.optionsWithChain(chain, nil))
		decision, err := router.Route(context.Background(), "anything",
			&modelrouter.RoutingConstraints{ExcludedProviders: []string{"ghost"}})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if decision.Provider != "test" {
			t.Errorf("Provider = %q, want %q", decision.Provider, "test")
		}
		assertCandidates(t, decision.Candidates, []modelrouter.ModelChoice{defaultChoice()})
	})

	t.Run("exclusion wins over allowance", func(t *testing.T) {
		router := mustNewRouter(t, f.optionsWithChain(chain, nil))
		decision, err := router.Route(context.Background(), "anything", &modelrouter.RoutingConstraints{
			AllowedProviders:  []string{"test", "ghost"},
			ExcludedProviders: []string{"ghost"},
		})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		assertCandidates(t, decision.Candidates, []modelrouter.ModelChoice{defaultChoice()})
	})

	t.Run("filters leaving no candidate name the configured chain", func(t *testing.T) {
		router := mustNewRouter(t, f.optionsWithChain(chain, nil))
		_, err := router.Route(context.Background(), "anything",
			&modelrouter.RoutingConstraints{AllowedProviders: []string{"nobody"}})
		if err == nil {
			t.Fatal("Route() error = nil, want *NoCandidateModelError")
		}
		want := "modelrouter: no candidate model: no configured candidate model satisfies the constraints " +
			"(candidates: test/vendor/cheap, ghost/vendor/ghost)"
		if got := err.Error(); got != want {
			t.Errorf("Route() error = %q, want %q", got, want)
		}
		// No catalog metadata was needed, so no catalog diagnostics appear.
		if strings.Contains(err.Error(), "eliminates all") {
			t.Errorf("error = %q, want no catalog diagnostics without a catalog-backed constraint", err)
		}
	})

	if got := f.catalog.Calls(); got != 0 {
		t.Errorf("catalog.Calls() = %d, want 0 (provider filters and selection need no catalog)", got)
	}
}

// TestRouteMetadataConstraintsFilterTheChain pins catalog-backed filtering: a
// metadata constraint fetches the catalogs once and drops candidates that
// violate it or are missing from the catalogs, preserving chain order.
func TestRouteMetadataConstraintsFilterTheChain(t *testing.T) {
	catalogModels := []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}
	chain := []modelrouter.ModelChoice{
		defaultChoice(),              // CHEAP: no vision, 8k context, blended 0.2
		choiceOf(testutil.EXPENSIVE), // EXPENSIVE: vision, 128k context, blended 10
		{Provider: "ghost", Model: "vendor/not-in-catalog"},
	}

	tests := []struct {
		name        string
		constraints modelrouter.RoutingConstraints
		want        []modelrouter.ModelChoice
	}{
		{
			name:        "needs vision drops text-only and unknown candidates",
			constraints: modelrouter.RoutingConstraints{NeedsVision: true},
			want:        []modelrouter.ModelChoice{choiceOf(testutil.EXPENSIVE)},
		},
		{
			name:        "min context drops small windows",
			constraints: modelrouter.RoutingConstraints{MinContextTokens: 100_000},
			want:        []modelrouter.ModelChoice{choiceOf(testutil.EXPENSIVE)},
		},
		{
			name:        "max price drops expensive and unknown candidates",
			constraints: modelrouter.RoutingConstraints{MaxPricePer1KTokens: float64Ptr(1)},
			want:        []modelrouter.ModelChoice{defaultChoice()},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newRouterFixture()
			f.catalog = &testutil.FakeCatalog{ModelsList: catalogModels}
			router := mustNewRouter(t, f.optionsWithChain(chain, nil))

			decision, err := router.Route(context.Background(), "anything", &tt.constraints)
			if err != nil {
				t.Fatalf("Route() error = %v", err)
			}
			assertCandidates(t, decision.Candidates, tt.want)
			if decision.ModelID != tt.want[0].Model || decision.Provider != tt.want[0].Provider {
				t.Errorf("winner = %s/%s, want the first surviving candidate %s/%s",
					decision.Provider, decision.ModelID, tt.want[0].Provider, tt.want[0].Model)
			}
			if got := f.catalog.Calls(); got != 1 {
				t.Errorf("catalog.Calls() = %d, want 1 (one fetch per constrained route)", got)
			}
			if got := f.classifier.Calls(); got != 1 {
				t.Errorf("classifier.Calls() = %d, want 1", got)
			}
		})
	}
}

func TestChooseModelRaisesWhenConstraintsLeaveNoCandidates(t *testing.T) {
	f := newRouterFixture()
	f.catalog.ModelsList = []modelrouter.ModelInfo{testutil.CHEAP} // CHEAP does not support vision
	router := f.newRouter(t)

	_, err := router.Route(context.Background(), "describe this image", &modelrouter.RoutingConstraints{NeedsVision: true})
	if err == nil {
		t.Fatal("Route() error = nil, want *NoCandidateModelError")
	}
	var noCandidate *modelrouter.NoCandidateModelError
	if !errors.As(err, &noCandidate) {
		t.Fatalf("Route() error = %v (%T), want *NoCandidateModelError", err, err)
	}
	if !errors.Is(err, modelrouter.ErrNoCandidate) {
		t.Errorf("errors.Is(err, ErrNoCandidate) = false; err = %v", err)
	}
}

func TestNoCandidateErrorNamesTheEliminatingConstraint(t *testing.T) {
	f := newRouterFixture()
	f.catalog.ModelsList = []modelrouter.ModelInfo{testutil.CHEAP} // CHEAP does not support vision
	router := f.newRouter(t)

	_, err := router.Route(context.Background(), "describe this image", &modelrouter.RoutingConstraints{NeedsVision: true})
	if err == nil {
		t.Fatal("Route() error = nil, want *NoCandidateModelError")
	}
	message := err.Error()
	if !strings.Contains(message, "needs_vision=true") {
		t.Errorf("message = %q, want it to name needs_vision=true", message)
	}
	if !strings.Contains(message, "0/1") {
		t.Errorf("message = %q, want the 0/1 survivor count", message)
	}
	// Defaulted, inactive constraints should not be named in the message.
	if strings.Contains(message, "min_context_tokens") {
		t.Errorf("message = %q, want inactive min_context_tokens not named", message)
	}
	if strings.Contains(message, "max_price_per_1k_tokens") {
		t.Errorf("message = %q, want inactive max_price_per_1k_tokens not named", message)
	}
}

// TestNoCandidateErrorNamesTheConfiguredCandidates pins the v2 message shape:
// the configured chain is always named, and the catalog diagnostics are
// appended when the catalogs were fetched.
func TestNoCandidateErrorNamesTheConfiguredCandidates(t *testing.T) {
	f := newRouterFixture()
	f.catalog.ModelsList = []modelrouter.ModelInfo{testutil.CHEAP} // CHEAP does not support vision
	router := f.newRouter(t)

	_, err := router.Route(context.Background(), "describe this image", &modelrouter.RoutingConstraints{NeedsVision: true})
	if err == nil {
		t.Fatal("Route() error = nil, want *NoCandidateModelError")
	}
	message := err.Error()
	if !strings.Contains(message, "candidates: test/vendor/cheap") {
		t.Errorf("message = %q, want it to name the configured candidates", message)
	}
	if !strings.Contains(message, "needs_vision=true eliminates all (0/1 models support vision)") {
		t.Errorf("message = %q, want the catalog diagnostics appended", message)
	}
}

func TestNoCandidateErrorNamesMinContextTokensWithLargestAvailable(t *testing.T) {
	f := newRouterFixture()
	router := f.newRouter(t)

	_, err := router.Route(context.Background(), "anything", &modelrouter.RoutingConstraints{MinContextTokens: 1_000_000})
	if err == nil {
		t.Fatal("Route() error = nil, want *NoCandidateModelError")
	}
	message := err.Error()
	if !strings.Contains(message, "min_context_tokens=1000000") {
		t.Errorf("message = %q, want min_context_tokens=1000000", message)
	}
	if !strings.Contains(message, "largest available: 128000") {
		t.Errorf("message = %q, want the largest available context", message)
	}
}

// --- Route: catalog stage ---------------------------------------------------

// TestRouteReturnsJoinedCatalogErrorsWithoutClassifying pins that catalog
// failures are never silently skipped when catalogs are needed: every failure
// is joined, no partial pool is routed, and the classifier is not called.
// Without a metadata constraint the catalogs are not consulted at all.
func TestRouteReturnsJoinedCatalogErrorsWithoutClassifying(t *testing.T) {
	t.Run("every failure is joined", func(t *testing.T) {
		good := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP}}
		failingA := &testutil.FakeCatalog{Err: &modelrouter.CatalogUnavailableError{Err: errors.New("catalog a down")}}
		failingB := &testutil.FakeCatalog{Err: &modelrouter.CatalogUnavailableError{Err: errors.New("catalog b down")}}
		classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
		router := mustNewRouter(t, modelrouter.RouterOptions{
			Catalogs:     []modelrouter.Catalog{good, failingA, failingB},
			Classifier:   classifier,
			DefaultModel: defaultChoicePtr(),
		})

		decision, err := router.Route(context.Background(), "anything", &modelrouter.RoutingConstraints{NeedsVision: true})
		if err == nil {
			t.Fatal("Route() error = nil, want joined catalog failures")
		}
		if !errors.Is(err, modelrouter.ErrCatalogUnavailable) {
			t.Errorf("errors.Is(err, ErrCatalogUnavailable) = false; err = %v", err)
		}
		message := err.Error()
		for _, want := range []string{"catalog a down", "catalog b down"} {
			if !strings.Contains(message, want) {
				t.Errorf("message = %q, want it to join %q", message, want)
			}
		}
		if decision.ModelID != "" || len(decision.Candidates) != 0 {
			t.Errorf("decision = %+v, want the zero decision when catalogs fail", decision)
		}
		if got := classifier.Calls(); got != 0 {
			t.Errorf("classifier.Calls() = %d, want 0 (no classification after a catalog failure)", got)
		}
	})

	t.Run("one failure discards the healthy catalog's models", func(t *testing.T) {
		failing := &testutil.FakeCatalog{Err: errors.New("only catalog down")}
		router := mustNewRouter(t, modelrouter.RouterOptions{
			Catalogs:     []modelrouter.Catalog{failing},
			Classifier:   &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}},
			DefaultModel: defaultChoicePtr(),
		})

		_, err := router.Route(context.Background(), "anything", &modelrouter.RoutingConstraints{MinContextTokens: 1})
		if err == nil {
			t.Fatal("Route() error = nil, want the catalog error")
		}
		if !strings.Contains(err.Error(), "only catalog down") {
			t.Errorf("error = %v, want the underlying cause named", err)
		}
	})

	t.Run("plain routes never consult the catalogs", func(t *testing.T) {
		broken := &testutil.FakeCatalog{Err: errors.New("catalog always down")}
		router := mustNewRouter(t, modelrouter.RouterOptions{
			Catalogs:     []modelrouter.Catalog{broken},
			Classifier:   &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}},
			DefaultModel: defaultChoicePtr(),
		})

		decision, err := router.Route(context.Background(), "anything", nil)
		if err != nil {
			t.Fatalf("Route() error = %v, want success (no constraint needs catalog metadata)", err)
		}
		if decision.ModelID != testutil.CHEAP.ID {
			t.Errorf("ModelID = %q, want %q", decision.ModelID, testutil.CHEAP.ID)
		}
		if got := broken.Calls(); got != 0 {
			t.Errorf("catalog.Calls() = %d, want 0 without metadata constraints", got)
		}
	})
}

// --- Route: classifier failures ---------------------------------------------

// TestRouteFailOpenUsesDefaultModel pins fail-open behavior: it routes on
// the configured default model and never fabricates a profile.
func TestRouteFailOpenUsesDefaultModel(t *testing.T) {
	f := newRouterFixture()
	f.classifier.Err = &modelrouter.ClassifierUnavailableError{Err: errors.New("boom")}
	opts := f.options()
	opts.OnClassifierError = modelrouter.OnClassifierErrorFailOpen
	router := mustNewRouter(t, opts)

	decision, err := router.Route(context.Background(), "anything", nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if decision.ModelID != "vendor/cheap" {
		t.Errorf("ModelID = %q, want the configured default %q", decision.ModelID, "vendor/cheap")
	}
	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceDefault)
	}
	if decision.Profile != nil {
		t.Errorf("Profile = %+v, want nil on the fail-open fallback (never fabricated)", decision.Profile)
	}
	assertCandidates(t, decision.Candidates, []modelrouter.ModelChoice{defaultChoice()})
	if decision.RawJevResponse != nil || decision.ClassifierUsage != nil {
		t.Errorf("RawJevResponse/ClassifierUsage = %v/%v, want nil (nothing was classified)",
			decision.RawJevResponse, decision.ClassifierUsage)
	}
}

// TestRouteFailOpenWithoutDefaultModelErrors pins the configuration gap on the
// unclassified path: fail-open needs a default model, whatever mappings exist
// for classified domains.
func TestRouteFailOpenWithoutDefaultModelErrors(t *testing.T) {
	t.Run("failing classifier", func(t *testing.T) {
		f := newRouterFixture()
		f.classifier.Err = &modelrouter.ClassifierUnavailableError{Err: errors.New("boom")}
		opts := f.options()
		opts.DefaultModel = nil // Models[DomainOther] stays configured
		router := mustNewRouter(t, opts)

		_, err := router.Route(context.Background(), "anything", nil)
		if err == nil {
			t.Fatal("Route() error = nil, want a no-default-model error")
		}
		if !strings.Contains(err.Error(), "no default model configured for unclassified request") {
			t.Errorf("error = %v, want it to name the missing default model", err)
		}
	})

	t.Run("nil classifier", func(t *testing.T) {
		f := newRouterFixture()
		opts := testutil.Options(f.catalog, nil, nil)
		opts.DefaultModel = nil
		router := mustNewRouter(t, opts)

		_, err := router.Route(context.Background(), "anything", nil)
		if err == nil {
			t.Fatal("Route() error = nil, want a no-default-model error")
		}
		if !strings.Contains(err.Error(), "no default model configured for unclassified request") {
			t.Errorf("error = %v, want it to name the missing default model", err)
		}
	})
}

func TestChooseModelFailClosedRaisesOnClassifierError(t *testing.T) {
	f := newRouterFixture()
	classifyErr := &modelrouter.ClassifierUnavailableError{Err: errors.New("boom")}
	f.classifier.Err = classifyErr
	opts := f.options()
	opts.OnClassifierError = modelrouter.OnClassifierErrorFailClosed
	router := mustNewRouter(t, opts)

	_, err := router.Route(context.Background(), "anything", nil)
	if err == nil {
		t.Fatal("Route() error = nil, want *ClassifierUnavailableError")
	}
	var unavailable *modelrouter.ClassifierUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("Route() error = %v (%T), want *ClassifierUnavailableError", err, err)
	}
	if !errors.Is(err, modelrouter.ErrClassifierUnavailable) {
		t.Errorf("errors.Is(err, ErrClassifierUnavailable) = false; err = %v", err)
	}
	if err != classifyErr {
		t.Errorf("fail_closed returned %v, want the classifier error unchanged (%v)", err, classifyErr)
	}
}

// TestNilClassifierFailOpenUsesDefaultModel pins that a nil classifier is
// treated as unavailable and fail-open routes on the configured default
// model, without fabricating a profile.
func TestNilClassifierFailOpenUsesDefaultModel(t *testing.T) {
	f := newRouterFixture()
	router := mustNewRouter(t, testutil.Options(f.catalog, nil, nil))

	decision, err := router.Route(context.Background(), "anything", nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if decision.ModelID != "vendor/cheap" {
		t.Errorf("ModelID = %q, want the default %q", decision.ModelID, "vendor/cheap")
	}
	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceDefault)
	}
	if decision.Profile != nil {
		t.Errorf("Profile = %+v, want nil (no fabricated profile)", decision.Profile)
	}
}

// TestNilClassifierFailClosedReturnsClassifierUnavailable pins that fail-closed
// surfaces a synthetic ClassifierUnavailableError naming the missing
// classifier.
func TestNilClassifierFailClosedReturnsClassifierUnavailable(t *testing.T) {
	f := newRouterFixture()
	opts := testutil.Options(f.catalog, nil, nil)
	opts.OnClassifierError = modelrouter.OnClassifierErrorFailClosed
	router := mustNewRouter(t, opts)

	_, err := router.Route(context.Background(), "anything", nil)
	if err == nil {
		t.Fatal("Route() error = nil, want *ClassifierUnavailableError")
	}
	var unavailable *modelrouter.ClassifierUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("Route() error = %v (%T), want *ClassifierUnavailableError", err, err)
	}
	if !errors.Is(err, modelrouter.ErrClassifierUnavailable) {
		t.Errorf("errors.Is(err, ErrClassifierUnavailable) = false; err = %v", err)
	}
	if !strings.Contains(err.Error(), "no classifier configured") {
		t.Errorf("error = %v, want it to name the missing classifier", err)
	}
}

// TestRoutePropagatesUnexpectedClassifierErrors pins the defensive contract:
// only *ClassifierUnavailableError failures are handled by OnClassifierError;
// anything else propagates unchanged regardless of the policy.
func TestRoutePropagatesUnexpectedClassifierErrors(t *testing.T) {
	plainErr := errors.New("unexpected classifier failure")
	f := newRouterFixture()
	opts := f.options()
	opts.Classifier = failingClassifier{err: plainErr}
	opts.OnClassifierError = modelrouter.OnClassifierErrorFailOpen
	router := mustNewRouter(t, opts)

	_, err := router.Route(context.Background(), "anything", nil)
	if err != plainErr {
		t.Errorf("Route() error = %v, want %v unchanged", err, plainErr)
	}
}

// TestRouteDecisionCarriesRawResponseAndUsage pins that the decision exposes
// the classifier's raw response and reported usage.
func TestRouteDecisionCarriesRawResponseAndUsage(t *testing.T) {
	raw := json.RawMessage(`{"answers":{"complexity":{"score":0}}}`)
	usage := &modelrouter.Usage{InputTokens: 12, OutputTokens: 3, TotalTokens: 15}
	f := newRouterFixture()
	f.classifier.Classification = modelrouter.Classification{
		Profile: testutil.CONFIDENT_TRIVIAL_PROFILE,
		Raw:     raw,
		Usage:   usage,
	}
	router := f.newRouter(t)

	decision, err := router.Route(context.Background(), "anything", nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if !bytes.Equal(decision.RawJevResponse, raw) {
		t.Errorf("RawJevResponse = %s, want %s", decision.RawJevResponse, raw)
	}
	if decision.ClassifierUsage != usage {
		t.Errorf("ClassifierUsage = %+v, want the classifier-reported usage %+v", decision.ClassifierUsage, usage)
	}

	t.Run("empty raw response becomes nil", func(t *testing.T) {
		f.classifier.Classification = modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}
		decision, err := f.newRouter(t).Route(context.Background(), "anything", nil)
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if decision.RawJevResponse != nil {
			t.Errorf("RawJevResponse = %s, want nil when the classifier returned none", decision.RawJevResponse)
		}
	})
}

// --- Complete ---------------------------------------------------------------

// TestCompleteCallsProviderWithTheChosenModel pins what the Router hands to
// the provider: the provider is an injected Completion, so the assertion is
// on the request the Router built, not on wire bytes. The HTTP request-shape
// assertion lives in the OpenRouter provider tests.
func TestCompleteCallsProviderWithTheChosenModel(t *testing.T) {
	f := newRouterFixture()
	router := f.newRouter(t)

	result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "what is 2+2?"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "4" {
		t.Errorf("result.Content = %q, want %q", result.Content, "4")
	}
	if result.RoutingDecision.ModelID != "vendor/cheap" {
		t.Errorf("result.RoutingDecision.ModelID = %q, want %q", result.RoutingDecision.ModelID, "vendor/cheap")
	}
	if result.RoutingDecision.Source != modelrouter.SourceClassification {
		t.Errorf("result.RoutingDecision.Source = %q, want %q",
			result.RoutingDecision.Source, modelrouter.SourceClassification)
	}

	last, ok := f.completion.LastRequest()
	if !ok {
		t.Fatal("completion received no request")
	}
	if last.Model != "vendor/cheap" {
		t.Errorf("provider request Model = %q, want %q", last.Model, "vendor/cheap")
	}
	if last.Provider != "test" {
		t.Errorf("provider request Provider = %q, want %q", last.Provider, "test")
	}
	if last.Prompt != "what is 2+2?" {
		t.Errorf("provider request Prompt = %q, want %q", last.Prompt, "what is 2+2?")
	}
	if f.classifier.Calls() != 1 {
		t.Errorf("classifier.Calls() = %d, want 1 (routing classified once)", f.classifier.Calls())
	}
}

// TestCompleteFallsBackThroughTheCandidateChain pins the fallback contract:
// attempt #1 fails, attempt #2 succeeds, and the decision reports the winner
// while keeping the whole chain.
func TestCompleteFallsBackThroughTheCandidateChain(t *testing.T) {
	f := newRouterFixture()
	attemptErr := &modelrouter.ProviderError{
		Provider:   testutil.CHEAP.Provider,
		Op:         "complete",
		StatusCode: 502,
		Err:        errors.New("first model down"),
	}
	f.completion.Errs = []error{attemptErr, nil}
	expensive := choiceOf(testutil.EXPENSIVE)
	router := mustNewRouter(t, f.optionsWithChain(
		[]modelrouter.ModelChoice{defaultChoice(), expensive}, nil))

	result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "what is 2+2?"})
	if err != nil {
		t.Fatalf("Complete() error = %v, want the second candidate to succeed", err)
	}
	if result.Content != "4" {
		t.Errorf("result.Content = %q, want %q", result.Content, "4")
	}
	if result.RoutingDecision.ModelID != expensive.Model || result.RoutingDecision.Provider != expensive.Provider {
		t.Errorf("RoutingDecision winner = %s/%s, want the second candidate %s/%s",
			result.RoutingDecision.Provider, result.RoutingDecision.ModelID, expensive.Provider, expensive.Model)
	}
	if result.RoutingDecision.Source != modelrouter.SourceClassification {
		t.Errorf("RoutingDecision.Source = %q, want %q",
			result.RoutingDecision.Source, modelrouter.SourceClassification)
	}
	assertCandidates(t, result.RoutingDecision.Candidates,
		[]modelrouter.ModelChoice{defaultChoice(), expensive})

	requests := f.completion.Requests()
	if len(requests) != 2 {
		t.Fatalf("provider attempts = %d, want 2 (chain: %v)", len(requests), requests)
	}
	if requests[0].Model != testutil.CHEAP.ID || requests[1].Model != testutil.EXPENSIVE.ID {
		t.Errorf("attempt models = %q, %q; want %q then %q",
			requests[0].Model, requests[1].Model, testutil.CHEAP.ID, testutil.EXPENSIVE.ID)
	}
	if requests[0].Prompt != "what is 2+2?" || requests[1].Prompt != "what is 2+2?" {
		t.Errorf("attempt prompts = %q, %q; want the caller's prompt passed through",
			requests[0].Prompt, requests[1].Prompt)
	}
	if got := f.classifier.Calls(); got != 1 {
		t.Errorf("classifier.Calls() = %d, want 1 (one classification for the whole chain)", got)
	}
}

// TestCompleteSingleCandidateFailureReturnsProviderErrorUnchanged pins the
// one-attempt guarantee: the caller sees the provider's error, unchanged and
// unwrapped, exactly as before the chain existed.
func TestCompleteSingleCandidateFailureReturnsProviderErrorUnchanged(t *testing.T) {
	f := newRouterFixture()
	providerErr := &modelrouter.ProviderError{
		Provider:   testutil.CHEAP.Provider,
		Op:         "complete",
		StatusCode: 502,
		Err:        errors.New("upstream down"),
	}
	f.completion.Err = providerErr
	router := f.newRouter(t) // one-candidate chain: [CHEAP]

	result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
	if err != providerErr {
		t.Errorf("Complete() error = %v, want the provider error unchanged (%v)", err, providerErr)
	}
	var gotErr *modelrouter.ProviderError
	if !errors.As(err, &gotErr) || gotErr != providerErr {
		t.Errorf("errors.As did not recover the original *ProviderError: %v", err)
	}
	if !errors.Is(err, modelrouter.ErrProvider) {
		t.Errorf("errors.Is(err, ErrProvider) = false; err = %v", err)
	}
	if result.Content != "" || result.RoutingDecision.ModelID != "" {
		t.Errorf("result = %+v, want the zero result on failure", result)
	}
	if got := len(f.completion.Requests()); got != 1 {
		t.Errorf("provider attempts = %d, want 1", got)
	}
}

// TestCompleteAllCandidatesFailIsWrapped pins the multi-attempt aggregation:
// the returned error names the attempt count, and errors.Is/As still reach
// every inner failure.
func TestCompleteAllCandidatesFailIsWrapped(t *testing.T) {
	f := newRouterFixture()
	causeA := errors.New("first model down")
	causeB := errors.New("second model down")
	f.completion.Errs = []error{
		&modelrouter.ProviderError{Provider: testutil.CHEAP.Provider, Op: "complete", Err: causeA},
		&modelrouter.ProviderError{Provider: testutil.CHEAP.Provider, Op: "complete", Err: causeB},
	}
	router := mustNewRouter(t, f.optionsWithChain(
		[]modelrouter.ModelChoice{defaultChoice(), choiceOf(testutil.EXPENSIVE)}, nil))

	_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
	if err == nil {
		t.Fatal("Complete() error = nil, want the joined attempt failures")
	}
	if !strings.HasPrefix(err.Error(), "modelrouter: all 2 candidate models failed") {
		t.Errorf("error = %q, want it to start with %q", err, "modelrouter: all 2 candidate models failed")
	}
	var providerErr *modelrouter.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("errors.As(err, *ProviderError) = false; err = %v (%T)", err, err)
	}
	if !errors.Is(err, causeA) || !errors.Is(err, causeB) {
		t.Errorf("errors.Is did not reach both attempts' causes; err = %v", err)
	}
	if got := len(f.completion.Requests()); got != 2 {
		t.Errorf("provider attempts = %d, want 2", got)
	}
}

// TestCompleteUnregisteredProviderAdvancesFallback pins that a chain entry
// with no registered Completion is an attempt failure like any other: the
// fallback reaches the next candidate instead of surfacing the registry gap.
func TestCompleteUnregisteredProviderAdvancesFallback(t *testing.T) {
	f := newRouterFixture()
	ghost := modelrouter.ModelChoice{Provider: "ghost", Model: "vendor/ghost"}
	router := mustNewRouter(t, f.optionsWithChain(
		[]modelrouter.ModelChoice{ghost, defaultChoice()}, nil))

	result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
	if err != nil {
		t.Fatalf("Complete() error = %v, want the registered candidate to win", err)
	}
	if result.RoutingDecision.ModelID != testutil.CHEAP.ID || result.RoutingDecision.Provider != testutil.CHEAP.Provider {
		t.Errorf("winner = %s/%s, want %s/%s (the ghost provider must be skipped)",
			result.RoutingDecision.Provider, result.RoutingDecision.ModelID,
			testutil.CHEAP.Provider, testutil.CHEAP.ID)
	}
	if requests := f.completion.Requests(); len(requests) != 1 {
		t.Errorf("provider attempts = %d, want 1 (only the registered provider is called); requests = %v",
			len(requests), requests)
	}

	t.Run("an all-unregistered chain fails with the registry error", func(t *testing.T) {
		f := newRouterFixture()
		ghost := modelrouter.ModelChoice{Provider: "ghost", Model: "vendor/ghost"}
		router := mustNewRouter(t, f.optionsWithChain([]modelrouter.ModelChoice{ghost}, nil))

		_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
		if !errors.Is(err, modelrouter.ErrProviderNotConfigured) {
			t.Errorf("errors.Is(err, ErrProviderNotConfigured) = false; err = %v", err)
		}
		var providerErr *modelrouter.ProviderError
		if !errors.As(err, &providerErr) || providerErr.Provider != "ghost" || providerErr.Op != "complete" {
			t.Errorf("error = %v, want a *ProviderError naming provider ghost op complete", err)
		}
	})
}

// TestCompleteWithExplicitModelSkipsRouting pins the explicit-model path: no
// classifier call, no fallback, and no decision is attached.
func TestCompleteWithExplicitModelSkipsRouting(t *testing.T) {
	f := newRouterFixture()
	router := f.newRouter(t)

	result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{
		Prompt:   "hello",
		Model:    "vendor/expensive",
		Provider: "test",
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if got := f.classifier.Calls(); got != 0 {
		t.Errorf("classifier.Calls() = %d, want 0 (routing must be skipped)", got)
	}
	last, ok := f.completion.LastRequest()
	if !ok {
		t.Fatal("completion received no request")
	}
	if last.Model != "vendor/expensive" || last.Provider != "test" {
		t.Errorf("provider request Model/Provider = %q/%q, want vendor/expensive/test", last.Model, last.Provider)
	}
	if result.RoutingDecision.ModelID != "" {
		t.Errorf("result.RoutingDecision.ModelID = %q, want empty (no routing happened)", result.RoutingDecision.ModelID)
	}

	t.Run("routing constraints are ignored on the explicit path", func(t *testing.T) {
		_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{
			Prompt:      "hello",
			Model:       "vendor/expensive",
			Provider:    "test",
			Constraints: &modelrouter.RoutingConstraints{NeedsVision: true, MinContextTokens: 1_000_000},
		})
		if err != nil {
			t.Fatalf("Complete() error = %v", err)
		}
		if got := f.classifier.Calls(); got != 0 {
			t.Errorf("classifier.Calls() = %d, want 0", got)
		}
	})

	t.Run("no fallback even when a candidate chain exists", func(t *testing.T) {
		f := newRouterFixture()
		failure := &modelrouter.ProviderError{Provider: "test", Op: "complete", Err: errors.New("model down")}
		f.completion.Err = failure
		router := mustNewRouter(t, f.optionsWithChain(
			[]modelrouter.ModelChoice{defaultChoice(), choiceOf(testutil.EXPENSIVE)}, nil))

		_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{
			Prompt:   "hello",
			Model:    testutil.CHEAP.ID,
			Provider: testutil.CHEAP.Provider,
		})
		if err != failure {
			t.Errorf("Complete() error = %v, want the provider error unchanged (%v)", err, failure)
		}
		if got := len(f.completion.Requests()); got != 1 {
			t.Errorf("provider attempts = %d, want 1 (explicit requests never fall back)", got)
		}
		if got := f.classifier.Calls(); got != 0 {
			t.Errorf("classifier.Calls() = %d, want 0", got)
		}
	})

	t.Run("explicit model requires a provider", func(t *testing.T) {
		_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{
			Prompt: "hello",
			Model:  "vendor/expensive",
		})
		if err == nil {
			t.Fatal("Complete() error = nil, want a provider-required error")
		}
		if !strings.Contains(err.Error(), "provider is required") {
			t.Errorf("error = %v, want it to mention the required provider", err)
		}
		if got := f.classifier.Calls(); got != 0 {
			t.Errorf("classifier.Calls() = %d, want 0", got)
		}
	})
}

// TestCompleteRequiresPromptOrMessages pins the input validation and the
// no-materialization contract: Messages-only requests pass through untouched.
func TestCompleteRequiresPromptOrMessages(t *testing.T) {
	f := newRouterFixture()
	router := f.newRouter(t)

	_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{})
	if err == nil {
		t.Fatal("Complete() error = nil, want a prompt-or-messages error")
	}
	if !strings.Contains(err.Error(), "prompt or messages required") {
		t.Errorf("error = %v, want it to mention prompt or messages", err)
	}
	if f.classifier.Calls() != 0 || f.catalog.Calls() != 0 {
		t.Errorf("classifier/catalog calls = %d/%d, want 0/0 (validation precedes routing)",
			f.classifier.Calls(), f.catalog.Calls())
	}

	messages := []modelrouter.Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
	}
	result, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Messages: messages})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.RoutingDecision.ModelID != "vendor/cheap" {
		t.Errorf("RoutingDecision.ModelID = %q, want %q", result.RoutingDecision.ModelID, "vendor/cheap")
	}
	last, ok := f.completion.LastRequest()
	if !ok {
		t.Fatal("completion received no request")
	}
	if len(last.Messages) != len(messages) {
		t.Errorf("provider request Messages = %d, want %d (passed through untouched)", len(last.Messages), len(messages))
	}
	if last.Prompt != "" {
		t.Errorf("provider request Prompt = %q, want empty (no {user, prompt} materialization)", last.Prompt)
	}
}

// TestCompleteClassifiesMessagesOnlyRequests pins that a Messages-only request
// classifies the last user message rather than an empty string.
func TestCompleteClassifiesMessagesOnlyRequests(t *testing.T) {
	tests := []struct {
		name     string
		req      modelrouter.CompletionRequest
		wantText string
	}{
		{"prompt wins", modelrouter.CompletionRequest{Prompt: "p", Messages: []modelrouter.Message{{Role: "user", Content: "m"}}}, "p"},
		{"last user message", modelrouter.CompletionRequest{Messages: []modelrouter.Message{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "first"},
			{Role: "assistant", Content: "ok"},
			{Role: "user", Content: "second"},
			{Role: "assistant", Content: "trailing"},
		}}, "second"},
		{"no user message falls back to last", modelrouter.CompletionRequest{Messages: []modelrouter.Message{
			{Role: "system", Content: "only system"},
		}}, "only system"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newRouterFixture()
			router := f.newRouter(t)
			if _, err := router.Complete(context.Background(), tt.req); err != nil {
				t.Fatalf("Complete() error = %v", err)
			}
			got := f.classifier.Prompts()
			if len(got) != 1 || got[0] != tt.wantText {
				t.Errorf("classified prompts = %q, want [%q]", got, tt.wantText)
			}
		})
	}
}

// TestCompleteUnknownProvider pins the ErrProviderNotConfigured mapping for
// explicit and routed providers.
func TestCompleteUnknownProvider(t *testing.T) {
	f := newRouterFixture()
	router := f.newRouter(t)

	_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{
		Prompt: "hi", Model: "vendor/expensive", Provider: "ghost",
	})
	if err == nil {
		t.Fatal("Complete() error = nil, want *ProviderError")
	}
	if !errors.Is(err, modelrouter.ErrProviderNotConfigured) {
		t.Errorf("errors.Is(err, ErrProviderNotConfigured) = false; err = %v", err)
	}
	if !errors.Is(err, modelrouter.ErrProvider) {
		t.Errorf("errors.Is(err, ErrProvider) = false; err = %v", err)
	}
	var providerErr *modelrouter.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("errors.As(err, *ProviderError) = false; err = %v (%T)", err, err)
	}
	if providerErr.Provider != "ghost" || providerErr.Op != "complete" {
		t.Errorf("ProviderError = %+v, want provider ghost op complete", providerErr)
	}
	if got := f.classifier.Calls(); got != 0 {
		t.Errorf("classifier.Calls() = %d, want 0 (explicit model skips routing)", got)
	}

	t.Run("routed provider with no completion registered", func(t *testing.T) {
		f := newRouterFixture()
		router := mustNewRouter(t, testutil.Options(f.catalog, f.classifier, nil))
		_, err := router.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"})
		if err == nil {
			t.Fatal("Complete() error = nil, want *ProviderError")
		}
		if !errors.Is(err, modelrouter.ErrProviderNotConfigured) {
			t.Errorf("errors.Is(err, ErrProviderNotConfigured) = false; err = %v", err)
		}
		var routedErr *modelrouter.ProviderError
		if !errors.As(err, &routedErr) {
			t.Fatalf("errors.As(err, *ProviderError) = false; err = %v (%T)", err, err)
		}
		if routedErr.Provider != "test" {
			t.Errorf("ProviderError.Provider = %q, want %q", routedErr.Provider, "test")
		}
		if got := f.classifier.Calls(); got != 1 {
			t.Errorf("classifier.Calls() = %d, want 1 (routing ran before dispatch failed)", got)
		}
	})
}

// --- New: validation and immutability ---------------------------------------

// TestNewValidation pins RouterOptions validation: the required catalog, the
// classifier-error policy, and the mapped-routing configuration (domains,
// chains, default model).
func TestNewValidation(t *testing.T) {
	validCatalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP}}
	validChain := map[modelrouter.Domain][]modelrouter.ModelChoice{modelrouter.DomainOther: {defaultChoice()}}

	tests := []struct {
		name    string
		opts    modelrouter.RouterOptions
		wantErr string
	}{
		{"no catalogs", modelrouter.RouterOptions{}, "at least one catalog"},
		{
			"invalid classifier error policy",
			modelrouter.RouterOptions{
				Catalogs:          []modelrouter.Catalog{validCatalog},
				Models:            validChain,
				OnClassifierError: "sideways",
			},
			"invalid OnClassifierError",
		},
		{
			"no models and no default",
			modelrouter.RouterOptions{Catalogs: []modelrouter.Catalog{validCatalog}},
			"at least one of Models or DefaultModel",
		},
		{
			"unknown domain",
			modelrouter.RouterOptions{
				Catalogs: []modelrouter.Catalog{validCatalog},
				Models:   map[modelrouter.Domain][]modelrouter.ModelChoice{"sideways": {defaultChoice()}},
			},
			"invalid domain",
		},
		{
			"empty chain",
			modelrouter.RouterOptions{
				Catalogs: []modelrouter.Catalog{validCatalog},
				Models:   map[modelrouter.Domain][]modelrouter.ModelChoice{modelrouter.DomainOther: {}},
			},
			"at least one model",
		},
		{
			"empty provider in a chain",
			modelrouter.RouterOptions{
				Catalogs: []modelrouter.Catalog{validCatalog},
				Models: map[modelrouter.Domain][]modelrouter.ModelChoice{
					modelrouter.DomainOther: {{Model: "m"}},
				},
			},
			`Models["other"][0].Provider must not be empty`,
		},
		{
			"empty model in a chain",
			modelrouter.RouterOptions{
				Catalogs: []modelrouter.Catalog{validCatalog},
				Models: map[modelrouter.Domain][]modelrouter.ModelChoice{
					modelrouter.DomainOther: {{Provider: "test"}},
				},
			},
			`Models["other"][0].Model must not be empty`,
		},
		{
			"empty default model provider",
			modelrouter.RouterOptions{
				Catalogs:     []modelrouter.Catalog{validCatalog},
				DefaultModel: &modelrouter.ModelChoice{Model: "m"},
			},
			"DefaultModel.Provider must not be empty",
		},
		{
			"empty default model id",
			modelrouter.RouterOptions{
				Catalogs:     []modelrouter.Catalog{validCatalog},
				DefaultModel: &modelrouter.ModelChoice{Provider: "test"},
			},
			"DefaultModel.Model must not be empty",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := modelrouter.New(tt.opts)
			if err == nil {
				t.Fatalf("New() error = nil, want an error containing %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("New() error = %v, want it to contain %q", err, tt.wantErr)
			}
		})
	}

	valid := []struct {
		name string
		opts modelrouter.RouterOptions
	}{
		{"models only", modelrouter.RouterOptions{Catalogs: []modelrouter.Catalog{validCatalog}, Models: validChain}},
		{"default model only", modelrouter.RouterOptions{Catalogs: []modelrouter.Catalog{validCatalog}, DefaultModel: defaultChoicePtr()}},
		{"models and default", modelrouter.RouterOptions{Catalogs: []modelrouter.Catalog{validCatalog}, Models: validChain, DefaultModel: defaultChoicePtr()}},
		{"explicit fail open", modelrouter.RouterOptions{Catalogs: []modelrouter.Catalog{validCatalog}, Models: validChain, OnClassifierError: modelrouter.OnClassifierErrorFailOpen}},
		{"explicit fail closed", modelrouter.RouterOptions{Catalogs: []modelrouter.Catalog{validCatalog}, Models: validChain, OnClassifierError: modelrouter.OnClassifierErrorFailClosed}},
		{"nil classifier is allowed", modelrouter.RouterOptions{Catalogs: []modelrouter.Catalog{validCatalog}, Models: validChain}},
	}
	for _, tt := range valid {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := modelrouter.New(tt.opts); err != nil {
				t.Errorf("New() error = %v, want nil", err)
			}
		})
	}
}

// TestNewCopiesModelsAndDefaultModel pins the immutability contract: New
// copies the caller's routing config, so later mutations cannot affect the
// Router.
func TestNewCopiesModelsAndDefaultModel(t *testing.T) {
	f := newRouterFixture()
	chain := []modelrouter.ModelChoice{defaultChoice()}
	def := modelrouter.ModelChoice{Provider: "test", Model: "vendor/default"}
	opts := modelrouter.RouterOptions{
		Catalogs:     []modelrouter.Catalog{f.catalog},
		Classifier:   f.classifier,
		Models:       map[modelrouter.Domain][]modelrouter.ModelChoice{modelrouter.DomainOther: chain},
		DefaultModel: &def,
	}
	router := mustNewRouter(t, opts)

	// Mutate everything the caller handed over after New: the map, the chain
	// slice elements, the slice itself and the default model value.
	opts.Models[modelrouter.DomainOther][0] = modelrouter.ModelChoice{Provider: "ghost", Model: "changed"}
	chain[0] = modelrouter.ModelChoice{Provider: "ghost", Model: "changed-too"}
	opts.Models[modelrouter.DomainCode] = []modelrouter.ModelChoice{{Provider: "ghost", Model: "added"}}
	delete(opts.Models, modelrouter.DomainOther)
	opts.Models = nil
	def.Provider, def.Model = "ghost", "changed-default"

	decision, err := router.Route(context.Background(), "anything", nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	assertCandidates(t, decision.Candidates,
		[]modelrouter.ModelChoice{defaultChoice(), {Provider: "test", Model: "vendor/default"}})
	if decision.ModelID != testutil.CHEAP.ID || decision.Provider != testutil.CHEAP.Provider {
		t.Errorf("winner = %s/%s, want the copied config's first candidate %s/%s",
			decision.Provider, decision.ModelID, testutil.CHEAP.Provider, testutil.CHEAP.ID)
	}
}

// TestRouteIsSafeForConcurrentUse exercises a shared Router from many
// goroutines; the race detector is the real assertion here.
func TestRouteIsSafeForConcurrentUse(t *testing.T) {
	f := newRouterFixture()
	router := f.newRouter(t)

	const goroutines = 20
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			decision, err := router.Route(context.Background(), "what is 2+2?", nil)
			if err != nil {
				errs <- err
				return
			}
			if decision.ModelID != "vendor/cheap" {
				errs <- fmt.Errorf("ModelID = %q, want %q", decision.ModelID, "vendor/cheap")
			}
			if decision.Profile == nil {
				errs <- errors.New("Profile = nil, want the classified profile")
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if got := f.classifier.Calls(); got != goroutines {
		t.Errorf("classifier.Calls() = %d, want %d", got, goroutines)
	}
	// v2: a plain route needs no catalog metadata, so the count stays at 0.
	if got := f.catalog.Calls(); got != 0 {
		t.Errorf("catalog.Calls() = %d, want 0 (no metadata constraints)", got)
	}
}
