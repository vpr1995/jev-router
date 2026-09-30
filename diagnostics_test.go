package modelrouter_test

import (
	"context"
	"strings"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/internal/testutil"
)

// assertDescribe pins the exact diagnostics output; DescribeEliminatingConstraints
// is pure string building, so exact assertions beat substring checks
// everywhere except the Router-level integration test below.
func assertDescribe(t *testing.T, all []modelrouter.ModelInfo, c modelrouter.RoutingConstraints, want string) {
	t.Helper()
	if got := modelrouter.DescribeEliminatingConstraints(all, c); got != want {
		t.Errorf("DescribeEliminatingConstraints() =\n  %q\nwant:\n  %q", got, want)
	}
}

func float64Ptr(v float64) *float64 { return &v }

// Covers the needs_vision branches.
func TestDescribeEliminatingConstraintsNeedsVision(t *testing.T) {
	t.Run("eliminates all", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP},
			modelrouter.RoutingConstraints{NeedsVision: true},
			"needs_vision=true eliminates all (0/1 models support vision)")
	})

	t.Run("some survive", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{NeedsVision: true},
			"needs_vision=true (1/2 models support vision)")
	})
}

// Covers the min_context_tokens branches, including the largest-available
// hint.
func TestDescribeEliminatingConstraintsMinContextTokens(t *testing.T) {
	t.Run("eliminates all with largest available", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{MinContextTokens: 1_000_000},
			"min_context_tokens=1000000 eliminates all (0/2 models qualify, largest available: 128000)")
	})

	t.Run("some qualify", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{MinContextTokens: 10_000},
			"min_context_tokens=10000 (1/2 models qualify)")
	})
}

// Covers the max_price_per_1k_tokens branches, including the
// cheapest-available hint (blended price).
func TestDescribeEliminatingConstraintsMaxPricePer1KTokens(t *testing.T) {
	t.Run("eliminates all with cheapest available", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{MaxPricePer1KTokens: float64Ptr(0.05)},
			"max_price_per_1k_tokens=0.05 eliminates all (0/2 models qualify, cheapest available: 0.2)")
	})

	t.Run("some qualify", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{MaxPricePer1KTokens: float64Ptr(1)},
			"max_price_per_1k_tokens=1 (1/2 models qualify)")
	})
}

// Covers the combination branch.
func TestDescribeEliminatingConstraintsCombination(t *testing.T) {
	t.Run("provider filter with no matches", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{AllowedProviders: []string{"ghost"}},
			"the combination of constraints eliminates all 2 candidate models: "+
				"RoutingConstraints{needs_vision: false, min_context_tokens: 0, max_price_per_1k_tokens: nil, allowed_providers: [ghost], excluded_providers: []}")
	})

	t.Run("individually survivable constraints together eliminate all", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{
				NeedsVision:         true,
				MinContextTokens:    100_000,
				MaxPricePer1KTokens: float64Ptr(1),
			},
			"needs_vision=true (1/2 models support vision); "+
				"min_context_tokens=100000 (1/2 models qualify); "+
				"max_price_per_1k_tokens=1 (1/2 models qualify)")
	})
}

// TestDescribeEliminatingConstraintsPerProviderSuffix covers the Go
// multi-provider extension: one "; per provider: <name> X/N" part per provider
// sorted by name, X counting models surviving every constraint.
func TestDescribeEliminatingConstraintsPerProviderSuffix(t *testing.T) {
	t.Run("two providers", func(t *testing.T) {
		models := []modelrouter.ModelInfo{
			{Provider: "a", ID: "vendor/a-text", PricePer1KPromptTokens: 0.1, PricePer1KCompletionTokens: 0.1, ContextLength: 8_000},
			{Provider: "b", ID: "vendor/b-vision", PricePer1KPromptTokens: 5, PricePer1KCompletionTokens: 5, ContextLength: 128_000, SupportsVision: true},
		}
		assertDescribe(t, models, modelrouter.RoutingConstraints{NeedsVision: true},
			"needs_vision=true (1/2 models support vision); per provider: a 0/1; per provider: b 1/1")
	})

	t.Run("survivors differ from the constraint-level count", func(t *testing.T) {
		models := []modelrouter.ModelInfo{
			{Provider: "a", ID: "vendor/a-text", PricePer1KPromptTokens: 0.1, PricePer1KCompletionTokens: 0.1, ContextLength: 8_000},
			{Provider: "b", ID: "vendor/b-vision", PricePer1KPromptTokens: 5, PricePer1KCompletionTokens: 5, ContextLength: 128_000, SupportsVision: true},
			{Provider: "b", ID: "vendor/b-text", PricePer1KPromptTokens: 0.2, PricePer1KCompletionTokens: 0.2, ContextLength: 8_000},
		}
		// needs_vision alone keeps 1/3; per provider, b keeps 1/2 because
		// only b-vision passes *all* constraints.
		assertDescribe(t, models, modelrouter.RoutingConstraints{NeedsVision: true},
			"needs_vision=true (1/3 models support vision); per provider: a 0/1; per provider: b 1/2")
	})

	t.Run("single provider adds no suffix", func(t *testing.T) {
		assertDescribe(t,
			[]modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE},
			modelrouter.RoutingConstraints{NeedsVision: true},
			"needs_vision=true (1/2 models support vision)")
	})
}

// Covers the empty-catalog branch.
func TestDescribeEliminatingConstraintsEmptyCatalog(t *testing.T) {
	want := "the catalog returned 0 candidate models"
	if got := modelrouter.DescribeEliminatingConstraints(nil, modelrouter.RoutingConstraints{NeedsVision: true}); got != want {
		t.Errorf("DescribeEliminatingConstraints(nil, ...) = %q, want %q", got, want)
	}
	if got := modelrouter.DescribeEliminatingConstraints([]modelrouter.ModelInfo{}, modelrouter.RoutingConstraints{}); got != want {
		t.Errorf("DescribeEliminatingConstraints(empty, ...) = %q, want %q", got, want)
	}
}

// TestRouteNoCandidateErrorIncludesPerProviderCounts ties the diagnostics into
// the Router error: a multi-provider chain eliminated by one constraint
// produces the configured-candidate list plus the full explanation. v2 note:
// the catalogs are fetched before classification, so the classifier does run.
func TestRouteNoCandidateErrorIncludesPerProviderCounts(t *testing.T) {
	catalogA := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{
		{Provider: "a", ID: "vendor/a-text", PricePer1KPromptTokens: 0.1, PricePer1KCompletionTokens: 0.1, ContextLength: 8_000},
	}}
	catalogB := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{
		{Provider: "b", ID: "vendor/b-text", PricePer1KPromptTokens: 0.2, PricePer1KCompletionTokens: 0.2, ContextLength: 8_000},
	}}
	classifier := &testutil.FakeClassifier{Classification: modelrouter.Classification{Profile: testutil.CONFIDENT_TRIVIAL_PROFILE}}
	router := mustNewRouter(t, modelrouter.RouterOptions{
		Catalogs:   []modelrouter.Catalog{catalogA, catalogB},
		Classifier: classifier,
		Models: map[modelrouter.Domain][]modelrouter.ModelChoice{
			modelrouter.DomainOther: {
				{Provider: "a", Model: "vendor/a-text"},
				{Provider: "b", Model: "vendor/b-text"},
			},
		},
	})

	_, err := router.Route(context.Background(), "describe this image", &modelrouter.RoutingConstraints{NeedsVision: true})
	if err == nil {
		t.Fatal("Route() error = nil, want *NoCandidateModelError")
	}
	want := "modelrouter: no candidate model: " +
		"no configured candidate model satisfies the constraints (candidates: a/vendor/a-text, b/vendor/b-text); " +
		"needs_vision=true eliminates all (0/2 models support vision); " +
		"per provider: a 0/1; per provider: b 0/1"
	if got := err.Error(); got != want {
		t.Errorf("Route() error = %q, want %q", got, want)
	}
	if got := classifier.Calls(); got != 1 {
		t.Errorf("classifier.Calls() = %d, want 1 (the fetch succeeds; the chain is filtered after classification)", got)
	}
	if !strings.Contains(err.Error(), "needs_vision=true") {
		t.Errorf("error = %q, want it to name the eliminating constraint", err)
	}
}
