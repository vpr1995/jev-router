package modelrouter_test

import (
	"reflect"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// sampleCatalog is four models covering the filter's constraint boundaries:
// vision support, context window, and blended price.
func sampleCatalog() []modelrouter.ModelInfo {
	return []modelrouter.ModelInfo{
		{
			Provider:                   "openrouter",
			ID:                         "openai/gpt-4o",
			PricePer1KPromptTokens:     2.5,
			PricePer1KCompletionTokens: 10.0,
			ContextLength:              128_000,
			SupportsVision:             true,
		},
		{
			Provider:                   "openrouter",
			ID:                         "anthropic/claude-haiku-4-5",
			PricePer1KPromptTokens:     0.8,
			PricePer1KCompletionTokens: 4.0,
			ContextLength:              200_000,
			SupportsVision:             true,
		},
		{
			Provider:                   "openrouter",
			ID:                         "deepseek/deepseek-r1",
			PricePer1KPromptTokens:     0.5,
			PricePer1KCompletionTokens: 2.1,
			ContextLength:              64_000,
			SupportsVision:             false,
		},
		{
			Provider:                   "openrouter",
			ID:                         "meta-llama/llama-3.1-8b-instruct",
			PricePer1KPromptTokens:     0.06,
			PricePer1KCompletionTokens: 0.06,
			ContextLength:              16_000,
			SupportsVision:             false,
		},
	}
}

func filterIDs(models []modelrouter.ModelInfo) []string {
	ids := make([]string, len(models))
	for i, model := range models {
		ids[i] = model.ID
	}
	return ids
}

func filterIDSet(models []modelrouter.ModelInfo) map[string]bool {
	set := make(map[string]bool, len(models))
	for _, model := range models {
		set[model.ID] = true
	}
	return set
}

func assertFilterIDSet(t *testing.T, got []modelrouter.ModelInfo, want ...string) {
	t.Helper()
	wantSet := make(map[string]bool, len(want))
	for _, id := range want {
		wantSet[id] = true
	}
	if gotSet := filterIDSet(got); len(got) != len(want) || !reflect.DeepEqual(gotSet, wantSet) {
		t.Errorf("filtered IDs = %v, want set %v", filterIDs(got), want)
	}
}

func TestFilterModelsDropsModelsMissingVisionSupport(t *testing.T) {
	filtered := modelrouter.FilterModels(sampleCatalog(), modelrouter.RoutingConstraints{NeedsVision: true})
	assertFilterIDSet(t, filtered, "openai/gpt-4o", "anthropic/claude-haiku-4-5")
}

func TestFilterModelsDropsModelsBelowMinContextOrAboveMaxPrice(t *testing.T) {
	maxPrice := 5.0
	filtered := modelrouter.FilterModels(sampleCatalog(), modelrouter.RoutingConstraints{
		MinContextTokens:    32_000,
		MaxPricePer1KTokens: &maxPrice,
	})

	// max_price_per_1k_tokens filters on blended (prompt + completion) price:
	// gpt-4o: 2.5+10.0=12.5 > 5.0 -> dropped; llama-3.1-8b: context 16k < 32k
	// -> dropped; claude-haiku (context 200k, blended 0.8+4.0=4.8) and
	// deepseek-r1 (context 64k, blended 0.5+2.1=2.6) satisfy both.
	assertFilterIDSet(t, filtered, "anthropic/claude-haiku-4-5", "deepseek/deepseek-r1")
}

func TestFilterModelsMaxPriceUsesBlendedPromptPlusCompletionPrice(t *testing.T) {
	// gpt-4o's prompt price alone (2.5) would pass a threshold of 3.0, but its
	// blended price (2.5+10.0=12.5) must not — proving filtering uses the
	// blended price, not the prompt price alone.
	maxPrice := 3.0
	filtered := modelrouter.FilterModels(sampleCatalog(), modelrouter.RoutingConstraints{MaxPricePer1KTokens: &maxPrice})

	assertFilterIDSet(t, filtered, "deepseek/deepseek-r1", "meta-llama/llama-3.1-8b-instruct")
	if filterIDSet(filtered)["openai/gpt-4o"] {
		t.Error("openai/gpt-4o kept despite blended price 12.5 exceeding max 3.0")
	}
}

// Extension: default constraints keep every model.
func TestFilterModelsDefaultConstraintsKeepEverything(t *testing.T) {
	filtered := modelrouter.FilterModels(sampleCatalog(), modelrouter.RoutingConstraints{})
	if len(filtered) != 4 {
		t.Errorf("len(filtered) = %d, want 4 with default constraints", len(filtered))
	}
}

// Extension (multi-provider): provider allow/exclude constraints.
func TestFilterModelsProviderAllowAndExclude(t *testing.T) {
	models := []modelrouter.ModelInfo{
		{Provider: "openrouter", ID: "or/one"},
		{Provider: "openai", ID: "oai/one"},
		{Provider: "bedrock", ID: "br/one"},
	}
	tests := []struct {
		name        string
		constraints modelrouter.RoutingConstraints
		want        []string
	}{
		{
			name: "empty allows all",
			want: []string{"or/one", "oai/one", "br/one"},
		},
		{
			name:        "allowed list restricts",
			constraints: modelrouter.RoutingConstraints{AllowedProviders: []string{"openai", "bedrock"}},
			want:        []string{"oai/one", "br/one"},
		},
		{
			name:        "excluded list removes",
			constraints: modelrouter.RoutingConstraints{ExcludedProviders: []string{"bedrock"}},
			want:        []string{"or/one", "oai/one"},
		},
		{
			name: "exclusion wins over allowance",
			constraints: modelrouter.RoutingConstraints{
				AllowedProviders:  []string{"openai"},
				ExcludedProviders: []string{"openai"},
			},
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filtered := modelrouter.FilterModels(models, tt.constraints)
			got := filterIDs(filtered)
			if len(got) != len(tt.want) {
				t.Fatalf("filtered IDs = %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("filtered IDs = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

// Boundaries are inclusive: exactly at the limit still passes.
func TestFilterModelsContextAndPriceBoundariesAreInclusive(t *testing.T) {
	models := sampleCatalog()
	deepseekBlended := models[2].BlendedPricePer1KTokens()

	filtered := modelrouter.FilterModels(models, modelrouter.RoutingConstraints{
		MinContextTokens:    64_000,
		MaxPricePer1KTokens: &deepseekBlended,
	})
	assertFilterIDSet(t, filtered, "deepseek/deepseek-r1")
}

// Extension: input order is preserved and neither the slice nor the models
// are mutated.
func TestFilterModelsPreservesOrderAndDoesNotMutateInput(t *testing.T) {
	input := sampleCatalog()
	snapshot := append([]modelrouter.ModelInfo(nil), input...)

	maxPrice := 5.0
	filtered := modelrouter.FilterModels(input, modelrouter.RoutingConstraints{MaxPricePer1KTokens: &maxPrice})

	// Input order is [gpt-4o, claude-haiku, deepseek-r1, llama]; gpt-4o
	// (blended 12.5) is dropped, the rest keep their relative order.
	want := []string{"anthropic/claude-haiku-4-5", "deepseek/deepseek-r1", "meta-llama/llama-3.1-8b-instruct"}
	if got := filterIDs(filtered); !reflect.DeepEqual(got, want) {
		t.Errorf("filtered order = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(input, snapshot) {
		t.Errorf("FilterModels mutated its input:\n got %v\nwant %v", input, snapshot)
	}
}

// Extension: an empty input yields a non-nil empty slice, so callers can
// range over the result unconditionally.
func TestFilterModelsEmptyInputReturnsNonNilEmpty(t *testing.T) {
	for _, input := range [][]modelrouter.ModelInfo{nil, {}} {
		filtered := modelrouter.FilterModels(input, modelrouter.RoutingConstraints{})
		if filtered == nil {
			t.Error("FilterModels returned nil, want non-nil empty slice")
		}
		if len(filtered) != 0 {
			t.Errorf("len(filtered) = %d, want 0", len(filtered))
		}
	}
}
