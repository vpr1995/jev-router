package modelrouter_test

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
)

func TestModelInfoHoldsExpectedFields(t *testing.T) {
	model := modelrouter.ModelInfo{
		Provider:                   "openrouter",
		ID:                         "openai/gpt-4o",
		PricePer1KPromptTokens:     2.5,
		PricePer1KCompletionTokens: 10.0,
		ContextLength:              128_000,
		SupportsVision:             true,
	}

	if model.ID != "openai/gpt-4o" {
		t.Errorf("ID = %q, want %q", model.ID, "openai/gpt-4o")
	}
	if !model.SupportsVision {
		t.Error("SupportsVision = false, want true")
	}
	if got := model.BlendedPricePer1KTokens(); got != 12.5 {
		t.Errorf("BlendedPricePer1KTokens() = %v, want 12.5", got)
	}
}

func TestModelInfoLabelFallsBackToID(t *testing.T) {
	model := modelrouter.ModelInfo{ID: "gpt-5.2"}
	if got := model.Label(); got != "gpt-5.2" {
		t.Errorf("Label() = %q, want ID fallback %q", got, "gpt-5.2")
	}

	model.DisplayID = "openai/gpt-5.2"
	if got := model.Label(); got != "openai/gpt-5.2" {
		t.Errorf("Label() = %q, want DisplayID %q", got, "openai/gpt-5.2")
	}
}

func TestRoutingConstraintsDefaults(t *testing.T) {
	constraints := modelrouter.RoutingConstraints{}

	if constraints.NeedsVision {
		t.Error("NeedsVision = true, want false")
	}
	if constraints.MinContextTokens != 0 {
		t.Errorf("MinContextTokens = %d, want 0", constraints.MinContextTokens)
	}
	if constraints.MaxPricePer1KTokens != nil {
		t.Errorf("MaxPricePer1KTokens = %v, want nil", *constraints.MaxPricePer1KTokens)
	}
	if constraints.AllowedProviders != nil {
		t.Errorf("AllowedProviders = %v, want nil", constraints.AllowedProviders)
	}
	if constraints.ExcludedProviders != nil {
		t.Errorf("ExcludedProviders = %v, want nil", constraints.ExcludedProviders)
	}
}

// Extension (multi-provider): provider allow/deny semantics.
func TestRoutingConstraintsAllowsProvider(t *testing.T) {
	tests := []struct {
		name        string
		constraints modelrouter.RoutingConstraints
		provider    string
		want        bool
	}{
		{"empty allows all", modelrouter.RoutingConstraints{}, "bedrock", true},
		{"allowed list restricts", modelrouter.RoutingConstraints{AllowedProviders: []string{"openrouter", "openai"}}, "bedrock", false},
		{"allowed list permits member", modelrouter.RoutingConstraints{AllowedProviders: []string{"openrouter", "openai"}}, "openai", true},
		{"excluded removes", modelrouter.RoutingConstraints{ExcludedProviders: []string{"bedrock"}}, "bedrock", false},
		{"excluded leaves others", modelrouter.RoutingConstraints{ExcludedProviders: []string{"bedrock"}}, "openrouter", true},
		{"exclusion wins over allowance", modelrouter.RoutingConstraints{AllowedProviders: []string{"openrouter"}, ExcludedProviders: []string{"openrouter"}}, "openrouter", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.constraints.AllowsProvider(tt.provider); got != tt.want {
				t.Errorf("AllowsProvider(%q) = %v, want %v", tt.provider, got, tt.want)
			}
		})
	}
}

func TestRequestProfileHoldsAllClassifiedFields(t *testing.T) {
	profile := modelrouter.RequestProfile{
		Domain:               modelrouter.DomainCode,
		DomainConfidence:     0.9,
		ComplexityScore:      1.5,
		ComplexityConfidence: 0.8,
		NeedsLongContext:     0.1,
		NeedsVision:          0.0,
		LatencySensitive:     0.2,
	}

	if profile.Domain != modelrouter.DomainCode {
		t.Errorf("Domain = %q, want %q", profile.Domain, modelrouter.DomainCode)
	}
	if profile.ComplexityScore != 1.5 {
		t.Errorf("ComplexityScore = %v, want 1.5", profile.ComplexityScore)
	}
}

func TestDomainConstantsMatchJevChoiceValues(t *testing.T) {
	want := map[modelrouter.Domain]string{
		modelrouter.DomainCode:          "code",
		modelrouter.DomainMathReasoning: "math_reasoning",
		modelrouter.DomainCreative:      "creative",
		modelrouter.DomainFactualLookup: "factual_lookup",
		modelrouter.DomainOther:         "other",
	}
	for domain, raw := range want {
		if string(domain) != raw {
			t.Errorf("domain constant = %q, want %q", string(domain), raw)
		}
	}
}

// Extension (v2 mapped routing): ModelChoice is the configuration unit and
// must round-trip through JSON for config and CLI consumers.
func TestModelChoiceJSONRoundTrip(t *testing.T) {
	choice := modelrouter.ModelChoice{Provider: "openrouter", Model: "openai/gpt-5.2"}

	data, err := json.Marshal(choice)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var decoded modelrouter.ModelChoice
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal(%s) error = %v", data, err)
	}
	if decoded != choice {
		t.Errorf("round trip = %+v, want %+v", decoded, choice)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("json.Unmarshal(%s) error = %v", data, err)
	}
	if raw["provider"] != choice.Provider || raw["model"] != choice.Model {
		t.Errorf("JSON shape = %v, want provider/model keys", raw)
	}
}

// Extension (v2 mapped routing): the five Jev domains are the only valid
// Models keys.
func TestIsValidDomain(t *testing.T) {
	for _, domain := range []modelrouter.Domain{
		modelrouter.DomainCode,
		modelrouter.DomainMathReasoning,
		modelrouter.DomainCreative,
		modelrouter.DomainFactualLookup,
		modelrouter.DomainOther,
	} {
		if !modelrouter.IsValidDomain(domain) {
			t.Errorf("IsValidDomain(%q) = false, want true", domain)
		}
	}
	for _, domain := range []modelrouter.Domain{"", "codex", "Code", "math", "other ", "vision"} {
		if modelrouter.IsValidDomain(domain) {
			t.Errorf("IsValidDomain(%q) = true, want false", domain)
		}
	}
}

// TestRoutingDecisionComposition pins the mapped-routing decision shape:
// Source + Candidates, alongside the classification profile and raw response.
func TestRoutingDecisionComposition(t *testing.T) {
	profile := modelrouter.RequestProfile{
		Domain:               modelrouter.DomainOther,
		DomainConfidence:     0.9,
		ComplexityScore:      0.0,
		ComplexityConfidence: 0.9,
	}
	candidates := []modelrouter.ModelChoice{
		{Provider: "test", Model: "vendor/cheap"},
		{Provider: "test", Model: "vendor/default"},
	}
	decision := modelrouter.RoutingDecision{
		ModelID:        "vendor/cheap",
		Provider:       "test",
		Source:         modelrouter.SourceClassification,
		Candidates:     candidates,
		Profile:        &profile,
		RawJevResponse: json.RawMessage(`{"answers":{}}`),
	}
	completion := modelrouter.CompletionResult{
		RoutingDecision: decision,
		Raw:             json.RawMessage(`{"id":"resp-1"}`),
	}

	if decision.ModelID != "vendor/cheap" {
		t.Errorf("ModelID = %q, want %q", decision.ModelID, "vendor/cheap")
	}
	if decision.Source != modelrouter.SourceClassification {
		t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceClassification)
	}
	if len(decision.Candidates) != 2 || decision.Candidates[1].Model != "vendor/default" {
		t.Errorf("Candidates = %+v, want the ordered chain", decision.Candidates)
	}
	var raw map[string]any
	if err := json.Unmarshal(completion.Raw, &raw); err != nil {
		t.Fatalf("unmarshal raw completion: %v", err)
	}
	if raw["id"] != "resp-1" {
		t.Errorf(`completion id = %v, want "resp-1"`, raw["id"])
	}

	// JSON consumers (server, CLI --json) must see the v2 decision shape:
	// source and candidates round-trip alongside the profile.
	encoded, err := json.Marshal(decision)
	if err != nil {
		t.Fatalf("json.Marshal(decision) error = %v", err)
	}
	var decoded modelrouter.RoutingDecision
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("json.Unmarshal(%s) error = %v", encoded, err)
	}
	if decoded.Source != decision.Source || !reflect.DeepEqual(decoded.Candidates, decision.Candidates) {
		t.Errorf("decision round trip = %+v, want source and candidates preserved", decoded)
	}
	if decoded.Profile == nil || *decoded.Profile != profile {
		t.Errorf("decision round trip profile = %+v, want %+v", decoded.Profile, profile)
	}
}

// TestRoutingDecisionProfileMayBeNilForDefaultFallbackRouting pins that the
// fallback path uses the configured default model, marked by SourceDefault,
// with a nil profile.
func TestRoutingDecisionProfileMayBeNilForDefaultFallbackRouting(t *testing.T) {
	decision := modelrouter.RoutingDecision{
		ModelID:        "vendor/cheap",
		Provider:       "test",
		Source:         modelrouter.SourceDefault,
		Candidates:     []modelrouter.ModelChoice{{Provider: "test", Model: "vendor/cheap"}},
		Profile:        nil,
		RawJevResponse: json.RawMessage(`{}`),
	}

	if decision.Profile != nil {
		t.Error("Profile != nil, want nil for the default-model fallback")
	}
	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceDefault)
	}
	if len(decision.Candidates) != 1 {
		t.Errorf("Candidates = %+v, want the default model as the only candidate", decision.Candidates)
	}
}

// TestErrorTypesAreErrors pins the error model: sentinel identifiers +
// typed carriers, all compatible with errors.Is/As.
func TestErrorTypesAreErrors(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		target error
	}{
		{"catalog unavailable", &modelrouter.CatalogUnavailableError{Err: errors.New("boom")}, modelrouter.ErrCatalogUnavailable},
		{"classifier unavailable", &modelrouter.ClassifierUnavailableError{Err: errors.New("boom")}, modelrouter.ErrClassifierUnavailable},
		{"no candidate model", &modelrouter.NoCandidateModelError{Message: "none"}, modelrouter.ErrNoCandidate},
		{"provider error", &modelrouter.ProviderError{Provider: "openai", Op: "complete", Err: errors.New("boom")}, modelrouter.ErrProvider},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !errors.Is(tc.err, tc.target) {
				t.Errorf("errors.Is(%T, %v) = false, want true", tc.err, tc.target)
			}
			var carrier interface{ Unwrap() error }
			if !errors.As(tc.err, &carrier) {
				t.Errorf("errors.As(%T, *interface{Unwrap()error}) = false, want true", tc.err)
			}
			if tc.err.Error() == "" {
				t.Error("Error() returned empty string")
			}
		})
	}
}

func TestProviderErrorKeepsCauseChain(t *testing.T) {
	cause := errors.New("connection reset")
	err := &modelrouter.ProviderError{Provider: "bedrock", Op: "complete", StatusCode: 502, Err: cause}

	if !errors.Is(err, cause) {
		t.Error("errors.Is(err, cause) = false, want true; cause chain must be preserved")
	}
	if !errors.Is(err, modelrouter.ErrProvider) {
		t.Error("errors.Is(err, ErrProvider) = false, want true")
	}
}
