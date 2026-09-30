// These tests exercise the assembled router against live OpenRouter APIs.
// They are environment gated: without OPENROUTER_API_KEY every test skips
// silently, and TestIntegrationCompleteOnLiveCheapModel additionally requires
// JEV_ROUTER_LIVE_COMPLETION=1 because it performs a billed completion. Keys
// are checked for presence only; nothing here logs, prints or persists their
// values. Environment-only mode needs a default model in v2
// (JEV_ROUTER_DEFAULT_MODEL); these tests set a cheap one when the caller has
// not exported a value.
package config_test

import (
	"context"
	"os"
	"testing"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/config"
)

// integrationPrompt is the routing probe: a small, unambiguous code task the
// live Jev classifier should place in the code domain.
const integrationPrompt = "Fix this off-by-one bug in a Go for-loop that skips the last element."

// requireOpenRouterKey skips the test when OPENROUTER_API_KEY is absent.
func requireOpenRouterKey(t *testing.T) {
	t.Helper()
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("requires OPENROUTER_API_KEY")
	}
}

// buildLiveIntegrationApp assembles the environment-only configuration
// (config.Load("")): defaults plus an OpenRouter provider keyed by
// OPENROUTER_API_KEY, which covers the catalog, Jev and completions. Build
// performs no network I/O, so failures here are configuration failures; the
// first live call happens in the tests themselves. The App is closed at test
// end.
func buildLiveIntegrationApp(t *testing.T) *config.App {
	t.Helper()
	requireOpenRouterKey(t)
	// Environment-only mode has no routing.models, so it needs a default
	// model; any cheap model the key can reach works. An explicitly exported
	// JEV_ROUTER_DEFAULT_MODEL wins (env > YAML > defaults); the provider is
	// pinned to openrouter, the only provider environment-only mode
	// configures.
	if os.Getenv(config.DefaultModelEnv) == "" {
		t.Setenv(config.DefaultModelEnv, "upstage/solar-mini4")
		t.Setenv(config.DefaultProviderEnv, "openrouter")
	}

	f, err := config.Load("")
	if err != nil {
		t.Fatalf("config.Load(\"\") error = %v", err)
	}
	app, err := config.Build(f)
	if err != nil {
		t.Fatalf("config.Build() error = %v", err)
	}
	t.Cleanup(func() {
		if err := app.Close(); err != nil {
			t.Errorf("app.Close() error = %v", err)
		}
	})
	return app
}

// TestIntegrationRouteWithLiveJev routes one prompt through the full live
// pipeline: OpenRouter catalog fetch, one Jev classification call and the
// configured default model (environment-only mode maps no domains, so even a
// classified request routes to the default). It pins the RoutingDecision
// invariants the offline tests fake: a real profile with sane values and a
// winner that matches the head of the candidate chain.
func TestIntegrationRouteWithLiveJev(t *testing.T) {
	app := buildLiveIntegrationApp(t)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	decision, err := app.Router.Route(ctx, integrationPrompt, nil)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}

	profile := decision.Profile
	if profile == nil {
		t.Fatal("decision.Profile = nil, want a live Jev classification (did the fail-open fallback run?)")
	}
	if !modelrouter.IsValidDomain(profile.Domain) {
		t.Errorf("Profile.Domain = %q, want one of code, math_reasoning, creative, factual_lookup, other", profile.Domain)
	}

	// Probabilities: both confidences and all three noul fields must be in
	// [0,1]. (RequestProfile exposes one noul field per noul question Jev
	// answers: needs_long_context, needs_vision, latency_sensitive.)
	for _, p := range []struct {
		name  string
		value float64
	}{
		{"domain_confidence", profile.DomainConfidence},
		{"complexity_confidence", profile.ComplexityConfidence},
		{"needs_long_context", profile.NeedsLongContext},
		{"needs_vision", profile.NeedsVision},
		{"latency_sensitive", profile.LatencySensitive},
	} {
		if p.value < 0 || p.value > 1 {
			t.Errorf("Profile.%s = %v, want 0 <= value <= 1", p.name, p.value)
		}
	}
	if s := profile.ComplexityScore; s < 0 || s > 2 {
		t.Errorf("Profile.ComplexityScore = %v, want 0 <= score <= 2", s)
	}

	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("decision.Source = %q, want %q (environment-only mode configures no domain chains)",
			decision.Source, modelrouter.SourceDefault)
	}
	if len(decision.Candidates) == 0 {
		t.Fatal("decision.Candidates is empty, want the default model as a candidate")
	}
	if decision.ModelID != decision.Candidates[0].Model {
		t.Errorf("decision.ModelID = %q, want candidates[0].Model = %q", decision.ModelID, decision.Candidates[0].Model)
	}
	if decision.Provider != decision.Candidates[0].Provider {
		t.Errorf("decision.Provider = %q, want candidates[0].Provider = %q", decision.Provider, decision.Candidates[0].Provider)
	}
	if want := os.Getenv(config.DefaultModelEnv); decision.ModelID != want {
		t.Errorf("decision.ModelID = %q, want the JEV_ROUTER_DEFAULT_MODEL value %q", decision.ModelID, want)
	}
	if len(decision.RawJevResponse) == 0 {
		t.Error("decision.RawJevResponse is empty, want the raw Jev response body")
	}
}

// TestIntegrationCompleteOnLiveCheapModel routes the same prompt and then
// completes it on the routed (here: default) model, pinning the round trip
// through the OpenRouter provider. It is doubly gated: OPENROUTER_API_KEY
// must be present and JEV_ROUTER_LIVE_COMPLETION=1 must explicitly opt in to
// the billed completion (MaxTokens=16 keeps the call tiny).
func TestIntegrationCompleteOnLiveCheapModel(t *testing.T) {
	requireOpenRouterKey(t)
	if os.Getenv("JEV_ROUTER_LIVE_COMPLETION") != "1" {
		t.Skip("requires JEV_ROUTER_LIVE_COMPLETION=1 (billed completion opt-in)")
	}
	app := buildLiveIntegrationApp(t)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Route first with the constraints Complete reuses: Complete routes again
	// itself (Model stays unset), but the first pass proves the routing stage
	// succeeds and records which model won.
	var constraints *modelrouter.RoutingConstraints
	decision, err := app.Router.Route(ctx, integrationPrompt, constraints)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}

	result, err := app.Router.Complete(ctx, modelrouter.CompletionRequest{
		Prompt:      integrationPrompt,
		Constraints: constraints,
		MaxTokens:   16,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v (routed to %s)", err, decision.ModelID)
	}
	if result.Content == "" && result.Usage == nil {
		t.Errorf("Complete() returned empty Content and nil Usage, want at least one (model %q at MaxTokens=16)", result.Model)
	}
	if result.Model == "" {
		t.Error("result.Model is empty, want the provider-native model ID")
	}
	if result.Provider != "openrouter" {
		t.Errorf("result.Provider = %q, want %q", result.Provider, "openrouter")
	}
	if result.RoutingDecision.ModelID != result.Model {
		t.Errorf("result.RoutingDecision.ModelID = %q, want the completing model %q",
			result.RoutingDecision.ModelID, result.Model)
	}
	if len(result.Raw) == 0 {
		t.Error("result.Raw is empty, want the provider-native response body")
	}
}
