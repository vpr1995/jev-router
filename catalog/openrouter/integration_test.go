package openrouter_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/vprprudhvi/jev-router/catalog/openrouter"
)

func requireOpenRouterKey(t *testing.T) string {
	t.Helper()
	key := os.Getenv("OPENROUTER_API_KEY")
	if key == "" {
		t.Skip("requires OPENROUTER_API_KEY")
	}
	return key
}

func TestIntegrationFetchLiveCatalog(t *testing.T) {
	key := requireOpenRouterKey(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	catalog := openrouter.New(key, openrouter.WithCacheTTL(5*time.Minute))
	defer func() { _ = catalog.Close() }()

	models, err := catalog.Models(ctx)
	if err != nil {
		t.Fatalf("Models() error = %v (live OpenRouter catalog fetch)", err)
	}
	if len(models) < 50 {
		t.Errorf("len(models) = %d, want >= 50 (OpenRouter serves hundreds)", len(models))
	}

	var visionCapable, longContext int
	for i, m := range models {
		if m.ID == "" {
			t.Errorf("models[%d].ID is empty, want a provider-native model ID", i)
		}
		if m.Provider != openrouter.ProviderName {
			t.Errorf("models[%d] (%s).Provider = %q, want %q", i, m.ID, m.Provider, openrouter.ProviderName)
		}
		if m.PricePer1KPromptTokens < 0 || m.PricePer1KCompletionTokens < 0 {
			t.Errorf("models[%d] (%s) has negative prices: prompt=%v completion=%v",
				i, m.ID, m.PricePer1KPromptTokens, m.PricePer1KCompletionTokens)
		}
		if m.ContextLength < 0 {
			t.Errorf("models[%d] (%s).ContextLength = %d, want >= 0", i, m.ID, m.ContextLength)
		}
		if m.SupportsVision {
			visionCapable++
		}
		if m.ContextLength >= 100000 {
			longContext++
		}
	}
	if visionCapable == 0 {
		t.Error("no model in the live catalog SupportsVision, want at least one")
	}
	if longContext == 0 {
		t.Error("no model in the live catalog has ContextLength >= 100000, want at least one")
	}

	// The second fetch is served from the 5-minute cache: same length, no
	// error, no request.
	cached, err := catalog.Models(ctx)
	if err != nil {
		t.Fatalf("second Models() error = %v, want the cached catalog", err)
	}
	if len(cached) != len(models) {
		t.Errorf("second Models() returned %d models, want %d (cache path)", len(cached), len(models))
	}
}

func TestIntegrationLegacyAndModernArchitectureShapesCoexist(t *testing.T) {
	key := requireOpenRouterKey(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	catalog := openrouter.New(key, openrouter.WithCacheTTL(5*time.Minute))
	defer func() { _ = catalog.Close() }()

	models, err := catalog.Models(ctx)
	if err != nil {
		t.Fatalf("Models() error = %v (live OpenRouter catalog fetch)", err)
	}

	var visionCapable, textOnly int
	for _, m := range models {
		if m.SupportsVision {
			visionCapable++
		} else {
			textOnly++
		}
	}
	if visionCapable == 0 {
		t.Error("no vision-capable model found, want models parsed from the modern/legacy vision shapes")
	}
	if textOnly == 0 {
		t.Error("no text-only model found, want non-vision entries to coexist with vision-capable ones")
	}
	t.Logf("live catalog: %d models, %d vision-capable, %d text-only", len(models), visionCapable, textOnly)
}
