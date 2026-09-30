package static_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vprprudhvi/jev-router/catalog/static"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func loadGood(t *testing.T, name string) *static.Catalog {
	t.Helper()
	catalog, err := static.Load(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("Load(%s) error = %v", name, err)
	}
	return catalog
}

// assertGoodCatalog checks the shared expectations for testdata/good.yaml and
// testdata/good.json (same content in both formats).
func assertGoodCatalog(t *testing.T, catalog *static.Catalog) {
	t.Helper()
	if got := catalog.Provider(); got != "openai" {
		t.Errorf("Provider() = %q, want %q", got, "openai")
	}

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("len(models) = %d, want 2", len(models))
	}

	first := models[0]
	if first.ID != "gpt-5.2" {
		t.Errorf("models[0].ID = %q, want %q", first.ID, "gpt-5.2")
	}
	if first.Provider != "openai" {
		t.Errorf("models[0].Provider = %q, want provider stamped from the file", first.Provider)
	}
	if first.DisplayID != "openai/gpt-5.2" {
		t.Errorf("models[0].DisplayID = %q, want %q", first.DisplayID, "openai/gpt-5.2")
	}
	if got := first.Label(); got != "openai/gpt-5.2" {
		t.Errorf("models[0].Label() = %q, want %q", got, "openai/gpt-5.2")
	}
	if first.ContextLength != 400_000 {
		t.Errorf("models[0].ContextLength = %d, want 400000", first.ContextLength)
	}
	if !first.SupportsVision {
		t.Error("models[0].SupportsVision = false, want true")
	}
	if !almostEqual(first.PricePer1KPromptTokens, 0.00125) {
		t.Errorf("models[0].PricePer1KPromptTokens = %v, want 0.00125", first.PricePer1KPromptTokens)
	}
	if !almostEqual(first.PricePer1KCompletionTokens, 0.01) {
		t.Errorf("models[0].PricePer1KCompletionTokens = %v, want 0.01", first.PricePer1KCompletionTokens)
	}

	second := models[1]
	// display_id is optional and defaults to id via ModelInfo.Label.
	if got := second.Label(); got != "gpt-5.2-mini" {
		t.Errorf("models[1].Label() = %q, want id fallback %q", got, "gpt-5.2-mini")
	}
	if second.ContextLength != 128_000 {
		t.Errorf("models[1].ContextLength = %d, want 128000", second.ContextLength)
	}
	if second.SupportsVision {
		t.Error("models[1].SupportsVision = true, want false")
	}
	if !almostEqual(second.PricePer1KPromptTokens, 0.00025) {
		t.Errorf("models[1].PricePer1KPromptTokens = %v, want 0.00025", second.PricePer1KPromptTokens)
	}
	if !almostEqual(second.PricePer1KCompletionTokens, 0.002) {
		t.Errorf("models[1].PricePer1KCompletionTokens = %v, want 0.002", second.PricePer1KCompletionTokens)
	}
}

func TestLoadYAML(t *testing.T) {
	assertGoodCatalog(t, loadGood(t, "good.yaml"))
}

func TestLoadJSON(t *testing.T) {
	assertGoodCatalog(t, loadGood(t, "good.json"))
}

// Extension: Parse accepts the JSON document too (JSON is a YAML subset).
func TestParseAcceptsJSON(t *testing.T) {
	catalog, err := static.Parse([]byte(`{"provider": "groq", "models": [{"id": "llama-3.3-70b", "context_length": 131072, "price_per_1k_prompt_tokens": 0.59, "price_per_1k_completion_tokens": 0.79}]}`))
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if got := catalog.Provider(); got != "groq" {
		t.Errorf("Provider() = %q, want %q", got, "groq")
	}
	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if len(models) != 1 || models[0].ID != "llama-3.3-70b" {
		t.Errorf("Models() = %v, want the single llama-3.3-70b entry", models)
	}
}

// Extension: unknown extensions fall back to the YAML parser.
func TestLoadUnknownExtensionFallsBackToYAML(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.conf")
	content := "provider: groq\nmodels:\n  - id: llama-3.3-70b\n    price_per_1k_prompt_tokens: 0.59\n    price_per_1k_completion_tokens: 0.79\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("writing temp catalog: %v", err)
	}

	catalog, err := static.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := catalog.Provider(); got != "groq" {
		t.Errorf("Provider() = %q, want %q", got, "groq")
	}
}

// Extension: missing files produce an error naming the path.
func TestLoadMissingFileErrorNamesPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.yaml")
	_, err := static.Load(path)
	if err == nil {
		t.Fatal("Load() error = nil, want read failure")
	}
	if !strings.Contains(err.Error(), "absent.yaml") {
		t.Errorf("Load() error = %q, want it to name the file", err)
	}
}

// Extension: validation failures name the file, model index, id and field.
func TestLoadValidationErrorNamesFileAndModel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.yaml")
	content := "provider: openai\nmodels:\n  - id: a\n  - id: b\n  - id: x\n    price_per_1k_prompt_tokens: -0.1\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("writing temp catalog: %v", err)
	}

	_, err := static.Load(path)
	want := `catalog file ` + path + `: model[2] ("x"): negative price_per_1k_prompt_tokens`
	if err == nil {
		t.Fatalf("Load() error = nil, want error containing %q", want)
	}
	if !strings.Contains(err.Error(), want) {
		t.Errorf("Load() error = %q, want containing %q", err, want)
	}
}

func TestParseValidationErrors(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantSub string
	}{
		{
			name:    "missing provider",
			input:   "models:\n  - id: a\n",
			wantSub: "provider",
		},
		{
			name:    "empty provider",
			input:   "provider: \"\"\nmodels:\n  - id: a\n",
			wantSub: "provider",
		},
		{
			name:    "empty models",
			input:   "provider: openai\nmodels: []\n",
			wantSub: "models must contain at least one model",
		},
		{
			name:    "missing models key",
			input:   "provider: openai\n",
			wantSub: "models must contain at least one model",
		},
		{
			name:    "missing id",
			input:   "provider: openai\nmodels:\n  - context_length: 1\n",
			wantSub: `model[0] (""): id is required`,
		},
		{
			name:    "duplicate id",
			input:   "provider: openai\nmodels:\n  - id: a\n  - id: a\n",
			wantSub: `model[1] ("a"): duplicate id`,
		},
		{
			name:    "negative context_length",
			input:   "provider: openai\nmodels:\n  - id: a\n    context_length: -1\n",
			wantSub: `model[0] ("a"): negative context_length`,
		},
		{
			name:    "negative prompt price",
			input:   "provider: openai\nmodels:\n  - id: a\n  - id: b\n  - id: x\n    price_per_1k_prompt_tokens: -0.1\n",
			wantSub: `model[2] ("x"): negative price_per_1k_prompt_tokens`,
		},
		{
			name:    "negative completion price",
			input:   "provider: openai\nmodels:\n  - id: x\n    price_per_1k_completion_tokens: -1\n",
			wantSub: `model[0] ("x"): negative price_per_1k_completion_tokens`,
		},
		{
			name:    "bad yaml syntax",
			input:   "provider: [unclosed\n",
			wantSub: "invalid YAML",
		},
		{
			name:    "bad yaml type",
			input:   "provider: openai\nmodels:\n  - id: a\n    context_length: not-a-number\n",
			wantSub: "invalid YAML",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := static.Parse([]byte(tt.input))
			if err == nil {
				t.Fatalf("Parse() error = nil, want error containing %q", tt.wantSub)
			}
			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("Parse() error = %q, want containing %q", err, tt.wantSub)
			}
		})
	}
}

// Extension: Models returns a defensive copy, so mutating the result cannot
// corrupt the loaded catalog.
func TestModelsReturnsDefensiveCopy(t *testing.T) {
	catalog := loadGood(t, "good.yaml")

	first, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	first[0].ID = "mutated"
	first[0].PricePer1KPromptTokens = 999

	second, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("second Models() error = %v", err)
	}
	if len(second) != 2 || second[0].ID != "gpt-5.2" {
		t.Errorf("second Models() = %v, want the uncorrupted catalog", second)
	}
	if !almostEqual(second[0].PricePer1KPromptTokens, 0.00125) {
		t.Errorf("second Models()[0].PricePer1KPromptTokens = %v, want 0.00125", second[0].PricePer1KPromptTokens)
	}
}
