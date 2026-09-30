package config_test

import (
	"context"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/config"
	"github.com/vprprudhvi/jev-router/provider"
)

func writeAssemblyFile(t *testing.T, dir, name, contents string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("os.WriteFile(%q) = %v", path, err)
	}
	return path
}

func assertMentions(t *testing.T, err error, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestBuildAllProviderKinds(t *testing.T) {
	unsetRoutingModelEnv(t)
	dir := t.TempDir()

	// One tiny catalog per keyed/bedrock/compatible provider. The ids are
	// referenced by the routing section below.
	catalog := func(provider, model string) string {
		return "provider: " + provider + "\nmodels:\n  - id: " + model + "\n" +
			"    context_length: 8192\n    price_per_1k_prompt_tokens: 0.001\n" +
			"    price_per_1k_completion_tokens: 0.002\n"
	}
	writeAssemblyFile(t, dir, "openai.yaml", catalog("openai", "openai-small"))
	writeAssemblyFile(t, dir, "anthropic.yaml", catalog("anthropic", "anthropic-small"))
	writeAssemblyFile(t, dir, "gemini.yaml", catalog("gemini", "gemini-small"))
	writeAssemblyFile(t, dir, "bedrock.yaml", catalog("bedrock", "bedrock-small"))
	writeAssemblyFile(t, dir, "groq.yaml", catalog("groq", "groq-small"))

	path := writeAssemblyFile(t, dir, "config.yaml", `providers:
  openrouter:
    api_key: test-openrouter-key
  openai:
    api_key: test-openai-key
    catalog: openai.yaml
  anthropic:
    api_key: test-anthropic-key
    catalog: anthropic.yaml
  gemini:
    api_key: test-gemini-key
    catalog: gemini.yaml
  bedrock:
    region: us-east-1
    catalog: bedrock.yaml
  compatible:
    - name: groq
      base_url: https://api.groq.com/openai/v1
      api_key: test-groq-key
      catalog: groq.yaml
routing:
  models:
    code:
      - provider: groq
        model: groq-small
  default_model:
    provider: openai
    model: openai-small
`)
	f, err := config.Load(path)
	if err != nil {
		t.Fatalf("config.Load(%q) = %v", path, err)
	}
	app, err := config.Build(f)
	if err != nil {
		t.Fatalf("config.Build() = %v", err)
	}
	if app.Router == nil {
		t.Error("app.Router = nil, want the assembled router")
	}
	want := []string{"anthropic", "bedrock", "gemini", "groq", "openai", "openrouter"}
	if !reflect.DeepEqual(app.Providers, want) {
		t.Errorf("app.Providers = %v, want %v (sorted)", app.Providers, want)
	}
	if app.File != f {
		t.Error("app.File does not point at the File it was built from")
	}
	if err := app.Close(); err != nil {
		t.Errorf("Close() = %v, want nil", err)
	}
	if err := app.Close(); err != nil {
		t.Errorf("second Close() = %v, want nil", err)
	}
}

const repeatCatalogYAML = `provider: repeat-local
models:
  - id: repeat-local-model
    context_length: 8192
    price_per_1k_prompt_tokens: 0.001
    price_per_1k_completion_tokens: 0.002
`

const repeatConfigYAML = `providers:
  compatible:
    - name: repeat-local
      base_url: http://127.0.0.1:0/v1
      api_key: x
      catalog: repeat.yaml
routing:
  default_model:
    provider: repeat-local
    model: repeat-local-model
`

func TestBuildIsRepeatable(t *testing.T) {
	unsetRoutingModelEnv(t)
	dir := t.TempDir()
	writeAssemblyFile(t, dir, "repeat.yaml", repeatCatalogYAML)
	path := writeAssemblyFile(t, dir, "config.yaml", repeatConfigYAML)
	f, err := config.Load(path)
	if err != nil {
		t.Fatalf("config.Load(%q) = %v", path, err)
	}

	first, err := config.Build(f)
	if err != nil {
		t.Fatalf("first Build() = %v", err)
	}
	second, err := config.Build(f)
	if err != nil {
		t.Fatalf("second Build() = %v", err)
	}
	if first.Router == nil || second.Router == nil {
		t.Fatalf("Build() routers = %v and %v, want both non-nil", first.Router, second.Router)
	}
	if err := first.Close(); err != nil {
		t.Errorf("first Close() = %v, want nil", err)
	}
	if err := second.Close(); err != nil {
		t.Errorf("second Close() = %v, want nil", err)
	}
}

const localCatalogYAML = `provider: local
models:
  - id: local-large
    context_length: 8192
    price_per_1k_prompt_tokens: 0.001
    price_per_1k_completion_tokens: 0.002
  - id: local-small
    context_length: 8192
    price_per_1k_prompt_tokens: 0.0001
    price_per_1k_completion_tokens: 0.0002
  - id: local-mid
    context_length: 8192
    price_per_1k_prompt_tokens: 0.0005
    price_per_1k_completion_tokens: 0.0007
`

func TestOfflineRouteThroughStaticCatalogOnly(t *testing.T) {
	unsetRoutingModelEnv(t)
	dir := t.TempDir()
	writeAssemblyFile(t, dir, "local.yaml", localCatalogYAML)
	path := writeAssemblyFile(t, dir, "config.yaml", `providers:
  compatible:
    - name: local
      base_url: http://127.0.0.1:0/v1
      api_key: x
      catalog: local.yaml
routing:
  default_model:
    provider: local
    model: local-small
`)
	f, err := config.Load(path)
	if err != nil {
		t.Fatalf("config.Load(%q) = %v", path, err)
	}

	// The fail-open warning for the missing classifier is expected; discard
	// it so the test output stays clean.
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	app, err := config.Build(f, config.WithLogger(logger))
	if err != nil {
		t.Fatalf("config.Build() = %v", err)
	}
	defer func() {
		if err := app.Close(); err != nil {
			t.Errorf("Close() = %v, want nil", err)
		}
	}()

	decision, err := app.Router.Route(context.Background(), "hello", nil)
	if err != nil {
		t.Fatalf("Route() = %v", err)
	}
	if decision.Profile != nil {
		t.Errorf("decision.Profile = %+v, want nil (no classifier configured)", decision.Profile)
	}
	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("decision.Source = %q, want %q", decision.Source, modelrouter.SourceDefault)
	}
	want := modelrouter.ModelChoice{Provider: "local", Model: "local-small"}
	if len(decision.Candidates) != 1 || decision.Candidates[0] != want {
		t.Errorf("decision.Candidates = %+v, want [%+v]", decision.Candidates, want)
	}
	if decision.ModelID != want.Model {
		t.Errorf("decision.ModelID = %q, want %q", decision.ModelID, want.Model)
	}
	if decision.Provider != want.Provider {
		t.Errorf("decision.Provider = %q, want %q", decision.Provider, want.Provider)
	}
}

func TestBuildValidationErrorNamesProvider(t *testing.T) {
	dir := t.TempDir()
	path := writeAssemblyFile(t, dir, "config.yaml", `providers:
  bedrock:
    catalog: bedrock.yaml
`)
	if _, err := config.Load(path); err == nil {
		t.Fatal("config.Load() = nil error, want the missing bedrock region to be rejected")
	} else {
		assertMentions(t, err, "region", "bedrock")
	}

	f := config.Default()
	f.Providers.Bedrock = &config.BedrockProvider{Catalog: "bedrock.yaml"} // Region deliberately empty
	app, err := config.Build(f)
	if err == nil {
		_ = app.Close()
		t.Fatal("config.Build() = nil error, want a validation error")
	}
	assertMentions(t, err, "region", "bedrock")
}

func TestBuildRejectsUnconfiguredModelProvider(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*config.File)
		want   string
	}{
		{
			"default_model",
			func(f *config.File) {
				f.Routing.DefaultModel = &config.ModelChoice{Provider: "gemini", Model: "gemini-model"}
			},
			`routing.default_model.provider "gemini": provider is not configured`,
		},
		{
			"models",
			func(f *config.File) {
				f.Routing.Models = map[string][]config.ModelChoice{
					"code": {{Provider: "anthropic", Model: "claude-model"}},
				}
				f.Routing.DefaultModel = nil
			},
			`routing.models.code[0].provider "anthropic": provider is not configured`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := config.Default()
			f.Providers.OpenRouter = &config.OpenRouter{KeyRef: config.KeyRef{APIKey: "test-key"}}
			tc.mutate(f)

			app, err := config.Build(f)
			if err == nil {
				_ = app.Close()
				t.Fatal("config.Build() = nil error, want an unconfigured provider error")
			}
			assertMentions(t, err, tc.want)
		})
	}
}

func TestCompatibleProvidersAreRegisteredOnce(t *testing.T) {
	unsetRoutingModelEnv(t)
	dir := t.TempDir()
	for _, name := range []string{"groq2", "ollama2"} {
		writeAssemblyFile(t, dir, name+".yaml", "provider: "+name+`
models:
  - id: `+name+`-model
    context_length: 8192
    price_per_1k_prompt_tokens: 0.001
    price_per_1k_completion_tokens: 0.002
`)
	}
	path := writeAssemblyFile(t, dir, "config.yaml", `providers:
  compatible:
    - name: groq2
      base_url: http://127.0.0.1:0/v1
      api_key: x
      catalog: groq2.yaml
    - name: ollama2
      base_url: http://127.0.0.1:0/v1
      api_key: x
      catalog: ollama2.yaml
routing:
  default_model:
    provider: groq2
    model: groq2-model
`)
	f, err := config.Load(path)
	if err != nil {
		t.Fatalf("config.Load(%q) = %v", path, err)
	}

	app, err := config.Build(f)
	if err != nil {
		t.Fatalf("config.Build() = %v", err)
	}
	for _, name := range []string{"groq2", "ollama2"} {
		if !provider.Registered(name) {
			t.Errorf("provider.Registered(%q) = false after Build, want true", name)
		}
	}

	second, err := config.Build(f)
	if err != nil {
		t.Fatalf("second Build() = %v (registered factories must be reused)", err)
	}
	if err := app.Close(); err != nil {
		t.Errorf("first Close() = %v, want nil", err)
	}
	if err := second.Close(); err != nil {
		t.Errorf("second Close() = %v, want nil", err)
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	f := config.Default()
	f.Providers.OpenRouter = &config.OpenRouter{KeyRef: config.KeyRef{APIKey: "test-key"}}
	f.Routing.DefaultModel = &config.ModelChoice{Provider: "openrouter", Model: "test/default-model"}

	app, err := config.Build(f)
	if err != nil {
		t.Fatalf("config.Build() = %v", err)
	}
	for i := 1; i <= 3; i++ {
		if err := app.Close(); err != nil {
			t.Errorf("Close() call %d = %v, want nil", i, err)
		}
	}
}
