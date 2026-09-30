package config_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vprprudhvi/jev-router/config"
)

// minimalConfig is a valid configuration used by tests that only care about
// individual fields. It carries a default_model because a configuration
// without any model mapping — routing.models or routing.default_model — is
// rejected.
const minimalConfig = `providers:
  openrouter:
    api_key: test-key
routing:
  default_model:
    provider: openrouter
    model: test/default-model
`

// minimalCatalog is a valid user catalog file. Tests that need a catalog
// path that exists on disk write it to a temp dir (config.Load validates
// catalog presence with a Stat; only static.Load reads the contents).
const minimalCatalog = `provider: test
models:
  - id: test-model
    context_length: 8192
    price_per_1k_prompt_tokens: 0.001
    price_per_1k_completion_tokens: 0.002
`

// unsetenv removes key for the duration of the test, restoring whatever was
// set before (t.Setenv alone cannot express "unset").
func unsetenv(t *testing.T, key string) {
	t.Helper()
	previous, had := os.LookupEnv(key)
	if err := os.Unsetenv(key); err != nil {
		t.Fatalf("os.Unsetenv(%q) = %v", key, err)
	}
	t.Cleanup(func() {
		if had {
			_ = os.Setenv(key, previous)
			return
		}
		_ = os.Unsetenv(key)
	})
}

// unsetRoutingModelEnv removes the routing default-model environment
// overrides for the duration of the test, so tests assert the file's or the
// built-in defaults instead of an ambient JEV_ROUTER_DEFAULT_MODEL.
func unsetRoutingModelEnv(t *testing.T) {
	t.Helper()
	unsetenv(t, config.DefaultModelEnv)
	unsetenv(t, config.DefaultProviderEnv)
}

// writeFile writes contents to a fresh temp file and returns its path.
func writeFile(t *testing.T, name, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("os.WriteFile(%q) = %v", path, err)
	}
	return path
}

// validFile returns a minimal valid configuration: documented defaults plus
// one OpenRouter provider with a literal key, a code-domain chain and a
// default model.
func validFile() *config.File {
	f := config.Default()
	f.Providers.OpenRouter = &config.OpenRouter{KeyRef: config.KeyRef{APIKey: "test-key"}}
	f.Routing.Models = map[string][]config.ModelChoice{
		"code": {{Provider: "openrouter", Model: "test/code-model"}},
	}
	f.Routing.DefaultModel = &config.ModelChoice{Provider: "openrouter", Model: "test/default-model"}
	return f
}

func TestResolve(t *testing.T) {
	t.Run("literal wins over env", func(t *testing.T) {
		t.Setenv("JEV_TEST_KEY", "from-env")
		got, err := config.KeyRef{APIKey: "literal", APIKeyEnv: "JEV_TEST_KEY"}.Resolve()
		if err != nil {
			t.Fatalf("Resolve() = %v", err)
		}
		if got != "literal" {
			t.Errorf("Resolve() = %q, want %q", got, "literal")
		}
	})
	t.Run("reads env when literal empty", func(t *testing.T) {
		t.Setenv("JEV_TEST_KEY", "from-env")
		got, err := config.KeyRef{APIKeyEnv: "JEV_TEST_KEY"}.Resolve()
		if err != nil {
			t.Fatalf("Resolve() = %v", err)
		}
		if got != "from-env" {
			t.Errorf("Resolve() = %q, want %q", got, "from-env")
		}
	})
	t.Run("errors naming unset env var", func(t *testing.T) {
		unsetenv(t, "JEV_TEST_MISSING_KEY")
		_, err := config.KeyRef{APIKeyEnv: "JEV_TEST_MISSING_KEY"}.Resolve()
		if err == nil {
			t.Fatal("Resolve() = nil error, want missing variable error")
		}
		if !strings.Contains(err.Error(), "JEV_TEST_MISSING_KEY") {
			t.Errorf("Resolve() error = %q, want it to name JEV_TEST_MISSING_KEY", err)
		}
	})
	t.Run("errors naming empty env var", func(t *testing.T) {
		t.Setenv("JEV_TEST_EMPTY_KEY", "")
		_, err := config.KeyRef{APIKeyEnv: "JEV_TEST_EMPTY_KEY"}.Resolve()
		if err == nil {
			t.Fatal("Resolve() = nil error, want empty variable error")
		}
		if !strings.Contains(err.Error(), "JEV_TEST_EMPTY_KEY") {
			t.Errorf("Resolve() error = %q, want it to name JEV_TEST_EMPTY_KEY", err)
		}
	})
	t.Run("errors when neither configured", func(t *testing.T) {
		_, err := config.KeyRef{}.Resolve()
		if err == nil {
			t.Fatal("Resolve() = nil error, want api_key or api_key_env error")
		}
		if !strings.Contains(err.Error(), "api_key or api_key_env required") {
			t.Errorf("Resolve() error = %q, want it to mention api_key or api_key_env required", err)
		}
	})
}

// TestLoadEnvOnlyDefaults pins environment-only mode: the documented
// defaults, with OPENROUTER_API_KEY covering Jev, the catalog and
// completions, and the default model coming from JEV_ROUTER_DEFAULT_MODEL.
func TestLoadEnvOnlyDefaults(t *testing.T) {
	unsetenv(t, config.ConfigPathEnv)
	unsetenv(t, config.ServerAddrEnv)
	unsetenv(t, config.DefaultProviderEnv)
	t.Setenv("OPENROUTER_API_KEY", "or-key")
	t.Setenv(config.DefaultModelEnv, "test/env-model")

	f, err := config.Load("")
	if err != nil {
		t.Fatalf("Load(\"\") = %v", err)
	}
	if f.Providers.OpenRouter == nil {
		t.Fatal("Providers.OpenRouter = nil, want the environment-only default provider")
	}
	key, err := f.Providers.OpenRouter.Resolve()
	if err != nil {
		t.Fatalf("Resolve() = %v", err)
	}
	if key != "or-key" {
		t.Errorf("Resolve() = %q, want %q", key, "or-key")
	}

	if got, want := f.Routing.CatalogCacheTTL.Value(), 600*time.Second; got != want {
		t.Errorf("CatalogCacheTTL = %v, want %v", got, want)
	}
	if got := f.Routing.OnClassifierError; got != "fail_open" {
		t.Errorf("OnClassifierError = %q, want %q", got, "fail_open")
	}
	if want := (config.ModelChoice{Provider: "openrouter", Model: "test/env-model"}); f.Routing.DefaultModel == nil || *f.Routing.DefaultModel != want {
		t.Errorf("DefaultModel = %+v, want %+v (provider defaults to openrouter)", f.Routing.DefaultModel, want)
	}
	if len(f.Routing.Models) != 0 {
		t.Errorf("Routing.Models = %v, want none in environment-only mode", f.Routing.Models)
	}
	if got := f.Classifier.BaseURL; got != "https://openrouter.ai" {
		t.Errorf("Classifier.BaseURL = %q, want %q", got, "https://openrouter.ai")
	}
	if got := f.Classifier.Path; got != "/api/v1/systemone" {
		t.Errorf("Classifier.Path = %q, want %q", got, "/api/v1/systemone")
	}
	if got := f.Classifier.Model; got != "~typesafe/jev-latest" {
		t.Errorf("Classifier.Model = %q, want %q", got, "~typesafe/jev-latest")
	}
	if got, want := f.Classifier.Timeout.Value(), 5*time.Second; got != want {
		t.Errorf("Classifier.Timeout = %v, want %v", got, want)
	}
	if got := f.Server.Addr; got != ":8080" {
		t.Errorf("Server.Addr = %q, want %q", got, ":8080")
	}
	if got, want := f.Server.ReadTimeout.Value(), 30*time.Second; got != want {
		t.Errorf("Server.ReadTimeout = %v, want %v", got, want)
	}
	if got := f.Telemetry.Enabled; got != "auto" {
		t.Errorf("Telemetry.Enabled = %q, want %q", got, "auto")
	}
}

func TestLoadEnvOnlyMissingKeyNamesVar(t *testing.T) {
	unsetenv(t, config.ConfigPathEnv)
	unsetenv(t, "OPENROUTER_API_KEY")

	_, err := config.Load("")
	if err == nil {
		t.Fatal("Load(\"\") = nil error, want missing OPENROUTER_API_KEY error")
	}
	if !strings.Contains(err.Error(), "OPENROUTER_API_KEY") {
		t.Fatalf("Load(\"\") error = %q, want it to name OPENROUTER_API_KEY", err)
	}
}

func TestLoadEnvOnlyEmptyKeyCountsAsMissing(t *testing.T) {
	unsetenv(t, config.ConfigPathEnv)
	t.Setenv("OPENROUTER_API_KEY", "")

	_, err := config.Load("")
	if err == nil {
		t.Fatal("Load(\"\") = nil error, want empty OPENROUTER_API_KEY to be rejected")
	}
	if !strings.Contains(err.Error(), "OPENROUTER_API_KEY") {
		t.Fatalf("Load(\"\") error = %q, want it to name OPENROUTER_API_KEY", err)
	}
}

// Environment-only mode needs a default model from JEV_ROUTER_DEFAULT_MODEL;
// without it Load fails with the routing error that names the variable.
func TestLoadEnvOnlyRequiresDefaultModel(t *testing.T) {
	unsetenv(t, config.ConfigPathEnv)
	unsetRoutingModelEnv(t)
	t.Setenv("OPENROUTER_API_KEY", "or-key")

	_, err := config.Load("")
	if err == nil {
		t.Fatal("Load(\"\") = nil error, want the missing default model to be rejected")
	}
	want := "routing.models or routing.default_model: at least one model mapping is required (set JEV_ROUTER_DEFAULT_MODEL in environment-only mode)"
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("Load(\"\") error = %q, want it to contain %q", err, want)
	}
}

// JEV_ROUTER_DEFAULT_MODEL supplies or overrides routing.default_model; the
// provider comes from JEV_ROUTER_DEFAULT_PROVIDER and defaults to
// openrouter. Env beats the file (flags > env > YAML > defaults).
func TestLoadDefaultModelEnvOverrides(t *testing.T) {
	dir := t.TempDir()
	writeAssemblyFile(t, dir, "anthropic.yaml", `provider: anthropic
models:
  - id: claude-test
    context_length: 8192
    price_per_1k_prompt_tokens: 0.001
    price_per_1k_completion_tokens: 0.002
`)
	path := writeAssemblyFile(t, dir, "config.yaml", `providers:
  openrouter:
    api_key: k
  anthropic:
    api_key: a
    catalog: anthropic.yaml
routing:
  default_model:
    provider: openrouter
    model: file/model
`)

	t.Run("overrides the file's default_model", func(t *testing.T) {
		unsetenv(t, config.DefaultProviderEnv)
		t.Setenv(config.DefaultModelEnv, "env/model")

		f, err := config.Load(path)
		if err != nil {
			t.Fatalf("Load(%q) = %v", path, err)
		}
		if want := (config.ModelChoice{Provider: "openrouter", Model: "env/model"}); f.Routing.DefaultModel == nil || *f.Routing.DefaultModel != want {
			t.Errorf("DefaultModel = %+v, want %+v", f.Routing.DefaultModel, want)
		}
	})

	t.Run("custom provider", func(t *testing.T) {
		t.Setenv(config.DefaultModelEnv, "env/model")
		t.Setenv(config.DefaultProviderEnv, "anthropic")

		f, err := config.Load(path)
		if err != nil {
			t.Fatalf("Load(%q) = %v", path, err)
		}
		if want := (config.ModelChoice{Provider: "anthropic", Model: "env/model"}); f.Routing.DefaultModel == nil || *f.Routing.DefaultModel != want {
			t.Errorf("DefaultModel = %+v, want %+v", f.Routing.DefaultModel, want)
		}
	})

	t.Run("sets when the file has none", func(t *testing.T) {
		unsetenv(t, config.DefaultProviderEnv)
		t.Setenv(config.DefaultModelEnv, "env/model")
		path := writeAssemblyFile(t, t.TempDir(), "config.yaml", `providers:
  openrouter:
    api_key: k
`)

		f, err := config.Load(path)
		if err != nil {
			t.Fatalf("Load(%q) = %v", path, err)
		}
		if want := (config.ModelChoice{Provider: "openrouter", Model: "env/model"}); f.Routing.DefaultModel == nil || *f.Routing.DefaultModel != want {
			t.Errorf("DefaultModel = %+v, want %+v (set from the environment)", f.Routing.DefaultModel, want)
		}
	})

	t.Run("unset env keeps the file value", func(t *testing.T) {
		unsetRoutingModelEnv(t)

		f, err := config.Load(path)
		if err != nil {
			t.Fatalf("Load(%q) = %v", path, err)
		}
		if want := (config.ModelChoice{Provider: "openrouter", Model: "file/model"}); f.Routing.DefaultModel == nil || *f.Routing.DefaultModel != want {
			t.Errorf("DefaultModel = %+v, want %+v", f.Routing.DefaultModel, want)
		}
	})
}

func TestValidateRejectsTypoedOnClassifierError(t *testing.T) {
	f := validFile()
	f.Routing.OnClassifierError = "fail_opne"

	err := f.Validate()
	if err == nil {
		t.Fatal("Validate() = nil, want on_classifier_error error")
	}
	if !strings.Contains(err.Error(), "on_classifier_error") {
		t.Fatalf("Validate() error = %q, want it to mention on_classifier_error", err)
	}
}

func TestValidateRejectsWrongCaseOnClassifierError(t *testing.T) {
	f := validFile()
	f.Routing.OnClassifierError = "Fail_Closed"

	err := f.Validate()
	if err == nil {
		t.Fatal("Validate() = nil, want on_classifier_error error")
	}
	if !strings.Contains(err.Error(), "on_classifier_error") {
		t.Fatalf("Validate() error = %q, want it to mention on_classifier_error", err)
	}
}

// Explicit values in every section must survive decoding and validation.
func TestLoadExplicitValuesHonored(t *testing.T) {
	unsetRoutingModelEnv(t)
	path := writeFile(t, "config.yaml", `providers:
  openrouter:
    api_key: literal-test-key
routing:
  catalog_cache_ttl: 90
  on_classifier_error: fail_closed
  models:
    code:
      - provider: openrouter
        model: test/code-model
    other:
      - provider: openrouter
        model: test/other-model
  default_model:
    provider: openrouter
    model: test/default-model
  allowed_providers: [openrouter]
  excluded_providers: [bedrock]
classifier:
  base_url: https://api.typesafe.ai
  path: /jev-latest
  model: jev-1.13
  timeout: 2s
server:
  addr: ":9090"
  read_timeout: 12s
telemetry:
  enabled: "off"
`)

	f, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load(%q) = %v", path, err)
	}
	if got, want := f.Routing.CatalogCacheTTL.Value(), 90*time.Second; got != want {
		t.Errorf("CatalogCacheTTL = %v, want %v (integer seconds)", got, want)
	}
	if got := f.Routing.OnClassifierError; got != "fail_closed" {
		t.Errorf("OnClassifierError = %q, want %q", got, "fail_closed")
	}
	if got := len(f.Routing.Models); got != 2 {
		t.Errorf("len(Routing.Models) = %d, want 2", got)
	}
	if chain := f.Routing.Models["code"]; len(chain) != 1 || chain[0] != (config.ModelChoice{Provider: "openrouter", Model: "test/code-model"}) {
		t.Errorf("Routing.Models[code] = %+v, want the openrouter test/code-model chain", chain)
	}
	if chain := f.Routing.Models["other"]; len(chain) != 1 || chain[0] != (config.ModelChoice{Provider: "openrouter", Model: "test/other-model"}) {
		t.Errorf("Routing.Models[other] = %+v, want the openrouter test/other-model chain", chain)
	}
	if want := (config.ModelChoice{Provider: "openrouter", Model: "test/default-model"}); f.Routing.DefaultModel == nil || *f.Routing.DefaultModel != want {
		t.Errorf("DefaultModel = %+v, want %+v", f.Routing.DefaultModel, want)
	}
	if got, want := f.Routing.AllowedProviders, []string{"openrouter"}; len(got) != 1 || got[0] != want[0] {
		t.Errorf("AllowedProviders = %v, want %v", got, want)
	}
	if got, want := f.Routing.ExcludedProviders, []string{"bedrock"}; len(got) != 1 || got[0] != want[0] {
		t.Errorf("ExcludedProviders = %v, want %v", got, want)
	}
	if got := f.Classifier.BaseURL; got != "https://api.typesafe.ai" {
		t.Errorf("Classifier.BaseURL = %q, want %q", got, "https://api.typesafe.ai")
	}
	if got := f.Classifier.Path; got != "/jev-latest" {
		t.Errorf("Classifier.Path = %q, want %q", got, "/jev-latest")
	}
	if got := f.Classifier.Model; got != "jev-1.13" {
		t.Errorf("Classifier.Model = %q, want %q", got, "jev-1.13")
	}
	if got, want := f.Classifier.Timeout.Value(), 2*time.Second; got != want {
		t.Errorf("Classifier.Timeout = %v, want %v", got, want)
	}
	if got := f.Server.Addr; got != ":9090" {
		t.Errorf("Server.Addr = %q, want %q", got, ":9090")
	}
	if got, want := f.Server.ReadTimeout.Value(), 12*time.Second; got != want {
		t.Errorf("Server.ReadTimeout = %v, want %v", got, want)
	}
	if got := f.Telemetry.Enabled; got != "off" {
		t.Errorf("Telemetry.Enabled = %q, want %q", got, "off")
	}
	if key, err := f.Providers.OpenRouter.Resolve(); err != nil || key != "literal-test-key" {
		t.Errorf("OpenRouter.Resolve() = %q, %v; want literal-test-key", key, err)
	}
}

// Strict decoding: unknown fields are rejected, wherever they appear.
func TestLoadRejectsUnknownFields(t *testing.T) {
	cases := []struct{ name, contents, wantErr string }{
		{
			"top-level section",
			"providers:\n  openrouter:\n    api_key: k\nbogus_section: true\n",
			"bogus_section",
		},
		{
			"nested provider field",
			"providers:\n  openrouter:\n    api_key: k\n    api_key_typo: x\n",
			"api_key_typo",
		},
		{
			"nested routing field",
			"providers:\n  openrouter:\n    api_key: k\nrouting:\n  on_classifier_eror: fail_open\n",
			"on_classifier_eror",
		},
		{
			// v2 removed the scorer; old configs fail the strict decoder.
			"removed scorer field",
			"providers:\n  openrouter:\n    api_key: k\nrouting:\n  confidence_threshold: 0.5\n",
			"confidence_threshold",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, "config.yaml", tc.contents)
			_, err := config.Load(path)
			if err == nil {
				t.Fatalf("Load() = nil error, want unknown field %q error", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Load() error = %q, want it to name %q", err, tc.wantErr)
			}
		})
	}
}

func TestLoadRejectsMalformedYAML(t *testing.T) {
	cases := []struct{ name, contents string }{
		{"unclosed flow sequence", "providers: [unclosed\n"},
		{"tab indentation", "providers:\n\topenrouter: {}\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, "config.yaml", tc.contents)
			if _, err := config.Load(path); err == nil {
				t.Fatal("Load() = nil error, want YAML syntax error")
			}
		})
	}
}

func TestDurationUnmarshal(t *testing.T) {
	cases := []struct {
		name     string
		yamlText string
		want     time.Duration
		wantErr  string
	}{
		{"duration string seconds", "timeout: 600s", 600 * time.Second, ""},
		{"duration string minutes", "timeout: 10m", 10 * time.Minute, ""},
		{"integer seconds", "timeout: 30", 30 * time.Second, ""},
		{"invalid string", "timeout: soon", 0, "invalid duration"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var doc struct {
				Timeout config.Duration `yaml:"timeout"`
			}
			err := yaml.Unmarshal([]byte(tc.yamlText), &doc)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("yaml.Unmarshal(%q) = nil error, want %q", tc.yamlText, tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("yaml.Unmarshal(%q) error = %q, want %q", tc.yamlText, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("yaml.Unmarshal(%q) = %v", tc.yamlText, err)
			}
			if got := doc.Timeout.Value(); got != tc.want {
				t.Errorf("Timeout.Value() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDurationMarshalEmitsString(t *testing.T) {
	out, err := yaml.Marshal(struct {
		Timeout config.Duration `yaml:"timeout"`
	}{Timeout: config.Duration(90 * time.Second)})
	if err != nil {
		t.Fatalf("yaml.Marshal = %v", err)
	}
	if got, want := strings.TrimSpace(string(out)), "timeout: 1m30s"; got != want {
		t.Errorf("yaml.Marshal = %q, want %q", got, want)
	}
}

// A partial file only overrides the fields it mentions.
func TestLoadPartialFileKeepsDefaults(t *testing.T) {
	unsetRoutingModelEnv(t)
	path := writeFile(t, "config.yaml", `providers:
  openrouter:
    api_key: partial-key
routing:
  models:
    code:
      - provider: openrouter
        model: partial/code-model
`)

	f, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load(%q) = %v", path, err)
	}
	if key, err := f.Providers.OpenRouter.Resolve(); err != nil || key != "partial-key" {
		t.Errorf("OpenRouter.Resolve() = %q, %v; want partial-key", key, err)
	}
	if chain := f.Routing.Models["code"]; len(chain) != 1 || chain[0].Model != "partial/code-model" {
		t.Errorf("Routing.Models[code] = %+v, want the file's chain", chain)
	}
	if f.Routing.DefaultModel != nil {
		t.Errorf("DefaultModel = %+v, want nil (the partial file does not set it)", f.Routing.DefaultModel)
	}
	if got, want := f.Routing.CatalogCacheTTL.Value(), 600*time.Second; got != want {
		t.Errorf("CatalogCacheTTL = %v, want default %v", got, want)
	}
	if got := f.Routing.OnClassifierError; got != "fail_open" {
		t.Errorf("OnClassifierError = %q, want default %q", got, "fail_open")
	}
	if got, want := f.Classifier.Timeout.Value(), 5*time.Second; got != want {
		t.Errorf("Classifier.Timeout = %v, want default %v", got, want)
	}
	if got := f.Classifier.Model; got != "~typesafe/jev-latest" {
		t.Errorf("Classifier.Model = %q, want default %q", got, "~typesafe/jev-latest")
	}
	if got := f.Server.Addr; got != ":8080" {
		t.Errorf("Server.Addr = %q, want default %q", got, ":8080")
	}
	if got := f.Telemetry.Enabled; got != "auto" {
		t.Errorf("Telemetry.Enabled = %q, want default %q", got, "auto")
	}
	if f.Providers.OpenAI != nil {
		t.Error("Providers.OpenAI != nil, want unconfigured")
	}
}

func TestValidateAcceptsValidFile(t *testing.T) {
	if err := validFile().Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}
}

// A literal key wins over api_key_env even when the variable is unset.
func TestValidateLiteralKeyWinsOverEnv(t *testing.T) {
	unsetenv(t, "JEV_TEST_ABSENT_KEY")
	f := config.Default()
	f.Providers.OpenRouter = &config.OpenRouter{KeyRef: config.KeyRef{
		APIKey:    "literal-key",
		APIKeyEnv: "JEV_TEST_ABSENT_KEY",
	}}
	f.Routing.DefaultModel = &config.ModelChoice{Provider: "openrouter", Model: "test/default-model"}
	if err := f.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil (literal key must not consult the missing env var)", err)
	}
}

func TestValidateTable(t *testing.T) {
	unsetenv(t, "JEV_TEST_ABSENT_KEY")

	// A catalog path that exists on disk: Validate Stats it, and only the
	// error under test should surface.
	catalog := writeFile(t, "catalog.yaml", minimalCatalog)

	validCompatible := func(name string) config.Compatible {
		return config.Compatible{
			Name:    name,
			BaseURL: "https://api.groq.com/openai/v1",
			KeyRef:  config.KeyRef{APIKey: "test-key"},
			Catalog: catalog,
		}
	}

	tests := []struct {
		name    string
		mutate  func(*config.File)
		wantErr string
	}{
		{
			"invalid on_classifier_error",
			func(f *config.File) { f.Routing.OnClassifierError = "fail_opne" },
			"on_classifier_error",
		},
		{
			"wrong-case on_classifier_error",
			func(f *config.File) { f.Routing.OnClassifierError = "Fail_Closed" },
			"on_classifier_error",
		},
		{
			"no model mapping",
			func(f *config.File) {
				f.Routing.Models = nil
				f.Routing.DefaultModel = nil
			},
			"routing.models or routing.default_model: at least one model mapping is required",
		},
		{
			"unknown domain key",
			func(f *config.File) {
				f.Routing.Models["bogus"] = []config.ModelChoice{{Provider: "openrouter", Model: "m"}}
			},
			"bogus",
		},
		{
			"empty chain",
			func(f *config.File) { f.Routing.Models["creative"] = []config.ModelChoice{} },
			"routing.models.creative",
		},
		{
			"model entry missing provider",
			func(f *config.File) {
				f.Routing.Models["code"] = append(f.Routing.Models["code"], config.ModelChoice{Model: "m"})
			},
			"routing.models.code[1].provider",
		},
		{
			"model entry missing model",
			func(f *config.File) {
				f.Routing.Models["code"] = append(f.Routing.Models["code"], config.ModelChoice{Provider: "openrouter"})
			},
			"routing.models.code[1].model",
		},
		{
			"default model missing provider",
			func(f *config.File) { f.Routing.DefaultModel = &config.ModelChoice{Model: "m"} },
			"routing.default_model.provider",
		},
		{
			"default model missing model",
			func(f *config.File) { f.Routing.DefaultModel = &config.ModelChoice{Provider: "openrouter"} },
			"routing.default_model.model",
		},
		{
			"models provider not configured",
			func(f *config.File) {
				f.Routing.Models["code"] = []config.ModelChoice{{Provider: "gemini", Model: "gemini-model"}}
			},
			`routing.models.code[0].provider "gemini"`,
		},
		{
			"default model provider not configured",
			func(f *config.File) {
				f.Routing.DefaultModel = &config.ModelChoice{Provider: "gemini", Model: "gemini-model"}
			},
			`routing.default_model.provider "gemini"`,
		},
		{
			"negative catalog cache ttl",
			func(f *config.File) { f.Routing.CatalogCacheTTL = config.Duration(-time.Second) },
			"catalog_cache_ttl",
		},
		{
			"no providers configured",
			func(f *config.File) { f.Providers = config.Providers{} },
			"no providers configured",
		},
		{
			"openai without catalog",
			func(f *config.File) {
				f.Providers.OpenAI = &config.Keyed{KeyRef: config.KeyRef{APIKey: "k"}}
			},
			"catalog is required",
		},
		{
			"openai catalog file missing",
			func(f *config.File) {
				f.Providers.OpenAI = &config.Keyed{
					KeyRef:  config.KeyRef{APIKey: "k"},
					Catalog: "no-such-catalog.yaml",
				}
			},
			"no-such-catalog.yaml",
		},
		{
			"gemini catalog file missing",
			func(f *config.File) {
				f.Providers.Gemini = &config.Keyed{
					KeyRef:  config.KeyRef{APIKey: "k"},
					Catalog: "no-such-gemini-catalog.yaml",
				}
			},
			"no-such-gemini-catalog.yaml",
		},
		{
			"openai key env missing",
			func(f *config.File) {
				f.Providers.OpenAI = &config.Keyed{
					KeyRef:  config.KeyRef{APIKeyEnv: "JEV_TEST_ABSENT_KEY"},
					Catalog: catalog,
				}
			},
			"JEV_TEST_ABSENT_KEY",
		},
		{
			"bedrock missing region",
			func(f *config.File) {
				f.Providers.Bedrock = &config.BedrockProvider{
					Catalog: catalog,
				}
			},
			"providers.bedrock.region",
		},
		{
			"bedrock missing catalog",
			func(f *config.File) {
				f.Providers.Bedrock = &config.BedrockProvider{Region: "us-east-1"}
			},
			"catalog is required",
		},
		{
			"compatible empty name",
			func(f *config.File) {
				f.Providers.Compatible = []config.Compatible{validCompatible("")}
			},
			"name is required",
		},
		{
			"compatible duplicate names",
			func(f *config.File) {
				f.Providers.Compatible = []config.Compatible{validCompatible("groq"), validCompatible("groq")}
			},
			"duplicate",
		},
		{
			"compatible key env missing",
			func(f *config.File) {
				c := validCompatible("groq")
				c.KeyRef = config.KeyRef{APIKeyEnv: "JEV_TEST_ABSENT_KEY"}
				f.Providers.Compatible = []config.Compatible{c}
			},
			"JEV_TEST_ABSENT_KEY",
		},
		{
			"compatible ftp base url",
			func(f *config.File) {
				c := validCompatible("groq")
				c.BaseURL = "ftp://api.groq.com/openai/v1"
				f.Providers.Compatible = []config.Compatible{c}
			},
			"http:// or https://",
		},
		{
			"compatible base url without scheme",
			func(f *config.File) {
				c := validCompatible("groq")
				c.BaseURL = "api.groq.com/openai/v1"
				f.Providers.Compatible = []config.Compatible{c}
			},
			"http:// or https://",
		},
		{
			"classifier path without leading slash",
			func(f *config.File) { f.Classifier.Path = "api/v1/systemone" },
			"classifier.path",
		},
		{
			"classifier base url not http",
			func(f *config.File) { f.Classifier.BaseURL = "ftp://openrouter.ai" },
			"classifier.base_url",
		},
		{
			"classifier timeout zero",
			func(f *config.File) { f.Classifier.Timeout = 0 },
			"classifier.timeout",
		},
		{
			"server addr empty",
			func(f *config.File) { f.Server.Addr = "" },
			"server.addr",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := validFile()
			tc.mutate(f)
			err := f.Validate()
			if err == nil {
				t.Fatalf("Validate() = nil, want error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate() error = %q, want substring %q", err, tc.wantErr)
			}
		})
	}
}

// The missing-mapping error is exact; on a File that was not loaded from disk
// (environment-only mode) it names the variable that supplies one.
func TestValidateMissingModelMappingMessage(t *testing.T) {
	f := validFile()
	f.Routing.Models = nil
	f.Routing.DefaultModel = nil

	err := f.Validate()
	if err == nil {
		t.Fatal("Validate() = nil, want a missing model mapping error")
	}
	want := "routing.models or routing.default_model: at least one model mapping is required (set JEV_ROUTER_DEFAULT_MODEL in environment-only mode)"
	if err.Error() != want {
		t.Errorf("Validate() error = %q, want exactly %q", err, want)
	}
}

// A file-loaded configuration gets the same error without the
// environment-only hint (the file could have supplied a mapping).
func TestLoadMissingModelMappingHasNoEnvHint(t *testing.T) {
	unsetRoutingModelEnv(t)
	path := writeFile(t, "config.yaml", `providers:
  openrouter:
    api_key: k
`)

	_, err := config.Load(path)
	if err == nil {
		t.Fatal("Load() = nil error, want a missing model mapping error")
	}
	if want := "routing.models or routing.default_model: at least one model mapping is required"; !strings.Contains(err.Error(), want) {
		t.Fatalf("Load() error = %q, want it to contain %q", err, want)
	}
	if strings.Contains(err.Error(), "environment-only") {
		t.Errorf("Load() error = %q, want no environment-only hint for a file-loaded config", err)
	}
}

// The unknown-domain error names the offending key and all five allowed
// domains.
func TestValidateUnknownDomainNamesAllowedValues(t *testing.T) {
	f := validFile()
	f.Routing.Models["bogus_domain"] = []config.ModelChoice{{Provider: "openrouter", Model: "m"}}

	err := f.Validate()
	if err == nil {
		t.Fatal("Validate() = nil, want an unknown domain error")
	}
	for _, want := range []string{"bogus_domain", "code", "math_reasoning", "creative", "factual_lookup", "other"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Validate() error = %q, want it to name %q", err, want)
		}
	}
}

func TestLoadServerAddrEnvOverride(t *testing.T) {
	unsetRoutingModelEnv(t)
	path := writeFile(t, "config.yaml", minimalConfig)

	t.Run("env overrides default", func(t *testing.T) {
		t.Setenv(config.ServerAddrEnv, ":9999")
		f, err := config.Load(path)
		if err != nil {
			t.Fatalf("Load(%q) = %v", path, err)
		}
		if got := f.Server.Addr; got != ":9999" {
			t.Errorf("Server.Addr = %q, want %q", got, ":9999")
		}
	})
	t.Run("empty env keeps file value", func(t *testing.T) {
		t.Setenv(config.ServerAddrEnv, "")
		f, err := config.Load(path)
		if err != nil {
			t.Fatalf("Load(%q) = %v", path, err)
		}
		if got := f.Server.Addr; got != ":8080" {
			t.Errorf("Server.Addr = %q, want default %q", got, ":8080")
		}
	})
}

func TestLoadUsesConfigPathEnv(t *testing.T) {
	unsetRoutingModelEnv(t)
	unsetenv(t, config.ServerAddrEnv)
	path := writeFile(t, "env-config.yaml", `providers:
  openrouter:
    api_key: env-file-key
routing:
  default_model:
    provider: openrouter
    model: test/default-model
server:
  addr: ":7777"
`)
	t.Setenv(config.ConfigPathEnv, path)

	f, err := config.Load("")
	if err != nil {
		t.Fatalf("Load(\"\") with %s = %v", config.ConfigPathEnv, err)
	}
	if got := f.Server.Addr; got != ":7777" {
		t.Errorf("Server.Addr = %q, want %q", got, ":7777")
	}
	if key, err := f.Providers.OpenRouter.Resolve(); err != nil || key != "env-file-key" {
		t.Errorf("OpenRouter.Resolve() = %q, %v; want env-file-key", key, err)
	}
}

func TestLoadMissingFileFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nope.yaml")
	_, err := config.Load(path)
	if err == nil {
		t.Fatal("Load() = nil error, want read error")
	}
	if !strings.Contains(err.Error(), "nope.yaml") {
		t.Fatalf("Load() error = %q, want it to name nope.yaml", err)
	}
}

func TestDirAndResolvePath(t *testing.T) {
	unsetRoutingModelEnv(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "router.yaml")
	if err := os.WriteFile(path, []byte(minimalConfig), 0o600); err != nil {
		t.Fatalf("os.WriteFile(%q) = %v", path, err)
	}
	f, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load(%q) = %v", path, err)
	}

	wantDir, err := filepath.Abs(dir)
	if err != nil {
		t.Fatalf("filepath.Abs = %v", err)
	}
	if got := f.Dir(); got != wantDir {
		t.Errorf("Dir() = %q, want %q", got, wantDir)
	}
	if got, want := f.ResolvePath("catalogs/x.yaml"), filepath.Join(wantDir, "catalogs", "x.yaml"); got != want {
		t.Errorf("ResolvePath(relative) = %q, want %q", got, want)
	}
	if got := f.ResolvePath("/abs/catalog.yaml"); got != "/abs/catalog.yaml" {
		t.Errorf("ResolvePath(absolute) = %q, want unchanged", got)
	}

	// A File that was not loaded from disk resolves like the working directory.
	var zero config.File
	if got := zero.Dir(); got != "" {
		t.Errorf("zero Dir() = %q, want empty", got)
	}
	if got, want := zero.ResolvePath("rel.yaml"), "rel.yaml"; got != want {
		t.Errorf("zero ResolvePath(relative) = %q, want %q", got, want)
	}
}
