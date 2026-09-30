// Package config loads and validates jev-router YAML configuration files.
//
// The schema is documented by the README's Configuration section. Loading
// always starts from Default, so a partial file only overrides the fields it
// mentions; environment variables (JEV_ROUTER_SERVER_ADDR,
// JEV_ROUTER_DEFAULT_MODEL/JEV_ROUTER_DEFAULT_PROVIDER) then apply on top.
// Catalog paths written in a file resolve relative to that file's directory,
// so a configuration file and its catalogs can be moved together.
//
// Precedence, highest first: command-line flags > environment > YAML >
// defaults. This package implements environment over YAML over defaults; the
// CLI layer applies flags before calling Load.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// Environment variables read by Load.
const (
	// ConfigPathEnv names the environment variable consulted by Load when
	// its path argument is empty.
	ConfigPathEnv = "JEV_ROUTER_CONFIG"
	// ServerAddrEnv overrides server.addr when set to a non-empty value.
	ServerAddrEnv = "JEV_ROUTER_SERVER_ADDR"
	// DefaultModelEnv supplies or overrides routing.default_model.model (and,
	// with DefaultProviderEnv, its provider) in both Load modes.
	DefaultModelEnv = "JEV_ROUTER_DEFAULT_MODEL"
	// DefaultProviderEnv names the provider of the model selected by
	// DefaultModelEnv; an empty value means "openrouter".
	DefaultProviderEnv = "JEV_ROUTER_DEFAULT_PROVIDER"
)

// Duration is a time.Duration that YAML-decodes from either a duration string
// ("600s", "10m", "5s") or an integer number of seconds, and YAML-encodes as
// a duration string.
type Duration time.Duration

// UnmarshalYAML implements yaml.Unmarshaler.
func (d *Duration) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.ScalarNode {
		return errors.New("duration: expected a scalar value (a duration string or integer seconds)")
	}
	if parsed, err := time.ParseDuration(value.Value); err == nil {
		*d = Duration(parsed)
		return nil
	}
	if value.Tag == "!!int" {
		var seconds int64
		if err := value.Decode(&seconds); err == nil {
			*d = Duration(time.Duration(seconds) * time.Second)
			return nil
		}
	}
	return fmt.Errorf("invalid duration %q: use a duration string such as \"600s\" or integer seconds", value.Value)
}

// MarshalYAML implements yaml.Marshaler, emitting the duration as a string
// such as "10m0s".
func (d Duration) MarshalYAML() (any, error) {
	return d.Value().String(), nil
}

// Value returns the duration as a time.Duration.
func (d Duration) Value() time.Duration { return time.Duration(d) }

// KeyRef names a provider credential either literally (APIKey) or through an
// environment variable (APIKeyEnv). Literal keys win; storing literal keys in
// a shared file is discouraged in favor of api_key_env.
type KeyRef struct {
	APIKey    string `yaml:"api_key"`
	APIKeyEnv string `yaml:"api_key_env"`
}

// Resolve returns the credential to use. A non-empty literal APIKey wins;
// otherwise the environment variable named by APIKeyEnv is read — an unset or
// empty variable, and a KeyRef with neither field set, are errors.
func (k KeyRef) Resolve() (string, error) {
	if k.APIKey != "" {
		return k.APIKey, nil
	}
	if k.APIKeyEnv == "" {
		return "", errors.New("api_key or api_key_env required")
	}
	value := os.Getenv(k.APIKeyEnv)
	if value == "" {
		return "", fmt.Errorf("environment variable %q is not set (or is empty)", k.APIKeyEnv)
	}
	return value, nil
}

// File is the root of a configuration file.
type File struct {
	Providers  Providers  `yaml:"providers"`
	Classifier Classifier `yaml:"classifier"`
	Routing    Routing    `yaml:"routing"`
	Server     Server     `yaml:"server"`
	Telemetry  Telemetry  `yaml:"telemetry"`

	// dir is the absolute directory of the file this configuration was
	// loaded from; see Dir and ResolvePath. Empty means "not loaded from a
	// file", in which case relative paths behave as cwd-relative.
	dir string
}

// Providers lists the providers to assemble. A nil pointer (and an empty
// Compatible slice) means the provider is not configured.
type Providers struct {
	OpenRouter *OpenRouter      `yaml:"openrouter"`
	OpenAI     *Keyed           `yaml:"openai"`
	Anthropic  *Keyed           `yaml:"anthropic"`
	Gemini     *Keyed           `yaml:"gemini"`
	Bedrock    *BedrockProvider `yaml:"bedrock"`
	Compatible []Compatible     `yaml:"compatible"`
}

// OpenRouter is the live-catalog provider: it has no catalog file, and its
// key is shared by Jev classification, the model catalog and completions.
type OpenRouter struct {
	KeyRef `yaml:",inline"`
}

// Keyed is a provider configured by a credential plus a catalog file
// (openai, anthropic, gemini).
type Keyed struct {
	KeyRef  `yaml:",inline"`
	Catalog string `yaml:"catalog"`
}

// BedrockProvider uses the standard AWS credential chain, so it has no
// credential field — only a region and a catalog file.
type BedrockProvider struct {
	Region  string `yaml:"region"`
	Catalog string `yaml:"catalog"`
}

// Compatible is an OpenAI-wire endpoint under a user-chosen provider name.
type Compatible struct {
	Name    string `yaml:"name"`
	BaseURL string `yaml:"base_url"`
	KeyRef  `yaml:",inline"`
	Catalog string `yaml:"catalog"`
}

// Classifier points at the Jev classification endpoint. One classification
// call guards each routing decision.
type Classifier struct {
	BaseURL string   `yaml:"base_url"`
	Path    string   `yaml:"path"`
	Model   string   `yaml:"model"`
	Timeout Duration `yaml:"timeout"`
}

// Routing is the routing policy applied to every request.
type Routing struct {
	// CatalogCacheTTL is how long a fetched OpenRouter catalog is served
	// from memory before it is refetched (default 600s).
	CatalogCacheTTL Duration `yaml:"catalog_cache_ttl"`
	// OnClassifierError is "fail_open" or "fail_closed" (default
	// "fail_open").
	OnClassifierError string `yaml:"on_classifier_error"`
	// Models maps each classified domain to an ordered chain of provider and
	// model candidates. Complete tries the entries in order, advancing to the
	// next one on any provider failure. Keys must be valid domains (code,
	// math_reasoning, creative, factual_lookup, other).
	Models map[string][]ModelChoice `yaml:"models"`
	// DefaultModel is used when the request cannot be classified, its domain
	// has no chain, or every chain candidate failed. At least one of Models
	// and DefaultModel must be present.
	DefaultModel *ModelChoice `yaml:"default_model"`
	// AllowedProviders and ExcludedProviders restrict the routing pool;
	// empty means no restriction.
	AllowedProviders  []string `yaml:"allowed_providers"`
	ExcludedProviders []string `yaml:"excluded_providers"`
}

// ModelChoice names one provider-native model in a routing chain entry.
type ModelChoice struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

// validDomains lists the five classified domains accepted as routing.models
// keys, for error messages; the authoritative check is
// modelrouter.IsValidDomain.
var validDomains = []string{
	string(modelrouter.DomainCode),
	string(modelrouter.DomainMathReasoning),
	string(modelrouter.DomainCreative),
	string(modelrouter.DomainFactualLookup),
	string(modelrouter.DomainOther),
}

// Server configures the HTTP server.
type Server struct {
	Addr        string   `yaml:"addr"`
	ReadTimeout Duration `yaml:"read_timeout"`
}

// Telemetry configures observability. Enabled accepts "auto" (default: on
// when OTEL_EXPORTER_OTLP_ENDPOINT is set), "on" or "off".
type Telemetry struct {
	Enabled string `yaml:"enabled"`
}

// Default returns the documented defaults: routing 600s / fail_open and no
// model mappings, the standard Jev endpoint at a 5s timeout, server :8080 +
// 30s read timeout, and telemetry "auto". No providers are configured; Load's
// environment-only mode adds a default OpenRouter provider.
func Default() *File {
	return &File{
		Classifier: Classifier{
			BaseURL: "https://openrouter.ai",
			Path:    "/api/v1/systemone",
			Model:   "~typesafe/jev-latest",
			Timeout: Duration(5 * time.Second),
		},
		Routing: Routing{
			CatalogCacheTTL:   Duration(600 * time.Second),
			OnClassifierError: string(modelrouter.OnClassifierErrorFailOpen),
		},
		Server: Server{
			Addr:        ":8080",
			ReadTimeout: Duration(30 * time.Second),
		},
		Telemetry: Telemetry{Enabled: "auto"},
	}
}

// Load reads, decodes and validates a configuration file.
//
// An empty path falls back to the JEV_ROUTER_CONFIG environment variable;
// when that is empty too, Load runs in environment-only mode: the documented
// defaults plus an OpenRouter provider keyed from OPENROUTER_API_KEY and a
// default model read from JEV_ROUTER_DEFAULT_MODEL. A missing or empty API
// key variable is a validation error naming it; without a default model the
// routing validation error names JEV_ROUTER_DEFAULT_MODEL.
//
// Files are decoded strictly — unknown fields and malformed YAML are errors —
// on top of Default, so absent fields keep their documented defaults. After
// decoding, JEV_ROUTER_SERVER_ADDR and JEV_ROUTER_DEFAULT_MODEL/
// JEV_ROUTER_DEFAULT_PROVIDER apply (in both modes), then Validate runs
// before Load returns; validation errors are wrapped with the file path.
func Load(path string) (*File, error) {
	if path == "" {
		path = os.Getenv(ConfigPathEnv)
	}
	if path == "" {
		f := Default()
		f.Providers.OpenRouter = &OpenRouter{KeyRef: KeyRef{APIKeyEnv: "OPENROUTER_API_KEY"}}
		applyEnvOverrides(f)
		if err := f.Validate(); err != nil {
			return nil, err
		}
		return f, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("config file %s: %w", path, err)
	}
	dir, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("config file %s: resolving directory: %w", path, err)
	}

	f := Default()
	f.dir = dir
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(f); err != nil {
		return nil, fmt.Errorf("config file %s: %w", path, err)
	}
	applyEnvOverrides(f)
	if err := f.Validate(); err != nil {
		return nil, fmt.Errorf("config file %s: %w", path, err)
	}
	return f, nil
}

// applyEnvOverrides applies JEV_ROUTER_* environment overrides on top of f.
// JEV_ROUTER_DEFAULT_MODEL supplies or replaces routing.default_model; the
// provider comes from JEV_ROUTER_DEFAULT_PROVIDER and defaults to
// "openrouter". Env wins over the file (flags > env > YAML > defaults).
func applyEnvOverrides(f *File) {
	if addr := os.Getenv(ServerAddrEnv); addr != "" {
		f.Server.Addr = addr
	}
	if model := os.Getenv(DefaultModelEnv); model != "" {
		provider := os.Getenv(DefaultProviderEnv)
		if provider == "" {
			provider = "openrouter"
		}
		f.Routing.DefaultModel = &ModelChoice{Provider: provider, Model: model}
	}
}

// Dir returns the directory containing the loaded config file, as an absolute
// path. It is empty for a File that was not loaded from a file (Default or an
// environment-only Load), which ResolvePath treats as the current working
// directory.
func (f *File) Dir() string { return f.dir }

// ResolvePath resolves a path written in the config file: absolute paths are
// returned unchanged, relative paths are resolved against Dir (or the current
// working directory when the file was not loaded from disk), and an empty
// path stays empty.
func (f *File) ResolvePath(rel string) string {
	if rel == "" || filepath.IsAbs(rel) {
		return rel
	}
	return filepath.Join(f.dir, rel)
}

// Validate checks the configuration for consistency and confirms every
// configured provider is usable: credentials resolve, catalogs exist (paths
// resolved against the config file's directory), URLs are absolute http/https
// and timeouts are positive. It also checks the routing model mappings:
// domains are known, chains and their entries are complete, at least one
// mapping exists, and every referenced provider is configured. It performs no
// I/O beyond reading the environment and os.Stat on catalog files.
func (f *File) Validate() error {
	r := &f.Routing
	policy := modelrouter.OnClassifierError(r.OnClassifierError)
	switch policy {
	case "", modelrouter.OnClassifierErrorFailOpen, modelrouter.OnClassifierErrorFailClosed:
	default:
		return fmt.Errorf("routing.on_classifier_error %q: must be %q, %q or empty",
			r.OnClassifierError, modelrouter.OnClassifierErrorFailOpen, modelrouter.OnClassifierErrorFailClosed)
	}
	if ttl := r.CatalogCacheTTL.Value(); ttl < 0 {
		return fmt.Errorf("routing.catalog_cache_ttl %v: must not be negative", ttl)
	}
	configured, err := f.validateProviders()
	if err != nil {
		return err
	}
	if err := f.validateRouting(r, configured); err != nil {
		return err
	}
	if err := validateHTTPURL("classifier.base_url", f.Classifier.BaseURL); err != nil {
		return err
	}
	if !strings.HasPrefix(f.Classifier.Path, "/") {
		return fmt.Errorf("classifier.path %q: must begin with %q", f.Classifier.Path, "/")
	}
	if timeout := f.Classifier.Timeout.Value(); timeout <= 0 {
		return fmt.Errorf("classifier.timeout %v: must be positive", timeout)
	}
	if f.Server.Addr == "" {
		return errors.New("server.addr is required and must be non-empty")
	}
	return nil
}

// validateProviders checks every configured provider and rejects an entirely
// empty provider section. It returns the configured provider names — the
// built-ins that were configured plus every compatible entry's name — so the
// routing checks can verify model references without duplicating this logic.
func (f *File) validateProviders() (map[string]bool, error) {
	configured := make(map[string]bool)
	if p := f.Providers.OpenRouter; p != nil {
		configured["openrouter"] = true
		if _, err := p.Resolve(); err != nil {
			return nil, fmt.Errorf("providers.openrouter: %w", err)
		}
	}
	if p := f.Providers.OpenAI; p != nil {
		configured["openai"] = true
		if err := f.validateKeyed("providers.openai", p); err != nil {
			return nil, err
		}
	}
	if p := f.Providers.Anthropic; p != nil {
		configured["anthropic"] = true
		if err := f.validateKeyed("providers.anthropic", p); err != nil {
			return nil, err
		}
	}
	if p := f.Providers.Gemini; p != nil {
		configured["gemini"] = true
		if err := f.validateKeyed("providers.gemini", p); err != nil {
			return nil, err
		}
	}
	if p := f.Providers.Bedrock; p != nil {
		configured["bedrock"] = true
		if p.Region == "" {
			return nil, errors.New("providers.bedrock.region: region is required and must be non-empty")
		}
		if err := f.checkCatalog("providers.bedrock.catalog", p.Catalog); err != nil {
			return nil, err
		}
	}
	seen := make(map[string]int, len(f.Providers.Compatible))
	for i, c := range f.Providers.Compatible {
		field := fmt.Sprintf("providers.compatible[%d]", i)
		if c.Name == "" {
			return nil, fmt.Errorf("%s.name: name is required and must be non-empty", field)
		}
		if first, dup := seen[c.Name]; dup {
			return nil, fmt.Errorf("%s.name: duplicate provider name %q (first defined at providers.compatible[%d])",
				field, c.Name, first)
		}
		seen[c.Name] = i
		if _, err := c.Resolve(); err != nil {
			return nil, fmt.Errorf("%s: %w", field, err)
		}
		if err := validateHTTPURL(field+".base_url", c.BaseURL); err != nil {
			return nil, err
		}
		if err := f.checkCatalog(field+".catalog", c.Catalog); err != nil {
			return nil, err
		}
		configured[c.Name] = true
	}
	if len(configured) == 0 {
		return nil, errors.New("no providers configured")
	}
	return configured, nil
}

// validateRouting checks the routing model mappings against the configured
// provider names: every domain is known, every chain is non-empty and every
// entry complete, the default model is complete when present, at least one
// mapping exists, and every referenced provider is configured.
func (f *File) validateRouting(r *Routing, configured map[string]bool) error {
	for domain, chain := range r.Models {
		if !modelrouter.IsValidDomain(modelrouter.Domain(domain)) {
			return fmt.Errorf("routing.models: invalid domain %q (want one of %s)",
				domain, strings.Join(validDomains, ", "))
		}
		if len(chain) == 0 {
			return fmt.Errorf("routing.models.%s: chain must have at least one entry", domain)
		}
		for i, choice := range chain {
			if choice.Provider == "" {
				return fmt.Errorf("routing.models.%s[%d].provider is required and must be non-empty", domain, i)
			}
			if choice.Model == "" {
				return fmt.Errorf("routing.models.%s[%d].model is required and must be non-empty", domain, i)
			}
			if !configured[choice.Provider] {
				return fmt.Errorf("routing.models.%s[%d].provider %q: provider is not configured", domain, i, choice.Provider)
			}
		}
	}
	if d := r.DefaultModel; d != nil {
		if d.Provider == "" {
			return errors.New("routing.default_model.provider is required and must be non-empty")
		}
		if d.Model == "" {
			return errors.New("routing.default_model.model is required and must be non-empty")
		}
		if !configured[d.Provider] {
			return fmt.Errorf("routing.default_model.provider %q: provider is not configured", d.Provider)
		}
	}
	if len(r.Models) == 0 && r.DefaultModel == nil {
		msg := "routing.models or routing.default_model: at least one model mapping is required"
		if f.dir == "" {
			msg += " (set JEV_ROUTER_DEFAULT_MODEL in environment-only mode)"
		}
		return errors.New(msg)
	}
	return nil
}

// validateKeyed checks a key-plus-catalog provider.
func (f *File) validateKeyed(field string, p *Keyed) error {
	if _, err := p.Resolve(); err != nil {
		return fmt.Errorf("%s: %w", field, err)
	}
	return f.checkCatalog(field+".catalog", p.Catalog)
}

// checkCatalog verifies that a provider's catalog path is set and, resolved
// against the config file's directory, names an existing file.
func (f *File) checkCatalog(field, catalog string) error {
	if catalog == "" {
		return fmt.Errorf("%s: catalog is required", field)
	}
	path := f.ResolvePath(catalog)
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s: catalog file %s: %w", field, path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("%s: catalog path %s is a directory, want a file", field, path)
	}
	return nil
}

// validateHTTPURL requires an absolute http:// or https:// URL.
func validateHTTPURL(field, raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%s %q: invalid URL: %w", field, raw, err)
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("%s %q: must be an absolute http:// or https:// URL", field, raw)
	}
	return nil
}
