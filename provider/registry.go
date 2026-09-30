// Package provider is a database/sql-style registry of completion factories.
//
//	provider/builtin.Register()        // openrouter, openai, ...
//	completion, err := provider.New("openrouter", provider.Config{
//		APIKey: os.Getenv("OPENROUTER_API_KEY"),
//	})
//
// Provider implementations live in provider/<name> packages and never get
// imported by the root package directly; provider/builtin wires the built-ins
// in. OpenAI-wire providers with a custom base URL (groq, ollama, vllm, ...)
// are constructed by the config layer through NewCompatible instead of
// Register, since their name is only known at configuration time.
package provider

import (
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"sync"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// Config carries everything a factory needs to construct a Completion.
//
// HTTPClient and Logger are optional and caller-owned: neither the registry
// nor the providers mutate, wrap or close them. A provider that constructs
// its own http.Client when HTTPClient is nil does not close an injected one
// instead.
type Config struct {
	// Name is the registry name of the provider ("openrouter", "groq", ...).
	// New fills it in from its name argument. Direct constructor calls may
	// leave it empty; built-in providers then fall back to their own name.
	Name string
	// APIKey is the provider credential. Every built-in provider requires a
	// non-empty key and reports a construction error otherwise.
	APIKey string
	// BaseURL overrides the provider endpoint. Unit tests point it at an
	// httptest.Server; NewCompatible functions require it.
	BaseURL string
	// Region is reserved for region-scoped providers (AWS Bedrock, P6b).
	// The completion providers implemented so far ignore it.
	Region string
	// HTTPClient, when non-nil, is used for every request and stays
	// caller-owned: it is never mutated or closed, and the provider does not
	// set a timeout on it. When nil the provider constructs its own
	// http.Client with NO timeout — completions are context-first, so pass a
	// context with a deadline to bound a call.
	HTTPClient *http.Client
	// Logger receives operational warnings. Nil means slog.Default().
	// Caller-owned: the provider never closes it.
	Logger *slog.Logger
}

// Factory constructs a Completion from cfg. The registry sets cfg.Name to the
// registered name before calling the factory, so factories can rely on it.
// Factories must be safe to call concurrently.
type Factory func(Config) (modelrouter.Completion, error)

var (
	registryMu sync.RWMutex
	registry   = map[string]Factory{}
)

// Register installs a provider factory under name. It mirrors database/sql's
// Register: call it from init functions or explicit assembly code
// (provider/builtin.Register) before any New call. Register panics on an
// empty name, a nil factory, or a name that is already registered — a
// duplicate registration is a programming error, not a runtime condition.
// Register is safe for concurrent use.
func Register(name string, factory Factory) {
	if name == "" {
		panic("provider: Register called with empty name")
	}
	if factory == nil {
		panic("provider: Register called with nil factory for provider " + name)
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, dup := registry[name]; dup {
		panic("provider: Register called twice for provider " + name)
	}
	registry[name] = factory
}

// Registered reports whether a factory is installed under name. It takes the
// registry lock for reading only and is safe for concurrent use — assembly
// code uses it to decide whether a factory still needs to be registered.
func Registered(name string) bool {
	registryMu.RLock()
	defer registryMu.RUnlock()
	_, ok := registry[name]
	return ok
}

// New constructs the Completion registered under name with cfg. cfg.Name is
// set from name before the factory runs.
//
// An unregistered name returns an error wrapping
// modelrouter.ErrProviderNotConfigured, so errors.Is(err,
// modelrouter.ErrProviderNotConfigured) identifies a configuration gap. The
// registry lock is not held while the factory runs, so a factory may itself
// call Register (or Names) without deadlocking.
func New(name string, cfg Config) (modelrouter.Completion, error) {
	registryMu.RLock()
	factory, ok := registry[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("provider: unknown provider %q: %w", name, modelrouter.ErrProviderNotConfigured)
	}
	cfg.Name = name
	completion, err := factory(cfg)
	if err != nil {
		return nil, err
	}
	if completion == nil {
		return nil, fmt.Errorf("provider: factory for %q returned a nil completion", name)
	}
	return completion, nil
}

// Names returns the registered provider names in sorted order. It returns an
// empty (non-nil) slice when nothing is registered.
func Names() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// WrapError builds the *modelrouter.ProviderError a Completion returns for a
// failed operation, so every provider wraps failures the same way.
func WrapError(providerName, op string, statusCode int, err error) error {
	return &modelrouter.ProviderError{
		Provider:   providerName,
		Op:         op,
		StatusCode: statusCode,
		Err:        err,
	}
}
