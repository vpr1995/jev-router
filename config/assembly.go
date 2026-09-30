package config

import (
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"

	modelrouter "github.com/vprprudhvi/jev-router"
	catalogopenrouter "github.com/vprprudhvi/jev-router/catalog/openrouter"
	"github.com/vprprudhvi/jev-router/catalog/static"
	"github.com/vprprudhvi/jev-router/classify"
	"github.com/vprprudhvi/jev-router/provider"
	"github.com/vprprudhvi/jev-router/provider/builtin"
	"github.com/vprprudhvi/jev-router/provider/openai"
)

// App is a fully assembled router: the configuration it was built from, the
// ready-to-serve Router and the components whose lifetime it owns.
//
// The Router itself closes nothing, so App does: Close releases the HTTP
// clients of the components Build constructed itself (the OpenRouter catalog
// and the Jev classifier). Static catalogs and completion providers hold no
// closable resources and are not closed.
type App struct {
	// File is the configuration this App was assembled from.
	File *File
	// Router is the assembled routing pipeline; never nil on success. It
	// keeps every catalog, classifier and completion alive, so Close the
	// App only when the Router is no longer in use.
	Router *modelrouter.Router
	// Providers are the configured provider names in sorted order: the
	// built-ins (openrouter, openai, anthropic, gemini, bedrock) that were
	// configured plus every compatible entry's name.
	Providers []string

	// closers are the Close methods to call, in construction order; Close
	// runs them in reverse.
	closers []func() error

	closeOnce sync.Once
	closeErr  error
}

// buildOptions collects the values set by BuildOption functions.
type buildOptions struct {
	logger *slog.Logger
}

// BuildOption configures Build. Options are applied in the order given,
// after the defaults.
type BuildOption func(*buildOptions)

// WithLogger sets the slog logger threaded through every component Build
// constructs: the Router, the OpenRouter catalog, the Jev classifier and the
// completion providers. Passing nil is ignored, keeping slog.Default().
func WithLogger(logger *slog.Logger) BuildOption {
	return func(o *buildOptions) {
		if logger == nil {
			return
		}
		o.logger = logger
	}
}

// Build assembles an App from f.
//
// Build registers the built-in provider factories first (idempotently) and
// then re-runs f.Validate, so a hand-built File that never passed through
// Load is still checked. Construction then follows the configuration:
//
//   - openrouter gets the live OpenRouter catalog, plus the Jev classifier
//     (same key, since it speaks OpenRouter's systemone route). Without it,
//     the Router receives a nil Classifier, treated as unavailable and
//     handled through routing.on_classifier_error.
//   - openai, anthropic, gemini and bedrock each get a provider.New
//     completion and a static catalog loaded from their configured file.
//   - every compatible entry gets a static catalog and an OpenAI-wire
//     completion; its factory is registered under the entry's name the first
//     time it is seen, and reused as-is if already registered.
//
// The Router receives the catalogs in sorted provider-name order together
// with the routing policy converted from the config schema. On any
// construction error, Build closes whatever it had already built and joins
// those cleanup errors into the returned error.
//
// The caller owns the returned App and must Close it when done.
func Build(f *File, opts ...BuildOption) (*App, error) {
	builtin.Register()

	if f == nil {
		return nil, errors.New("config: Build requires a non-nil File")
	}
	if err := f.Validate(); err != nil {
		return nil, err
	}

	options := buildOptions{logger: slog.Default()}
	for _, opt := range opts {
		if opt != nil {
			opt(&options)
		}
	}
	logger := options.logger
	if logger == nil {
		logger = slog.Default()
	}

	catalogs := make(map[string]modelrouter.Catalog)
	completions := make(map[string]modelrouter.Completion)
	names := make([]string, 0)
	var closers []func() error
	var classifier modelrouter.Classifier

	// add records one assembled provider; a compatible entry that reuses a
	// built-in name replaces that provider's catalog and completion instead
	// of appearing twice.
	add := func(name string, catalog modelrouter.Catalog, completion modelrouter.Completion) {
		if _, exists := catalogs[name]; !exists {
			names = append(names, name)
		}
		catalogs[name] = catalog
		completions[name] = completion
	}
	// fail aborts Build, releasing whatever was constructed so far.
	fail := func(err error) (*App, error) {
		return nil, errors.Join(err, closeAll(closers))
	}

	if p := f.Providers.OpenRouter; p != nil {
		key, err := p.Resolve()
		if err != nil {
			return fail(fmt.Errorf("providers.openrouter: %w", err))
		}
		catalog := catalogopenrouter.New(key,
			catalogopenrouter.WithCacheTTL(f.Routing.CatalogCacheTTL.Value()),
			catalogopenrouter.WithLogger(logger))
		completion, err := provider.New("openrouter", provider.Config{APIKey: key, Logger: logger})
		if err != nil {
			return fail(fmt.Errorf("providers.openrouter: %w", err))
		}
		jev := classify.New(key,
			classify.WithBaseURL(f.Classifier.BaseURL),
			classify.WithPath(f.Classifier.Path),
			classify.WithModel(f.Classifier.Model),
			classify.WithTimeout(f.Classifier.Timeout.Value()),
			classify.WithLogger(logger))
		classifier = jev
		add("openrouter", catalog, completion)
		closers = append(closers, catalog.Close, jev.Close)
	}

	for _, entry := range []struct {
		name string
		p    *Keyed
	}{
		{"openai", f.Providers.OpenAI},
		{"anthropic", f.Providers.Anthropic},
		{"gemini", f.Providers.Gemini},
	} {
		if entry.p == nil {
			continue
		}
		catalog, completion, err := buildKeyed(entry.name, entry.p, f, logger)
		if err != nil {
			return fail(err)
		}
		add(entry.name, catalog, completion)
	}

	if p := f.Providers.Bedrock; p != nil {
		catalog, err := static.Load(f.ResolvePath(p.Catalog))
		if err != nil {
			return fail(fmt.Errorf("providers.bedrock: %w", err))
		}
		completion, err := provider.New("bedrock", provider.Config{Region: p.Region, Logger: logger})
		if err != nil {
			return fail(fmt.Errorf("providers.bedrock: %w", err))
		}
		add("bedrock", catalog, completion)
	}

	for i := range f.Providers.Compatible {
		entry := &f.Providers.Compatible[i]
		field := fmt.Sprintf("providers.compatible[%d] (%s)", i, entry.Name)
		key, err := entry.Resolve()
		if err != nil {
			return fail(fmt.Errorf("%s: %w", field, err))
		}
		if !provider.Registered(entry.Name) {
			provider.Register(entry.Name, func(cfg provider.Config) (modelrouter.Completion, error) {
				return openai.NewCompatible(cfg.Name, cfg)
			})
		}
		catalog, err := static.Load(f.ResolvePath(entry.Catalog))
		if err != nil {
			return fail(fmt.Errorf("%s: %w", field, err))
		}
		completion, err := provider.New(entry.Name, provider.Config{
			APIKey:  key,
			BaseURL: entry.BaseURL,
			Logger:  logger,
		})
		if err != nil {
			return fail(fmt.Errorf("%s: %w", field, err))
		}
		add(entry.Name, catalog, completion)
	}

	sort.Strings(names)
	catalogList := make([]modelrouter.Catalog, 0, len(names))
	for _, name := range names {
		catalogList = append(catalogList, catalogs[name])
	}

	router, err := modelrouter.New(modelrouter.RouterOptions{
		Catalogs:          catalogList,
		Classifier:        classifier,
		Completions:       completions,
		Logger:            logger,
		OnClassifierError: modelrouter.OnClassifierError(f.Routing.OnClassifierError),
		Models:            routerModels(f.Routing.Models),
		DefaultModel:      routerDefaultModel(f.Routing.DefaultModel),
	})
	if err != nil {
		return fail(fmt.Errorf("building router: %w", err))
	}

	return &App{
		File:      f,
		Router:    router,
		Providers: names,
		closers:   closers,
	}, nil
}

// routerModels converts the configuration's string-keyed model chains into
// the Router's typed-domain schema, copying every chain so the Router owns
// its own values.
func routerModels(models map[string][]ModelChoice) map[modelrouter.Domain][]modelrouter.ModelChoice {
	if len(models) == 0 {
		return nil
	}
	converted := make(map[modelrouter.Domain][]modelrouter.ModelChoice, len(models))
	for domain, chain := range models {
		choices := make([]modelrouter.ModelChoice, 0, len(chain))
		for _, choice := range chain {
			choices = append(choices, modelrouter.ModelChoice{Provider: choice.Provider, Model: choice.Model})
		}
		converted[modelrouter.Domain(domain)] = choices
	}
	return converted
}

// routerDefaultModel converts the configuration's default model, if any.
func routerDefaultModel(choice *ModelChoice) *modelrouter.ModelChoice {
	if choice == nil {
		return nil
	}
	return &modelrouter.ModelChoice{Provider: choice.Provider, Model: choice.Model}
}

// buildKeyed assembles one key-plus-static-catalog provider (openai,
// anthropic, gemini): the credential is resolved, the catalog loaded from
// the configured file and the completion constructed through the registry.
func buildKeyed(name string, p *Keyed, f *File, logger *slog.Logger) (modelrouter.Catalog, modelrouter.Completion, error) {
	key, err := p.Resolve()
	if err != nil {
		return nil, nil, fmt.Errorf("providers.%s: %w", name, err)
	}
	catalog, err := static.Load(f.ResolvePath(p.Catalog))
	if err != nil {
		return nil, nil, fmt.Errorf("providers.%s: %w", name, err)
	}
	completion, err := provider.New(name, provider.Config{APIKey: key, Logger: logger})
	if err != nil {
		return nil, nil, fmt.Errorf("providers.%s: %w", name, err)
	}
	return catalog, completion, nil
}

// Close releases the resources Build acquired — the OpenRouter catalog's and
// the Jev classifier's HTTP clients, when configured — in reverse
// construction order, joining every non-nil error.
//
// Close is idempotent and safe on a nil *App.
func (a *App) Close() error {
	if a == nil {
		return nil
	}
	a.closeOnce.Do(func() {
		a.closeErr = closeAll(a.closers)
	})
	return a.closeErr
}

// closeAll runs closers in reverse order and joins non-nil errors.
func closeAll(closers []func() error) error {
	var errs []error
	for i := len(closers) - 1; i >= 0; i-- {
		if closers[i] == nil {
			continue
		}
		if err := closers[i](); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
