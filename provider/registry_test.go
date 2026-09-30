package provider_test

import (
	"context"
	"errors"
	"sort"
	"strings"
	"sync"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
)

// stubCompletion is the toy Completion the registry tests register.
type stubCompletion struct {
	name string
}

func (s *stubCompletion) Complete(context.Context, modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	return modelrouter.CompletionResult{Provider: s.name, Model: "stub"}, nil
}

var _ modelrouter.Completion = (*stubCompletion)(nil)

func mustPanic(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		t.Helper()
		recovered := recover()
		if recovered == nil {
			t.Fatalf("expected panic containing %q, got no panic", want)
		}
		if msg, ok := recovered.(string); !ok || !strings.Contains(msg, want) {
			t.Fatalf("panic = %v, want a string containing %q", recovered, want)
		}
	}()
	fn()
}

func TestRegisterAndNew(t *testing.T) {
	const name = "registry-test-toy"
	var gotConfig provider.Config
	provider.Register(name, func(cfg provider.Config) (modelrouter.Completion, error) {
		gotConfig = cfg
		return &stubCompletion{name: cfg.Name}, nil
	})

	completion, err := provider.New(name, provider.Config{APIKey: "test-key", BaseURL: "https://example.test"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if completion == nil {
		t.Fatal("New() returned a nil completion")
	}
	if gotConfig.Name != name {
		t.Errorf("factory cfg.Name = %q, want %q (registry must set it)", gotConfig.Name, name)
	}
	if gotConfig.APIKey != "test-key" {
		t.Errorf("factory cfg.APIKey = %q, want %q", gotConfig.APIKey, "test-key")
	}
	if gotConfig.BaseURL != "https://example.test" {
		t.Errorf("factory cfg.BaseURL = %q, want %q", gotConfig.BaseURL, "https://example.test")
	}

	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "m"})
	if err != nil {
		t.Fatalf("stub Complete() error = %v", err)
	}
	if result.Provider != name {
		t.Errorf("stub result.Provider = %q, want %q", result.Provider, name)
	}
}

func TestNamesSorted(t *testing.T) {
	provider.Register("registry-test-zzz", func(provider.Config) (modelrouter.Completion, error) {
		return &stubCompletion{name: "zzz"}, nil
	})
	provider.Register("registry-test-aaa", func(provider.Config) (modelrouter.Completion, error) {
		return &stubCompletion{name: "aaa"}, nil
	})

	names := provider.Names()
	if !sort.StringsAreSorted(names) {
		t.Errorf("Names() = %v, want sorted order", names)
	}
	zi, ai := indexOf(names, "registry-test-zzz"), indexOf(names, "registry-test-aaa")
	if ai < 0 || zi < 0 {
		t.Fatalf("Names() = %v, missing the two test registrations", names)
	}
	if ai >= zi {
		t.Errorf("Names() = %v: %q should sort before %q", names, "registry-test-aaa", "registry-test-zzz")
	}
}

func indexOf(names []string, want string) int {
	for i, name := range names {
		if name == want {
			return i
		}
	}
	return -1
}

func TestRegisterDuplicatePanics(t *testing.T) {
	const name = "registry-test-duplicate"
	factory := func(provider.Config) (modelrouter.Completion, error) {
		return &stubCompletion{name: name}, nil
	}
	provider.Register(name, factory)
	mustPanic(t, "twice", func() {
		provider.Register(name, factory)
	})
}

func TestRegisterNilFactoryPanics(t *testing.T) {
	mustPanic(t, "nil factory", func() {
		provider.Register("registry-test-nil", nil)
	})
}

func TestRegisterEmptyNamePanics(t *testing.T) {
	mustPanic(t, "empty name", func() {
		provider.Register("", func(provider.Config) (modelrouter.Completion, error) {
			return &stubCompletion{name: "unnamed"}, nil
		})
	})
}

func TestNewUnknownProviderWrapsErrProviderNotConfigured(t *testing.T) {
	completion, err := provider.New("registry-test-missing", provider.Config{})
	if err == nil {
		t.Fatalf("New() = %v, want an error", completion)
	}
	if completion != nil {
		t.Errorf("New() completion = %v, want nil", completion)
	}
	if !errors.Is(err, modelrouter.ErrProviderNotConfigured) {
		t.Errorf("errors.Is(err, ErrProviderNotConfigured) = false, err = %v", err)
	}
}

func TestNewConcurrent(t *testing.T) {
	const name = "registry-test-concurrent"
	provider.Register(name, func(cfg provider.Config) (modelrouter.Completion, error) {
		return &stubCompletion{name: cfg.Name}, nil
	})

	const goroutines = 20
	var wg sync.WaitGroup
	completions := make(chan modelrouter.Completion, goroutines)
	errs := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			completion, err := provider.New(name, provider.Config{APIKey: "test-key"})
			if err != nil {
				errs <- err
				return
			}
			completions <- completion
		}()
	}
	wg.Wait()
	close(errs)
	close(completions)

	for err := range errs {
		t.Errorf("concurrent New() error = %v", err)
	}
	count := 0
	for completion := range completions {
		if completion == nil {
			t.Error("concurrent New() returned a nil completion")
		}
		count++
	}
	if count != goroutines {
		t.Errorf("got %d completions, want %d", count, goroutines)
	}
}
