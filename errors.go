package modelrouter

import (
	"errors"
	"fmt"
)

// Sentinel errors. Every typed error below satisfies errors.Is against its
// sentinel and errors.As against its concrete type, while preserving the
// underlying cause chain.
var (
	// ErrCatalogUnavailable means a catalog could not be fetched and no
	// cached copy was available.
	ErrCatalogUnavailable = errors.New("catalog unavailable")
	// ErrClassifierUnavailable means the Jev classification call failed.
	ErrClassifierUnavailable = errors.New("classifier unavailable")
	// ErrNoCandidate means constraints eliminated every candidate model.
	ErrNoCandidate = errors.New("no candidate model")
	// ErrProvider wraps failures from a specific provider operation.
	ErrProvider = errors.New("provider error")
)

// CatalogUnavailableError is returned when a catalog cannot be fetched and no
// cached copy exists. Fetches that fail while a stale cache is warm downgrade
// to a logged warning instead.
type CatalogUnavailableError struct{ Err error }

func (e *CatalogUnavailableError) Error() string {
	if e.Err == nil {
		return "modelrouter: catalog unavailable"
	}
	return "modelrouter: catalog unavailable: " + e.Err.Error()
}

func (e *CatalogUnavailableError) Is(target error) bool {
	return target == ErrCatalogUnavailable || errors.Is(e.Err, target)
}

func (e *CatalogUnavailableError) Unwrap() error { return e.Err }

// ClassifierUnavailableError is returned when the Jev classification call
// fails, times out, or returns an unparseable response.
type ClassifierUnavailableError struct{ Err error }

func (e *ClassifierUnavailableError) Error() string {
	if e.Err == nil {
		return "modelrouter: classifier unavailable"
	}
	return "modelrouter: classifier unavailable: " + e.Err.Error()
}

func (e *ClassifierUnavailableError) Is(target error) bool {
	return target == ErrClassifierUnavailable || errors.Is(e.Err, target)
}

func (e *ClassifierUnavailableError) Unwrap() error { return e.Err }

// NoCandidateModelError is returned when constraints eliminate every
// candidate. Message names the eliminating constraint(s); see the diagnostics
// helpers for construction.
type NoCandidateModelError struct{ Message string }

func (e *NoCandidateModelError) Error() string {
	if e.Message == "" {
		return "modelrouter: no candidate model"
	}
	return "modelrouter: no candidate model: " + e.Message
}

func (e *NoCandidateModelError) Is(target error) bool { return target == ErrNoCandidate }

func (e *NoCandidateModelError) Unwrap() error { return ErrNoCandidate }

// ProviderError wraps a failure from a specific provider operation (catalog
// fetch, completion) while preserving the underlying cause.
type ProviderError struct {
	Provider   string
	Op         string
	StatusCode int
	Err        error
}

func (e *ProviderError) Error() string {
	msg := "modelrouter: provider"
	if e.Provider != "" {
		msg += " " + e.Provider
	}
	if e.Op != "" {
		msg += " " + e.Op
	}
	if e.StatusCode != 0 {
		msg += fmt.Sprintf(" (status %d)", e.StatusCode)
	}
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *ProviderError) Is(target error) bool {
	return target == ErrProvider || errors.Is(e.Err, target)
}

func (e *ProviderError) Unwrap() error { return e.Err }
