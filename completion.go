// This file defines the provider-agnostic completion surface the Router
// dispatches to: the Router routes to a provider-native model ID and hands
// execution to that provider's Completion implementation (see provider/*).
package modelrouter

import (
	"context"
	"errors"
)

// Message is one chat message in a completion request.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest is a provider-agnostic completion request.
//
// Model and Provider are filled in by the Router from the routing decision
// when the caller routes implicitly; a caller that sets Model explicitly must
// also set Provider, and routing (including classification) is skipped.
// Constraints is only read by Router.Complete to route — providers ignore it.
type CompletionRequest struct {
	// Prompt is a single-string prompt. Either Prompt or Messages must be
	// set; the Router passes whichever the caller set through untouched (it
	// never materializes one from the other) and providers map it onto their
	// native request shape.
	Prompt string `json:"prompt,omitempty"`
	// Messages is a full chat message list, for callers that need roles.
	Messages []Message `json:"messages,omitempty"`
	// Constraints are the routing constraints applied when the Router picks
	// the model. Only used by the Router; providers ignore this field.
	Constraints *RoutingConstraints `json:"constraints,omitempty"`
	// Model is the provider-native model ID. Set by the Router from the
	// routing decision, or explicitly by the caller together with Provider.
	Model string `json:"model,omitempty"`
	// Provider names the Completion implementation that executes this
	// request. Set by the Router from the routing decision, or explicitly by
	// the caller together with Model.
	Provider string `json:"provider,omitempty"`
	// MaxTokens caps generated tokens; 0 means the provider default.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Temperature overrides sampling temperature; nil means the provider
	// default.
	Temperature *float64 `json:"temperature,omitempty"`
}

// Completion executes completions for one provider.
//
// Implementations must be safe for concurrent use when a Router is shared.
// Providers own any wrapping of their own transport failures (the plan's
// convention: provider API errors propagate wrapped in *ProviderError where
// translation occurs, and the Router passes them through otherwise).
// Providers must leave CompletionResult.RoutingDecision empty — the Router
// owns that field and always overwrites it.
type Completion interface {
	// Complete runs a completion.
	Complete(ctx context.Context, req CompletionRequest) (CompletionResult, error)
}

// ErrProviderNotConfigured means no Completion is registered for the provider
// required by the request (routed or explicit). It lives here, next to the
// Completion interface, deliberately: it concerns completion dispatch only,
// not the catalog/classifier error families in errors.go.
var ErrProviderNotConfigured = errors.New("provider not configured")
