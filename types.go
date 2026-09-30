// Package modelrouter is a multi-provider LLM router driven by explicit,
// configuration-provided model choices.
//
// Each request passes through:
//
//  1. one narrow classification call to TypeSafe's Jev (by default through
//     OpenRouter; it never sees the model list),
//  2. selection of the configured candidate chain for the classified domain
//     (RouterOptions.Models), falling back to RouterOptions.DefaultModel when
//     the request is unclassified or its domain is unmapped, and
//  3. hard-constraint filtering (provider allow/exclude always; vision,
//     context and price checks against the catalogs only when requested)
//
// before the primary candidate is returned as a RoutingDecision. Complete
// executes the selected chain in order, advancing to the next candidate on
// any provider failure.
package modelrouter

import "encoding/json"

// Domain is the primary domain of a request, as classified by Jev.
type Domain string

// Domain values mirror the Jev choice criteria.
const (
	DomainCode          Domain = "code"
	DomainMathReasoning Domain = "math_reasoning"
	DomainCreative      Domain = "creative"
	DomainFactualLookup Domain = "factual_lookup"
	DomainOther         Domain = "other"
)

// IsValidDomain reports whether d is one of the five classified domains.
func IsValidDomain(d Domain) bool {
	switch d {
	case DomainCode, DomainMathReasoning, DomainCreative, DomainFactualLookup, DomainOther:
		return true
	}
	return false
}

// ModelInfo describes one routable model from a provider catalog.
type ModelInfo struct {
	// Provider is the registry name of the provider that serves this model
	// (e.g. "openrouter", "bedrock", "openai").
	Provider string `json:"provider"`
	// ID is the provider-native model identifier used in completion requests.
	ID string `json:"id"`
	// DisplayID is an optional human-readable identifier (for example
	// "openai/gpt-5.2" in a user-supplied catalog). Label falls back to ID.
	DisplayID string `json:"display_id,omitempty"`
	// PricePer1KPromptTokens is USD per 1,000 prompt tokens.
	PricePer1KPromptTokens float64 `json:"price_per_1k_prompt_tokens"`
	// PricePer1KCompletionTokens is USD per 1,000 completion tokens.
	PricePer1KCompletionTokens float64 `json:"price_per_1k_completion_tokens"`
	// ContextLength is the maximum context window in tokens.
	ContextLength int `json:"context_length"`
	// SupportsVision reports whether the model accepts image input.
	SupportsVision bool `json:"supports_vision"`
}

// BlendedPricePer1KTokens returns prompt + completion price per 1k tokens.
//
// Cost ranking and price filtering use the blended price because
// prompt-token price alone understates real cost for models whose completion
// tokens are priced much higher than prompt tokens.
func (m ModelInfo) BlendedPricePer1KTokens() float64 {
	return m.PricePer1KPromptTokens + m.PricePer1KCompletionTokens
}

// Label returns DisplayID when set, otherwise ID.
func (m ModelInfo) Label() string {
	if m.DisplayID != "" {
		return m.DisplayID
	}
	return m.ID
}

// RoutingConstraints are hard requirements applied to the catalog before
// classification runs, so an impossible constraint never wastes a Jev call.
type RoutingConstraints struct {
	NeedsVision      bool `json:"needs_vision,omitempty"`
	MinContextTokens int  `json:"min_context_tokens,omitempty"`
	// MaxPricePer1KTokens filters on the blended price per 1k tokens.
	MaxPricePer1KTokens *float64 `json:"max_price_per_1k_tokens,omitempty"`
	// AllowedProviders, when non-empty, restricts routing to these providers.
	AllowedProviders []string `json:"allowed_providers,omitempty"`
	// ExcludedProviders removes these providers from routing; exclusion wins
	// over AllowedProviders.
	ExcludedProviders []string `json:"excluded_providers,omitempty"`
}

// AllowsProvider reports whether the named provider passes the provider filters.
func (c RoutingConstraints) AllowsProvider(name string) bool {
	for _, p := range c.ExcludedProviders {
		if p == name {
			return false
		}
	}
	if len(c.AllowedProviders) == 0 {
		return true
	}
	for _, p := range c.AllowedProviders {
		if p == name {
			return true
		}
	}
	return false
}

// RequestProfile is the result of classifying a request with Jev.
type RequestProfile struct {
	Domain               Domain  `json:"domain"`
	DomainConfidence     float64 `json:"domain_confidence"`
	ComplexityScore      float64 `json:"complexity_score"` // 0..2
	ComplexityConfidence float64 `json:"complexity_confidence"`
	NeedsLongContext     float64 `json:"needs_long_context"`
	NeedsVision          float64 `json:"needs_vision"`
	LatencySensitive     float64 `json:"latency_sensitive"`
}

// ModelChoice names one provider-native model: the configured unit of model
// selection.
type ModelChoice struct {
	// Provider is the registry name of the provider ("openrouter",
	// "anthropic", a compatible entry's name, ...).
	Provider string `json:"provider"`
	// Model is the provider-native model identifier passed through to the
	// provider on completion requests.
	Model string `json:"model"`
}

// RoutingDecision.Source values.
const (
	// SourceClassification marks a decision whose chain came from
	// Models[classified domain].
	SourceClassification = "classification"
	// SourceDefault marks a decision that used DefaultModel because the
	// request was unclassified or its domain had no configured chain.
	SourceDefault = "default"
)

// RoutingDecision is the outcome of routing: the selected model, the ordered
// candidate chain Complete will attempt, and the classification
// context behind the choice.
type RoutingDecision struct {
	// ModelID and Provider name the selected model: the first candidate at
	// Route time, and the successful candidate when Complete fell back
	// through the chain.
	ModelID  string `json:"model_id"`
	Provider string `json:"provider"`
	// Source is SourceClassification or SourceDefault.
	Source string `json:"source"`
	// Candidates is the ordered attempt chain after constraint filtering:
	// the configured chain for the classified domain with DefaultModel
	// appended when set (and not already present), or just DefaultModel when
	// the request is unclassified or its domain is unmapped.
	Candidates []ModelChoice `json:"candidates"`
	// Profile is nil when no real classification was available (the classifier
	// call failed and on_classifier_error=fail_open); the decision then used
	// DefaultModel.
	Profile *RequestProfile `json:"profile,omitempty"`
	// RawJevResponse is the raw Jev response body, kept for debugging.
	RawJevResponse json.RawMessage `json:"raw_jev_response,omitempty"`
	// ClassifierUsage is set when the classifier endpoint reports usage.
	ClassifierUsage *Usage `json:"classifier_usage,omitempty"`
}

// CompletionResult pairs a provider completion with the routing decision that
// produced it.
type CompletionResult struct {
	RoutingDecision RoutingDecision `json:"routing_decision"`
	Model           string          `json:"model"`
	Provider        string          `json:"provider"`
	// Content is the text of the first choice.
	Content      string `json:"content"`
	FinishReason string `json:"finish_reason,omitempty"`
	Usage        *Usage `json:"usage,omitempty"`
	// Raw is the provider-native response body.
	Raw json.RawMessage `json:"raw,omitempty"`
}

// Usage reports token usage and, when the provider exposes it, cost in USD.
type Usage struct {
	InputTokens  int      `json:"input_tokens"`
	OutputTokens int      `json:"output_tokens"`
	TotalTokens  int      `json:"total_tokens"`
	CostUSD      *float64 `json:"cost_usd,omitempty"`
}
