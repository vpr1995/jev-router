// This file declares the Catalog interface and the hard-constraint filter the
// router consumes. Catalog lives in the root package — rather than in a
// catalog subpackage — so provider catalog implementations can import the
// core types without creating an import cycle.
package modelrouter

import "context"

// Catalog supplies the models a provider can route to.
//
// Implementations may cache or refetch internally. The returned slice is
// treated as read-only by callers; implementations that serve mutable state
// must return a copy.
type Catalog interface {
	// Models returns the provider's currently routable models.
	Models(ctx context.Context) ([]ModelInfo, error)
}

// FilterModels applies the hard constraints in c to models, preserving the
// input order. It never mutates models or c. A model is kept only when it
// passes the provider filter, vision requirement, minimum context and
// maximum blended price — whichever of those are active. An empty input
// yields a non-nil, empty result.
func FilterModels(models []ModelInfo, c RoutingConstraints) []ModelInfo {
	filtered := make([]ModelInfo, 0, len(models))
	for _, model := range models {
		if !c.AllowsProvider(model.Provider) {
			continue
		}
		if c.NeedsVision && !model.SupportsVision {
			continue
		}
		if model.ContextLength < c.MinContextTokens {
			continue
		}
		if c.MaxPricePer1KTokens != nil && model.BlendedPricePer1KTokens() > *c.MaxPricePer1KTokens {
			continue
		}
		filtered = append(filtered, model)
	}
	return filtered
}
