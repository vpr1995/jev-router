// This file builds the diagnostic message for a NoCandidateModelError: it
// names the constraint(s) actually responsible for eliminating every
// candidate, instead of dumping the whole constraints object (which includes
// defaults that never eliminate anything).
package modelrouter

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// DescribeEliminatingConstraints explains why filtering all with c left no
// candidates.
//
// It reports, per active (non-default) constraint, how many models would
// survive it on their own, naming the elimination explicitly when a
// constraint alone wipes out the pool. When no single constraint explains it
// (for example a combination of individually-passing constraints, or the
// provider filters), it says so and renders the full constraints. When all
// spans more than one provider, a "per provider: <name> X/N" part is appended
// for each, sorted by name, where X survives every constraint and N is that
// provider's total. An empty catalog says so directly.
func DescribeEliminatingConstraints(all []ModelInfo, c RoutingConstraints) string {
	n := len(all)
	if n == 0 {
		return "the catalog returned 0 candidate models"
	}

	parts := make([]string, 0, 4)

	if c.NeedsVision {
		survivors := 0
		for _, m := range all {
			if m.SupportsVision {
				survivors++
			}
		}
		if survivors == 0 {
			parts = append(parts, fmt.Sprintf(
				"needs_vision=true eliminates all (0/%d models support vision)", n))
		} else {
			parts = append(parts, fmt.Sprintf(
				"needs_vision=true (%d/%d models support vision)", survivors, n))
		}
	}

	if c.MinContextTokens > 0 {
		survivors, largest := 0, 0
		for _, m := range all {
			if m.ContextLength > largest {
				largest = m.ContextLength
			}
			if m.ContextLength >= c.MinContextTokens {
				survivors++
			}
		}
		if survivors == 0 {
			parts = append(parts, fmt.Sprintf(
				"min_context_tokens=%d eliminates all (0/%d models qualify, largest available: %d)",
				c.MinContextTokens, n, largest))
		} else {
			parts = append(parts, fmt.Sprintf(
				"min_context_tokens=%d (%d/%d models qualify)", c.MinContextTokens, survivors, n))
		}
	}

	if c.MaxPricePer1KTokens != nil {
		survivors, cheapest := 0, math.Inf(1)
		for _, m := range all {
			price := m.BlendedPricePer1KTokens()
			if price < cheapest {
				cheapest = price
			}
			if price <= *c.MaxPricePer1KTokens {
				survivors++
			}
		}
		if survivors == 0 {
			parts = append(parts, fmt.Sprintf(
				"max_price_per_1k_tokens=%g eliminates all (0/%d models qualify, cheapest available: %g)",
				*c.MaxPricePer1KTokens, n, cheapest))
		} else {
			parts = append(parts, fmt.Sprintf(
				"max_price_per_1k_tokens=%g (%d/%d models qualify)",
				*c.MaxPricePer1KTokens, survivors, n))
		}
	}

	if len(parts) == 0 {
		// No active constraint alone explains it: it must be the combination
		// of constraints, including the provider filters.
		parts = append(parts, fmt.Sprintf(
			"the combination of constraints eliminates all %d candidate models: %s",
			n, formatConstraints(c)))
	}

	if providers := distinctProviders(all); len(providers) > 1 {
		surviving := make(map[string]int, len(providers))
		for _, m := range FilterModels(all, c) {
			surviving[m.Provider]++
		}
		totals := make(map[string]int, len(providers))
		for _, m := range all {
			totals[m.Provider]++
		}
		for _, provider := range providers {
			parts = append(parts, fmt.Sprintf(
				"per provider: %s %d/%d", provider, surviving[provider], totals[provider]))
		}
	}

	return strings.Join(parts, "; ")
}

// distinctProviders returns the distinct provider names in all, sorted
// ascending for deterministic output.
func distinctProviders(all []ModelInfo) []string {
	seen := make(map[string]struct{}, len(all))
	names := make([]string, 0, len(all))
	for _, m := range all {
		if _, ok := seen[m.Provider]; ok {
			continue
		}
		seen[m.Provider] = struct{}{}
		names = append(names, m.Provider)
	}
	sort.Strings(names)
	return names
}

// formatConstraints renders constraints for the combination-case message.
func formatConstraints(c RoutingConstraints) string {
	price := "nil"
	if c.MaxPricePer1KTokens != nil {
		price = fmt.Sprintf("%g", *c.MaxPricePer1KTokens)
	}
	return fmt.Sprintf(
		"RoutingConstraints{needs_vision: %t, min_context_tokens: %d, max_price_per_1k_tokens: %s, allowed_providers: %v, excluded_providers: %v}",
		c.NeedsVision, c.MinContextTokens, price, c.AllowedProviders, c.ExcludedProviders)
}
