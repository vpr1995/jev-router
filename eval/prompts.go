// Package eval implements the jev-router evaluation harness: it routes a
// fixed set of hand-labeled prompts through a Router and reports how many
// decisions classified into the expected domain, alongside optional
// price-savings (when a model pool is supplied — see Options.Models) and a
// per-domain breakdown, with support for repeat runs and aggregate accuracy.
// The `jev-router eval` subcommand is the intended caller; the package
// itself performs no I/O beyond the Router it is handed and its Logger.
package eval

import modelrouter "github.com/vprprudhvi/jev-router"

// Prompt is one labeled evaluation prompt: the text to route plus the human
// expectation for the resulting decision.
//
// The expectation is domain-only: classification correctness is the one
// measurable per-prompt property (price savings remain a reportable
// pool-relative statistic, not a pass/fail check).
type Prompt struct {
	// Prompt is the request text routed verbatim.
	Prompt string `json:"prompt"`
	// ExpectedDomain is the domain the classifier is expected to report.
	ExpectedDomain modelrouter.Domain `json:"expected_domain"`
}

// LabeledPrompts returns the eight hand-labeled evaluation prompts. A fresh
// slice is returned on every call, so a caller mutating the result cannot
// poison later runs.
func LabeledPrompts() []Prompt {
	return []Prompt{
		{
			Prompt:         "What is the capital of France?",
			ExpectedDomain: modelrouter.DomainFactualLookup,
		},
		{
			Prompt:         "Summarize this sentence: 'The cat sat on the mat.'",
			ExpectedDomain: modelrouter.DomainFactualLookup,
		},
		{
			Prompt:         "Write a haiku about autumn leaves.",
			ExpectedDomain: modelrouter.DomainCreative,
		},
		{
			Prompt:         "Fix this off-by-one bug in a Python for-loop that skips the last element.",
			ExpectedDomain: modelrouter.DomainCode,
		},
		{
			Prompt: "Design a distributed rate limiter that works correctly across " +
				"multiple regions with clock skew, handles burst traffic, and " +
				"degrades gracefully when the coordination service is down. " +
				"Walk through the tradeoffs of at least two approaches.",
			ExpectedDomain: modelrouter.DomainCode,
		},
		{
			Prompt: "Prove that the square root of 2 is irrational, and then explain " +
				"why the same proof technique does not directly work for the " +
				"square root of 4.",
			ExpectedDomain: modelrouter.DomainMathReasoning,
		},
		{
			Prompt: "Here's a 40-step logic puzzle involving five houses, five " +
				"nationalities, and five pets — work through the full deduction " +
				"chain and identify who owns the fish.",
			ExpectedDomain: modelrouter.DomainMathReasoning,
		},
		{
			Prompt:         "Translate 'good morning' into Portuguese.",
			ExpectedDomain: modelrouter.DomainFactualLookup,
		},
	}
}
