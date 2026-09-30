// This file implements the evaluation run: routing every labeled prompt,
// translating each decision into a PromptResult and aggregating the results
// into a Report. See prompts.go for the labeled prompt set.
package eval

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// Router is the narrow slice of *modelrouter.Router the harness needs: one
// routing call per prompt. The real Router satisfies it — pinned by a
// compile-time check in eval_test.go — and tests substitute a scripted fake.
type Router interface {
	Route(ctx context.Context, prompt string, constraints *modelrouter.RoutingConstraints) (modelrouter.RoutingDecision, error)
}

// Options configures Run.
type Options struct {
	// MinAccuracy is the accuracy threshold Report.Passed is computed
	// against. Zero is valid: Passed then only demands at least one
	// evaluated decision (Evaluated > 0, see Report.Passed). The CLI
	// defaults it to 0.75.
	MinAccuracy float64
	// Repeat is how many times the whole prompt set is routed. Values below
	// 1 are treated as 1.
	Repeat int
	// Logger receives one warning per run that skipped prompts for lack of a
	// classification profile. Nil means slog.Default().
	Logger *slog.Logger
	// Models is an optional model pool used ONLY to compute the report's
	// price fields — the CLI passes the Router.ListModels output here.
	// Entries are matched against each decision by Provider+ModelID for the
	// chosen price; the median and frontier come from the pool's blended
	// prices (see priceStats). When Models is empty every price field stays
	// zero: no pool means no prices, and accuracy is unaffected.
	Models []modelrouter.ModelInfo
}

// PromptResult is the outcome of evaluating one prompt once.
//
// Blended prices are prompt + completion price per 1k tokens, the same cost
// model the constraints use, and are computed only when Options.Models
// supplies a model pool: without one every price field stays zero. A Skipped
// result (no profile) leaves the domain and price fields zero too.
type PromptResult struct {
	Prompt         string `json:"prompt"`
	Domain         string `json:"domain"`
	ModelID        string `json:"model_id"`
	Provider       string `json:"provider"`
	ExpectedDomain string `json:"expected_domain"`

	DomainConfidence float64 `json:"domain_confidence"`
	ComplexityScore  float64 `json:"complexity_score"`

	DomainOK bool `json:"domain_ok"`
	// Matched is DomainOK: classification correctness is the only
	// per-prompt expectation the harness checks.
	Matched bool `json:"matched"`
	// Skipped marks a decision without a profile (the classifier failed and
	// fail_open selected the default model). Skipped results are excluded
	// from every evaluated aggregate.
	Skipped bool `json:"skipped"`

	// ChosenBlendedPrice is the blended price of decision.ModelID/Provider
	// as found in Options.Models, or 0 when the pool does not contain it (or
	// no pool was supplied).
	ChosenBlendedPrice float64 `json:"chosen_blended_price"`
	// MedianBlendedPrice is the upper median (sorted[len/2]) of the model
	// pool's blended prices, or 0 when no pool was supplied.
	MedianBlendedPrice float64 `json:"median_blended_price"`
	// FrontierBlendedPrice is the model pool's highest blended price, or 0
	// when no pool was supplied.
	FrontierBlendedPrice float64 `json:"frontier_blended_price"`

	// SavingsVsFrontier is 1 - Chosen/Frontier when both are positive: an
	// unknown chosen model (price 0) or a zero frontier deliberately report
	// no savings rather than a division artefact.
	SavingsVsFrontier float64 `json:"savings_vs_frontier"`

	// RouteDuration is the wall time of the single Route call; JSON sees
	// nanoseconds (encoding/json renders time.Duration as an integer).
	RouteDuration time.Duration `json:"route_duration"`
}

// RunReport aggregates one repetition of the whole prompt set.
type RunReport struct {
	Results []PromptResult `json:"results"`
	// Evaluated counts results with a profile; Matched those whose
	// classified domain matched the human expectation; Skipped the
	// profileless ones.
	Evaluated int `json:"evaluated"`
	Matched   int `json:"matched"`
	Skipped   int `json:"skipped"`
	// Accuracy is Matched/Evaluated (0 when Evaluated == 0); MeanSavings the
	// mean SavingsVsFrontier over the evaluated results (0 when there are
	// none, including when no model pool was supplied).
	Accuracy    float64 `json:"accuracy"`
	MeanSavings float64 `json:"mean_savings"`
}

// DomainStats aggregates the evaluated results expected in one domain.
type DomainStats struct {
	Evaluated int     `json:"evaluated"`
	Matched   int     `json:"matched"`
	Accuracy  float64 `json:"accuracy"`
}

// Report is the full eval outcome: one RunReport per repetition plus the
// cross-run aggregate.
type Report struct {
	Runs []RunReport `json:"runs"`
	// Evaluated, Matched and Skipped are summed over the runs.
	Evaluated int `json:"evaluated"`
	Matched   int `json:"matched"`
	Skipped   int `json:"skipped"`
	// Accuracy is Matched/Evaluated over all runs (0 when Evaluated == 0);
	// MeanSavings is the mean SavingsVsFrontier over every evaluated result
	// (0 when no model pool was supplied or every frontier was zero).
	Accuracy    float64 `json:"accuracy"`
	MeanSavings float64 `json:"mean_savings"`
	// MinAccuracy is Options.MinAccuracy, echoed for the report's benefit.
	// Passed is Evaluated > 0 && Accuracy >= MinAccuracy: a zero threshold
	// still demands at least one evaluated decision, so an all-skipped run
	// can never pass.
	MinAccuracy float64 `json:"min_accuracy"`
	Passed      bool    `json:"passed"`
	// PerDomain aggregates by ExpectedDomain over evaluated results only;
	// domains whose prompts were all skipped get no entry.
	PerDomain map[string]DomainStats `json:"per_domain"`
}

// Run routes every prompt, Repeat times, and aggregates the results.
//
//   - a decision without a profile (the classifier failed and fail_open
//     selected the default model) marks the result Skipped and is excluded
//     from every evaluated aggregate (Evaluated, Accuracy, MeanSavings,
//     PerDomain) — there is no classified domain to check;
//   - a Route error aborts the whole run immediately and is returned
//     unchanged; a partial report would silently bias the accuracy, so the
//     returned Report is the zero value.
//
// Options.Repeat runs the whole prompt set repeatedly for stability checks:
// Report.Runs carries one RunReport per repetition, and the top-level fields
// aggregate across all of them (summed counts, Accuracy = Matched/Evaluated,
// MeanSavings = the mean SavingsVsFrontier over every evaluated result).
func Run(ctx context.Context, r Router, prompts []Prompt, opts Options) (Report, error) {
	if r == nil {
		return Report{}, errors.New("eval: nil router")
	}
	repeat := opts.Repeat
	if repeat < 1 {
		repeat = 1
	}
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	report := Report{
		Runs:        make([]RunReport, 0, repeat),
		MinAccuracy: opts.MinAccuracy,
		PerDomain:   make(map[string]DomainStats),
	}
	var savingsSum float64
	var savingsCount int
	for i := 0; i < repeat; i++ {
		run := RunReport{Results: make([]PromptResult, 0, len(prompts))}
		for _, prompt := range prompts {
			result, err := evaluate(ctx, r, prompt, opts.Models)
			if err != nil {
				return Report{}, err
			}
			run.Results = append(run.Results, result)
			if result.Skipped {
				run.Skipped++
				continue
			}
			run.Evaluated++
			savingsSum += result.SavingsVsFrontier
			savingsCount++
			if result.Matched {
				run.Matched++
			}
			stats := report.PerDomain[result.ExpectedDomain]
			stats.Evaluated++
			if result.Matched {
				stats.Matched++
			}
			report.PerDomain[result.ExpectedDomain] = stats
		}
		run.Accuracy = ratio(run.Matched, run.Evaluated)
		run.MeanSavings = meanSavings(run.Results)
		if run.Skipped > 0 {
			logger.Warn("eval: prompts skipped without a classification profile",
				"skipped", run.Skipped, "evaluated", run.Evaluated)
		}
		report.Runs = append(report.Runs, run)
		report.Evaluated += run.Evaluated
		report.Matched += run.Matched
		report.Skipped += run.Skipped
	}
	for domain, stats := range report.PerDomain {
		stats.Accuracy = ratio(stats.Matched, stats.Evaluated)
		report.PerDomain[domain] = stats
	}
	report.Accuracy = ratio(report.Matched, report.Evaluated)
	if savingsCount > 0 {
		report.MeanSavings = savingsSum / float64(savingsCount)
	}
	report.Passed = report.Evaluated > 0 && report.Accuracy >= opts.MinAccuracy
	return report, nil
}

// evaluate routes one prompt and translates the decision into a PromptResult.
//
// A nil profile marks the result Skipped and returns before any price math:
// the fail-open path selected the default model, so there is no classified
// domain to check and no chosen chain to price. A profile without a
// candidate chain is an invalid decision (the real Router returns
// *NoCandidateModelError before one escapes) and is reported as an error
// rather than evaluating a decision nothing could be completed on.
func evaluate(ctx context.Context, r Router, prompt Prompt, models []modelrouter.ModelInfo) (PromptResult, error) {
	result := PromptResult{
		Prompt:         prompt.Prompt,
		ExpectedDomain: string(prompt.ExpectedDomain),
	}
	started := time.Now()
	decision, err := r.Route(ctx, prompt.Prompt, nil)
	result.RouteDuration = time.Since(started)
	if err != nil {
		return PromptResult{}, err
	}
	result.ModelID = decision.ModelID
	result.Provider = decision.Provider
	profile := decision.Profile
	if profile == nil {
		result.Skipped = true
		return result, nil
	}
	if len(decision.Candidates) == 0 {
		return PromptResult{}, fmt.Errorf("eval: routing decision for prompt %q has no candidate models", prompt.Prompt)
	}
	result.Domain = string(profile.Domain)
	result.DomainConfidence = profile.DomainConfidence
	result.ComplexityScore = profile.ComplexityScore
	result.DomainOK = profile.Domain == prompt.ExpectedDomain
	result.Matched = result.DomainOK

	// Prices need a pool: without one the price fields keep their zero
	// values (documented on Options.Models).
	if len(models) > 0 {
		result.ChosenBlendedPrice, result.MedianBlendedPrice, result.FrontierBlendedPrice = priceStats(decision, models)
		if result.FrontierBlendedPrice > 0 && result.ChosenBlendedPrice > 0 {
			result.SavingsVsFrontier = 1 - result.ChosenBlendedPrice/result.FrontierBlendedPrice
		}
	}
	return result, nil
}

// priceStats matches the decision's chosen model against the supplied pool
// and summarizes the pool's blended prices: chosen is the matching entry's
// blended price (0 when the model is absent; the first match wins if the
// pool repeats it), median the upper median (sorted[len/2]) and frontier the
// maximum. The caller guarantees a non-empty pool.
func priceStats(decision modelrouter.RoutingDecision, models []modelrouter.ModelInfo) (chosen, median, frontier float64) {
	prices := make([]float64, 0, len(models))
	found := false
	for _, model := range models {
		price := model.BlendedPricePer1KTokens()
		prices = append(prices, price)
		if !found && model.Provider == decision.Provider && model.ID == decision.ModelID {
			chosen, found = price, true
		}
	}
	sort.Float64s(prices)
	// Upper median: sorted[len/2], not the average of the two middle values.
	median = prices[len(prices)/2]
	frontier = prices[len(prices)-1]
	return chosen, median, frontier
}

// ratio returns matched/evaluated, or 0 when evaluated is 0.
func ratio(matched, evaluated int) float64 {
	if evaluated == 0 {
		return 0
	}
	return float64(matched) / float64(evaluated)
}

// meanSavings returns the mean SavingsVsFrontier over the evaluated
// (non-skipped) results, or 0 when there are none.
func meanSavings(results []PromptResult) float64 {
	var sum float64
	var count int
	for _, result := range results {
		if result.Skipped {
			continue
		}
		sum += result.SavingsVsFrontier
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}
