// These tests exercise the eval harness offline with a scripted fake router:
// no configuration, no network, no classifier. They pin the core semantics —
// skipped prompts, domain matching, the "(no profile)" rule — and the
// extensions: price savings from a supplied model pool, repeat runs,
// aggregates, per-domain stats and the JSON round-trip.
package eval_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/eval"
)

// Compile-time proof that the real Router satisfies the harness's interface:
// if Route's signature ever drifts, this line stops compiling.
var _ eval.Router = (*modelrouter.Router)(nil)

// scriptedRouter is the fake Router: it answers from byPrompt, from err, or
// from the fn override (which also sees the 1-based call number, for repeat
// tests), and counts calls.
type scriptedRouter struct {
	byPrompt map[string]modelrouter.RoutingDecision
	err      error
	fn       func(call int, prompt string) (modelrouter.RoutingDecision, error)

	mu    sync.Mutex
	calls int
}

// Route implements eval.Router.
func (r *scriptedRouter) Route(_ context.Context, prompt string, _ *modelrouter.RoutingConstraints) (modelrouter.RoutingDecision, error) {
	r.mu.Lock()
	r.calls++
	call := r.calls
	r.mu.Unlock()

	if r.fn != nil {
		return r.fn(call, prompt)
	}
	if r.err != nil {
		return modelrouter.RoutingDecision{}, r.err
	}
	decision, ok := r.byPrompt[prompt]
	if !ok {
		return modelrouter.RoutingDecision{}, fmt.Errorf("scriptedRouter: unscripted prompt %q", prompt)
	}
	return decision, nil
}

// callCount reports how many Route calls were served.
func (r *scriptedRouter) callCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls
}

// routerFor maps each prompt onto its scripted decision, positionally.
func routerFor(prompts []eval.Prompt, decisions ...modelrouter.RoutingDecision) *scriptedRouter {
	byPrompt := make(map[string]modelrouter.RoutingDecision, len(prompts))
	for i, prompt := range prompts {
		byPrompt[prompt.Prompt] = decisions[i]
	}
	return &scriptedRouter{byPrompt: byPrompt}
}

// discardLogger silences the harness's skip warnings.
func discardLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// profile builds a confidently-classified RequestProfile.
func profile(domain modelrouter.Domain, complexity float64) *modelrouter.RequestProfile {
	return &modelrouter.RequestProfile{
		Domain:               domain,
		DomainConfidence:     0.9,
		ComplexityScore:      complexity,
		ComplexityConfidence: 0.9,
	}
}

// testProvider is the provider every scripted decision and pool entry uses,
// so the two match by Provider+ModelID without extra bookkeeping.
const testProvider = "test"

// choice builds one candidate model in testProvider.
func choice(model string) modelrouter.ModelChoice {
	return modelrouter.ModelChoice{Provider: testProvider, Model: model}
}

// decision builds a decision whose candidate chain is the given model names,
// in order. ModelID/Provider echo the first candidate exactly as Route sets
// them; an empty chain leaves both empty. A nil profile is the fail-open
// shape — no classification, default-selected model — so Source becomes
// "default".
func decision(prof *modelrouter.RequestProfile, models ...string) modelrouter.RoutingDecision {
	chain := make([]modelrouter.ModelChoice, 0, len(models))
	for _, model := range models {
		chain = append(chain, choice(model))
	}
	d := modelrouter.RoutingDecision{
		Source:     modelrouter.SourceClassification,
		Candidates: chain,
		Profile:    prof,
	}
	if prof == nil {
		d.Source = modelrouter.SourceDefault
	}
	if len(chain) > 0 {
		d.ModelID = chain[0].Model
		d.Provider = chain[0].Provider
	}
	return d
}

// pool builds an Options.Models pool in testProvider: entry i is model-i
// with the given blended price split evenly between prompt and completion
// tokens, so tests state one number per model while the blended sum is still
// exercised.
func pool(prices ...float64) []modelrouter.ModelInfo {
	models := make([]modelrouter.ModelInfo, 0, len(prices))
	for i, price := range prices {
		models = append(models, modelrouter.ModelInfo{
			Provider:                   testProvider,
			ID:                         fmt.Sprintf("model-%d", i),
			PricePer1KPromptTokens:     price / 2,
			PricePer1KCompletionTokens: price / 2,
		})
	}
	return models
}

// run evaluates prompts against router, requiring success.
func run(t *testing.T, router eval.Router, prompts []eval.Prompt, opts eval.Options) eval.Report {
	t.Helper()
	if opts.Logger == nil {
		opts.Logger = discardLogger()
	}
	report, err := eval.Run(context.Background(), router, prompts, opts)
	if err != nil {
		t.Fatalf("eval.Run(...) = %v", err)
	}
	return report
}

// closeEnough compares floating-point results within a tight tolerance.
func closeEnough(got, want float64) bool { return math.Abs(got-want) < 1e-9 }

// TestLabeledPromptsArePinnedWordForWord pins the eight labeled prompts word
// for word (including the em dash in the logic-puzzle prompt), so a stray
// edit cannot silently change what the eval harness measures.
func TestLabeledPromptsArePinnedWordForWord(t *testing.T) {
	want := []eval.Prompt{
		{Prompt: "What is the capital of France?", ExpectedDomain: modelrouter.DomainFactualLookup},
		{Prompt: "Summarize this sentence: 'The cat sat on the mat.'", ExpectedDomain: modelrouter.DomainFactualLookup},
		{Prompt: "Write a haiku about autumn leaves.", ExpectedDomain: modelrouter.DomainCreative},
		{Prompt: "Fix this off-by-one bug in a Python for-loop that skips the last element.", ExpectedDomain: modelrouter.DomainCode},
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
		{Prompt: "Translate 'good morning' into Portuguese.", ExpectedDomain: modelrouter.DomainFactualLookup},
	}
	got := eval.LabeledPrompts()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("LabeledPrompts() = %#v, want the fixed eight prompts %#v", got, want)
	}

	// The returned slice must be a fresh copy: mutating it cannot leak into
	// later calls.
	got[0].Prompt = "mutated"
	if eval.LabeledPrompts()[0].Prompt != "What is the capital of France?" {
		t.Error("LabeledPrompts() shares state between calls")
	}
}

// TestRunCountsSkippedSeparately pins the "(no profile)" rule: a
// decision without a profile is skipped, excluded from Evaluated (hence from
// accuracy) and carries no domain or price data — even when a model pool is
// supplied.
func TestRunCountsSkippedSeparately(t *testing.T) {
	prompts := []eval.Prompt{
		{Prompt: "classified", ExpectedDomain: modelrouter.DomainCode},
		{Prompt: "fallback", ExpectedDomain: modelrouter.DomainCreative},
	}
	router := routerFor(prompts,
		decision(profile(modelrouter.DomainCode, 0), "model-0", "model-1"),
		decision(nil, "model-0"),
	)
	report := run(t, router, prompts, eval.Options{Models: pool(1, 2)})

	if len(report.Runs) != 1 {
		t.Fatalf("len(Runs) = %d, want 1", len(report.Runs))
	}
	results := report.Runs[0].Results
	if len(results) != 2 {
		t.Fatalf("len(Results) = %d, want 2", len(results))
	}
	if results[0].Skipped || !results[0].Matched {
		t.Errorf("results[0] = %+v, want a matched evaluation", results[0])
	}
	if results[0].ChosenBlendedPrice == 0 {
		t.Error("results[0].ChosenBlendedPrice = 0, want the pool price of the chosen model")
	}
	skipped := results[1]
	if !skipped.Skipped {
		t.Error("results[1].Skipped = false, want true (profile nil)")
	}
	if skipped.Matched || skipped.DomainOK {
		t.Errorf("results[1] = %+v, want every expectation flag false when skipped", skipped)
	}
	if skipped.Domain != "" || skipped.DomainConfidence != 0 || skipped.ComplexityScore != 0 {
		t.Errorf("results[1] carries profile data %+v, want none", skipped)
	}
	if skipped.ModelID != "model-0" || skipped.Provider != testProvider {
		t.Errorf("results[1] model/provider = %q/%q, want the default choice model-0/%s", skipped.ModelID, skipped.Provider, testProvider)
	}
	if skipped.ChosenBlendedPrice != 0 || skipped.MedianBlendedPrice != 0 ||
		skipped.FrontierBlendedPrice != 0 || skipped.SavingsVsFrontier != 0 {
		t.Errorf("results[1] price data = %+v, want zero despite the supplied pool", skipped)
	}
	if report.Evaluated != 1 || report.Matched != 1 || report.Skipped != 1 {
		t.Errorf("report counts = evaluated %d, matched %d, skipped %d, want 1/1/1",
			report.Evaluated, report.Matched, report.Skipped)
	}
	if report.Runs[0].Accuracy != 1 {
		t.Errorf("run accuracy = %v, want 1 (the skipped prompt must not reduce it)", report.Runs[0].Accuracy)
	}
}

// TestMatchedIsDomainMatch pins Matched = DomainOK: with the scorer and the
// cheapness check retired, classification correctness is the only per-prompt
// expectation left.
func TestMatchedIsDomainMatch(t *testing.T) {
	prompts := []eval.Prompt{
		{Prompt: "match", ExpectedDomain: modelrouter.DomainCode},
		{Prompt: "mismatch", ExpectedDomain: modelrouter.DomainCode},
	}
	router := routerFor(prompts,
		decision(profile(modelrouter.DomainCode, 0), "model-0"),
		decision(profile(modelrouter.DomainMathReasoning, 0), "model-0"),
	)
	report := run(t, router, prompts, eval.Options{})

	results := report.Runs[0].Results
	if !results[0].DomainOK || !results[0].Matched {
		t.Errorf("results[0] = domainOK %t matched %t, want true/true", results[0].DomainOK, results[0].Matched)
	}
	if results[1].DomainOK || results[1].Matched {
		t.Errorf("results[1] = domainOK %t matched %t, want false/false", results[1].DomainOK, results[1].Matched)
	}
	for i, result := range results {
		if result.Matched != result.DomainOK {
			t.Errorf("results[%d].Matched = %t, want Matched == DomainOK (%t)", i, result.Matched, result.DomainOK)
		}
	}
	if report.Evaluated != 2 || report.Matched != 1 {
		t.Errorf("counts = %d evaluated, %d matched, want 2/1", report.Evaluated, report.Matched)
	}
	if !closeEnough(report.Accuracy, 0.5) {
		t.Errorf("accuracy = %v, want 0.5", report.Accuracy)
	}
}

// TestSavingsFromModelPool pins the pool-based price math: the chosen price
// is looked up by Provider+ModelID (0 when absent), the median is the upper
// median of the pool (sorted[len/2], the retired ranked-list rule) and the
// frontier the pool maximum. Savings are 1 - chosen/frontier only when both
// are positive — an unknown chosen model or a zero frontier reports 0 — and
// an empty pool leaves every price field zero.
func TestSavingsFromModelPool(t *testing.T) {
	tests := []struct {
		name         string
		chain        []string
		models       []modelrouter.ModelInfo
		wantChosen   float64
		wantMedian   float64
		wantFrontier float64
		wantSavings  float64
	}{
		{
			name:         "chosen found in the pool",
			chain:        []string{"model-0"},
			models:       pool(0.2, 0.5, 1, 2),
			wantChosen:   0.2,
			wantMedian:   1,
			wantFrontier: 2,
			wantSavings:  0.9,
		},
		{
			name:         "upper median of an even pool",
			chain:        []string{"model-0"},
			models:       pool(1, 7),
			wantChosen:   1,
			wantMedian:   7,
			wantFrontier: 7,
			wantSavings:  1 - 1.0/7,
		},
		{
			name:         "chosen equal to the frontier",
			chain:        []string{"model-1"},
			models:       pool(1, 3),
			wantChosen:   3,
			wantMedian:   3,
			wantFrontier: 3,
			wantSavings:  0,
		},
		{
			name:         "single-entry pool",
			chain:        []string{"model-0"},
			models:       pool(4),
			wantChosen:   4,
			wantMedian:   4,
			wantFrontier: 4,
			wantSavings:  0,
		},
		{
			name:         "chosen absent from the pool reports no chosen price",
			chain:        []string{"ghost"},
			models:       pool(1, 2),
			wantChosen:   0,
			wantMedian:   2,
			wantFrontier: 2,
			wantSavings:  0,
		},
		{
			name:         "no pool leaves every price field zero",
			chain:        []string{"model-0"},
			models:       nil,
			wantChosen:   0,
			wantMedian:   0,
			wantFrontier: 0,
			wantSavings:  0,
		},
		{
			name:         "zero frontier yields no savings",
			chain:        []string{"model-0"},
			models:       pool(0, 0),
			wantChosen:   0,
			wantMedian:   0,
			wantFrontier: 0,
			wantSavings:  0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prompts := []eval.Prompt{{Prompt: tc.name, ExpectedDomain: modelrouter.DomainCode}}
			router := routerFor(prompts, decision(profile(modelrouter.DomainCode, 0), tc.chain...))
			report := run(t, router, prompts, eval.Options{Models: tc.models})
			result := report.Runs[0].Results[0]
			if !closeEnough(result.ChosenBlendedPrice, tc.wantChosen) {
				t.Errorf("ChosenBlendedPrice = %v, want %v", result.ChosenBlendedPrice, tc.wantChosen)
			}
			if !closeEnough(result.MedianBlendedPrice, tc.wantMedian) {
				t.Errorf("MedianBlendedPrice = %v, want %v", result.MedianBlendedPrice, tc.wantMedian)
			}
			if !closeEnough(result.FrontierBlendedPrice, tc.wantFrontier) {
				t.Errorf("FrontierBlendedPrice = %v, want %v", result.FrontierBlendedPrice, tc.wantFrontier)
			}
			if !closeEnough(result.SavingsVsFrontier, tc.wantSavings) {
				t.Errorf("SavingsVsFrontier = %v, want %v", result.SavingsVsFrontier, tc.wantSavings)
			}
			if !closeEnough(report.MeanSavings, tc.wantSavings) {
				t.Errorf("MeanSavings = %v, want %v", report.MeanSavings, tc.wantSavings)
			}
		})
	}
}

// TestChosenPriceIsBlended pins that the chosen price is prompt + completion
// price per 1k tokens, not the prompt price alone.
func TestChosenPriceIsBlended(t *testing.T) {
	models := []modelrouter.ModelInfo{{
		Provider:                   testProvider,
		ID:                         "model-0",
		PricePer1KPromptTokens:     0.3,
		PricePer1KCompletionTokens: 0.7,
	}}
	prompts := []eval.Prompt{{Prompt: "blend", ExpectedDomain: modelrouter.DomainCode}}
	router := routerFor(prompts, decision(profile(modelrouter.DomainCode, 0), "model-0"))
	result := run(t, router, prompts, eval.Options{Models: models}).Runs[0].Results[0]
	if !closeEnough(result.ChosenBlendedPrice, 1) {
		t.Errorf("ChosenBlendedPrice = %v, want 1 (0.3 prompt + 0.7 completion)", result.ChosenBlendedPrice)
	}
	if !closeEnough(result.SavingsVsFrontier, 0) {
		t.Errorf("SavingsVsFrontier = %v, want 0 (the chosen model is the only pool entry)", result.SavingsVsFrontier)
	}
}

// TestRepeatAggregation runs the prompt set twice and pins that Report.Runs
// holds one RunReport per repetition while the top-level counts, accuracy
// and Passed aggregate across runs.
func TestRepeatAggregation(t *testing.T) {
	prompts := []eval.Prompt{{Prompt: "repeat me", ExpectedDomain: modelrouter.DomainCode}}
	// Odd calls match (domain code), even calls miss the expected domain
	// (creative): one matched and one unmatched decision in total.
	router := &scriptedRouter{fn: func(call int, _ string) (modelrouter.RoutingDecision, error) {
		domain := modelrouter.DomainCode
		if call%2 == 0 {
			domain = modelrouter.DomainCreative
		}
		return decision(profile(domain, 0), "model-0"), nil
	}}
	report := run(t, router, prompts, eval.Options{Repeat: 2, MinAccuracy: 0.75})

	if len(report.Runs) != 2 {
		t.Fatalf("len(Runs) = %d, want 2 (one per repetition)", len(report.Runs))
	}
	if router.callCount() != 2 {
		t.Errorf("Route calls = %d, want 2", router.callCount())
	}
	if report.Runs[0].Matched != 1 || report.Runs[1].Matched != 0 {
		t.Errorf("per-run matched = %d/%d, want 1/0", report.Runs[0].Matched, report.Runs[1].Matched)
	}
	if !closeEnough(report.Runs[0].Accuracy, 1) || !closeEnough(report.Runs[1].Accuracy, 0) {
		t.Errorf("per-run accuracy = %v/%v, want 1/0", report.Runs[0].Accuracy, report.Runs[1].Accuracy)
	}
	if report.Evaluated != 2 || report.Matched != 1 || report.Skipped != 0 {
		t.Errorf("report counts = %d/%d/%d evaluated/matched/skipped, want 2/1/0",
			report.Evaluated, report.Matched, report.Skipped)
	}
	if !closeEnough(report.Accuracy, 0.5) {
		t.Errorf("report accuracy = %v, want 0.5", report.Accuracy)
	}
	if report.Passed {
		t.Error("report passed = true, want false (0.5 < min-accuracy 0.75)")
	}
}

// TestRepeatBelowOneIsTreatedAsOne pins the Repeat < 1 -> 1 rule for both
// zero and negative values.
func TestRepeatBelowOneIsTreatedAsOne(t *testing.T) {
	for _, repeat := range []int{0, -3} {
		prompts := []eval.Prompt{{Prompt: "once", ExpectedDomain: modelrouter.DomainCode}}
		router := routerFor(prompts, decision(profile(modelrouter.DomainCode, 0), "model-0"))
		report := run(t, router, prompts, eval.Options{Repeat: repeat})
		if len(report.Runs) != 1 {
			t.Errorf("Repeat %d: len(Runs) = %d, want 1", repeat, len(report.Runs))
		}
		if router.callCount() != 1 {
			t.Errorf("Repeat %d: Route calls = %d, want 1", repeat, router.callCount())
		}
	}
}

// TestPerDomainStats pins the per-domain breakdown: evaluated results only,
// keyed by expected domain, with accuracy recomputed per domain. A domain
// whose only prompt was skipped gets no entry (nothing was evaluated).
func TestPerDomainStats(t *testing.T) {
	prompts := []eval.Prompt{
		{Prompt: "code match", ExpectedDomain: modelrouter.DomainCode},
		{Prompt: "code miss", ExpectedDomain: modelrouter.DomainCode},
		{Prompt: "creative match", ExpectedDomain: modelrouter.DomainCreative},
		{Prompt: "skipped", ExpectedDomain: modelrouter.DomainMathReasoning},
	}
	router := routerFor(prompts,
		decision(profile(modelrouter.DomainCode, 0), "model-0"),
		decision(profile(modelrouter.DomainCreative, 0), "model-0"),
		decision(profile(modelrouter.DomainCreative, 0), "model-0"),
		decision(nil, "model-0"),
	)
	report := run(t, router, prompts, eval.Options{})

	want := map[string]eval.DomainStats{
		"code":     {Evaluated: 2, Matched: 1, Accuracy: 0.5},
		"creative": {Evaluated: 1, Matched: 1, Accuracy: 1},
	}
	if !reflect.DeepEqual(report.PerDomain, want) {
		t.Errorf("PerDomain = %#v, want %#v", report.PerDomain, want)
	}
}

// TestPassedSemantics pins Passed = Evaluated > 0 && Accuracy >= MinAccuracy:
// the threshold is inclusive, and a zero threshold still demands at least one
// evaluated decision.
func TestPassedSemantics(t *testing.T) {
	matched := decision(profile(modelrouter.DomainCode, 0), "model-0")
	mismatched := decision(profile(modelrouter.DomainCreative, 0), "model-0")
	skipped := decision(nil, "model-0")

	tests := []struct {
		name       string
		decisions  []modelrouter.RoutingDecision
		minAcc     float64
		wantPassed bool
	}{
		{"meets threshold", []modelrouter.RoutingDecision{matched}, 0.75, true},
		{"below threshold", []modelrouter.RoutingDecision{mismatched}, 0.75, false},
		{"zero accuracy fails a positive threshold", []modelrouter.RoutingDecision{mismatched}, 0.1, false},
		{"exactly at threshold", []modelrouter.RoutingDecision{matched, mismatched}, 0.5, true},
		{"nothing evaluated never passes", []modelrouter.RoutingDecision{skipped}, 0, false},
		{"zero accuracy passes a zero threshold once evaluated", []modelrouter.RoutingDecision{mismatched}, 0, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prompts := make([]eval.Prompt, len(tc.decisions))
			for i := range tc.decisions {
				prompts[i] = eval.Prompt{Prompt: fmt.Sprintf("prompt %d", i), ExpectedDomain: modelrouter.DomainCode}
			}
			report := run(t, routerFor(prompts, tc.decisions...), prompts, eval.Options{MinAccuracy: tc.minAcc})
			if report.Passed != tc.wantPassed {
				t.Errorf("Passed = %t (accuracy %v, evaluated %d, min %v), want %t",
					report.Passed, report.Accuracy, report.Evaluated, tc.minAcc, tc.wantPassed)
			}
			if !closeEnough(report.MinAccuracy, tc.minAcc) {
				t.Errorf("MinAccuracy = %v, want %v", report.MinAccuracy, tc.minAcc)
			}
		})
	}
}

// TestRouteErrorAborts pins the abort rule: a Route error ends the run
// immediately, is returned unchanged and yields the zero Report — no
// partial accuracy.
func TestRouteErrorAborts(t *testing.T) {
	errBoom := errors.New("boom")
	prompts := []eval.Prompt{
		{Prompt: "first", ExpectedDomain: modelrouter.DomainCode},
		{Prompt: "second", ExpectedDomain: modelrouter.DomainCode},
		{Prompt: "third", ExpectedDomain: modelrouter.DomainCode},
	}
	router := &scriptedRouter{fn: func(call int, _ string) (modelrouter.RoutingDecision, error) {
		if call == 2 {
			return modelrouter.RoutingDecision{}, errBoom
		}
		return decision(profile(modelrouter.DomainCode, 0), "model-0"), nil
	}}
	report, err := eval.Run(context.Background(), router, prompts, eval.Options{Logger: discardLogger()})
	if !errors.Is(err, errBoom) {
		t.Fatalf("eval.Run(...) = %v, want errBoom", err)
	}
	if !reflect.DeepEqual(report, eval.Report{}) {
		t.Errorf("report = %#v, want the zero Report after an aborted run", report)
	}
	if router.callCount() != 2 {
		t.Errorf("Route calls = %d, want 2 (must stop at the first error)", router.callCount())
	}
}

// TestNilRouterIsAnError documents the defensive guard: a nil Router is
// reported, not panicked on.
func TestNilRouterIsAnError(t *testing.T) {
	if _, err := eval.Run(context.Background(), nil, nil, eval.Options{}); err == nil {
		t.Error("eval.Run(nil router) = nil error, want one")
	}
}

// TestDecisionWithoutCandidatesIsAnError documents the defensive guard for a
// decision that carries a profile but no candidate chain: the harness reports
// it instead of evaluating a decision nothing could be completed on. (The
// real Router returns *NoCandidateModelError before producing one.)
func TestDecisionWithoutCandidatesIsAnError(t *testing.T) {
	prompt := eval.Prompt{Prompt: "no candidates", ExpectedDomain: modelrouter.DomainCode}
	router := routerFor([]eval.Prompt{prompt}, modelrouter.RoutingDecision{
		ModelID: "ghost",
		Source:  modelrouter.SourceClassification,
		Profile: profile(modelrouter.DomainCode, 0),
	})
	report, err := eval.Run(context.Background(), router, []eval.Prompt{prompt}, eval.Options{Logger: discardLogger()})
	if err == nil || !strings.Contains(err.Error(), "no candidate models") {
		t.Errorf("eval.Run(...) = %v, want a no-candidate-models error", err)
	}
	if !reflect.DeepEqual(report, eval.Report{}) {
		t.Errorf("report = %#v, want the zero Report after an invalid decision", report)
	}
}

// TestSkippedDecisionNeedsNoCandidates pins the ordering of the two guards: a
// nil-profile decision is Skipped (the fail-open shape) even when it carries
// no candidates, because no domain chain was selected to validate.
func TestSkippedDecisionNeedsNoCandidates(t *testing.T) {
	prompt := eval.Prompt{Prompt: "fail open", ExpectedDomain: modelrouter.DomainCode}
	router := routerFor([]eval.Prompt{prompt}, modelrouter.RoutingDecision{
		Source: modelrouter.SourceDefault,
	})
	report := run(t, router, []eval.Prompt{prompt}, eval.Options{})
	result := report.Runs[0].Results[0]
	if !result.Skipped {
		t.Errorf("result = %+v, want Skipped", result)
	}
	if report.Evaluated != 0 || report.Skipped != 1 {
		t.Errorf("counts = %d evaluated, %d skipped, want 0/1", report.Evaluated, report.Skipped)
	}
}

// TestReportJSONRoundTrip marshals a populated Report and unmarshals it back:
// the JSON report is the CLI's machine-readable artifact, so its shape must
// round-trip losslessly.
func TestReportJSONRoundTrip(t *testing.T) {
	prompts := []eval.Prompt{
		{Prompt: "matched", ExpectedDomain: modelrouter.DomainCode},
		{Prompt: "skipped", ExpectedDomain: modelrouter.DomainCreative},
	}
	router := routerFor(prompts,
		decision(profile(modelrouter.DomainCode, 1), "model-0"),
		decision(nil, "model-1"),
	)
	report := run(t, router, prompts, eval.Options{Models: pool(0.2, 10)})

	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("json.Marshal(report) = %v", err)
	}
	for _, key := range []string{
		`"mean_savings"`, `"per_domain"`, `"min_accuracy"`, `"savings_vs_frontier"`,
		`"chosen_blended_price"`, `"median_blended_price"`, `"frontier_blended_price"`, `"route_duration"`,
	} {
		if !strings.Contains(string(data), key) {
			t.Errorf("report JSON is missing key %s:\n%s", key, data)
		}
	}
	// The cheapness keys are gone for good (v2 breaking change, plan §4):
	// their absence is part of the contract, not an accident.
	for _, key := range []string{`"picked_cheap"`, `"cheap_ok"`, `"expect_cheap"`} {
		if strings.Contains(string(data), key) {
			t.Errorf("report JSON still contains retired key %s:\n%s", key, data)
		}
	}
	var got eval.Report
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report) = %v", err)
	}
	if !reflect.DeepEqual(got, report) {
		t.Errorf("round-tripped report = %#v, want %#v", got, report)
	}
}
