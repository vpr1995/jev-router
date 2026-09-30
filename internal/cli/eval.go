// This file implements the eval command.
//
// `jev-router eval` routes the eight labeled prompts (see the eval
// package) through the assembled Router and reports how many decisions
// matched the human expectations, the mean price savings against the frontier
// of the model pool fetched via ListModels (the savings line reads "n/a" when
// that pool is unavailable) and — when any prompt was skipped for lack of a
// classification profile — the skipped count. With a Jev-capable provider
// configured it makes real, billed classifier calls: this is the only CLI
// command designed to spend money.
//
// Output shapes (stdout only; logs stay on stderr):
//
//	(default)     one human table per run followed by its summary lines;
//	              with --repeat N > 1 an aggregate summary closes the output
//	--json FILE   the human output, plus the indented JSON report in FILE
//	--json -      the JSON report on stdout INSTEAD of the human table
//	              (the route --json convention: piped stdout stays parseable)
//
// Exit codes: 1 for runtime failures — including a run that evaluated
// nothing (every prompt skipped, normally a missing classifier); 3 when the
// accuracy misses --min-accuracy; 0 otherwise.
package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"unicode/utf8"

	"github.com/vprprudhvi/jev-router/eval"
)

// errEvalThreshold marks a run whose accuracy fell below --min-accuracy;
// finish maps it to exit code 3. evalThresholdError carries a user-ready
// message while Is reports the sentinel, mirroring usageError.
var errEvalThreshold = errors.New("eval accuracy below --min-accuracy")

type evalThresholdError struct{ msg string }

func (e *evalThresholdError) Error() string { return e.msg }

func (e *evalThresholdError) Is(target error) bool { return target == errEvalThreshold }

// evalTableSeparator is the ruler width between the table header and rows.
const evalTableSeparator = 150

// promptPreviewWidth is the prompt-column width: prompts are sliced to 67
// characters.
const promptPreviewWidth = 67

// runEval implements `jev-router eval [flags]`.
func runEval(args []string, stdout, stderr io.Writer) error {
	fs := newFlagSet("eval")
	global := registerGlobalFlags(fs)
	configPath := fs.String("config", "", "configuration file (default: $JEV_ROUTER_CONFIG, else environment-only defaults)")
	jsonOut := fs.String("json", "", `write the JSON report to this file ("-" = stdout, replacing the human output)`)
	minAccuracy := fs.Float64("min-accuracy", 0.75, "minimum accuracy required to pass (below it, exit 3)")
	repeat := fs.Int("repeat", 1, "route the whole prompt set this many times")

	positional, err := parseCommandFlags(fs, args)
	if err != nil {
		return err
	}
	if len(positional) > 0 {
		return usageErrorf("eval: unexpected argument %q", positional[0])
	}
	logger, err := global.logger(stderr)
	if err != nil {
		return err
	}
	// Flag values are validated before the configuration loads, so a bad
	// invocation can never reach the (billed) classifier.
	if *repeat < 1 {
		return usageErrorf("eval: --repeat must be at least 1, got %d", *repeat)
	}
	if math.IsNaN(*minAccuracy) || *minAccuracy < 0 || *minAccuracy > 1 {
		return usageErrorf("eval: --min-accuracy must be between 0 and 1, got %g", *minAccuracy)
	}

	app, err := buildApp(*configPath, logger)
	if err != nil {
		return err
	}
	defer closeApp(logger, app)

	// The model pool feeds the report's price fields only, so a failed fetch
	// is a warning, never a run failure: the accuracy checks and the exit
	// codes are untouched, and without a pool the price fields stay zero
	// (see eval.Options.Models). ListModels is the same catalog stage Route
	// uses, and the catalogs cache their fetch, so the live case costs one
	// round trip up front.
	ctx := context.Background()
	poolAvailable := true
	models, err := app.Router.ListModels(ctx, nil)
	if err != nil {
		logger.Warn("model pool unavailable; the report's price savings will read zero", "error", err)
		poolAvailable = false
		models = nil
	}

	report, err := eval.Run(ctx, app.Router, eval.LabeledPrompts(), eval.Options{
		MinAccuracy: *minAccuracy,
		Repeat:      *repeat,
		Logger:      logger,
		Models:      models,
	})
	if err != nil {
		return err
	}

	// The report is emitted before the exit checks: a failing run still
	// produces its artifact (and its human table), which is exactly what CI
	// wants to archive.
	if *jsonOut != "" {
		if err := writeEvalJSON(*jsonOut, stdout, report); err != nil {
			return err
		}
	}
	if *jsonOut != "-" {
		printEvalReport(stdout, report, poolAvailable)
	}

	if report.Evaluated == 0 {
		return fmt.Errorf("no prompts evaluated — classifier unavailable? (%d prompts skipped)", report.Skipped)
	}
	if !report.Passed {
		return &evalThresholdError{msg: fmt.Sprintf(
			"eval: accuracy %.2f below --min-accuracy %.2f (%d/%d decisions matched)",
			report.Accuracy, report.MinAccuracy, report.Matched, report.Evaluated)}
	}
	return nil
}

// writeEvalJSON writes the indented JSON report to stdout when dest is "-",
// otherwise to the named file (mode 0644: the report is a shareable CI
// artifact, not a secret).
func writeEvalJSON(dest string, stdout io.Writer, report eval.Report) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("encoding JSON: %w", err)
	}
	data = append(data, '\n')
	if dest == "-" {
		if _, err := stdout.Write(data); err != nil {
			return fmt.Errorf("writing output: %w", err)
		}
		return nil
	}
	if err := os.WriteFile(dest, data, 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", dest, err)
	}
	return nil
}

// printEvalReport writes the human-readable eval output: one table per run
// with its summary lines, followed — for multi-run invocations — by the
// cross-run aggregate summary. poolAvailable reports whether the model pool
// behind the price fields could be fetched; when it could not, the savings
// line says so instead of printing a zero that reads as "no savings".
func printEvalReport(w io.Writer, report eval.Report, poolAvailable bool) {
	for i, run := range report.Runs {
		if len(report.Runs) > 1 {
			_, _ = fmt.Fprintf(w, "run %d/%d\n\n", i+1, len(report.Runs))
		}
		printEvalTable(w, run.Results)
		printEvalSummary(w, "", run.Matched, run.Evaluated, run.Skipped, run.Accuracy, run.MeanSavings, poolAvailable)
		if i < len(report.Runs)-1 {
			_, _ = fmt.Fprintln(w)
		}
	}
	if len(report.Runs) > 1 {
		printEvalSummary(w, "aggregate: ", report.Matched, report.Evaluated, report.Skipped, report.Accuracy, report.MeanSavings, poolAvailable)
	}
}

// printEvalTable writes one run's per-prompt table: prompt preview, domain,
// domain confidence, complexity score, chosen model and the verdict — "yes"
// when the decision matched the expectations, "NO" when it did not, and
// "skipped" for a profileless (fail-open) decision, whose other columns show
// placeholders instead of meaningless zeros.
func printEvalTable(w io.Writer, results []eval.PromptResult) {
	_, _ = fmt.Fprintf(w, "%-67s  %-16s  %5s  %6s  %-35s  %s\n",
		"prompt", "domain", "conf", "compl", "model chosen", "as expected?")
	_, _ = fmt.Fprintln(w, strings.Repeat("-", evalTableSeparator))
	for _, result := range results {
		preview := truncateRunes(result.Prompt, promptPreviewWidth)
		if result.Skipped {
			_, _ = fmt.Fprintf(w, "%-67s  %-16s  %5s  %6s  %-35s  %s\n",
				preview, "(no profile)", "-", "-", "-", "skipped")
			continue
		}
		verdict := "NO"
		if result.Matched {
			verdict = "yes"
		}
		_, _ = fmt.Fprintf(w, "%-67s  %-16s  %5.2f  %6.2f  %-35s  %s\n",
			preview, result.Domain, result.DomainConfidence, result.ComplexityScore, result.ModelID, verdict)
	}
	_, _ = fmt.Fprintln(w, strings.Repeat("-", evalTableSeparator))
}

// printEvalSummary writes one set of summary lines: the match ratio with its
// accuracy, the mean price savings against the pool's frontier as a
// percentage — or an n/a line when no model pool was available — and, only
// when relevant, the skipped-prompt count. With prefix "aggregate: " the
// same lines summarise every repetition at once.
func printEvalSummary(w io.Writer, prefix string, matched, evaluated, skipped int, accuracy, meanSavings float64, poolAvailable bool) {
	_, _ = fmt.Fprintf(w, "%s%d/%d decisions matched expectations (accuracy %.2f)\n", prefix, matched, evaluated, accuracy)
	if poolAvailable {
		_, _ = fmt.Fprintf(w, "%smean price savings vs frontier: %.1f%%\n", prefix, meanSavings*100)
	} else {
		_, _ = fmt.Fprintf(w, "%smean price savings vs frontier: n/a (model pool unavailable)\n", prefix)
	}
	if skipped > 0 {
		_, _ = fmt.Fprintf(w, "%sskipped: %d\n", prefix, skipped)
	}
}

// truncateRunes truncates s to at most n runes (a code-point slice), leaving
// short strings untouched. Byte slicing could cut a multi-byte rune in half,
// and some prompts include an em dash.
func truncateRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return string([]rune(s)[:n])
}
