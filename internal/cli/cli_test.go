// These tests drive the CLI in-process. They are entirely offline: every
// configuration and catalog lives in t.TempDir, credentials are literal
// values, and the only network endpoint is an httptest server implementing
// the OpenAI chat-completions wire format. Nothing here touches OpenRouter,
// Jev or any other real service.
package cli_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/config"
	"github.com/vprprudhvi/jev-router/eval"
	"github.com/vprprudhvi/jev-router/internal/cli"
)

// run executes the CLI in-process and returns its exit code and streams.
func run(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := cli.Run(args, &stdout, &stderr)
	return code, stdout.String(), stderr.String()
}

// writeFile writes contents into dir/name and returns the full path.
func writeFile(t *testing.T, dir, name, contents string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("os.WriteFile(%q) = %v", path, err)
	}
	return path
}

// localCatalogYAML is a three-model, one-provider catalog. Blended prices are
// 0.003 (local-large), 0.0015 (local-mid) and 0.0003 (local-small): the
// models command prints them, and eval prices its savings from them.
const localCatalogYAML = `provider: local
models:
  - id: local-large
    context_length: 8192
    price_per_1k_prompt_tokens: 0.001
    price_per_1k_completion_tokens: 0.002
  - id: local-small
    context_length: 4096
    price_per_1k_prompt_tokens: 0.0001
    price_per_1k_completion_tokens: 0.0002
  - id: local-mid
    context_length: 16384
    price_per_1k_prompt_tokens: 0.0005
    price_per_1k_completion_tokens: 0.001
`

// localConfig writes a temp config with one OpenAI-wire provider named
// "local" (which also means no Jev classifier: only providers.openrouter
// builds one) plus the catalog it references. Model selection is explicit
// configuration, so the config carries a default model — local-small — which
// every classifier-less route falls back to (SourceDefault). extra is
// appended verbatim, so tests can add sections such as telemetry.
func localConfig(t *testing.T, baseURL, extra string) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "catalog.yaml", localCatalogYAML)
	configYAML := fmt.Sprintf(`providers:
  compatible:
    - name: local
      base_url: %s
      api_key: test-key
      catalog: catalog.yaml
routing:
  default_model:
    provider: local
    model: local-small
%s`, baseURL, extra)
	return writeFile(t, dir, "config.yaml", configYAML)
}

// fakeJevAnswersJSON mirrors classify/jev_test.go's valid answers fixture: the
// parser needs a choice+confidence domain, a score+confidence complexity and
// the three noul values, and ignores the extra fields the real endpoint
// returns.
const fakeJevAnswersJSON = `{
	"domain": {"choice": "code", "probabilities": {"code": 0.9}, "confidence": 0.9},
	"complexity": {"score": 1.4, "legend": {}, "probabilities": {}, "confidence": 0.75},
	"needs_long_context": {"noul": 0.1},
	"needs_vision": {"noul": 0.05},
	"latency_sensitive": {"noul": 0.6}
}`

// newFakeJev serves the systemone wire format the Jev classifier speaks, so a
// test configuration can point classifier.base_url at it instead of
// OpenRouter.
func newFakeJev(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/systemone" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"answers": %s}`, fakeJevAnswersJSON)
	}))
	t.Cleanup(server.Close)
	return server
}

// classifiedConfig writes a temp config whose classifier answers domain
// "code" from the fake Jev server: providers.openrouter is what makes Build
// construct a classifier at all (the classifier shares the OpenRouter key),
// and classifier.base_url redirects it to the local test endpoint. Routing maps
// code to an explicit two-entry chain and keeps a default model, so the
// decided chain is [openrouter/vendor/coder, local/local-mid] plus the
// appended default. Route only consults catalogs for metadata constraints,
// and these tests apply provider filters only, so the live OpenRouter
// catalog is never fetched: the suite stays offline.
func classifiedConfig(t *testing.T, jevURL string) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "catalog.yaml", localCatalogYAML)
	configYAML := fmt.Sprintf(`providers:
  openrouter:
    api_key: test-key
  compatible:
    - name: local
      base_url: http://127.0.0.1:0/v1
      api_key: test-key
      catalog: catalog.yaml
classifier:
  base_url: %s
  path: /api/v1/systemone
routing:
  models:
    code:
      - provider: openrouter
        model: vendor/coder
      - provider: local
        model: local-mid
  default_model:
    provider: local
    model: local-small
`, jevURL)
	return writeFile(t, dir, "config.yaml", configYAML)
}

func TestRunUsageErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"no arguments", nil},
		{"unknown command", []string{"frobnicate"}},
		{"route without prompt", []string{"route"}},
		{"route with only flags", []string{"route", "--json"}},
		{"complete without prompt", []string{"complete"}},
		{"validate without config", []string{"validate"}},
		{"bad log level", []string{"models", "--log-level", "loud"}},
		{"bad log format", []string{"models", "--log-format", "xml"}},
		{"unknown flag", []string{"route", "--frobnicate"}},
		{"serve with extra argument", []string{"serve", "extra"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			code, stdout, stderr := run(t, tc.args...)
			if code != 2 {
				t.Errorf("Run(%q) = %d, want 2", tc.args, code)
			}
			if stdout != "" {
				t.Errorf("Run(%q) wrote %q to stdout, want nothing", tc.args, stdout)
			}
			if !strings.Contains(stderr, "jev-router:") {
				t.Errorf("Run(%q) stderr = %q, want a \"jev-router: \" report", tc.args, stderr)
			}
		})
	}
}

// TestEvalUsageErrors covers the eval command's usage contract (it was an
// unknown command until the harness landed in P10). Every case below fails
// during flag parsing or value validation — before the configuration is
// loaded — so none of them can reach the network.
func TestEvalUsageErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"repeat zero", []string{"eval", "--repeat", "0"}, "--repeat"},
		{"repeat negative", []string{"eval", "--repeat", "-1"}, "--repeat"},
		{"repeat not a number", []string{"eval", "--repeat", "many"}, "repeat"},
		{"min accuracy negative", []string{"eval", "--min-accuracy", "-0.1"}, "min-accuracy"},
		{"min accuracy above one", []string{"eval", "--min-accuracy", "1.5"}, "min-accuracy"},
		{"min accuracy NaN", []string{"eval", "--min-accuracy", "NaN"}, "min-accuracy"},
		{"unexpected argument", []string{"eval", "extra"}, "unexpected argument"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			code, stdout, stderr := run(t, tc.args...)
			if code != 2 {
				t.Errorf("Run(%q) = %d, want 2", tc.args, code)
			}
			if stdout != "" {
				t.Errorf("Run(%q) stdout = %q, want nothing", tc.args, stdout)
			}
			if !strings.Contains(stderr, tc.want) {
				t.Errorf("Run(%q) stderr = %q, want a %q report", tc.args, stderr, tc.want)
			}
		})
	}
}

// TestEvalOfflineAllSkipped pins the classifier-less offline behaviour: with
// a static-catalog-only configuration there is no Jev classifier, every
// Route call fails open with a nil profile, so every prompt is skipped and
// the command exits 1 with a clear message instead of pretending to have
// evaluated anything. The human table still lists one "skipped" row per
// labeled prompt, so the failure is diagnosable. (A real run — with a
// classifier — makes billed calls; that path belongs to the gated eval runs,
// never this suite.)
func TestEvalOfflineAllSkipped(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	code, stdout, stderr := run(t, "eval", "--config", configPath, "--log-level", "error")
	if code != 1 {
		t.Fatalf("Run(eval) = %d, want 1 (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stderr, "no prompts evaluated") {
		t.Errorf("stderr = %q, want the no-prompts-evaluated report", stderr)
	}
	if !strings.Contains(stderr, "classifier unavailable") {
		t.Errorf("stderr = %q, want the classifier hint", stderr)
	}
	promptCount := len(eval.LabeledPrompts())
	if got := strings.Count(stdout, "skipped"); got != promptCount+1 {
		t.Errorf("stdout has %d \"skipped\" occurrences, want %d (one row per prompt plus the summary):\n%s",
			got, promptCount+1, stdout)
	}
	if !strings.Contains(stdout, "0/0 decisions matched expectations (accuracy 0.00)") {
		t.Errorf("stdout = %q, want the zero-evaluated summary", stdout)
	}
	// The model pool is available (the catalog is static) but nothing was
	// evaluated, so the mean savings stay zero.
	if !strings.Contains(stdout, "mean price savings vs frontier: 0.0%") {
		t.Errorf("stdout = %q, want the zero-savings summary", stdout)
	}
}

// TestEvalJSONReportFile exercises the --json file path offline, where it is
// reachable: with every prompt skipped the command still exits 1, but the
// report file is written first so CI can archive the failing run. The static
// catalog lets the ListModels price pool load, but skipped decisions carry no
// price data at all: every price field, and the mean savings, must read zero.
func TestEvalJSONReportFile(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	out := filepath.Join(t.TempDir(), "report.json")
	code, _, stderr := run(t, "eval", "--config", configPath, "--json", out, "--log-level", "error")
	if code != 1 {
		t.Fatalf("Run(eval --json) = %d, want 1 (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stderr, "no prompts evaluated") {
		t.Errorf("stderr = %q, want the no-prompts-evaluated report", stderr)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("reading the report file: %v", err)
	}
	var report eval.Report
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v\n%s", err, data)
	}
	if report.Evaluated != 0 {
		t.Errorf("report evaluated = %d, want 0", report.Evaluated)
	}
	if want := len(eval.LabeledPrompts()); report.Skipped != want {
		t.Errorf("report skipped = %d, want %d", report.Skipped, want)
	}
	if report.Passed {
		t.Error("report passed = true, want false when nothing was evaluated")
	}
	if len(report.Runs) != 1 {
		t.Errorf("report runs = %d, want 1 (the default --repeat)", len(report.Runs))
	}
	if report.MeanSavings != 0 {
		t.Errorf("report mean savings = %v, want 0 when every prompt was skipped", report.MeanSavings)
	}
	results := report.Runs[0].Results
	if len(results) != len(eval.LabeledPrompts()) {
		t.Fatalf("run results = %d, want %d", len(results), len(eval.LabeledPrompts()))
	}
	for i, result := range results {
		if result.ChosenBlendedPrice != 0 || result.MedianBlendedPrice != 0 ||
			result.FrontierBlendedPrice != 0 || result.SavingsVsFrontier != 0 {
			t.Errorf("result %d carries price data (chosen %v, median %v, frontier %v, savings %v), want zeros for a skipped decision",
				i, result.ChosenBlendedPrice, result.MedianBlendedPrice, result.FrontierBlendedPrice, result.SavingsVsFrontier)
		}
	}
}

// TestEvalJSONStdout exercises `--json -`: the report goes to stdout INSTEAD
// of the human table. It is reachable offline because the report is emitted
// before the exit checks (a failing run still produces its artifact), so the
// all-skipped configuration covers it end to end: exit 1, no human summary
// on stdout and a parseable JSON report there.
func TestEvalJSONStdout(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	code, stdout, stderr := run(t, "eval", "--config", configPath, "--json", "-", "--log-level", "error")
	if code != 1 {
		t.Fatalf("Run(eval --json -) = %d, want 1 (stderr: %s)", code, stderr)
	}
	if strings.Contains(stdout, "decisions matched expectations") {
		t.Errorf("stdout contains the human summary, want JSON only:\n%s", stdout)
	}
	var report eval.Report
	if err := json.Unmarshal([]byte(stdout), &report); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\n%s", err, stdout)
	}
	if report.Evaluated != 0 {
		t.Errorf("report evaluated = %d, want 0", report.Evaluated)
	}
	if want := len(eval.LabeledPrompts()); report.Skipped != want {
		t.Errorf("report skipped = %d, want %d", report.Skipped, want)
	}
	if report.MeanSavings != 0 {
		t.Errorf("report mean savings = %v, want 0", report.MeanSavings)
	}
}

// TestRunNeverPanicsOnFlagOnlyArguments feeds the argument shapes most likely
// to trip a parser: flags with no value, bare separators, dangling values.
// All must be usage errors, never panics or hangs.
func TestRunNeverPanicsOnFlagOnlyArguments(t *testing.T) {
	for _, args := range [][]string{
		{"--log-level"},
		{"--json"},
		{"-x"},
		{"--"},
		{"route", "--"},
		{"validate", "--config"},
	} {
		code, _, stderr := run(t, args...)
		if code != 2 {
			t.Errorf("Run(%q) = %d, want 2", args, code)
		}
		if stderr == "" {
			t.Errorf("Run(%q) produced no stderr report", args)
		}
	}
}

// TestHelpExitsZero documents the deliberate addition to the flag contract:
// -h/--help prints the usage on stdout and exits 0 (a help request is not a
// usage error).
func TestHelpExitsZero(t *testing.T) {
	code, stdout, stderr := run(t, "route", "--help")
	if code != 0 {
		t.Errorf("Run(route --help) = %d, want 0", code)
	}
	if !strings.Contains(stdout, "usage:") {
		t.Errorf("stdout = %q, want the usage summary", stdout)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty", stderr)
	}
}

func TestRouteHumanOutput(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	// --log-level error silences the fail-open warning that is expected here
	// (no OpenRouter provider means no Jev classifier, so routing fails open
	// onto the configured default model and logs a warning at the default
	// level).
	code, stdout, stderr := run(t, "route", "explain quantum tunnelling", "--config", configPath, "--log-level", "error")
	if code != 0 {
		t.Fatalf("Run(route) = %d, want 0 (stderr: %s)", code, stderr)
	}
	want := "model: local-small\n" +
		"provider: local\n" +
		"source: default\n" +
		"profile: none (classifier unavailable; default model used)\n" +
		"candidates:\n" +
		"  1. local/local-small\n"
	if stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
	if strings.Contains(stdout, "ranked") || strings.Contains(stdout, "$") {
		t.Errorf("stdout = %q, want neither the retired ranked list nor any price", stdout)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

func TestRouteJSONOutput(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	// Flags before the positional argument exercise the interspersed parser.
	code, stdout, stderr := run(t, "route", "--json", "--config", configPath, "--log-level", "error", "route this prompt")
	if code != 0 {
		t.Fatalf("Run(route --json) = %d, want 0 (stderr: %s)", code, stderr)
	}
	var decision modelrouter.RoutingDecision
	if err := json.Unmarshal([]byte(stdout), &decision); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\n%s", err, stdout)
	}
	if decision.ModelID != "local-small" {
		t.Errorf("ModelID = %q, want %q", decision.ModelID, "local-small")
	}
	if decision.Provider != "local" {
		t.Errorf("Provider = %q, want %q", decision.Provider, "local")
	}
	if decision.Source != modelrouter.SourceDefault {
		t.Errorf("Source = %q, want %q", decision.Source, modelrouter.SourceDefault)
	}
	wantCandidates := []modelrouter.ModelChoice{{Provider: "local", Model: "local-small"}}
	if !reflect.DeepEqual(decision.Candidates, wantCandidates) {
		t.Errorf("Candidates = %+v, want %+v", decision.Candidates, wantCandidates)
	}
	if decision.Profile != nil {
		t.Errorf("Profile = %+v, want nil (fail-open fallback)", decision.Profile)
	}
	// The retired ranked list must not reappear in the JSON shape.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(stdout), &fields); err != nil {
		t.Fatalf("stdout is not valid JSON: %v", err)
	}
	if _, present := fields["ranked"]; present {
		t.Errorf("JSON carries the retired \"ranked\" field:\n%s", stdout)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

// TestRouteFailOpenLogsWarning pins the default (info) logging behaviour:
// without a classifier the Router warns and the CLI emits it as a structured
// stderr log line — JSON, when --log-format=json asks for it.
func TestRouteFailOpenLogsWarning(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	code, _, stderr := run(t, "route", "--config", configPath, "--log-format", "json", "hello")
	if code != 0 {
		t.Fatalf("Run(route) = %d, want 0 (stderr: %s)", code, stderr)
	}
	line := strings.TrimSpace(stderr)
	if line == "" {
		t.Fatal("stderr = empty, want the fail-open classifier warning")
	}
	var record map[string]any
	if err := json.Unmarshal([]byte(line), &record); err != nil {
		t.Fatalf("stderr is not a JSON log line: %v (%q)", err, line)
	}
	if msg, _ := record["msg"].(string); !strings.Contains(msg, "classifier unavailable") {
		t.Errorf("log msg = %q, want the classifier fallback warning", msg)
	}
}

// TestRouteClassifiedOutput pins the classified path end to end: the fake
// Jev endpoint answers domain "code", so selection uses the configured code
// chain — with the default model appended as its last-resort fallback — and
// the human output reports source classification plus every candidate in
// order. Only provider filters are involved, so Route never consults the
// catalogs and the test stays offline.
func TestRouteClassifiedOutput(t *testing.T) {
	jev := newFakeJev(t)
	configPath := classifiedConfig(t, jev.URL)
	code, stdout, stderr := run(t, "route", "write a Go function", "--config", configPath, "--log-level", "error")
	if code != 0 {
		t.Fatalf("Run(route) = %d, want 0 (stderr: %s)", code, stderr)
	}
	want := "model: vendor/coder\n" +
		"provider: openrouter\n" +
		"source: classification\n" +
		"domain: code (confidence 0.90)\n" +
		"complexity: 1.40 (confidence 0.75)\n" +
		"candidates:\n" +
		"  1. openrouter/vendor/coder\n" +
		"  2. local/local-mid\n" +
		"  3. local/local-small\n"
	if stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

// TestRouteProviderFilterChangesTheChoice checks that --excluded-providers
// filters the selected chain before the first candidate is picked: excluding
// openrouter drops vendor/coder, so local-mid leads the two remaining
// candidates.
func TestRouteProviderFilterChangesTheChoice(t *testing.T) {
	jev := newFakeJev(t)
	configPath := classifiedConfig(t, jev.URL)
	code, stdout, stderr := run(t, "route", "--config", configPath, "--excluded-providers", "openrouter", "--log-level", "error", "write a Go function")
	if code != 0 {
		t.Fatalf("Run(route --excluded-providers) = %d, want 0 (stderr: %s)", code, stderr)
	}
	want := "model: local-mid\n" +
		"provider: local\n" +
		"source: classification\n" +
		"domain: code (confidence 0.90)\n" +
		"complexity: 1.40 (confidence 0.75)\n" +
		"candidates:\n" +
		"  1. local/local-mid\n" +
		"  2. local/local-small\n"
	if stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

// TestRouteMetadataConstraintKeepsTheDefaultModel exercises the catalog-backed
// constraint path offline: --min-context makes Route fetch the (static)
// catalogs, and the default model's 4096-token window satisfies the minimum,
// so the decision is unchanged.
func TestRouteMetadataConstraintKeepsTheDefaultModel(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	code, stdout, stderr := run(t, "route", "--config", configPath, "--min-context", "4096", "--log-level", "error", "hello")
	if code != 0 {
		t.Fatalf("Run(route --min-context 4096) = %d, want 0 (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stdout, "model: local-small") {
		t.Errorf("stdout = %q, want local-small to survive the context minimum", stdout)
	}
}

func TestRouteNoCandidateExitsOne(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	// The constraint needs catalog metadata, so Route fetches the (static)
	// catalog and local-small's 4096-token window eliminates the only
	// candidate: the report names the configured candidate.
	code, stdout, stderr := run(t, "route", "--config", configPath, "--min-context", "8192", "--log-level", "error", "hello")
	if code != 1 {
		t.Errorf("Run(route impossible constraints) = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "no candidate") {
		t.Errorf("stderr = %q, want the no-candidate report", stderr)
	}
	if !strings.Contains(stderr, "local/local-small") {
		t.Errorf("stderr = %q, want the configured candidate named", stderr)
	}
}

// TestRouteEnvironmentOnlyModeNeedsDefaultModel pins the environment-only
// mode through the CLI: without --config (and without JEV_ROUTER_CONFIG) the
// configuration is built from defaults plus the environment, where v2 takes
// the default model from JEV_ROUTER_DEFAULT_MODEL — the validation error
// names it, so the fix is obvious. Validation fails before any request is
// made, so the test stays offline.
func TestRouteEnvironmentOnlyModeNeedsDefaultModel(t *testing.T) {
	t.Setenv(config.ConfigPathEnv, "")
	t.Setenv("OPENROUTER_API_KEY", "test-key")
	t.Setenv(config.DefaultModelEnv, "")
	t.Setenv(config.DefaultProviderEnv, "")

	code, stdout, stderr := run(t, "route", "hello", "--log-level", "error")
	if code != 1 {
		t.Fatalf("Run(route) = %d, want 1 (stderr: %s)", code, stderr)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "JEV_ROUTER_DEFAULT_MODEL") {
		t.Errorf("stderr = %q, want the env-only hint naming JEV_ROUTER_DEFAULT_MODEL", stderr)
	}
}

func TestModelsJSONOutput(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	code, stdout, stderr := run(t, "models", "--config", configPath, "--json")
	if code != 0 {
		t.Fatalf("Run(models --json) = %d, want 0 (stderr: %s)", code, stderr)
	}
	var document struct {
		Models []modelrouter.ModelInfo `json:"models"`
	}
	if err := json.Unmarshal([]byte(stdout), &document); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\n%s", err, stdout)
	}
	if len(document.Models) != 3 {
		t.Fatalf("len(models) = %d, want 3", len(document.Models))
	}
	seen := map[string]bool{"local-small": false, "local-mid": false, "local-large": false}
	for _, model := range document.Models {
		if model.Provider != "local" {
			t.Errorf("Provider = %q, want %q", model.Provider, "local")
		}
		if _, ok := seen[model.ID]; ok {
			seen[model.ID] = true
		}
	}
	for id, found := range seen {
		if !found {
			t.Errorf("model %q missing from the JSON output", id)
		}
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

func TestModelsHumanOutput(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	code, stdout, stderr := run(t, "models", "--config", configPath)
	if code != 0 {
		t.Fatalf("Run(models) = %d, want 0 (stderr: %s)", code, stderr)
	}
	for _, want := range []string{
		"local\tlocal-small\t$0.0003/1k\t4096\tfalse",
		"local\tlocal-mid\t$0.0015/1k\t16384\tfalse",
		"local\tlocal-large\t$0.003/1k\t8192\tfalse",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout missing %q:\n%s", want, stdout)
		}
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

func TestValidateGoodConfig(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "")
	code, stdout, stderr := run(t, "validate", "--config", configPath)
	if code != 0 {
		t.Fatalf("Run(validate) = %d, want 0 (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stdout, "OK") || !strings.Contains(stdout, "local") {
		t.Errorf("stdout = %q, want the OK report naming local", stdout)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

func TestValidateBadConfig(t *testing.T) {
	dir := t.TempDir()
	// Bedrock without a region: config.Validate rejects it.
	path := writeFile(t, dir, "config.yaml", "providers:\n  bedrock:\n    catalog: catalog.yaml\n")
	code, stdout, stderr := run(t, "validate", "--config", path)
	if code != 1 {
		t.Errorf("Run(validate bad config) = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "region") {
		t.Errorf("stderr = %q, want the missing-region report", stderr)
	}
}

func TestValidateNonexistentConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.yaml")
	code, stdout, stderr := run(t, "validate", "--config", path)
	if code != 1 {
		t.Errorf("Run(validate missing file) = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "config file") {
		t.Errorf("stderr = %q, want a config-file report", stderr)
	}
}

// TestValidateMalformedCatalog relies on validate running Build, not just
// Load: Load only stats a catalog file, while Build parses it, so a malformed
// catalog is a validation failure (exit 1). The config needs a model mapping
// to get past the first validation pass.
func TestValidateMalformedCatalog(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "broken.yaml", `provider: local
models:
  - id: broken
    context_length: 8192
    price_per_1k_prompt_tokens: -1
`)
	path := writeFile(t, dir, "config.yaml", `providers:
  compatible:
    - name: local
      base_url: http://127.0.0.1:0/v1
      api_key: test-key
      catalog: broken.yaml
routing:
  default_model:
    provider: local
    model: broken
`)
	code, stdout, stderr := run(t, "validate", "--config", path)
	if code != 1 {
		t.Errorf("Run(validate malformed catalog) = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "negative price") {
		t.Errorf("stderr = %q, want the catalog price report", stderr)
	}
}

// TestValidateRoutingErrors checks the v2 model-mapping validation through
// the CLI: the mapping requirement, the five-domain key set, the
// configured-provider references and the strict decoder that rejects the
// retired scorer keys all surface as exit-1 reports naming the offending
// field.
func TestValidateRoutingErrors(t *testing.T) {
	base := `providers:
  compatible:
    - name: local
      base_url: http://127.0.0.1:0/v1
      api_key: test-key
      catalog: catalog.yaml
`
	tests := []struct {
		name    string
		routing string
		want    string
	}{
		{"missing mapping", "", "routing.models or routing.default_model"},
		{
			"unknown domain",
			"routing:\n  models:\n    bogus:\n      - provider: local\n        model: local-small\n",
			`invalid domain "bogus"`,
		},
		{
			"unconfigured provider",
			"routing:\n  default_model:\n    provider: gemini\n    model: gemini-model\n",
			`routing.default_model.provider "gemini": provider is not configured`,
		},
		{
			"removed scorer key",
			"routing:\n  weight_domain: 0.9\n",
			"weight_domain",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeFile(t, dir, "catalog.yaml", localCatalogYAML)
			path := writeFile(t, dir, "config.yaml", base+tc.routing)
			code, stdout, stderr := run(t, "validate", "--config", path)
			if code != 1 {
				t.Errorf("Run(validate) = %d, want 1 (stderr: %s)", code, stderr)
			}
			if stdout != "" {
				t.Errorf("stdout = %q, want nothing", stdout)
			}
			if !strings.Contains(stderr, tc.want) {
				t.Errorf("stderr = %q, want a %q report", stderr, tc.want)
			}
		})
	}
}

// TestValidateMultiProviderConfig validates a configuration that declares
// every provider kind with all referenced catalogs on disk; it must be
// offline (Build constructs lazy clients and loads local catalogs only).
func TestValidateMultiProviderConfig(t *testing.T) {
	// The file's default_model must win over any ambient environment override.
	t.Setenv(config.DefaultModelEnv, "")
	t.Setenv(config.DefaultProviderEnv, "")

	dir := t.TempDir()
	catalog := func(provider, model string) string {
		return "provider: " + provider + "\nmodels:\n  - id: " + model + "\n" +
			"    context_length: 8192\n    price_per_1k_prompt_tokens: 0.001\n" +
			"    price_per_1k_completion_tokens: 0.002\n"
	}
	for provider, model := range map[string]string{
		"openai":    "openai-small",
		"anthropic": "anthropic-small",
		"gemini":    "gemini-small",
		"bedrock":   "bedrock-small",
		"groq":      "groq-small",
	} {
		writeFile(t, dir, provider+".yaml", catalog(provider, model))
	}
	path := writeFile(t, dir, "config.yaml", `providers:
  openrouter:
    api_key: test-openrouter-key
  openai:
    api_key: test-openai-key
    catalog: openai.yaml
  anthropic:
    api_key: test-anthropic-key
    catalog: anthropic.yaml
  gemini:
    api_key: test-gemini-key
    catalog: gemini.yaml
  bedrock:
    region: us-east-1
    catalog: bedrock.yaml
  compatible:
    - name: groq
      base_url: https://api.groq.com/openai/v1
      api_key: test-groq-key
      catalog: groq.yaml
routing:
  models:
    code:
      - provider: groq
        model: groq-small
  default_model:
    provider: openai
    model: openai-small
`)

	code, stdout, stderr := run(t, "validate", "--config", path)
	if code != 0 {
		t.Fatalf("Run(validate) = %d, want 0 (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stdout, "anthropic, bedrock, gemini, groq, openai, openrouter") {
		t.Errorf("stdout = %q, want every provider in sorted order", stdout)
	}
	if !strings.Contains(stdout, "(6 catalogs)") {
		t.Errorf("stdout = %q, want the catalog count", stdout)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

func TestVersion(t *testing.T) {
	code, stdout, stderr := run(t, "version")
	if code != 0 {
		t.Errorf("Run(version) = %d, want 0", code)
	}
	if !strings.Contains(stdout, "jev-router") {
		t.Errorf("stdout = %q, want the version line", stdout)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty", stderr)
	}
}

// TestCompleteMissingPrompt is the usage-only completion test (no network).
func TestCompleteMissingPrompt(t *testing.T) {
	code, stdout, stderr := run(t, "complete")
	if code != 2 {
		t.Errorf("Run(complete) = %d, want 2", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if stderr == "" {
		t.Error("stderr = empty, want a usage report")
	}
}

// --- completion tests against a fake OpenAI-wire endpoint ------------------

const fakeCompletionJSON = `{
	"id": "chatcmpl-1",
	"object": "chat.completion",
	"created": 1,
	"model": "local-small",
	"choices": [{
		"index": 0,
		"message": {"role": "assistant", "content": "Hello!"},
		"finish_reason": "stop"
	}],
	"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
}`

// fakeOpenAI is an httptest-backed OpenAI-wire endpoint. It records the last
// request body and answers chat completions.
type fakeOpenAI struct {
	server *httptest.Server

	mu   sync.Mutex
	body map[string]any
}

func newFakeOpenAI(t *testing.T) *fakeOpenAI {
	t.Helper()
	fake := &fakeOpenAI{}
	fake.server = httptest.NewServer(http.HandlerFunc(fake.handle))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *fakeOpenAI) handle(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/chat/completions" {
		http.NotFound(w, r)
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	f.body = body
	f.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, fakeCompletionJSON)
}

// lastBody returns the most recent decoded request body.
func (f *fakeOpenAI) lastBody(t *testing.T) map[string]any {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.body == nil {
		t.Fatal("no completion request recorded")
	}
	return f.body
}

// requestMessages decodes body["messages"] into a comparable shape.
func requestMessages(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	raw, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("messages = %#v, want a list", body["messages"])
	}
	out := make([]map[string]any, 0, len(raw))
	for _, entry := range raw {
		message, ok := entry.(map[string]any)
		if !ok {
			t.Fatalf("message = %#v, want an object", entry)
		}
		out = append(out, message)
	}
	return out
}

func TestCompleteHumanOutput(t *testing.T) {
	fake := newFakeOpenAI(t)
	configPath := localConfig(t, fake.server.URL+"/v1", "")
	code, stdout, stderr := run(t, "complete", "say hello", "--config", configPath, "--log-level", "error")
	if code != 0 {
		t.Fatalf("Run(complete) = %d, want 0 (stderr: %s)", code, stderr)
	}
	if stdout != "Hello!\n" {
		t.Errorf("stdout = %q, want %q", stdout, "Hello!\n")
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}

	body := fake.lastBody(t)
	if body["model"] != "local-small" {
		t.Errorf("request model = %v, want %q", body["model"], "local-small")
	}
	want := []map[string]any{{"role": "user", "content": "say hello"}}
	if got := requestMessages(t, body); !reflect.DeepEqual(got, want) {
		t.Errorf("request messages = %#v, want %#v", got, want)
	}
}

func TestCompleteJSONOutput(t *testing.T) {
	fake := newFakeOpenAI(t)
	configPath := localConfig(t, fake.server.URL+"/v1", "")
	// Flag before the positional argument: interspersed parsing again.
	code, stdout, stderr := run(t, "complete", "--json", "--config", configPath, "--log-level", "error", "say hello")
	if code != 0 {
		t.Fatalf("Run(complete --json) = %d, want 0 (stderr: %s)", code, stderr)
	}
	var result modelrouter.CompletionResult
	if err := json.Unmarshal([]byte(stdout), &result); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\n%s", err, stdout)
	}
	if result.Content != "Hello!" {
		t.Errorf("Content = %q, want %q", result.Content, "Hello!")
	}
	if result.Model != "local-small" || result.Provider != "local" {
		t.Errorf("model/provider = %q/%q, want local-small/local", result.Model, result.Provider)
	}
	if result.Usage == nil || result.Usage.TotalTokens != 5 {
		t.Errorf("Usage = %+v, want total 5 tokens", result.Usage)
	}
	if result.RoutingDecision.ModelID != "local-small" {
		t.Errorf("RoutingDecision.ModelID = %q, want %q", result.RoutingDecision.ModelID, "local-small")
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty on success", stderr)
	}
}

func TestCompleteSystemMessage(t *testing.T) {
	fake := newFakeOpenAI(t)
	configPath := localConfig(t, fake.server.URL+"/v1", "")
	code, _, stderr := run(t, "complete", "say hello", "--config", configPath, "--system", "be brief", "--log-level", "error")
	if code != 0 {
		t.Fatalf("Run(complete --system) = %d, want 0 (stderr: %s)", code, stderr)
	}
	got := requestMessages(t, fake.lastBody(t))
	want := []map[string]any{
		{"role": "system", "content": "be brief"},
		{"role": "user", "content": "say hello"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("request messages = %#v, want %#v", got, want)
	}
}

func TestCompleteMaxTokensAndTemperature(t *testing.T) {
	fake := newFakeOpenAI(t)
	configPath := localConfig(t, fake.server.URL+"/v1", "")
	code, _, stderr := run(t, "complete", "say hello", "--config", configPath,
		"--max-tokens", "7", "--temperature", "0.25", "--log-level", "error")
	if code != 0 {
		t.Fatalf("Run(complete --max-tokens --temperature) = %d, want 0 (stderr: %s)", code, stderr)
	}
	body := fake.lastBody(t)
	if body["max_tokens"] != float64(7) {
		t.Errorf("request max_tokens = %v, want 7", body["max_tokens"])
	}
	if body["temperature"] != 0.25 {
		t.Errorf("request temperature = %v, want 0.25", body["temperature"])
	}
}

// TestCompleteStreamFlagRemoved pins the flag contract after streaming was
// dropped from the core: --stream no longer exists, so the flag package
// rejects it as an unknown flag — exit 2 with the usage summary on stderr,
// exactly like any other unknown flag.
func TestCompleteStreamFlagRemoved(t *testing.T) {
	code, stdout, stderr := run(t, "complete", "say hello", "--stream")
	if code != 2 {
		t.Errorf("Run(complete --stream) = %d, want 2", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "flag provided but not defined: -stream") {
		t.Errorf("stderr = %q, want the unknown-flag report naming --stream", stderr)
	}
	if !strings.Contains(stderr, "usage:") {
		t.Errorf("stderr = %q, want the usage summary", stderr)
	}
}

// --- serve -----------------------------------------------------------------

func TestServeConfigErrors(t *testing.T) {
	t.Run("nonexistent config", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "missing.yaml")
		code, stdout, stderr := run(t, "serve", "--config", path)
		if code != 1 {
			t.Errorf("Run(serve missing config) = %d, want 1", code)
		}
		if stdout != "" {
			t.Errorf("stdout = %q, want nothing", stdout)
		}
		if !strings.Contains(stderr, "config file") {
			t.Errorf("stderr = %q, want a config-file report", stderr)
		}
	})
	t.Run("invalid config", func(t *testing.T) {
		dir := t.TempDir()
		path := writeFile(t, dir, "config.yaml", "providers:\n  bedrock:\n    catalog: catalog.yaml\n")
		code, _, stderr := run(t, "serve", "--config", path)
		if code != 1 {
			t.Errorf("Run(serve invalid config) = %d, want 1", code)
		}
		if !strings.Contains(stderr, "region") {
			t.Errorf("stderr = %q, want the missing-region report", stderr)
		}
	})
	t.Run("addr flag is accepted", func(t *testing.T) {
		// Parse-level check: the flag must be known; the command still fails
		// at config loading, so it never starts listening (no hang).
		path := filepath.Join(t.TempDir(), "missing.yaml")
		code, _, stderr := run(t, "serve", "--config", path, "--addr", "127.0.0.1:9999")
		if code != 1 {
			t.Errorf("Run(serve --addr) = %d, want 1", code)
		}
		if strings.Contains(stderr, "flag provided but not defined") {
			t.Errorf("stderr = %q, --addr must be a known flag", stderr)
		}
	})
}

// TestServeRejectsUnknownTelemetryValue covers the mapping failure end to
// end: the config loads, but telemetry.enabled is neither auto/on/off (nor
// their accepted synonyms), so serve exits 1 before starting the server.
func TestServeRejectsUnknownTelemetryValue(t *testing.T) {
	configPath := localConfig(t, "http://127.0.0.1:0/v1", "\ntelemetry:\n  enabled: sometimes\n")
	code, stdout, stderr := run(t, "serve", "--config", configPath)
	if code != 1 {
		t.Errorf("Run(serve unknown telemetry) = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "telemetry.enabled") {
		t.Errorf("stderr = %q, want the telemetry mapping report", stderr)
	}
}

func TestMapTelemetryEnabled(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"", "auto"},
		{"auto", "auto"},
		{"on", "true"},
		{"off", "false"},
		{"true", "true"},
		{"false", "false"},
	}
	for _, tc := range tests {
		got, err := cli.MapTelemetryEnabled(tc.in)
		if err != nil {
			t.Errorf("MapTelemetryEnabled(%q) = error %v, want %q", tc.in, err, tc.want)
			continue
		}
		if got != tc.want {
			t.Errorf("MapTelemetryEnabled(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	// telemetry.Setup is strict: anything else must be rejected, not
	// silently defaulted.
	if got, err := cli.MapTelemetryEnabled("sometimes"); err == nil {
		t.Errorf("MapTelemetryEnabled(%q) = %q, want an error", "sometimes", got)
	}
}
