// This file implements the route, models, validate and version commands.
package cli

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strconv"
	"strings"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/config"
)

// runRoute implements `jev-router route <prompt> [flags]`: it loads the
// configuration, routes the prompt and prints the decision — indented JSON
// with --json, otherwise a short human summary.
func runRoute(args []string, stdout, stderr io.Writer) error {
	fs := newFlagSet("route")
	global := registerGlobalFlags(fs)
	configPath := fs.String("config", "", "configuration file (default: $JEV_ROUTER_CONFIG, else environment-only defaults)")
	constraints := registerConstraintFlags(fs)
	asJSON := fs.Bool("json", false, "print the routing decision as indented JSON")

	positional, err := parseCommandFlags(fs, args)
	if err != nil {
		return err
	}
	logger, err := global.logger(stderr)
	if err != nil {
		return err
	}
	prompt := strings.Join(positional, " ")
	if prompt == "" {
		return usageErrorf("route: missing prompt")
	}

	app, err := buildApp(*configPath, logger)
	if err != nil {
		return err
	}
	defer closeApp(logger, app)

	decision, err := app.Router.Route(context.Background(), prompt, constraints.routing(fs))
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(stdout, decision)
	}
	printDecision(stdout, decision)
	return nil
}

// printDecision writes the human-readable route output: the winning model,
// its provider, whether selection used a classified chain or the default
// model, the classification profile (or the fail-open note) and the full
// ordered candidate chain Complete will attempt. There are no prices or
// scores to print: selection is explicit configuration, not ranking.
func printDecision(w io.Writer, decision modelrouter.RoutingDecision) {
	_, _ = fmt.Fprintf(w, "model: %s\n", decision.ModelID)
	_, _ = fmt.Fprintf(w, "provider: %s\n", decision.Provider)
	_, _ = fmt.Fprintf(w, "source: %s\n", decision.Source)
	if profile := decision.Profile; profile != nil {
		_, _ = fmt.Fprintf(w, "domain: %s (confidence %.2f)\n", profile.Domain, profile.DomainConfidence)
		_, _ = fmt.Fprintf(w, "complexity: %.2f (confidence %.2f)\n", profile.ComplexityScore, profile.ComplexityConfidence)
	} else {
		_, _ = fmt.Fprintln(w, "profile: none (classifier unavailable; default model used)")
	}
	_, _ = fmt.Fprintln(w, "candidates:")
	for i, candidate := range decision.Candidates {
		_, _ = fmt.Fprintf(w, "  %d. %s/%s\n", i+1, candidate.Provider, candidate.Model)
	}
}

// formatPrice renders a per-1k price with six significant digits: enough to
// keep the usual 0.0001–0.01 values readable, while hiding binary
// representation noise (0.0001+0.0002 prints as 0.00030000000000000003
// with the default %g). JSON output keeps the exact float values.
func formatPrice(price float64) string {
	return strconv.FormatFloat(price, 'g', 6, 64)
}

// modelsDocument is the --json shape of `jev-router models`.
type modelsDocument struct {
	Models []modelrouter.ModelInfo `json:"models"`
}

// runModels implements `jev-router models [flags]`: it lists the models the
// configured catalogs serve after applying the constraint flags.
func runModels(args []string, stdout, stderr io.Writer) error {
	fs := newFlagSet("models")
	global := registerGlobalFlags(fs)
	configPath := fs.String("config", "", "configuration file (default: $JEV_ROUTER_CONFIG, else environment-only defaults)")
	constraints := registerConstraintFlags(fs)
	asJSON := fs.Bool("json", false, "print the model list as JSON")

	positional, err := parseCommandFlags(fs, args)
	if err != nil {
		return err
	}
	if len(positional) > 0 {
		return usageErrorf("models: unexpected argument %q", positional[0])
	}
	logger, err := global.logger(stderr)
	if err != nil {
		return err
	}

	app, err := buildApp(*configPath, logger)
	if err != nil {
		return err
	}
	defer closeApp(logger, app)

	models, err := app.Router.ListModels(context.Background(), constraints.routing(fs))
	if err != nil {
		return err
	}
	if *asJSON {
		if models == nil {
			// An empty JSON array, not null: consumers can iterate unconditionally.
			models = []modelrouter.ModelInfo{}
		}
		return printJSON(stdout, modelsDocument{Models: models})
	}
	printModels(stdout, models)
	return nil
}

// printModels writes the human-readable model list: one tab-separated line
// per model (provider, id, blended price per 1k tokens, context, vision).
func printModels(w io.Writer, models []modelrouter.ModelInfo) {
	for _, model := range models {
		_, _ = fmt.Fprintf(w, "%s\t%s\t$%s/1k\t%d\t%t\n",
			model.Provider, model.ID, formatPrice(model.BlendedPricePer1KTokens()),
			model.ContextLength, model.SupportsVision)
	}
}

// runValidate implements `jev-router validate --config PATH`: it loads and
// assembles the configuration (Load + Build), which checks the routing
// policy, every provider's credentials and every catalog file. Build is
// deliberately included — Load only stats catalog files, while Build parses
// them — and it stays offline: network components construct lazy clients.
func runValidate(args []string, stdout, stderr io.Writer) error {
	fs := newFlagSet("validate")
	global := registerGlobalFlags(fs)
	configPath := fs.String("config", "", "configuration file to validate (required)")

	positional, err := parseCommandFlags(fs, args)
	if err != nil {
		return err
	}
	if len(positional) > 0 {
		return usageErrorf("validate: unexpected argument %q", positional[0])
	}
	if *configPath == "" {
		return usageErrorf("validate: --config is required")
	}
	logger, err := global.logger(stderr)
	if err != nil {
		return err
	}

	app, err := buildApp(*configPath, logger)
	if err != nil {
		return err
	}
	defer closeApp(logger, app)

	_, _ = fmt.Fprintf(stdout, "OK: providers: %s (%d catalogs)\n",
		strings.Join(app.Providers, ", "), len(app.Providers))
	return nil
}

// runVersion implements `jev-router version`. The global flags are parsed
// (and validated) on every subcommand, so `version --log-level loud` is a
// usage error rather than a silent success.
func runVersion(args []string, stdout io.Writer) error {
	fs := newFlagSet("version")
	global := registerGlobalFlags(fs)
	if _, err := parseCommandFlags(fs, args); err != nil {
		return err
	}
	if _, err := global.logger(io.Discard); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "jev-router %s\n", version)
	return nil
}

// buildApp loads the configuration at path (empty means config.Load(""):
// $JEV_ROUTER_CONFIG, else environment-only defaults) and assembles it with
// the command's logger threaded through every component.
func buildApp(path string, logger *slog.Logger) (*config.App, error) {
	f, err := config.Load(path)
	if err != nil {
		return nil, err
	}
	return config.Build(f, config.WithLogger(logger))
}
