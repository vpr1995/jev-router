// Package cli implements the jev-router command-line interface.
//
// Run executes exactly one subcommand and returns the process exit code; it
// never calls os.Exit, so every command is testable in-process, and it reads
// state only through os.Getenv, so tests can isolate configuration with
// t.Setenv. Arguments are argv-style (os.Args[1:]).
//
// Exit codes: 0 success, 1 runtime/API failure, 2 usage error, 3 eval
// accuracy below --min-accuracy. Failures are reported on stderr as
// "jev-router: <err>"; usage errors additionally print the command summary,
// which -h/--help prints to stdout instead. Operational logs go to stderr
// through the *slog.Logger configured by the global --log-level/--log-format
// flags, so stdout carries command output only and JSON output stays
// pipeable.
//
// Commands: route, complete, models, validate, serve, eval, version.
package cli

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"strings"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/config"
)

// version is reported by `jev-router version`. A constant (not a link-time
// variable) because release engineering is out of scope; a
// later -ldflags build can override it without changing this interface.
const version = "dev"

// ErrUsage marks command-line usage errors: a missing or unknown command,
// missing arguments, unknown flags and invalid flag values. Run maps them to
// exit code 2. Internal command functions return errors wrapping ErrUsage
// through usageError, so callers of the package can classify failures with
// errors.Is.
var ErrUsage = errors.New("usage error")

// usageError is a usage failure whose message is already user-ready: Error
// returns just the message (Run prints it after the "jev-router: " prefix),
// and Is reports ErrUsage without polluting the message.
type usageError struct{ msg string }

func (e *usageError) Error() string { return e.msg }

func (e *usageError) Is(target error) bool { return target == ErrUsage }

// usageErrorf builds a usage error.
func usageErrorf(format string, args ...any) error {
	return &usageError{msg: fmt.Sprintf(format, args...)}
}

// usageText is printed after a usage error (stderr) and for -h/--help
// (stdout).
const usageText = `usage: jev-router <command> [flags]

commands:
  route <prompt>       route a prompt to the best model
  complete <prompt>    route, then complete
  models               list the routable models
  validate --config PATH
                       validate a configuration file and its catalogs
  serve                serve the HTTP API
  eval                 run the labeled-prompt routing evaluation
  version              print the version

global flags: --log-level debug|info|warn|error, --log-format text|json
`

// Run executes one CLI invocation and returns its exit code (0, 1, 2 or 3;
// see the package documentation). stdout receives command output, stderr
// receives error reports and logs. Nil writers are tolerated and discarded.
func Run(args []string, stdout, stderr io.Writer) int {
	if stdout == nil {
		stdout = io.Discard
	}
	if stderr == nil {
		stderr = io.Discard
	}

	if len(args) == 0 {
		return finish(stdout, stderr, usageErrorf("missing command"))
	}

	command, rest := args[0], args[1:]
	var err error
	switch command {
	case "route":
		err = runRoute(rest, stdout, stderr)
	case "complete":
		err = runComplete(rest, stdout, stderr)
	case "models":
		err = runModels(rest, stdout, stderr)
	case "validate":
		err = runValidate(rest, stdout, stderr)
	case "serve":
		err = runServe(rest, stdout, stderr)
	case "eval":
		err = runEval(rest, stdout, stderr)
	case "version":
		err = runVersion(rest, stdout)
	default:
		err = usageErrorf("unknown command %q", command)
	}
	return finish(stdout, stderr, err)
}

// finish maps a command result onto the exit codes: nil is 0; -h/--help
// (flag.ErrHelp) prints the usage to stdout and exits 0; a usage error prints
// the message and the usage to stderr and exits 2; everything else is a
// runtime failure, reported on stderr, exit 1.
func finish(stdout, stderr io.Writer, err error) int {
	switch {
	case err == nil:
		return 0
	case errors.Is(err, flag.ErrHelp):
		_, _ = fmt.Fprint(stdout, usageText)
		return 0
	case errors.Is(err, ErrUsage):
		_, _ = fmt.Fprintf(stderr, "jev-router: %v\n", err)
		_, _ = fmt.Fprint(stderr, usageText)
		return 2
	case errors.Is(err, errEvalThreshold):
		_, _ = fmt.Fprintf(stderr, "jev-router: %v\n", err)
		return 3
	default:
		_, _ = fmt.Fprintf(stderr, "jev-router: %v\n", err)
		return 1
	}
}

// newFlagSet returns a FlagSet for one command. Output is discarded because
// Run formats every error itself (and tests assert on stderr).
func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return fs
}

// globalFlags are the flags every subcommand accepts.
type globalFlags struct {
	level  *string
	format *string
}

// registerGlobalFlags registers --log-level and --log-format on fs.
func registerGlobalFlags(fs *flag.FlagSet) *globalFlags {
	return &globalFlags{
		level:  fs.String("log-level", "info", "log level: debug, info, warn or error"),
		format: fs.String("log-format", "text", "log format: text or json"),
	}
}

// logger builds the stderr *slog.Logger selected by the global flags. Invalid
// values are usage errors (exit 2).
func (g *globalFlags) logger(stderr io.Writer) (*slog.Logger, error) {
	level, err := parseLevel(*g.level)
	if err != nil {
		return nil, err
	}
	options := &slog.HandlerOptions{Level: level}
	var handler slog.Handler
	switch *g.format {
	case "text":
		handler = slog.NewTextHandler(stderr, options)
	case "json":
		handler = slog.NewJSONHandler(stderr, options)
	default:
		return nil, usageErrorf("invalid --log-format %q (want %q or %q)", *g.format, "text", "json")
	}
	return slog.New(handler), nil
}

// parseLevel maps the --log-level value onto a slog.Level.
func parseLevel(value string) (slog.Level, error) {
	switch value {
	case "debug":
		return slog.LevelDebug, nil
	case "info":
		return slog.LevelInfo, nil
	case "warn":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, usageErrorf("invalid --log-level %q (want %q, %q, %q or %q)",
			value, "debug", "info", "warn", "error")
	}
}

// parseCommandFlags parses fs against args, allowing flags and positional
// arguments to be interleaved: the flag package stops at the first
// positional, while the CLI must accept `route "prompt" --json` as well as
// `route --json "prompt"`. -h/--help surfaces as flag.ErrHelp unchanged, and
// every other parse failure becomes a usage error.
func parseCommandFlags(fs *flag.FlagSet, args []string) ([]string, error) {
	var positional []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, normalizeFlagError(err)
		}
		args = fs.Args()
		if len(args) == 0 {
			return positional, nil
		}
		positional = append(positional, args[0])
		args = args[1:]
	}
}

// normalizeFlagError keeps flag.ErrHelp recognisable (it turns into the usage
// on stdout) and wraps everything else (unknown flag, missing value, ...) as
// a usage error.
func normalizeFlagError(err error) error {
	if errors.Is(err, flag.ErrHelp) {
		return err
	}
	return usageErrorf("%v", err)
}

// constraintFlags holds the routing-constraint flags shared by route,
// complete and models.
type constraintFlags struct {
	needsVision       *bool
	minContext        *int
	maxPrice          *float64
	providers         *string
	excludedProviders *string
}

// registerConstraintFlags registers the shared constraint flags on fs.
func registerConstraintFlags(fs *flag.FlagSet) *constraintFlags {
	return &constraintFlags{
		needsVision:       fs.Bool("needs-vision", false, "require vision-capable models"),
		minContext:        fs.Int("min-context", 0, "minimum context window in tokens"),
		maxPrice:          fs.Float64("max-price", 0, "maximum blended price per 1k tokens in USD"),
		providers:         fs.String("providers", "", "comma-separated providers to allow (default: every configured provider)"),
		excludedProviders: fs.String("excluded-providers", "", "comma-separated providers to exclude"),
	}
}

// routing converts the parsed flags into RoutingConstraints. fs must already
// have parsed args; the --max-price flag is detected through fs.Visit so an
// explicit 0 is distinguishable from "not set" (a 0 maximum legitimately
// eliminates everything, while the default must not filter at all).
func (f *constraintFlags) routing(fs *flag.FlagSet) *modelrouter.RoutingConstraints {
	constraints := &modelrouter.RoutingConstraints{
		NeedsVision:       *f.needsVision,
		MinContextTokens:  *f.minContext,
		AllowedProviders:  splitList(*f.providers),
		ExcludedProviders: splitList(*f.excludedProviders),
	}
	fs.Visit(func(fk *flag.Flag) {
		if fk.Name == "max-price" {
			price := *f.maxPrice
			constraints.MaxPricePer1KTokens = &price
		}
	})
	return constraints
}

// splitList splits a comma-separated flag value, trimming spaces and
// dropping empty entries, so "a, b" and "a,,b" both yield [a b].
func splitList(value string) []string {
	var out []string
	for _, part := range strings.Split(value, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

// printJSON writes v as indented JSON followed by a newline.
func printJSON(w io.Writer, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("encoding JSON: %w", err)
	}
	if _, err := fmt.Fprintln(w, string(data)); err != nil {
		return fmt.Errorf("writing output: %w", err)
	}
	return nil
}

// closeApp releases the assembled configuration, logging — never failing on —
// close errors: once the command's work is done, a cleanup failure cannot
// change the result.
func closeApp(logger *slog.Logger, app *config.App) {
	if err := app.Close(); err != nil {
		logger.Warn("closing configuration", "error", err)
	}
}
