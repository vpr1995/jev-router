// This file implements the complete command.
package cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// runComplete implements `jev-router complete <prompt> [flags]`: it routes
// the prompt and executes the completion on the winning provider.
//
// Output shapes (stdout only; logs stay on stderr):
//
//	(default)            the completion content plus a trailing newline
//	--json               the full CompletionResult as indented JSON
//
// With --system the request carries two messages (system, then user);
// without it, the prompt travels as a single user prompt.
func runComplete(args []string, stdout, stderr io.Writer) error {
	fs := newFlagSet("complete")
	global := registerGlobalFlags(fs)
	configPath := fs.String("config", "", "configuration file (default: $JEV_ROUTER_CONFIG, else environment-only defaults)")
	constraints := registerConstraintFlags(fs)
	system := fs.String("system", "", "prepend a system message with this text")
	asJSON := fs.Bool("json", false, "print the result as JSON")
	maxTokens := fs.Int("max-tokens", 0, "cap generated tokens (0 = provider default)")
	temperature := fs.Float64("temperature", 0, "sampling temperature (unset = provider default)")

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
		return usageErrorf("complete: missing prompt")
	}

	app, err := buildApp(*configPath, logger)
	if err != nil {
		return err
	}
	defer closeApp(logger, app)

	request := modelrouter.CompletionRequest{
		Constraints: constraints.routing(fs),
		MaxTokens:   *maxTokens,
	}
	if *system != "" {
		request.Messages = []modelrouter.Message{
			{Role: "system", Content: *system},
			{Role: "user", Content: prompt},
		}
	} else {
		request.Prompt = prompt
	}
	// fs.Visit distinguishes an explicit `--temperature 0` from "unset" (the
	// provider default), matching the --max-price handling.
	fs.Visit(func(fk *flag.Flag) {
		if fk.Name == "temperature" {
			value := *temperature
			request.Temperature = &value
		}
	})

	ctx := context.Background()
	result, err := app.Router.Complete(ctx, request)
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(stdout, result)
	}
	_, _ = fmt.Fprintln(stdout, result.Content)
	return nil
}
