// Command jev-router is the command-line interface of the jev-router
// multi-provider model router: route and complete prompts, list models,
// validate configuration and serve the HTTP API.
package main

import (
	"os"

	"github.com/vprprudhvi/jev-router/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
