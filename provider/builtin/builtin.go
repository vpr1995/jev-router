// Package builtin registers the built-in completion providers with the
// provider registry.
//
//	builtin.Register()
//	completion, err := provider.New("openrouter", provider.Config{
//		APIKey: os.Getenv("OPENROUTER_API_KEY"),
//	})
//
// Compatible instances (Groq, Ollama, vLLM, LM Studio, ...) are not
// registered here: their names come from user configuration, so the config
// layer constructs them with openai.NewCompatible instead.
package builtin

import (
	"sync"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
	"github.com/vprprudhvi/jev-router/provider/anthropic"
	"github.com/vprprudhvi/jev-router/provider/bedrock"
	"github.com/vprprudhvi/jev-router/provider/gemini"
	"github.com/vprprudhvi/jev-router/provider/openai"
	"github.com/vprprudhvi/jev-router/provider/openrouter"
)

// registerOnce makes Register idempotent: assembly code (config.Build) may
// run more than once in a single process, and provider.Register panics on a
// duplicate registration.
var registerOnce sync.Once

// Register installs the built-in provider factories: "openrouter", "openai",
// "anthropic", "gemini" and "bedrock".
//
// It is idempotent and safe for concurrent use — the five registrations run
// exactly once no matter how many times it is called — so assembly code may
// call it unconditionally before provider.New. Each factory adapts a
// constructor's concrete *Completion return type to the interface
// provider.Factory expects; Go function types are not covariant, so the
// wrapping is required.
func Register() {
	registerOnce.Do(func() {
		provider.Register("openrouter", func(cfg provider.Config) (modelrouter.Completion, error) {
			return openrouter.New(cfg)
		})
		provider.Register("openai", func(cfg provider.Config) (modelrouter.Completion, error) {
			return openai.New(cfg)
		})
		provider.Register("anthropic", func(cfg provider.Config) (modelrouter.Completion, error) {
			return anthropic.New(cfg)
		})
		provider.Register("gemini", func(cfg provider.Config) (modelrouter.Completion, error) {
			return gemini.New(cfg)
		})
		provider.Register("bedrock", func(cfg provider.Config) (modelrouter.Completion, error) {
			return bedrock.New(cfg)
		})
	})
}
