// Package anthropic implements modelrouter.Completion with the official
// Anthropic Go SDK (github.com/anthropics/anthropic-sdk-go).
//
// Notable SDK quirks this provider works around: the Messages API requires
// max_tokens on every request (defaultMaxTokens fills an unset one), there is
// no "system" role (system messages join into the top-level System
// parameter), and usage carries no total so TotalTokens is InputTokens +
// OutputTokens.
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
)

const (
	defaultName      = "anthropic"
	defaultMaxTokens = 4096
)

// Completion executes completions against Anthropic's Messages API. It is
// safe for concurrent use.
type Completion struct {
	name   string
	client anthropicsdk.Client
}

var _ modelrouter.Completion = (*Completion)(nil)

// New builds a Completion from cfg. APIKey is required. BaseURL overrides the
// SDK's default (https://api.anthropic.com). An injected HTTPClient is
// caller-owned and never closed; when nil the SDK constructs its own, so pass
// a context with a deadline to bound a call.
func New(cfg provider.Config) (*Completion, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("anthropic: APIKey is required")
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey)}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &Completion{name: name, client: anthropicsdk.NewClient(opts...)}, nil
}

// Complete runs a Messages API request.
func (c *Completion) Complete(ctx context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	params, err := buildParams(req)
	if err != nil {
		return modelrouter.CompletionResult{}, err
	}
	message, err := c.client.Messages.New(ctx, params)
	if err != nil {
		return modelrouter.CompletionResult{}, c.providerError("complete", err)
	}
	return c.mapMessage(req, message), nil
}

// buildParams maps the router request onto MessageNewParams: system messages
// go to the top-level system parameter, user/assistant messages keep their
// order, and an unset MaxTokens becomes defaultMaxTokens.
func buildParams(req modelrouter.CompletionRequest) (anthropicsdk.MessageNewParams, error) {
	if req.Model == "" {
		return anthropicsdk.MessageNewParams{}, errors.New("anthropic: model is required")
	}
	if req.Prompt == "" && len(req.Messages) == 0 {
		return anthropicsdk.MessageNewParams{}, errors.New("anthropic: prompt or messages required")
	}
	params := anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.Model(req.Model),
		MaxTokens: defaultMaxTokens,
	}
	if req.MaxTokens > 0 {
		params.MaxTokens = int64(req.MaxTokens)
	}
	if req.Temperature != nil {
		params.Temperature = anthropicsdk.Float(*req.Temperature)
	}
	if len(req.Messages) == 0 {
		params.Messages = []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock(req.Prompt)),
		}
		return params, nil
	}
	var system []string
	for _, message := range req.Messages {
		switch message.Role {
		case "system":
			system = append(system, message.Content)
		case "user":
			params.Messages = append(params.Messages, anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock(message.Content)))
		case "assistant":
			params.Messages = append(params.Messages, anthropicsdk.NewAssistantMessage(anthropicsdk.NewTextBlock(message.Content)))
		default:
			return anthropicsdk.MessageNewParams{}, fmt.Errorf("anthropic: unsupported message role %q (want \"system\", \"user\" or \"assistant\")", message.Role)
		}
	}
	if len(params.Messages) == 0 {
		return anthropicsdk.MessageNewParams{}, errors.New("anthropic: at least one user or assistant message required")
	}
	if len(system) > 0 {
		params.System = []anthropicsdk.TextBlockParam{{Text: strings.Join(system, "\n\n")}}
	}
	return params, nil
}

// mapMessage translates a Messages response; a nil message or no text blocks
// yield an empty Content instead of a panic. RoutingDecision is left zero:
// the Router owns that field.
func (c *Completion) mapMessage(req modelrouter.CompletionRequest, message *anthropicsdk.Message) modelrouter.CompletionResult {
	out := modelrouter.CompletionResult{
		Model:    req.Model,
		Provider: c.name,
	}
	if message == nil {
		return out
	}
	if raw := message.RawJSON(); raw != "" {
		out.Raw = json.RawMessage(raw)
	}
	out.Content = textOfBlocks(message.Content)
	out.FinishReason = string(message.StopReason)
	out.Usage = usageOf(message.Usage)
	return out
}

// textOfBlocks concatenates the text of every "text" content block.
func textOfBlocks(blocks []anthropicsdk.ContentBlockUnion) string {
	var text strings.Builder
	for _, block := range blocks {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	return text.String()
}

// usageOf maps SDK usage; nil when the response carried no token counts.
func usageOf(usage anthropicsdk.Usage) *modelrouter.Usage {
	if !usage.JSON.InputTokens.Valid() && !usage.JSON.OutputTokens.Valid() {
		return nil
	}
	inputTokens := int(usage.InputTokens)
	outputTokens := int(usage.OutputTokens)
	return &modelrouter.Usage{
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		TotalTokens:  inputTokens + outputTokens,
	}
}

// providerError wraps a provider failure as a *modelrouter.ProviderError.
func (c *Completion) providerError(op string, err error) error {
	return provider.WrapError(c.name, op, statusCode(err), err)
}

// statusCode extracts the HTTP status an SDK error responded with
// (*anthropic.Error carries it directly; other errors yield 0).
func statusCode(err error) int {
	var apiErr *anthropicsdk.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode
	}
	return 0
}
