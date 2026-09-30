// Package openai implements modelrouter.Completion with the official OpenAI
// Go SDK (github.com/openai/openai-go).
//
// NewCompatible builds the same provider for other OpenAI-wire endpoints
// (groq, ollama, vllm, LM Studio, ...), which just require a BaseURL and a
// custom name.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/respjson"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
)

const defaultName = "openai"

// Completion executes completions against OpenAI's (or an OpenAI-compatible)
// chat completions API. It is safe for concurrent use.
type Completion struct {
	name   string
	client openaisdk.Client
}

var _ modelrouter.Completion = (*Completion)(nil)

// New builds a Completion from cfg. APIKey is required. BaseURL overrides the
// SDK's default (https://api.openai.com/v1). An injected HTTPClient is
// caller-owned and never closed; when nil the provider constructs its own
// with no timeout, so pass a context with a deadline to bound a call.
func New(cfg provider.Config) (*Completion, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("openai: APIKey is required")
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	opts := []option.RequestOption{
		option.WithAPIKey(cfg.APIKey),
		option.WithHTTPClient(httpClient),
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &Completion{name: name, client: openaisdk.NewClient(opts...)}, nil
}

// NewCompatible builds a Completion for an OpenAI-wire provider reachable at
// cfg.BaseURL (groq, ollama, vllm, LM Studio, ...); cfg.BaseURL is required.
// name is the provider name used in results and errors and wins over
// cfg.Name; when both are empty the package default ("openai") applies.
func NewCompatible(name string, cfg provider.Config) (*Completion, error) {
	if cfg.BaseURL == "" {
		return nil, errors.New("openai: BaseURL is required for OpenAI-compatible providers")
	}
	if name != "" {
		cfg.Name = name
	}
	return New(cfg)
}

// Complete runs a chat completion.
func (c *Completion) Complete(ctx context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	params, err := buildParams(req)
	if err != nil {
		return modelrouter.CompletionResult{}, err
	}
	completion, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return modelrouter.CompletionResult{}, c.providerError("complete", err)
	}
	return c.mapCompletion(req, completion), nil
}

// buildParams maps the router request onto ChatCompletionNewParams.
func buildParams(req modelrouter.CompletionRequest) (openaisdk.ChatCompletionNewParams, error) {
	if req.Model == "" {
		return openaisdk.ChatCompletionNewParams{}, errors.New("openai: model is required")
	}
	if req.Prompt == "" && len(req.Messages) == 0 {
		return openaisdk.ChatCompletionNewParams{}, errors.New("openai: prompt or messages required")
	}
	messages, err := buildMessages(req)
	if err != nil {
		return openaisdk.ChatCompletionNewParams{}, err
	}
	params := openaisdk.ChatCompletionNewParams{
		Model:    openaisdk.ChatModel(req.Model),
		Messages: messages,
	}
	if req.MaxTokens > 0 {
		params.MaxTokens = openaisdk.Int(int64(req.MaxTokens))
	}
	if req.Temperature != nil {
		params.Temperature = openaisdk.Float(*req.Temperature)
	}
	return params, nil
}

// buildMessages maps the request onto SDK messages. A prompt-only request
// becomes a single user message; a message list keeps its order and roles.
func buildMessages(req modelrouter.CompletionRequest) ([]openaisdk.ChatCompletionMessageParamUnion, error) {
	if len(req.Messages) == 0 {
		return []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage(req.Prompt)}, nil
	}
	messages := make([]openaisdk.ChatCompletionMessageParamUnion, 0, len(req.Messages))
	for _, message := range req.Messages {
		switch message.Role {
		case "system":
			messages = append(messages, openaisdk.SystemMessage(message.Content))
		case "user":
			messages = append(messages, openaisdk.UserMessage(message.Content))
		case "assistant":
			messages = append(messages, openaisdk.AssistantMessage(message.Content))
		default:
			return nil, fmt.Errorf("openai: unsupported message role %q (want \"system\", \"user\" or \"assistant\")", message.Role)
		}
	}
	return messages, nil
}

// mapCompletion translates a chat completion; a nil completion or empty
// choices yield an empty Content instead of a panic. RoutingDecision is left
// zero: the Router owns that field.
func (c *Completion) mapCompletion(req modelrouter.CompletionRequest, completion *openaisdk.ChatCompletion) modelrouter.CompletionResult {
	out := modelrouter.CompletionResult{
		Model:    req.Model,
		Provider: c.name,
	}
	if completion == nil {
		return out
	}
	if raw := completion.RawJSON(); raw != "" {
		out.Raw = json.RawMessage(raw)
	}
	if len(completion.Choices) > 0 {
		choice := completion.Choices[0]
		out.Content = choice.Message.Content
		out.FinishReason = choice.FinishReason
	}
	if completion.JSON.Usage.Valid() {
		out.Usage = mapCompletionUsage(completion.Usage)
	}
	return out
}

// mapCompletionUsage translates SDK usage. OpenAI reports tokens only, so
// CostUSD stays nil unless the server added a "cost" extra field (OpenRouter
// and similar OpenAI-compatible APIs do).
func mapCompletionUsage(usage openaisdk.CompletionUsage) *modelrouter.Usage {
	out := &modelrouter.Usage{
		InputTokens:  int(usage.PromptTokens),
		OutputTokens: int(usage.CompletionTokens),
		TotalTokens:  int(usage.TotalTokens),
	}
	if cost, ok := extraFieldFloat(usage.JSON.ExtraFields, "cost"); ok {
		out.CostUSD = &cost
	}
	return out
}

// extraFieldFloat reads a numeric JSON field the SDK did not model.
func extraFieldFloat(fields map[string]respjson.Field, name string) (float64, bool) {
	field, ok := fields[name]
	if !ok || !field.Valid() {
		return 0, false
	}
	var value float64
	if err := json.Unmarshal([]byte(field.Raw()), &value); err != nil {
		return 0, false
	}
	return value, true
}

// providerError wraps a provider failure as a *modelrouter.ProviderError.
func (c *Completion) providerError(op string, err error) error {
	return provider.WrapError(c.name, op, statusCode(err), err)
}

// statusCode extracts the HTTP status an SDK error responded with
// (*openai.Error carries it directly; other errors yield 0).
func statusCode(err error) int {
	var apiErr *openaisdk.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode
	}
	return 0
}
