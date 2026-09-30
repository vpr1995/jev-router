// Package openrouter implements modelrouter.Completion with the official
// OpenRouter Go SDK (github.com/OpenRouterTeam/go-sdk).
//
// Most SDK failures arrive as typed sdkerrors values that do not carry the
// HTTP status; statusCode maps the ones Chat.Send generates back to their
// status codes. OpenRouter-specific USD cost, when reported, rides on
// components.ChatUsage.Cost.
package openrouter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	openroutersdk "github.com/OpenRouterTeam/go-sdk"
	"github.com/OpenRouterTeam/go-sdk/models/components"
	"github.com/OpenRouterTeam/go-sdk/models/operations"
	"github.com/OpenRouterTeam/go-sdk/models/sdkerrors"
	"github.com/OpenRouterTeam/go-sdk/optionalnullable"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
)

// defaultName is the provider name used when Config.Name is empty (direct
// construction without the registry; provider.New always sets it).
const defaultName = "openrouter"

// Completion executes completions against OpenRouter's OpenAI-compatible
// chat completions API. It is safe for concurrent use.
type Completion struct {
	name   string
	logger *slog.Logger
	sdk    *openroutersdk.OpenRouter
}

var _ modelrouter.Completion = (*Completion)(nil)

// New builds a Completion from cfg. APIKey is required. BaseURL overrides the
// SDK's production server (https://openrouter.ai/api/v1). An injected
// HTTPClient is caller-owned and never closed; when nil the provider
// constructs its own with no timeout, so pass a context with a deadline to
// bound a call.
func New(cfg provider.Config) (*Completion, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("openrouter: APIKey is required")
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	opts := []openroutersdk.SDKOption{
		openroutersdk.WithSecurity(cfg.APIKey),
		openroutersdk.WithClient(httpClient),
	}
	if cfg.BaseURL != "" {
		opts = append(opts, openroutersdk.WithServerURL(cfg.BaseURL))
	}
	return &Completion{name: name, logger: logger, sdk: openroutersdk.New(opts...)}, nil
}

// Complete runs a chat completion.
func (c *Completion) Complete(ctx context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	chatRequest, err := buildRequest(req)
	if err != nil {
		return modelrouter.CompletionResult{}, err
	}
	res, err := c.sdk.Chat.Send(ctx, chatRequest, nil)
	if err != nil {
		return modelrouter.CompletionResult{}, c.providerError("complete", err)
	}
	if res == nil || res.Type != operations.SendChatCompletionRequestResponseTypeChatResult {
		return modelrouter.CompletionResult{}, c.unexpectedResponse("complete", res)
	}
	return c.mapChatResult(req, res.ChatResult), nil
}

// buildRequest maps the router request onto components.ChatRequest.
func buildRequest(req modelrouter.CompletionRequest) (components.ChatRequest, error) {
	if req.Model == "" {
		return components.ChatRequest{}, errors.New("openrouter: model is required")
	}
	if req.Prompt == "" && len(req.Messages) == 0 {
		return components.ChatRequest{}, errors.New("openrouter: prompt or messages required")
	}
	messages, err := buildMessages(req)
	if err != nil {
		return components.ChatRequest{}, err
	}
	chatRequest := components.ChatRequest{
		Model:    openroutersdk.Pointer(req.Model),
		Messages: messages,
	}
	if req.MaxTokens > 0 {
		chatRequest.MaxTokens = optionalnullable.From(openroutersdk.Int64(int64(req.MaxTokens)))
	}
	if req.Temperature != nil {
		chatRequest.Temperature = optionalnullable.From(openroutersdk.Float64(*req.Temperature))
	}
	return chatRequest, nil
}

// buildMessages maps the request onto SDK messages. A prompt-only request
// becomes a single user message; a message list keeps its order and roles.
func buildMessages(req modelrouter.CompletionRequest) ([]components.ChatMessages, error) {
	if len(req.Messages) == 0 {
		return []components.ChatMessages{components.CreateChatMessagesUser(components.ChatUserMessage{
			Role:    components.ChatUserMessageRoleUser,
			Content: components.CreateChatUserMessageContentStr(req.Prompt),
		})}, nil
	}
	messages := make([]components.ChatMessages, 0, len(req.Messages))
	for _, message := range req.Messages {
		switch message.Role {
		case "system":
			messages = append(messages, components.CreateChatMessagesSystem(components.ChatSystemMessage{
				Role:    components.ChatSystemMessageRoleSystem,
				Content: components.CreateChatSystemMessageContentStr(message.Content),
			}))
		case "user":
			messages = append(messages, components.CreateChatMessagesUser(components.ChatUserMessage{
				Role:    components.ChatUserMessageRoleUser,
				Content: components.CreateChatUserMessageContentStr(message.Content),
			}))
		case "assistant":
			// ChatAssistantMessage.Content is OptionalNullable in the SDK
			// because the same type models responses, where content may be
			// null behind tool calls; requests always set it.
			content := components.CreateChatAssistantMessageContentStr(message.Content)
			messages = append(messages, components.CreateChatMessagesAssistant(components.ChatAssistantMessage{
				Role:    components.ChatAssistantMessageRoleAssistant,
				Content: optionalnullable.From(&content),
			}))
		default:
			return nil, fmt.Errorf("openrouter: unsupported message role %q (want \"system\", \"user\" or \"assistant\")", message.Role)
		}
	}
	return messages, nil
}

// mapChatResult translates a chat result; a nil result or empty choices
// yield an empty Content instead of a panic. RoutingDecision is left zero:
// the Router owns that field.
func (c *Completion) mapChatResult(req modelrouter.CompletionRequest, result *components.ChatResult) modelrouter.CompletionResult {
	out := modelrouter.CompletionResult{
		Model:    req.Model,
		Provider: c.name,
	}
	if result == nil {
		return out
	}
	if raw, err := json.Marshal(result); err != nil {
		c.logger.Warn("openrouter: could not re-marshal chat result for Raw", "error", err)
	} else {
		out.Raw = raw
	}
	if len(result.Choices) > 0 {
		choice := result.Choices[0]
		out.Content = assistantText(choice.Message)
		if choice.FinishReason != nil {
			out.FinishReason = string(*choice.FinishReason)
		}
	}
	out.Usage = mapUsage(result.Usage)
	return out
}

// assistantText extracts the plain-text content of an assistant message.
// Unset/null content (for example tool-only replies) and array content yield
// "" — the router never panics on a missing choice.
func assistantText(message components.ChatAssistantMessage) string {
	content, ok := message.Content.GetOrZero()
	if !ok || content.Str == nil {
		return ""
	}
	return *content.Str
}

// mapUsage translates SDK usage; CostUSD is nil when absent or null.
func mapUsage(usage *components.ChatUsage) *modelrouter.Usage {
	if usage == nil {
		return nil
	}
	out := &modelrouter.Usage{
		InputTokens:  int(usage.PromptTokens),
		OutputTokens: int(usage.CompletionTokens),
		TotalTokens:  int(usage.TotalTokens),
	}
	if cost, ok := usage.Cost.Get(); ok && cost != nil {
		out.CostUSD = cost
	}
	return out
}

// providerError wraps a provider failure as a *modelrouter.ProviderError.
func (c *Completion) providerError(op string, err error) error {
	return provider.WrapError(c.name, op, statusCode(err), err)
}

// unexpectedResponse reports a response union member the provider does not
// map (the provider only consumes ChatResult responses).
func (c *Completion) unexpectedResponse(op string, res *operations.SendChatCompletionRequestResponse) error {
	kind := "<nil>"
	if res != nil {
		kind = string(res.Type)
	}
	return provider.WrapError(c.name, op, 0, fmt.Errorf("openrouter: unexpected response type %q", kind))
}

// statusCode extracts the HTTP status code an SDK error responded with.
// sdkerrors.APIError carries it directly; the per-status typed errors
// (BadRequestResponseError, UnauthorizedResponseError, ...) do not, so they
// are mapped back to the statuses Chat.Send generates them for.
func statusCode(err error) int {
	var apiErr *sdkerrors.APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode
	}
	switch {
	case isError[*sdkerrors.BadRequestResponseError](err):
		return http.StatusBadRequest
	case isError[*sdkerrors.UnauthorizedResponseError](err):
		return http.StatusUnauthorized
	case isError[*sdkerrors.PaymentRequiredResponseError](err):
		return http.StatusPaymentRequired
	case isError[*sdkerrors.ForbiddenResponseError](err):
		return http.StatusForbidden
	case isError[*sdkerrors.NotFoundResponseError](err):
		return http.StatusNotFound
	case isError[*sdkerrors.RequestTimeoutResponseError](err):
		return http.StatusRequestTimeout
	case isError[*sdkerrors.PayloadTooLargeResponseError](err):
		return http.StatusRequestEntityTooLarge
	case isError[*sdkerrors.UnprocessableEntityResponseError](err):
		return http.StatusUnprocessableEntity
	case isError[*sdkerrors.TooManyRequestsResponseError](err):
		return http.StatusTooManyRequests
	case isError[*sdkerrors.InternalServerResponseError](err):
		return http.StatusInternalServerError
	case isError[*sdkerrors.BadGatewayResponseError](err):
		return http.StatusBadGateway
	case isError[*sdkerrors.ServiceUnavailableResponseError](err):
		return http.StatusServiceUnavailable
	case isError[*sdkerrors.EdgeNetworkTimeoutResponseError](err):
		return 524 // Cloudflare/OpenRouter edge timeout
	case isError[*sdkerrors.ProviderOverloadedResponseError](err):
		return 529 // OpenRouter provider overloaded
	default:
		return 0
	}
}

// isError reports whether err (or anything it wraps) is a *T.
func isError[T error](err error) bool {
	var target T
	return errors.As(err, &target)
}
