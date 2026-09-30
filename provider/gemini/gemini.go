// Package gemini implements modelrouter.Completion with the official Google
// Gen AI Go SDK (google.golang.org/genai), talking to the Gemini Developer
// API (Backend is pinned so an environment variable cannot redirect this
// provider to Vertex AI).
//
// Notable SDK quirks this provider works around: roles are "user"/"model"
// rather than "user"/"assistant", there is no system role (system messages
// join into GenerateContentConfig.SystemInstruction), and the response
// carries no raw JSON (Raw is a re-encoding of the decoded response).
package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	genaisdk "google.golang.org/genai"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
)

// defaultName is the provider name used when Config.Name is empty (direct
// construction without the registry; provider.New always sets it).
const defaultName = "gemini"

// Completion executes completions against the Gemini Developer API. It is
// safe for concurrent use.
type Completion struct {
	name   string
	logger *slog.Logger
	client *genaisdk.Client
}

var _ modelrouter.Completion = (*Completion)(nil)

// New builds a Completion from cfg. APIKey is required. BaseURL overrides the
// SDK's default (https://generativelanguage.googleapis.com). An injected
// HTTPClient is caller-owned and never closed; when nil the SDK constructs
// its own, so pass a context with a deadline to bound a call.
func New(cfg provider.Config) (*Completion, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("gemini: APIKey is required")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	clientConfig := &genaisdk.ClientConfig{
		APIKey:     cfg.APIKey,
		Backend:    genaisdk.BackendGeminiAPI,
		HTTPClient: cfg.HTTPClient,
	}
	if cfg.BaseURL != "" {
		clientConfig.HTTPOptions = genaisdk.HTTPOptions{BaseURL: cfg.BaseURL}
	}
	client, err := genaisdk.NewClient(context.Background(), clientConfig)
	if err != nil {
		return nil, fmt.Errorf("gemini: constructing client: %w", err)
	}
	return &Completion{name: name, logger: logger, client: client}, nil
}

// Complete runs a generateContent call.
func (c *Completion) Complete(ctx context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	model, contents, config, err := buildRequest(req)
	if err != nil {
		return modelrouter.CompletionResult{}, err
	}
	response, err := c.client.Models.GenerateContent(ctx, model, contents, config)
	if err != nil {
		return modelrouter.CompletionResult{}, c.providerError("complete", err)
	}
	return c.mapResponse(req, response), nil
}

// buildRequest maps the router request onto the Gen AI request shape:
// assistant messages become "model" role contents, system messages become
// the system instruction (joined with "\n\n").
func buildRequest(req modelrouter.CompletionRequest) (string, []*genaisdk.Content, *genaisdk.GenerateContentConfig, error) {
	if req.Model == "" {
		return "", nil, nil, errors.New("gemini: model is required")
	}
	if req.Prompt == "" && len(req.Messages) == 0 {
		return "", nil, nil, errors.New("gemini: prompt or messages required")
	}
	var (
		contents          []*genaisdk.Content
		system            []string
		systemInstruction *genaisdk.Content
	)
	if len(req.Messages) == 0 {
		contents = []*genaisdk.Content{genaisdk.NewContentFromText(req.Prompt, genaisdk.RoleUser)}
	}
	for _, message := range req.Messages {
		switch message.Role {
		case "system":
			system = append(system, message.Content)
		case "user":
			contents = append(contents, genaisdk.NewContentFromText(message.Content, genaisdk.RoleUser))
		case "assistant":
			contents = append(contents, genaisdk.NewContentFromText(message.Content, genaisdk.RoleModel))
		default:
			return "", nil, nil, fmt.Errorf("gemini: unsupported message role %q (want \"system\", \"user\" or \"assistant\")", message.Role)
		}
	}
	if len(req.Messages) > 0 && len(contents) == 0 {
		return "", nil, nil, errors.New("gemini: at least one user or assistant message required")
	}
	if len(system) > 0 {
		systemInstruction = &genaisdk.Content{Parts: []*genaisdk.Part{{Text: strings.Join(system, "\n\n")}}}
	}
	config := &genaisdk.GenerateContentConfig{SystemInstruction: systemInstruction}
	if req.MaxTokens > 0 {
		config.MaxOutputTokens = int32(req.MaxTokens)
	}
	if req.Temperature != nil {
		config.Temperature = genaisdk.Ptr(float32(*req.Temperature))
	}
	return req.Model, contents, config, nil
}

// mapResponse translates a generateContent response; a nil response or no
// candidates yield an empty Content instead of a panic. RoutingDecision is
// left zero: the Router owns that field.
func (c *Completion) mapResponse(req modelrouter.CompletionRequest, response *genaisdk.GenerateContentResponse) modelrouter.CompletionResult {
	out := modelrouter.CompletionResult{
		Model:    req.Model,
		Provider: c.name,
	}
	if response == nil {
		return out
	}
	if raw, err := json.Marshal(response); err != nil {
		c.logger.Warn("gemini: could not re-marshal response for Raw", "error", err)
	} else {
		out.Raw = raw
	}
	if len(response.Candidates) > 0 && response.Candidates[0] != nil {
		candidate := response.Candidates[0]
		out.Content = textOfContent(candidate.Content)
		out.FinishReason = string(candidate.FinishReason)
	}
	out.Usage = usageOf(response.UsageMetadata)
	return out
}

// textOfContent concatenates the text of every part of one content message.
func textOfContent(content *genaisdk.Content) string {
	if content == nil {
		return ""
	}
	var text strings.Builder
	for _, part := range content.Parts {
		if part != nil {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

// usageOf maps the SDK usage metadata; nil when the response carried none.
func usageOf(metadata *genaisdk.GenerateContentResponseUsageMetadata) *modelrouter.Usage {
	if metadata == nil {
		return nil
	}
	inputTokens := int(metadata.PromptTokenCount)
	outputTokens := int(metadata.CandidatesTokenCount)
	totalTokens := int(metadata.TotalTokenCount)
	if totalTokens == 0 {
		totalTokens = inputTokens + outputTokens
	}
	return &modelrouter.Usage{
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		TotalTokens:  totalTokens,
	}
}

// providerError wraps a provider failure as a *modelrouter.ProviderError.
func (c *Completion) providerError(op string, err error) error {
	return provider.WrapError(c.name, op, statusCode(err), err)
}

// statusCode extracts the HTTP status a genai API error responded with
// (genai.APIError.Code; other errors yield 0).
func statusCode(err error) int {
	var apiErr genaisdk.APIError
	if errors.As(err, &apiErr) {
		return apiErr.Code
	}
	return 0
}
