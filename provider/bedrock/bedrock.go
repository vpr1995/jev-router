// Package bedrock implements modelrouter.Completion with the official AWS
// SDK for Go v2 (service/bedrockruntime) Converse API. Bedrock authenticates
// through the AWS credential chain, so Config.APIKey is unused; only Region
// is required.
//
// converseAPI is a deliberate test seam: an unexported interface holding just
// the Converse call, which *bedrockruntime.Client satisfies directly and
// which newWithAPI lets white-box tests fake without HTTP.
package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
)

// defaultName is the provider name used when Config.Name is empty (direct
// construction without the registry; provider.New always sets it).
const defaultName = "bedrock"

// converseAPI is the one Bedrock Runtime call the provider needs.
type converseAPI interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
}

// The production client satisfies the seam as-is.
var _ converseAPI = (*bedrockruntime.Client)(nil)

// Completion executes completions against Amazon Bedrock's Converse API. It
// is safe for concurrent use.
type Completion struct {
	name   string
	logger *slog.Logger
	api    converseAPI
}

var _ modelrouter.Completion = (*Completion)(nil)

// New builds a Completion from cfg. Region is required; credentials come from
// the AWS default chain (environment, shared config, IMDS, ...). An injected
// HTTPClient is caller-owned and never closed. Loading the AWS config is
// local (no network), so New uses context.Background() internally.
func New(cfg provider.Config) (*Completion, error) {
	if cfg.Region == "" {
		return nil, errors.New("bedrock: Region is required")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	loadOptions := []func(*config.LoadOptions) error{config.WithRegion(cfg.Region)}
	if cfg.HTTPClient != nil {
		loadOptions = append(loadOptions, config.WithHTTPClient(cfg.HTTPClient))
	}
	awsConfig, err := config.LoadDefaultConfig(context.Background(), loadOptions...)
	if err != nil {
		return nil, fmt.Errorf("bedrock: loading AWS configuration: %w", err)
	}
	return newWithAPI(name, logger, bedrockruntime.NewFromConfig(awsConfig)), nil
}

// newWithAPI builds a Completion around an injected converseAPI, for
// white-box tests that fake the Bedrock Runtime calls.
func newWithAPI(name string, logger *slog.Logger, api converseAPI) *Completion {
	if name == "" {
		name = defaultName
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Completion{name: name, logger: logger, api: api}
}

// Complete runs a Converse call.
func (c *Completion) Complete(ctx context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	input, err := buildConverseInput(req)
	if err != nil {
		return modelrouter.CompletionResult{}, err
	}
	output, err := c.api.Converse(ctx, input)
	if err != nil {
		return modelrouter.CompletionResult{}, c.providerError("complete", err)
	}
	return c.mapConverse(req, output), nil
}

// buildConverseInput maps the router request onto ConverseInput: system
// messages become the top-level system parameter (joined with "\n\n"), and a
// prompt-only request becomes a single user message.
func buildConverseInput(req modelrouter.CompletionRequest) (*bedrockruntime.ConverseInput, error) {
	if req.Model == "" {
		return nil, errors.New("bedrock: model is required")
	}
	if req.Prompt == "" && len(req.Messages) == 0 {
		return nil, errors.New("bedrock: prompt or messages required")
	}
	input := &bedrockruntime.ConverseInput{ModelId: aws.String(req.Model)}
	if len(req.Messages) == 0 {
		input.Messages = []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: req.Prompt}},
		}}
	} else {
		var system []string
		for _, message := range req.Messages {
			switch message.Role {
			case "system":
				system = append(system, message.Content)
			case "user", "assistant":
				role := types.ConversationRoleUser
				if message.Role == "assistant" {
					role = types.ConversationRoleAssistant
				}
				input.Messages = append(input.Messages, types.Message{
					Role:    role,
					Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: message.Content}},
				})
			default:
				return nil, fmt.Errorf("bedrock: unsupported message role %q (want \"system\", \"user\" or \"assistant\")", message.Role)
			}
		}
		if len(input.Messages) == 0 {
			return nil, errors.New("bedrock: at least one user or assistant message required")
		}
		if len(system) > 0 {
			input.System = []types.SystemContentBlock{&types.SystemContentBlockMemberText{Value: strings.Join(system, "\n\n")}}
		}
	}
	if req.MaxTokens > 0 || req.Temperature != nil {
		inference := &types.InferenceConfiguration{}
		if req.MaxTokens > 0 {
			inference.MaxTokens = aws.Int32(int32(req.MaxTokens))
		}
		if req.Temperature != nil {
			inference.Temperature = aws.Float32(float32(*req.Temperature))
		}
		input.InferenceConfig = inference
	}
	return input, nil
}

// mapConverse translates a Converse output; a nil output or a non-message
// variant yield an empty Content instead of a panic. RoutingDecision is left
// zero: the Router owns that field.
func (c *Completion) mapConverse(req modelrouter.CompletionRequest, output *bedrockruntime.ConverseOutput) modelrouter.CompletionResult {
	out := modelrouter.CompletionResult{
		Model:    req.Model,
		Provider: c.name,
	}
	if output == nil {
		return out
	}
	if raw, err := json.Marshal(output); err != nil {
		c.logger.Warn("bedrock: could not re-marshal output for Raw", "error", err)
	} else {
		out.Raw = raw
	}
	if member, ok := output.Output.(*types.ConverseOutputMemberMessage); ok {
		out.Content = textOfBlocks(member.Value.Content)
	}
	out.FinishReason = string(output.StopReason)
	out.Usage = usageOf(output.Usage)
	return out
}

// textOfBlocks concatenates the text of every text content block.
func textOfBlocks(blocks []types.ContentBlock) string {
	var text strings.Builder
	for _, block := range blocks {
		if member, ok := block.(*types.ContentBlockMemberText); ok {
			text.WriteString(member.Value)
		}
	}
	return text.String()
}

// usageOf maps SDK token usage; nil when the SDK reported none.
func usageOf(usage *types.TokenUsage) *modelrouter.Usage {
	if usage == nil {
		return nil
	}
	if usage.InputTokens == nil && usage.OutputTokens == nil && usage.TotalTokens == nil {
		return nil
	}
	var inputTokens, outputTokens, totalTokens int
	if usage.InputTokens != nil {
		inputTokens = int(*usage.InputTokens)
	}
	if usage.OutputTokens != nil {
		outputTokens = int(*usage.OutputTokens)
	}
	if usage.TotalTokens != nil {
		totalTokens = int(*usage.TotalTokens)
	}
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

// statusCode extracts the HTTP status an SDK error responded with
// (*awshttp.ResponseError carries it; other errors yield 0).
func statusCode(err error) int {
	var responseErr *awshttp.ResponseError
	if errors.As(err, &responseErr) {
		return responseErr.HTTPStatusCode()
	}
	return 0
}
