package bedrock

// White-box tests: the unexported converseAPI seam and newWithAPI exist
// exactly so these tests can run without HTTP or AWS credentials.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
)

// fakeAPI records the inputs it was called with and returns canned outputs.
type fakeAPI struct {
	converseOut *bedrockruntime.ConverseOutput
	converseErr error

	lastConverseInput *bedrockruntime.ConverseInput
}

var _ converseAPI = (*fakeAPI)(nil)

func (f *fakeAPI) Converse(_ context.Context, params *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.lastConverseInput = params
	return f.converseOut, f.converseErr
}

func messageOutput(content string, stopReason types.StopReason, usage *types.TokenUsage) *bedrockruntime.ConverseOutput {
	return &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Role:    types.ConversationRoleAssistant,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: content}},
		}},
		StopReason: stopReason,
		Usage:      usage,
	}
}

func TestCompleteMapsRequestAndResponse(t *testing.T) {
	api := &fakeAPI{converseOut: messageOutput("Hello world", types.StopReasonEndTurn, &types.TokenUsage{
		InputTokens:  aws.Int32(10),
		OutputTokens: aws.Int32(4),
		TotalTokens:  aws.Int32(14),
	})}
	completion := newWithAPI("bedrock", nil, api)
	temperature := 0.25
	request := modelrouter.CompletionRequest{
		Model: "anthropic.claude-test-v1:0",
		Messages: []modelrouter.Message{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
			{Role: "system", Content: "and exact"},
		},
		MaxTokens:   128,
		Temperature: &temperature,
	}
	// Snapshot the caller's request to prove it is never mutated.
	before := append([]modelrouter.Message(nil), request.Messages...)

	result, err := completion.Complete(context.Background(), request)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	input := api.lastConverseInput
	if input == nil {
		t.Fatal("Converse was not called")
	}
	if input.ModelId == nil || *input.ModelId != "anthropic.claude-test-v1:0" {
		t.Errorf("ModelId = %v, want anthropic.claude-test-v1:0", input.ModelId)
	}
	if len(input.Messages) != 2 {
		t.Fatalf("Messages = %+v, want 2", input.Messages)
	}
	if input.Messages[0].Role != types.ConversationRoleUser {
		t.Errorf("message[0].Role = %q, want user", input.Messages[0].Role)
	}
	text, ok := input.Messages[0].Content[0].(*types.ContentBlockMemberText)
	if !ok || text.Value != "hi" {
		t.Errorf("message[0].Content = %+v, want text %q", input.Messages[0].Content, "hi")
	}
	if input.Messages[1].Role != types.ConversationRoleAssistant {
		t.Errorf("message[1].Role = %q, want assistant", input.Messages[1].Role)
	}
	if len(input.System) != 1 {
		t.Fatalf("System = %+v, want one block", input.System)
	}
	system, ok := input.System[0].(*types.SystemContentBlockMemberText)
	if !ok || system.Value != "be brief\n\nand exact" {
		t.Errorf("System = %+v, want %q", input.System, "be brief\n\nand exact")
	}
	if input.InferenceConfig == nil {
		t.Fatal("InferenceConfig = nil, want maxTokens and temperature")
	}
	if input.InferenceConfig.MaxTokens == nil || *input.InferenceConfig.MaxTokens != 128 {
		t.Errorf("MaxTokens = %v, want 128", input.InferenceConfig.MaxTokens)
	}
	if input.InferenceConfig.Temperature == nil || *input.InferenceConfig.Temperature != 0.25 {
		t.Errorf("Temperature = %v, want 0.25", input.InferenceConfig.Temperature)
	}
	if result.Content != "Hello world" {
		t.Errorf("Content = %q, want %q", result.Content, "Hello world")
	}
	if result.FinishReason != "end_turn" {
		t.Errorf("FinishReason = %q, want end_turn", result.FinishReason)
	}
	if result.Model != "anthropic.claude-test-v1:0" || result.Provider != "bedrock" {
		t.Errorf("Model/Provider = %q/%q, want native/bedrock", result.Model, result.Provider)
	}
	if result.Usage == nil || result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 4 || result.Usage.TotalTokens != 14 {
		t.Errorf("Usage = %+v, want 10/4/14", result.Usage)
	}
	if !strings.Contains(string(result.Raw), "Hello world") {
		t.Errorf("Raw = %s, want it to contain the generated text", result.Raw)
	}
	if !reflect.DeepEqual(request.Messages, before) {
		t.Errorf("caller request mutated: %+v, want %+v", request.Messages, before)
	}
}

func TestCompletePromptOnlySkipsInferenceConfig(t *testing.T) {
	api := &fakeAPI{converseOut: messageOutput("ok", types.StopReasonEndTurn, &types.TokenUsage{
		InputTokens:  aws.Int32(7),
		OutputTokens: aws.Int32(5),
	})}
	completion := newWithAPI("bedrock", nil, api)
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{
		Model:  "amazon.nova-test-v1:0",
		Prompt: "hello",
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	input := api.lastConverseInput
	if len(input.Messages) != 1 || input.Messages[0].Role != types.ConversationRoleUser {
		t.Errorf("Messages = %+v, want one user message", input.Messages)
	}
	if len(input.System) != 0 {
		t.Errorf("System = %+v, want none", input.System)
	}
	if input.InferenceConfig != nil {
		t.Errorf("InferenceConfig = %+v, want nil when nothing was set", input.InferenceConfig)
	}
	// TotalTokens was omitted, so it falls back to input+output.
	if result.Usage == nil || result.Usage.TotalTokens != 12 {
		t.Errorf("Usage = %+v, want total 12", result.Usage)
	}
}

func TestCompleteNoUsage(t *testing.T) {
	api := &fakeAPI{converseOut: messageOutput("ok", types.StopReasonEndTurn, nil)}
	completion := newWithAPI("bedrock", nil, api)
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "m", Prompt: "hi"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Usage != nil {
		t.Errorf("Usage = %+v, want nil when the service reports none", result.Usage)
	}
}

func TestCompleteErrorMapsProviderError(t *testing.T) {
	apiErr := &awshttp.ResponseError{
		ResponseError: &smithyhttp.ResponseError{
			Response: &smithyhttp.Response{Response: &http.Response{StatusCode: http.StatusBadRequest}},
			Err:      errors.New("bad request"),
		},
	}
	api := &fakeAPI{converseErr: fmt.Errorf("operation error Bedrock Runtime: Converse: %w", apiErr)}
	completion := newWithAPI("bedrock", nil, api)
	_, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "m", Prompt: "hi"})
	if err == nil {
		t.Fatal("Complete() error = nil, want a provider error")
	}
	if !errors.Is(err, modelrouter.ErrProvider) {
		t.Errorf("errors.Is(err, ErrProvider) = false, err = %v", err)
	}
	var providerErr *modelrouter.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("errors.As(*ProviderError) = false, err = %v", err)
	}
	if providerErr.Provider != "bedrock" || providerErr.Op != "complete" || providerErr.StatusCode != http.StatusBadRequest {
		t.Errorf("ProviderError = %+v, want bedrock/complete/400", providerErr)
	}
}

func TestCompleteNonMessageOutputIsEmptyContent(t *testing.T) {
	// The union currently has one variant; a nil Output must not panic.
	api := &fakeAPI{converseOut: &bedrockruntime.ConverseOutput{StopReason: types.StopReasonEndTurn}}
	completion := newWithAPI("bedrock", nil, api)
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "m", Prompt: "hi"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "" {
		t.Errorf("Content = %q, want empty", result.Content)
	}
}

func TestNewValidatesRegion(t *testing.T) {
	if _, err := New(provider.Config{}); err == nil {
		t.Fatal("New() error = nil, want Region validation")
	}
	// A region is enough: the default credential chain is resolved lazily,
	// so construction is offline.
	completion, err := New(provider.Config{Region: "us-east-1"})
	if err != nil {
		t.Fatalf("New(region) error = %v", err)
	}
	if completion.name != "bedrock" {
		t.Errorf("name = %q, want bedrock", completion.name)
	}
}

func TestValidationErrorsArePlain(t *testing.T) {
	completion := newWithAPI("bedrock", nil, &fakeAPI{})
	cases := []struct {
		name    string
		request modelrouter.CompletionRequest
	}{
		{"model", modelrouter.CompletionRequest{Prompt: "hi"}},
		{"prompt or messages", modelrouter.CompletionRequest{Model: "m"}},
		{"unsupported role", modelrouter.CompletionRequest{Model: "m", Messages: []modelrouter.Message{{Role: "tool", Content: "x"}}}},
		{"system only", modelrouter.CompletionRequest{Model: "m", Messages: []modelrouter.Message{{Role: "system", Content: "x"}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := completion.Complete(context.Background(), tc.request); err == nil {
				t.Error("Complete() error = nil, want a validation error")
			} else if errors.Is(err, modelrouter.ErrProvider) {
				t.Errorf("validation error wrapped as provider error: %v", err)
			}
		})
	}
}
