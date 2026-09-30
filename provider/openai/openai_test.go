package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
	"github.com/vprprudhvi/jev-router/provider/openai"
)

const testAPIKey = "test-key"

// newCompletion starts a fake OpenAI server and points the provider at its
// /v1 root through option.WithBaseURL (the SDK appends /chat/completions).
// Error tests use 400/401 bodies: the SDK retries 408/409/429/5xx only.
func newCompletion(t *testing.T, handler http.Handler) *openai.Completion {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	completion, err := openai.New(provider.Config{
		Name:    "openai",
		APIKey:  testAPIKey,
		BaseURL: server.URL + "/v1",
	})
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	return completion
}

// capturedRequest is the decoded shape of the requests the provider sends.
type capturedRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	MaxTokens   *int     `json:"max_tokens"`
	Temperature *float64 `json:"temperature"`
}

func decodeRequest(t *testing.T, r *http.Request) capturedRequest {
	t.Helper()
	var body capturedRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decoding request body: %v", err)
	}
	return body
}

func checkRequestEnvelope(t *testing.T, r *http.Request) {
	t.Helper()
	if r.Method != http.MethodPost {
		t.Errorf("method = %q, want POST", r.Method)
	}
	if r.URL.Path != "/v1/chat/completions" {
		t.Errorf("path = %q, want /v1/chat/completions", r.URL.Path)
	}
	if got := r.Header.Get("Authorization"); got != "Bearer "+testAPIKey {
		t.Errorf("Authorization = %q, want %q", got, "Bearer "+testAPIKey)
	}
}

func TestCompleteMapsResponse(t *testing.T) {
	const responseBody = `{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"created": 1,
		"model": "gpt-test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "hello world", "refusal": null},
			"logprobs": null,
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkRequestEnvelope(t, r)
		body := decodeRequest(t, r)
		if body.Model != "gpt-test" {
			t.Errorf("body model = %q, want %q", body.Model, "gpt-test")
		}
		if len(body.Messages) != 2 || body.Messages[0].Role != "system" || body.Messages[1].Role != "user" {
			t.Errorf("body messages = %+v, want [system user]", body.Messages)
		}
		if body.MaxTokens == nil || *body.MaxTokens != 32 {
			t.Errorf("body max_tokens = %v, want 32", body.MaxTokens)
		}
		if body.Temperature == nil || *body.Temperature != 0.5 {
			t.Errorf("body temperature = %v, want 0.5", body.Temperature)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, responseBody)
	}))

	temperature := 0.5
	req := modelrouter.CompletionRequest{
		Messages: []modelrouter.Message{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "hello"},
		},
		Model:       "gpt-test",
		MaxTokens:   32,
		Temperature: &temperature,
	}
	// Snapshot to prove the provider never mutates the caller's request.
	before := req
	before.Messages = append([]modelrouter.Message(nil), req.Messages...)

	result, err := completion.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "hello world" {
		t.Errorf("Content = %q, want %q", result.Content, "hello world")
	}
	if result.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, "stop")
	}
	if result.Model != "gpt-test" {
		t.Errorf("Model = %q, want the request model %q", result.Model, "gpt-test")
	}
	if result.Provider != "openai" {
		t.Errorf("Provider = %q, want %q", result.Provider, "openai")
	}
	if result.Usage == nil {
		t.Fatal("Usage = nil, want token usage")
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 || result.Usage.TotalTokens != 15 {
		t.Errorf("Usage = %+v, want 10/5/15", result.Usage)
	}
	if result.Usage.CostUSD != nil {
		t.Errorf("Usage.CostUSD = %v, want nil (OpenAI reports no cost)", result.Usage.CostUSD)
	}
	var raw map[string]any
	if err := json.Unmarshal(result.Raw, &raw); err != nil {
		t.Errorf("Raw is not valid JSON: %v (%s)", err, result.Raw)
	} else if raw["id"] != "chatcmpl-1" {
		t.Errorf("Raw id = %v, want chatcmpl-1", raw["id"])
	}
	if !reflect.DeepEqual(result.RoutingDecision, modelrouter.RoutingDecision{}) {
		t.Errorf("RoutingDecision = %+v, want zero value (the Router owns it)", result.RoutingDecision)
	}
	if !reflect.DeepEqual(req, before) {
		t.Errorf("Complete() mutated the caller's request: got %+v, want %+v", req, before)
	}
}

func TestCompletePromptBecomesSingleUserMessage(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkRequestEnvelope(t, r)
		body := decodeRequest(t, r)
		if len(body.Messages) != 1 {
			t.Fatalf("messages = %+v, want one message", body.Messages)
		}
		if body.Messages[0].Role != "user" || body.Messages[0].Content != "hello" {
			t.Errorf("message = %+v, want {user hello}", body.Messages[0])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"ok","refusal":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))

	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{
		Prompt: "hello",
		Model:  "gpt-test",
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "ok" {
		t.Errorf("Content = %q, want %q", result.Content, "ok")
	}
}

func TestCompleteWithoutChoicesDoesNotPanic(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-test","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`)
	}))
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi", Model: "gpt-test"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "" {
		t.Errorf("Content = %q, want empty", result.Content)
	}
	if result.FinishReason != "" {
		t.Errorf("FinishReason = %q, want empty", result.FinishReason)
	}
}

func TestCompleteErrorWrappedInProviderError(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkRequestEnvelope(t, r)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}`)
	}))

	_, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi", Model: "gpt-test"})
	if err == nil {
		t.Fatal("Complete() error = nil, want a 401 provider error")
	}
	if !errors.Is(err, modelrouter.ErrProvider) {
		t.Errorf("errors.Is(err, ErrProvider) = false, err = %v", err)
	}
	var providerErr *modelrouter.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("errors.As(*ProviderError) = false, err = %v", err)
	}
	if providerErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("StatusCode = %d, want 401", providerErr.StatusCode)
	}
	if providerErr.Provider != "openai" {
		t.Errorf("Provider = %q, want %q", providerErr.Provider, "openai")
	}
	if providerErr.Op != "complete" {
		t.Errorf("Op = %q, want %q", providerErr.Op, "complete")
	}
}

func TestNewCompatibleRequiresBaseURL(t *testing.T) {
	completion, err := openai.NewCompatible("groq", provider.Config{APIKey: testAPIKey})
	if err == nil {
		t.Fatalf("NewCompatible() = %v, want an error without BaseURL", completion)
	}
}

func TestNewCompatibleUsesBaseURLAndName(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/custom/v1/chat/completions" {
			t.Errorf("path = %q, want /custom/v1/chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+testAPIKey {
			t.Errorf("Authorization = %q, want %q", got, "Bearer "+testAPIKey)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"llama-3","choices":[{"index":0,"message":{"role":"assistant","content":"hi from groq","refusal":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`)
	}))
	t.Cleanup(server.Close)

	completion, err := openai.NewCompatible("groq", provider.Config{
		APIKey:  testAPIKey,
		BaseURL: server.URL + "/custom/v1",
	})
	if err != nil {
		t.Fatalf("NewCompatible() error = %v", err)
	}
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi", Model: "llama-3"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Provider != "groq" {
		t.Errorf("Provider = %q, want the compatible provider name %q", result.Provider, "groq")
	}
	if result.Content != "hi from groq" {
		t.Errorf("Content = %q, want %q", result.Content, "hi from groq")
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	completion, err := openai.New(provider.Config{Name: "openai"})
	if err == nil {
		t.Fatalf("New() = %v, want an error for an empty API key", completion)
	}
}

func TestCompleteRequiresModel(t *testing.T) {
	completion, err := openai.New(provider.Config{Name: "openai", APIKey: testAPIKey})
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	if _, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"}); err == nil {
		t.Error("Complete() error = nil, want an error for an empty model")
	}
	if _, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "m"}); err == nil {
		t.Error("Complete() error = nil, want an error when prompt and messages are both empty")
	}
}
