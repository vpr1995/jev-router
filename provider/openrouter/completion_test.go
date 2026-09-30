package openrouter_test

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
	"github.com/vprprudhvi/jev-router/provider/openrouter"
)

const testAPIKey = "test-key"

// newCompletion starts a fake OpenRouter server and points the provider at it
// through the SDK server-URL override (Config.BaseURL).
func newCompletion(t *testing.T, handler http.Handler) *openrouter.Completion {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	completion, err := openrouter.New(provider.Config{
		Name:    "openrouter",
		APIKey:  testAPIKey,
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("openrouter.New() error = %v", err)
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
	if r.URL.Path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", r.URL.Path)
	}
	if got := r.Header.Get("Authorization"); got != "Bearer "+testAPIKey {
		t.Errorf("Authorization = %q, want %q", got, "Bearer "+testAPIKey)
	}
}

func TestCompleteMapsResponse(t *testing.T) {
	const responseBody = `{
		"id": "gen-1",
		"object": "chat.completion",
		"created": 1,
		"model": "vendor/model",
		"choices": [{
			"index": 0,
			"finish_reason": "stop",
			"message": {"role": "assistant", "content": "hello world"}
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.00123}
	}`
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkRequestEnvelope(t, r)
		body := decodeRequest(t, r)
		if body.Model != "vendor/model" {
			t.Errorf("body model = %q, want %q", body.Model, "vendor/model")
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
		Model:       "vendor/model",
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
	if result.Model != "vendor/model" {
		t.Errorf("Model = %q, want the request model %q", result.Model, "vendor/model")
	}
	if result.Provider != "openrouter" {
		t.Errorf("Provider = %q, want %q", result.Provider, "openrouter")
	}
	if result.Usage == nil {
		t.Fatal("Usage = nil, want token usage")
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 || result.Usage.TotalTokens != 15 {
		t.Errorf("Usage = %+v, want 10/5/15", result.Usage)
	}
	if result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0.00123 {
		t.Errorf("Usage.CostUSD = %v, want 0.00123", result.Usage.CostUSD)
	}
	var raw map[string]any
	if err := json.Unmarshal(result.Raw, &raw); err != nil {
		t.Errorf("Raw is not valid JSON: %v (%s)", err, result.Raw)
	} else if raw["id"] != "gen-1" {
		t.Errorf("Raw id = %v, want gen-1", raw["id"])
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
		_, _ = fmt.Fprint(w, `{"id":"1","object":"chat.completion","created":1,"model":"vendor/model","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`)
	}))

	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{
		Prompt: "hello",
		Model:  "vendor/model",
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "ok" {
		t.Errorf("Content = %q, want %q", result.Content, "ok")
	}
	if result.Usage != nil {
		t.Errorf("Usage = %+v, want nil when the provider reports none", result.Usage)
	}
}

func TestCompleteWithoutChoicesDoesNotPanic(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"1","object":"chat.completion","created":1,"model":"vendor/model","choices":[]}`)
	}))
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi", Model: "vendor/model"})
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
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"message":"invalid model"}}`)
	}))

	_, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi", Model: "nope"})
	if err == nil {
		t.Fatal("Complete() error = nil, want a 400 provider error")
	}
	if !errors.Is(err, modelrouter.ErrProvider) {
		t.Errorf("errors.Is(err, ErrProvider) = false, err = %v", err)
	}
	var providerErr *modelrouter.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("errors.As(*ProviderError) = false, err = %v", err)
	}
	if providerErr.StatusCode != http.StatusBadRequest {
		t.Errorf("StatusCode = %d, want 400", providerErr.StatusCode)
	}
	if providerErr.Provider != "openrouter" {
		t.Errorf("Provider = %q, want %q", providerErr.Provider, "openrouter")
	}
	if providerErr.Op != "complete" {
		t.Errorf("Op = %q, want %q", providerErr.Op, "complete")
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	completion, err := openrouter.New(provider.Config{Name: "openrouter"})
	if err == nil {
		t.Fatalf("New() = %v, want an error for an empty API key", completion)
	}
}

func TestCompleteRequiresModel(t *testing.T) {
	completion, err := openrouter.New(provider.Config{Name: "openrouter", APIKey: testAPIKey})
	if err != nil {
		t.Fatalf("openrouter.New() error = %v", err)
	}
	if _, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Prompt: "hi"}); err == nil {
		t.Error("Complete() error = nil, want an error for an empty model")
	}
	if _, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "m"}); err == nil {
		t.Error("Complete() error = nil, want an error when prompt and messages are both empty")
	}
}
