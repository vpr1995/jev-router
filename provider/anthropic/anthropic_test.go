package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/provider"
	"github.com/vprprudhvi/jev-router/provider/anthropic"
)

const testAPIKey = "test-key"

// newCompletion starts a fake Anthropic server and points the provider at it
// through option.WithBaseURL (the SDK appends /v1/messages). Error tests use
// 400/401 bodies: the SDK retries 408/409/429/5xx only.
func newCompletion(t *testing.T, handler http.Handler) *anthropic.Completion {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	completion, err := anthropic.New(provider.Config{
		Name:    "anthropic",
		APIKey:  testAPIKey,
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("anthropic.New() error = %v", err)
	}
	return completion
}

// capturedRequest is the decoded shape of the requests the provider sends.
type capturedRequest struct {
	Model       string   `json:"model"`
	MaxTokens   *int64   `json:"max_tokens"`
	Temperature *float64 `json:"temperature"`
	System      []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"system"`
	Messages []struct {
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"messages"`
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
	if r.URL.Path != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", r.URL.Path)
	}
	if got := r.Header.Get("x-api-key"); got != testAPIKey {
		t.Errorf("x-api-key = %q, want %q", got, testAPIKey)
	}
	if got := r.Header.Get("anthropic-version"); got == "" {
		t.Error("anthropic-version header missing")
	}
}

// writeJSON answers with an application/json body (the SDK rejects
// text/plain responses).
func writeJSON(t *testing.T, w http.ResponseWriter, body string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, body)
}

func TestCompleteMapsResponse(t *testing.T) {
	const responseBody = `{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "claude-test",
		"content": [
			{"type": "text", "text": "Hello "},
			{"type": "text", "text": "world"}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 4}
	}`
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkRequestEnvelope(t, r)
		body := decodeRequest(t, r)
		if body.Model != "claude-test" {
			t.Errorf("body model = %q, want %q", body.Model, "claude-test")
		}
		if body.MaxTokens == nil || *body.MaxTokens != 128 {
			t.Errorf("body max_tokens = %v, want 128", body.MaxTokens)
		}
		if body.Temperature == nil || *body.Temperature != 0.25 {
			t.Errorf("body temperature = %v, want 0.25", body.Temperature)
		}
		if len(body.System) != 1 || body.System[0].Text != "be brief\n\nand exact" {
			t.Errorf("body system = %+v, want one block %q", body.System, "be brief\n\nand exact")
		}
		if len(body.Messages) != 2 {
			t.Fatalf("body messages = %+v, want 2", body.Messages)
		}
		if body.Messages[0].Role != "user" || body.Messages[0].Content[0].Text != "hi" {
			t.Errorf("message[0] = %+v, want user/hi", body.Messages[0])
		}
		if body.Messages[1].Role != "assistant" || body.Messages[1].Content[0].Text != "hello" {
			t.Errorf("message[1] = %+v, want assistant/hello", body.Messages[1])
		}
		writeJSON(t, w, responseBody)
	}))
	temperature := 0.25
	request := modelrouter.CompletionRequest{
		Model: "claude-test",
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
	if result.Content != "Hello world" {
		t.Errorf("Content = %q, want %q", result.Content, "Hello world")
	}
	if result.FinishReason != "end_turn" {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, "end_turn")
	}
	if result.Model != "claude-test" || result.Provider != "anthropic" {
		t.Errorf("Model/Provider = %q/%q, want claude-test/anthropic", result.Model, result.Provider)
	}
	if result.Usage == nil {
		t.Fatal("Usage = nil, want 10/4/14")
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 4 || result.Usage.TotalTokens != 14 {
		t.Errorf("Usage = %+v, want 10/4/14", result.Usage)
	}
	if !strings.Contains(string(result.Raw), `"end_turn"`) {
		t.Errorf("Raw = %s, want it to contain the stop reason", result.Raw)
	}
	if !reflect.DeepEqual(request.Messages, before) {
		t.Errorf("caller request mutated: %+v, want %+v", request.Messages, before)
	}
}

func TestCompleteDefaultsMaxTokens(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeRequest(t, r)
		if body.MaxTokens == nil || *body.MaxTokens != 4096 {
			t.Errorf("body max_tokens = %v, want the 4096 default", body.MaxTokens)
		}
		writeJSON(t, w, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "claude-test", Prompt: "hi"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "ok" {
		t.Errorf("Content = %q, want %q", result.Content, "ok")
	}
}

func TestCompletePromptOnlyBecomesUserMessage(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeRequest(t, r)
		if len(body.Messages) != 1 || body.Messages[0].Role != "user" || body.Messages[0].Content[0].Text != "hello" {
			t.Errorf("body messages = %+v, want one user message %q", body.Messages, "hello")
		}
		if len(body.System) != 0 {
			t.Errorf("body system = %+v, want none", body.System)
		}
		writeJSON(t, w, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`)
	}))
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "claude-test", Prompt: "hello"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Usage != nil {
		t.Errorf("Usage = %+v, want nil when the response reports none", result.Usage)
	}
}

func TestCompleteErrorMapsProviderError(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`)
	}))
	_, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "claude-test", Prompt: "hi"})
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
	if providerErr.Provider != "anthropic" || providerErr.Op != "complete" || providerErr.StatusCode != http.StatusBadRequest {
		t.Errorf("ProviderError = %+v, want anthropic/complete/400", providerErr)
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	if _, err := anthropic.New(provider.Config{}); err == nil {
		t.Fatal("New() error = nil, want APIKey validation")
	}
}

func TestValidationErrorsArePlain(t *testing.T) {
	completion, err := anthropic.New(provider.Config{APIKey: testAPIKey})
	if err != nil {
		t.Fatalf("anthropic.New() error = %v", err)
	}
	cases := []struct {
		name    string
		request modelrouter.CompletionRequest
	}{
		{"model", modelrouter.CompletionRequest{Prompt: "hi"}},
		{"prompt or messages", modelrouter.CompletionRequest{Model: "claude-test"}},
		{"unsupported role", modelrouter.CompletionRequest{Model: "claude-test", Messages: []modelrouter.Message{{Role: "tool", Content: "x"}}}},
		{"system only", modelrouter.CompletionRequest{Model: "claude-test", Messages: []modelrouter.Message{{Role: "system", Content: "x"}}}},
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
