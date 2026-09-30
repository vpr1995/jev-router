package gemini_test

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
	"github.com/vprprudhvi/jev-router/provider/gemini"
)

const testAPIKey = "test-key"

// newCompletion starts a fake Gemini server and points the provider at it
// through genai.ClientConfig.HTTPOptions.BaseURL (requests hit
// "<base>/v1beta/models/<model>:generateContent"). Error tests use 400/401
// bodies to avoid SDK retries.
func newCompletion(t *testing.T, handler http.Handler) *gemini.Completion {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	completion, err := gemini.New(provider.Config{
		Name:    "gemini",
		APIKey:  testAPIKey,
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("gemini.New() error = %v", err)
	}
	return completion
}

// capturedRequest is the decoded shape of the requests the provider sends.
type capturedRequest struct {
	Contents []struct {
		Role  string `json:"role"`
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"contents"`
	SystemInstruction *struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"systemInstruction"`
	GenerationConfig *struct {
		Temperature     *float64 `json:"temperature"`
		MaxOutputTokens *int32   `json:"maxOutputTokens"`
	} `json:"generationConfig"`
}

func decodeRequest(t *testing.T, r *http.Request) capturedRequest {
	t.Helper()
	var body capturedRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decoding request body: %v", err)
	}
	return body
}

func checkRequestEnvelope(t *testing.T, r *http.Request, wantPath string) {
	t.Helper()
	if r.Method != http.MethodPost {
		t.Errorf("method = %q, want POST", r.Method)
	}
	if r.URL.Path != wantPath {
		t.Errorf("path = %q, want %q", r.URL.Path, wantPath)
	}
	if got := r.Header.Get("x-goog-api-key"); got != testAPIKey {
		t.Errorf("x-goog-api-key = %q, want %q", got, testAPIKey)
	}
}

// writeJSON answers with an application/json body.
func writeJSON(t *testing.T, w http.ResponseWriter, body string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, body)
}

func TestCompleteMapsResponse(t *testing.T) {
	const responseBody = `{
		"candidates": [{
			"content": {"role": "model", "parts": [{"text": "Hello world"}]},
			"finishReason": "STOP"
		}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14}
	}`
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkRequestEnvelope(t, r, "/v1beta/models/gemini-test:generateContent")
		body := decodeRequest(t, r)
		if len(body.Contents) != 2 {
			t.Fatalf("body contents = %+v, want 2", body.Contents)
		}
		if body.Contents[0].Role != "user" || body.Contents[0].Parts[0].Text != "hi" {
			t.Errorf("contents[0] = %+v, want user/hi", body.Contents[0])
		}
		if body.Contents[1].Role != "model" || body.Contents[1].Parts[0].Text != "hello" {
			t.Errorf("contents[1] = %+v, want model/hello (assistant maps to model)", body.Contents[1])
		}
		if body.SystemInstruction == nil || len(body.SystemInstruction.Parts) != 1 || body.SystemInstruction.Parts[0].Text != "be brief\n\nand exact" {
			t.Errorf("systemInstruction = %+v, want one part %q", body.SystemInstruction, "be brief\n\nand exact")
		}
		if body.GenerationConfig == nil {
			t.Fatal("generationConfig = nil, want temperature and maxOutputTokens")
		}
		if body.GenerationConfig.Temperature == nil || *body.GenerationConfig.Temperature != 0.25 {
			t.Errorf("temperature = %v, want 0.25", body.GenerationConfig.Temperature)
		}
		if body.GenerationConfig.MaxOutputTokens == nil || *body.GenerationConfig.MaxOutputTokens != 128 {
			t.Errorf("maxOutputTokens = %v, want 128", body.GenerationConfig.MaxOutputTokens)
		}
		writeJSON(t, w, responseBody)
	}))
	temperature := 0.25
	request := modelrouter.CompletionRequest{
		Model: "gemini-test",
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
	if result.FinishReason != "STOP" {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, "STOP")
	}
	if result.Model != "gemini-test" || result.Provider != "gemini" {
		t.Errorf("Model/Provider = %q/%q, want gemini-test/gemini", result.Model, result.Provider)
	}
	if result.Usage == nil {
		t.Fatal("Usage = nil, want 10/4/14")
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 4 || result.Usage.TotalTokens != 14 {
		t.Errorf("Usage = %+v, want 10/4/14", result.Usage)
	}
	if !strings.Contains(string(result.Raw), `"Hello world"`) {
		t.Errorf("Raw = %s, want it to contain the generated text", result.Raw)
	}
	if !reflect.DeepEqual(request.Messages, before) {
		t.Errorf("caller request mutated: %+v, want %+v", request.Messages, before)
	}
}

func TestCompletePromptOnly(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeRequest(t, r)
		if len(body.Contents) != 1 || body.Contents[0].Role != "user" || body.Contents[0].Parts[0].Text != "hello" {
			t.Errorf("body contents = %+v, want one user content %q", body.Contents, "hello")
		}
		if body.SystemInstruction != nil {
			t.Errorf("systemInstruction = %+v, want nil", body.SystemInstruction)
		}
		if body.GenerationConfig == nil || body.GenerationConfig.MaxOutputTokens != nil || body.GenerationConfig.Temperature != nil {
			t.Errorf("generationConfig = %+v, want no overrides", body.GenerationConfig)
		}
		writeJSON(t, w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	result, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "gemini-test", Prompt: "hello"})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.Content != "ok" {
		t.Errorf("Content = %q, want %q", result.Content, "ok")
	}
	if result.Usage != nil {
		t.Errorf("Usage = %+v, want nil when the response reports none", result.Usage)
	}
}

func TestCompleteErrorMapsProviderError(t *testing.T) {
	completion := newCompletion(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"message":"bad request","status":"INVALID_ARGUMENT"}}`)
	}))
	_, err := completion.Complete(context.Background(), modelrouter.CompletionRequest{Model: "gemini-test", Prompt: "hi"})
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
	if providerErr.Provider != "gemini" || providerErr.Op != "complete" || providerErr.StatusCode != http.StatusBadRequest {
		t.Errorf("ProviderError = %+v, want gemini/complete/400", providerErr)
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	if _, err := gemini.New(provider.Config{}); err == nil {
		t.Fatal("New() error = nil, want APIKey validation")
	}
}

func TestValidationErrorsArePlain(t *testing.T) {
	completion, err := gemini.New(provider.Config{APIKey: testAPIKey})
	if err != nil {
		t.Fatalf("gemini.New() error = %v", err)
	}
	cases := []struct {
		name    string
		request modelrouter.CompletionRequest
	}{
		{"model", modelrouter.CompletionRequest{Prompt: "hi"}},
		{"prompt or messages", modelrouter.CompletionRequest{Model: "gemini-test"}},
		{"unsupported role", modelrouter.CompletionRequest{Model: "gemini-test", Messages: []modelrouter.Message{{Role: "tool", Content: "x"}}}},
		{"system only", modelrouter.CompletionRequest{Model: "gemini-test", Messages: []modelrouter.Message{{Role: "system", Content: "x"}}}},
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
