package server_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/server"
)

func chatResult() modelrouter.CompletionResult {
	return modelrouter.CompletionResult{
		RoutingDecision: modelrouter.RoutingDecision{ModelID: "vendor/cheap", Provider: "openrouter", Source: modelrouter.SourceClassification},
		Model:           "vendor/cheap",
		Provider:        "openrouter",
		Content:         "Paris.",
		FinishReason:    "end_turn",
		Usage:           &modelrouter.Usage{InputTokens: 7, OutputTokens: 2},
	}
}

// TestOpenAISDKRoundTrip drives the handler with the official OpenAI Go SDK,
// which fails on any response-shape mismatch.
func TestOpenAISDKRoundTrip(t *testing.T) {
	fr := &fakeRouter{completeResult: chatResult()}
	ts := newTestEnv(t, fr)
	client := openai.NewClient(
		option.WithBaseURL(ts.URL+"/v1"),
		option.WithAPIKey("ignored"),
		option.WithMaxRetries(0),
	)

	completion, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model: "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.DeveloperMessage("be brief"),
			openai.UserMessage("capital of France?"),
		},
		MaxCompletionTokens: openai.Int(64),
		Temperature:         openai.Float(0.2),
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New() error = %v", err)
	}
	if got := completion.Choices[0].Message.Content; got != "Paris." {
		t.Errorf("content = %q, want %q", got, "Paris.")
	}
	if completion.Model != "vendor/cheap" {
		t.Errorf("model = %q, want the served model", completion.Model)
	}
	if got := completion.Choices[0].FinishReason; got != "stop" {
		t.Errorf("finish_reason = %q, want stop (end_turn normalized)", got)
	}
	if u := completion.Usage; u.PromptTokens != 7 || u.CompletionTokens != 2 || u.TotalTokens != 9 {
		t.Errorf("usage = %+v, want 7/2/9", u)
	}
	if !strings.HasPrefix(completion.ID, "chatcmpl-") || completion.Object != "chat.completion" || completion.Created == 0 {
		t.Errorf("envelope = id %q object %q created %d", completion.ID, completion.Object, completion.Created)
	}

	reqs := fr.completeCalls()
	if len(reqs) != 1 {
		t.Fatalf("Complete calls = %d, want 1", len(reqs))
	}
	got := reqs[0]
	if len(got.Messages) != 2 || got.Messages[0].Role != "system" || got.Messages[1].Content != "capital of France?" {
		t.Errorf("messages = %+v, want developer mapped to system", got.Messages)
	}
	if got.MaxTokens != 64 || got.Temperature == nil || *got.Temperature != 0.2 {
		t.Errorf("max_tokens/temperature = %d/%v", got.MaxTokens, got.Temperature)
	}
}

func TestOpenAISDKErrorAndModels(t *testing.T) {
	fr := &fakeRouter{completeErr: modelrouter.ErrNoCandidate}
	ts := newTestEnv(t, fr)
	client := openai.NewClient(option.WithBaseURL(ts.URL+"/v1"), option.WithAPIKey("x"), option.WithMaxRetries(0))

	_, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model:    "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want *openai.Error", err)
	}
	if apiErr.StatusCode != http.StatusUnprocessableEntity || apiErr.Code != "no_candidate" || apiErr.Message == "" {
		t.Errorf("error = status %d code %q message %q", apiErr.StatusCode, apiErr.Code, apiErr.Message)
	}

	page, err := client.Models.List(context.Background())
	if err != nil {
		t.Fatalf("Models.List() error = %v", err)
	}
	if len(page.Data) != 1 || page.Data[0].ID != "auto" {
		t.Errorf("models = %+v, want the single auto model", page.Data)
	}
	model, err := client.Models.Get(context.Background(), "auto")
	if err != nil || model.ID != "auto" {
		t.Errorf("Models.Get(auto) = %+v, %v", model, err)
	}
	if _, err := client.Models.Get(context.Background(), "gpt-4o"); !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusNotFound {
		t.Errorf("Models.Get(gpt-4o) error = %v, want 404", err)
	}
}

func TestChatCompletionsRejections(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantCode  string
		wantParam string
	}{
		{"stream", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, "streaming_not_supported", "stream"},
		{"n>1", `{"n":2,"messages":[{"role":"user","content":"hi"}]}`, "unsupported_parameter", "n"},
		{"tools", `{"tools":[{"type":"function"}],"messages":[{"role":"user","content":"hi"}]}`, "unsupported_parameter", "tools"},
		{"functions", `{"functions":[{"name":"f"}],"messages":[{"role":"user","content":"hi"}]}`, "unsupported_parameter", "tools"},
		{"no messages", `{"model":"m"}`, "invalid_request", "messages"},
		{"tool role", `{"messages":[{"role":"tool","content":"x"}]}`, "unsupported_parameter", "messages[0].role"},
		{"image part", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]}`, "unsupported_parameter", "messages[0].content"},
		{"bad content", `{"messages":[{"role":"user","content":42}]}`, "unsupported_parameter", "messages[0].content"},
		{"json_object", `{"response_format":{"type":"json_object"},"messages":[{"role":"user","content":"hi"}]}`, "unsupported_parameter", "response_format"},
		{"json_schema", `{"response_format":{"type":"json_schema"},"messages":[{"role":"user","content":"hi"}]}`, "unsupported_parameter", "response_format"},
		{"logprobs", `{"logprobs":true,"messages":[{"role":"user","content":"hi"}]}`, "unsupported_parameter", "logprobs"},
		{"stop", `{"stop":["END"],"messages":[{"role":"user","content":"hi"}]}`, "unsupported_parameter", "stop"},
		{"null content", `{"messages":[{"role":"assistant","content":null}]}`, "invalid_request", "messages[0].content"},
		{"empty content", `{"messages":[{"role":"user","content":""}]}`, "invalid_request", "messages[0].content"},
		{"negative max_tokens", `{"max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}`, "invalid_request", "max_tokens"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fr := &fakeRouter{completeResult: chatResult()}
			ts := newTestEnv(t, fr)
			resp, body := do(t, ts, http.MethodPost, "/v1/chat/completions", tt.body, nil)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (body %s)", resp.StatusCode, body)
			}
			got := decodeJSON[struct {
				Error struct {
					Message string  `json:"message"`
					Type    string  `json:"type"`
					Param   *string `json:"param"`
					Code    *string `json:"code"`
				} `json:"error"`
			}](t, body).Error
			if got.Type != "invalid_request_error" || got.Message == "" ||
				got.Code == nil || *got.Code != tt.wantCode || got.Param == nil || *got.Param != tt.wantParam {
				t.Errorf("error = %+v, want code %q param %q", got, tt.wantCode, tt.wantParam)
			}
			if len(fr.completeCalls()) != 0 {
				t.Error("router was called for a rejected request")
			}
		})
	}
}

func TestChatCompletionsAcceptsOpenAIQuirks(t *testing.T) {
	fr := &fakeRouter{completeResult: chatResult()}
	ts := newTestEnv(t, fr)
	body := `{"model":"gpt-4o","n":1,"stream":false,"tools":[],"top_p":0.9,"user":"u","response_format":{"type":"text"},
		"messages":[
			{"role":"system","content":[{"type":"text","text":"be "},{"type":"text","text":"brief"}]},
			{"role":"user","content":"hi"},
			{"role":"assistant","content":"ok"},
			{"role":"user","content":[{"type":"text","text":"again"}]}
		],"max_tokens":10,"max_completion_tokens":20}`
	resp, raw := do(t, ts, http.MethodPost, "/v1/chat/completions", body, map[string]string{"Authorization": "Bearer sk-anything"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, body %s", resp.StatusCode, raw)
	}
	if resp.Header.Get("X-Jev-Router-Provider") != "openrouter" || resp.Header.Get("X-Jev-Router-Model") != "vendor/cheap" ||
		resp.Header.Get("X-Jev-Router-Source") != "classification" {
		t.Errorf("routing headers = %v", resp.Header)
	}
	got := fr.completeCalls()[0]
	if got.Messages[0].Content != "be brief" || got.Messages[2].Content != "ok" || got.Messages[3].Content != "again" {
		t.Errorf("messages = %+v", got.Messages)
	}
	if got.MaxTokens != 20 {
		t.Errorf("MaxTokens = %d, want max_completion_tokens (20) to win", got.MaxTokens)
	}
}

// TestServerErrorsAreNotRetriedBySDK guards against retry amplification: the
// SDK's default retries would otherwise re-run the whole candidate chain.
func TestServerErrorsAreNotRetriedBySDK(t *testing.T) {
	fr := &fakeRouter{completeErr: &modelrouter.ProviderError{Provider: "p", Op: "complete", Err: errors.New("down")}}
	ts := newTestEnv(t, fr)
	client := openai.NewClient(option.WithBaseURL(ts.URL+"/v1"), option.WithAPIKey("x"))
	_, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model:    "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusBadGateway {
		t.Fatalf("error = %v, want a 502 *openai.Error", err)
	}
	if n := len(fr.completeCalls()); n != 1 {
		t.Errorf("Complete calls = %d, want 1 (no SDK retries)", n)
	}
}

func TestChatCompletionsConstraintsExtension(t *testing.T) {
	fr := &fakeRouter{completeResult: chatResult()}
	ts := newTestEnv(t, fr)
	do(t, ts, http.MethodPost, "/v1/chat/completions",
		`{"messages":[{"role":"user","content":"hi"}],"constraints":{"needs_vision":true}}`, nil)
	if c := fr.completeCalls()[0].Constraints; c == nil || !c.NeedsVision {
		t.Errorf("constraints = %+v, want needs_vision", c)
	}
}

func TestCompatPathsUseOpenAIEnvelopeForTransportErrors(t *testing.T) {
	fr := &fakeRouter{}
	ts := newTestEnv(t, fr)

	resp, body := do(t, ts, http.MethodGet, "/v1/chat/completions", "", nil)
	if resp.StatusCode != http.StatusMethodNotAllowed || resp.Header.Get("Allow") != http.MethodPost {
		t.Errorf("GET chat: status %d Allow %q", resp.StatusCode, resp.Header.Get("Allow"))
	}
	requireJSONKeys(t, body, "error")
	if !strings.Contains(string(body), `"code":"method_not_allowed"`) {
		t.Errorf("body = %s, want OpenAI envelope with method_not_allowed code", body)
	}

	resp, _ = do(t, ts, http.MethodPost, "/v1/models", "{}", nil)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("POST models status = %d, want 405", resp.StatusCode)
	}
}

func TestRequestBodyLimit(t *testing.T) {
	srv, err := server.New(server.Options{Router: &fakeRouter{completeResult: chatResult()}, MaxBodyBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/v1/chat/completions", "/v1/complete", "/v1/route"} {
		rec := httptest.NewRecorder()
		req, _ := http.NewRequest(http.MethodPost, path, strings.NewReader(`{"prompt":"`+strings.Repeat("x", 200)+`"}`))
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusRequestEntityTooLarge {
			t.Errorf("%s: status = %d, want 413 (body %s)", path, rec.Code, rec.Body.String())
		}
	}
}

func TestNormalizedFinishReasonsOnTheWire(t *testing.T) {
	for reason, want := range map[string]string{
		"stop": "stop", "STOP": "stop", "": "stop", "tool_use": "stop", "weird": "stop",
		"length": "length", "max_tokens": "length", "MAX_TOKENS": "length",
		"content_filter": "content_filter", "SAFETY": "content_filter", "guardrail_intervened": "content_filter",
	} {
		res := chatResult()
		res.FinishReason = reason
		ts := newTestEnv(t, &fakeRouter{completeResult: res})
		_, body := do(t, ts, http.MethodPost, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
		got := decodeJSON[struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}](t, body)
		if got.Choices[0].FinishReason != want {
			t.Errorf("finish %q -> %q, want %q", reason, got.Choices[0].FinishReason, want)
		}
	}
}
