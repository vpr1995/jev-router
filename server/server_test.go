// Black-box tests for the HTTP server: endpoint routing, request validation,
// error-status mapping, middleware (recovery, request IDs, logging) and
// graceful shutdown. They use httptest.NewServer(srv.Handler()) so requests
// go through the real net/http server, exactly as deployments will.
package server_test

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/server"
)

// --- test doubles -----------------------------------------------------------

// fakeRouter is a configurable server.Router. Canned responses are set before
// the httptest server starts; calls are recorded under a mutex.
type fakeRouter struct {
	routeDecision modelrouter.RoutingDecision
	routeErr      error
	routePanic    any

	completeResult modelrouter.CompletionResult
	completeErr    error

	mu               sync.Mutex
	routePrompts     []string
	routeConstraints []*modelrouter.RoutingConstraints
	completeReqs     []modelrouter.CompletionRequest
}

var _ server.Router = (*fakeRouter)(nil)

func (f *fakeRouter) Route(_ context.Context, prompt string, constraints *modelrouter.RoutingConstraints) (modelrouter.RoutingDecision, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.routePanic != nil {
		panic(f.routePanic)
	}
	f.routePrompts = append(f.routePrompts, prompt)
	f.routeConstraints = append(f.routeConstraints, constraints)
	return f.routeDecision, f.routeErr
}

func (f *fakeRouter) Complete(_ context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.completeReqs = append(f.completeReqs, req)
	return f.completeResult, f.completeErr
}

func (f *fakeRouter) routeCalls() ([]string, []*modelrouter.RoutingConstraints) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.routePrompts...), append([]*modelrouter.RoutingConstraints(nil), f.routeConstraints...)
}

func (f *fakeRouter) completeCalls() []modelrouter.CompletionRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]modelrouter.CompletionRequest(nil), f.completeReqs...)
}

// --- helpers ----------------------------------------------------------------

// newTestEnv builds the server around router and starts an httptest server
// on srv.Handler(). The fake must be fully configured before calling this.
func newTestEnv(t *testing.T, router server.Router) *httptest.Server {
	t.Helper()
	srv, err := server.New(server.Options{
		Router: router,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("server.New() error = %v", err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

// do performs an HTTP request against ts, returning the response (body
// already read and closed) and the raw body.
func do(t *testing.T, ts *httptest.Server, method, path, body string, headers map[string]string) (*http.Response, []byte) {
	t.Helper()
	var reqBody io.Reader
	if body != "" {
		reqBody = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, ts.URL+path, reqBody)
	if err != nil {
		t.Fatalf("http.NewRequest(%s %s) error = %v", method, path, err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for name, value := range headers {
		req.Header.Set(name, value)
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("client.Do(%s %s) error = %v", method, path, err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading %s %s response error = %v", method, path, err)
	}
	return resp, data
}

// decodeJSON unmarshals data into T, failing the test on malformed JSON.
func decodeJSON[T any](t *testing.T, data []byte) T {
	t.Helper()
	var value T
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatalf("json.Unmarshal(%q) error = %v", data, err)
	}
	return value
}

// requireJSONKeys unmarshals data as a JSON object, fails the test when that
// is not possible or any of keys is missing, and returns the raw fields so
// callers can descend into them. It pins wire keys (the JSON tags) without
// pinning the whole object, which stays free to grow.
func requireJSONKeys(t *testing.T, data []byte, keys ...string) map[string]json.RawMessage {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		t.Fatalf("json.Unmarshal(%q) error = %v, want a JSON object", data, err)
	}
	for _, key := range keys {
		if _, ok := object[key]; !ok {
			t.Errorf("JSON object %s is missing key %q", data, key)
		}
	}
	return object
}

// errorBody mirrors the server's unexported error payload.
type errorBody struct {
	Error string `json:"error"`
	Code  string `json:"code"`
}

func requireStatus(t *testing.T, resp *http.Response, want int) {
	t.Helper()
	if resp.StatusCode != want {
		t.Fatalf("status = %d, want %d", resp.StatusCode, want)
	}
}

func requireContentType(t *testing.T, resp *http.Response, want string) {
	t.Helper()
	if got := resp.Header.Get("Content-Type"); got != want {
		t.Errorf("Content-Type = %q, want %q", got, want)
	}
}

// sampleDecision is a routing decision with enough detail to assert JSON
// round-tripping (model, provider, source and candidate chain).
func sampleDecision() modelrouter.RoutingDecision {
	return modelrouter.RoutingDecision{
		ModelID:  "vendor/cheap",
		Provider: "test",
		Source:   modelrouter.SourceClassification,
		Candidates: []modelrouter.ModelChoice{
			{Provider: "test", Model: "vendor/cheap"},
		},
	}
}

// --- /v1/route --------------------------------------------------------------

func TestRouteHappyPath(t *testing.T) {
	fake := &fakeRouter{routeDecision: sampleDecision()}
	ts := newTestEnv(t, fake)

	resp, body := do(t, ts, http.MethodPost, "/v1/route",
		`{"prompt":"what is 2+2?","constraints":{"min_context_tokens":8000}}`, nil)
	requireStatus(t, resp, http.StatusOK)
	requireContentType(t, resp, "application/json")

	decision := decodeJSON[modelrouter.RoutingDecision](t, body)
	if decision.ModelID != "vendor/cheap" || decision.Provider != "test" {
		t.Errorf("decision = {ModelID:%q Provider:%q}, want vendor/cheap/test", decision.ModelID, decision.Provider)
	}
	if decision.Source != modelrouter.SourceClassification {
		t.Errorf("decision.Source = %q, want %q", decision.Source, modelrouter.SourceClassification)
	}
	wantChoice := modelrouter.ModelChoice{Provider: "test", Model: "vendor/cheap"}
	if len(decision.Candidates) != 1 || decision.Candidates[0] != wantChoice {
		t.Errorf("decision.Candidates = %+v, want one entry %+v", decision.Candidates, wantChoice)
	}

	// Round-trip guard for the v2 wire shape: the serialized decision carries
	// source and the ordered candidates chain, whose entries use the
	// provider/model keys.
	raw := requireJSONKeys(t, body, "source", "candidates")
	var rawCandidates []json.RawMessage
	if err := json.Unmarshal(raw["candidates"], &rawCandidates); err != nil {
		t.Fatalf("candidates %s is not a JSON array: %v", raw["candidates"], err)
	}
	if len(rawCandidates) != 1 {
		t.Fatalf("candidates = %s, want one entry", raw["candidates"])
	}
	requireJSONKeys(t, rawCandidates[0], "provider", "model")

	prompts, constraints := fake.routeCalls()
	if len(prompts) != 1 || prompts[0] != "what is 2+2?" {
		t.Errorf("recorded prompts = %v, want [what is 2+2?]", prompts)
	}
	if len(constraints) != 1 || constraints[0] == nil || constraints[0].MinContextTokens != 8000 {
		t.Errorf("recorded constraints = %+v, want min_context_tokens 8000", constraints)
	}
}

func TestRouteRejectsInvalidRequests(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string // expected error substring
	}{
		{name: "empty prompt", body: `{"prompt":""}`, want: "prompt"},
		{name: "missing prompt field", body: `{}`, want: "prompt"},
		{name: "invalid JSON", body: `{"prompt":`, want: "invalid JSON"},
		{name: "empty body", body: ``, want: "invalid JSON"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &fakeRouter{}
			ts := newTestEnv(t, fake)

			resp, body := do(t, ts, http.MethodPost, "/v1/route", tc.body, nil)
			requireStatus(t, resp, http.StatusBadRequest)
			requireContentType(t, resp, "application/json")

			errBody := decodeJSON[errorBody](t, body)
			if errBody.Code != "invalid_request" {
				t.Errorf("code = %q, want invalid_request", errBody.Code)
			}
			if !strings.Contains(errBody.Error, tc.want) {
				t.Errorf("error = %q, want it to mention %q", errBody.Error, tc.want)
			}
			if prompts, _ := fake.routeCalls(); len(prompts) != 0 {
				t.Errorf("router was called (%v) for an invalid request", prompts)
			}
		})
	}
}

func TestRouteErrorMapping(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{
			name:       "no candidate maps to 422",
			err:        &modelrouter.NoCandidateModelError{Message: "nothing matches"},
			wantStatus: http.StatusUnprocessableEntity,
			wantCode:   "no_candidate",
		},
		{
			name:       "catalog unavailable maps to 502",
			err:        &modelrouter.CatalogUnavailableError{Err: errors.New("catalog down")},
			wantStatus: http.StatusBadGateway,
			wantCode:   "catalog_unavailable",
		},
		{
			name:       "classifier unavailable maps to 502",
			err:        &modelrouter.ClassifierUnavailableError{Err: errors.New("jev down")},
			wantStatus: http.StatusBadGateway,
			wantCode:   "classifier_unavailable",
		},
		{
			name:       "unknown error maps to 500",
			err:        errors.New("kaboom"),
			wantStatus: http.StatusInternalServerError,
			wantCode:   "internal",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &fakeRouter{routeErr: tc.err}
			ts := newTestEnv(t, fake)

			resp, body := do(t, ts, http.MethodPost, "/v1/route", `{"prompt":"hi"}`, nil)
			requireStatus(t, resp, tc.wantStatus)
			requireContentType(t, resp, "application/json")

			errBody := decodeJSON[errorBody](t, body)
			if errBody.Code != tc.wantCode {
				t.Errorf("code = %q, want %q", errBody.Code, tc.wantCode)
			}
			if errBody.Error == "" {
				t.Error("error message is empty, want the router error text")
			}
		})
	}
}

// --- /v1/complete -----------------------------------------------------------

func TestCompleteHappyPath(t *testing.T) {
	fake := &fakeRouter{
		completeResult: modelrouter.CompletionResult{
			Model:           "vendor/cheap",
			Provider:        "test",
			Content:         "4",
			FinishReason:    "stop",
			Usage:           &modelrouter.Usage{InputTokens: 3, OutputTokens: 1, TotalTokens: 4},
			RoutingDecision: sampleDecision(),
		},
	}
	ts := newTestEnv(t, fake)

	resp, body := do(t, ts, http.MethodPost, "/v1/complete",
		`{"prompt":"what is 2+2?","max_tokens":16,"temperature":0.2}`, nil)
	requireStatus(t, resp, http.StatusOK)
	requireContentType(t, resp, "application/json")

	result := decodeJSON[modelrouter.CompletionResult](t, body)
	if result.Content != "4" || result.Model != "vendor/cheap" || result.Provider != "test" {
		t.Errorf("result = %+v, want content 4 from vendor/cheap/test", result)
	}
	if result.Usage == nil || result.Usage.TotalTokens != 4 {
		t.Errorf("result.Usage = %+v, want total_tokens 4", result.Usage)
	}
	if result.RoutingDecision.ModelID != "vendor/cheap" {
		t.Errorf("routing_decision.model_id = %q, want vendor/cheap", result.RoutingDecision.ModelID)
	}

	requests := fake.completeCalls()
	if len(requests) != 1 {
		t.Fatalf("router saw %d completion requests, want 1", len(requests))
	}
	req := requests[0]
	if req.Prompt != "what is 2+2?" || req.MaxTokens != 16 {
		t.Errorf("router request = {Prompt:%q MaxTokens:%d}, want prompt and max_tokens passed through", req.Prompt, req.MaxTokens)
	}
	if req.Temperature == nil || *req.Temperature != 0.2 {
		t.Errorf("router request Temperature = %v, want 0.2", req.Temperature)
	}
}

// TestCompleteIgnoresLegacyStreamField pins the compatibility behavior after
// streaming was removed everywhere: a request body carrying the old
// "stream":true field is decoded like any other unrecognized JSON field — it
// is ignored, so the response is the full CompletionResult JSON, not a stream
// and not a rejection.
func TestCompleteIgnoresLegacyStreamField(t *testing.T) {
	fake := &fakeRouter{
		completeResult: modelrouter.CompletionResult{
			Model:           "vendor/cheap",
			Provider:        "test",
			Content:         "4",
			FinishReason:    "stop",
			RoutingDecision: sampleDecision(),
		},
	}
	ts := newTestEnv(t, fake)

	resp, body := do(t, ts, http.MethodPost, "/v1/complete",
		`{"prompt":"what is 2+2?","stream":true}`, nil)
	requireStatus(t, resp, http.StatusOK)
	requireContentType(t, resp, "application/json")

	// The whole result is on the wire — content and the routing decision are
	// both present — so the legacy field neither switched the response to a
	// stream nor suppressed the JSON body.
	requireJSONKeys(t, body, "content", "routing_decision")

	result := decodeJSON[modelrouter.CompletionResult](t, body)
	if result.Content != "4" {
		t.Errorf("result.Content = %q, want 4", result.Content)
	}
	if result.RoutingDecision.ModelID != "vendor/cheap" {
		t.Errorf("routing_decision.model_id = %q, want vendor/cheap", result.RoutingDecision.ModelID)
	}

	requests := fake.completeCalls()
	if len(requests) != 1 || requests[0].Prompt != "what is 2+2?" {
		t.Fatalf("router requests = %+v, want one with prompt what is 2+2?", requests)
	}
}

func TestCompleteAcceptsMessages(t *testing.T) {
	fake := &fakeRouter{completeResult: modelrouter.CompletionResult{Content: "hi"}}
	ts := newTestEnv(t, fake)

	resp, _ := do(t, ts, http.MethodPost, "/v1/complete",
		`{"messages":[{"role":"user","content":"hello"}]}`, nil)
	requireStatus(t, resp, http.StatusOK)

	requests := fake.completeCalls()
	if len(requests) != 1 || len(requests[0].Messages) != 1 {
		t.Fatalf("router requests = %+v, want one request with one message", requests)
	}
	if msg := requests[0].Messages[0]; msg.Role != "user" || msg.Content != "hello" {
		t.Errorf("message = %+v, want user:hello", msg)
	}
}

func TestCompleteRejectsMissingPromptAndMessages(t *testing.T) {
	fake := &fakeRouter{}
	ts := newTestEnv(t, fake)

	for _, body := range []string{`{}`, `{"messages":[]}`, `{"prompt":""}`} {
		resp, data := do(t, ts, http.MethodPost, "/v1/complete", body, nil)
		requireStatus(t, resp, http.StatusBadRequest)
		errBody := decodeJSON[errorBody](t, data)
		if errBody.Code != "invalid_request" {
			t.Errorf("body %q: code = %q, want invalid_request", body, errBody.Code)
		}
	}
	if requests := fake.completeCalls(); len(requests) != 0 {
		t.Errorf("router was called %d times for invalid requests", len(requests))
	}
}

func TestCompleteErrorMappingPreservesProviderErrors(t *testing.T) {
	providerErr := &modelrouter.ProviderError{
		Provider:   "test",
		Op:         "complete",
		StatusCode: 500,
		Err:        errors.New("upstream exploded"),
	}
	cases := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{name: "provider error maps to 502", err: providerErr, wantStatus: http.StatusBadGateway, wantCode: "provider_error"},
		{name: "unknown error maps to 500", err: errors.New("marker"), wantStatus: http.StatusInternalServerError, wantCode: "internal"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &fakeRouter{completeErr: tc.err}
			ts := newTestEnv(t, fake)

			resp, body := do(t, ts, http.MethodPost, "/v1/complete", `{"prompt":"hi"}`, nil)
			requireStatus(t, resp, tc.wantStatus)
			requireContentType(t, resp, "application/json")

			errBody := decodeJSON[errorBody](t, body)
			if errBody.Code != tc.wantCode {
				t.Errorf("code = %q, want %q", errBody.Code, tc.wantCode)
			}
		})
	}
}

// --- health, 404 and 405 ----------------------------------------------------

func TestHealth(t *testing.T) {
	ts := newTestEnv(t, &fakeRouter{})

	resp, body := do(t, ts, http.MethodGet, "/healthz", "", nil)
	requireStatus(t, resp, http.StatusOK)
	requireContentType(t, resp, "application/json")

	payload := decodeJSON[struct {
		Status string `json:"status"`
	}](t, body)
	if payload.Status != "ok" {
		t.Errorf("/healthz status = %q, want %q", payload.Status, "ok")
	}
}

func TestUnknownPathIs404(t *testing.T) {
	ts := newTestEnv(t, &fakeRouter{})

	resp, body := do(t, ts, http.MethodGet, "/v1/nope", "", nil)
	requireStatus(t, resp, http.StatusNotFound)
	requireContentType(t, resp, "application/json")
	if errBody := decodeJSON[errorBody](t, body); errBody.Code != "not_found" {
		t.Errorf("code = %q, want not_found", errBody.Code)
	}
}

func TestWrongMethodIs405WithAllowHeader(t *testing.T) {
	ts := newTestEnv(t, &fakeRouter{})

	cases := []struct {
		method    string
		path      string
		wantAllow string
	}{
		{method: http.MethodGet, path: "/v1/route", wantAllow: "POST"},
		{method: http.MethodGet, path: "/v1/complete", wantAllow: "POST"},
		{method: http.MethodPost, path: "/healthz", wantAllow: "GET"},
	}
	for _, tc := range cases {
		resp, body := do(t, ts, tc.method, tc.path, "", nil)
		requireStatus(t, resp, http.StatusMethodNotAllowed)
		if got := resp.Header.Get("Allow"); got != tc.wantAllow {
			t.Errorf("%s %s Allow = %q, want %q", tc.method, tc.path, got, tc.wantAllow)
		}
		requireContentType(t, resp, "application/json")
		if errBody := decodeJSON[errorBody](t, body); errBody.Code != "method_not_allowed" {
			t.Errorf("%s %s code = %q, want method_not_allowed", tc.method, tc.path, errBody.Code)
		}
	}
}

// --- middleware -------------------------------------------------------------

func TestRequestIDPassthroughAndGeneration(t *testing.T) {
	ts := newTestEnv(t, &fakeRouter{})

	resp, _ := do(t, ts, http.MethodGet, "/healthz", "", map[string]string{"X-Request-ID": "abc-123"})
	if got := resp.Header.Get("X-Request-ID"); got != "abc-123" {
		t.Errorf("X-Request-ID = %q, want the caller value abc-123", got)
	}

	resp, _ = do(t, ts, http.MethodGet, "/healthz", "", nil)
	generated := resp.Header.Get("X-Request-ID")
	if len(generated) != 32 {
		t.Fatalf("generated X-Request-ID = %q (len %d), want 32 hex characters", generated, len(generated))
	}
	if _, err := hex.DecodeString(generated); err != nil {
		t.Errorf("generated X-Request-ID %q is not hex: %v", generated, err)
	}

	resp, _ = do(t, ts, http.MethodGet, "/healthz", "", nil)
	if second := resp.Header.Get("X-Request-ID"); second == generated {
		t.Error("two generated request IDs were identical; want fresh randomness per request")
	}
}

func TestPanicRecoveryReturns500AndServerKeepsServing(t *testing.T) {
	fake := &fakeRouter{routePanic: "kaboom"}
	ts := newTestEnv(t, fake)

	resp, body := do(t, ts, http.MethodPost, "/v1/route", `{"prompt":"hi"}`, nil)
	requireStatus(t, resp, http.StatusInternalServerError)
	requireContentType(t, resp, "application/json")
	if errBody := decodeJSON[errorBody](t, body); errBody.Code != "internal" {
		t.Errorf("code = %q, want internal", errBody.Code)
	}

	resp, body = do(t, ts, http.MethodGet, "/healthz", "", nil)
	requireStatus(t, resp, http.StatusOK)
	payload := decodeJSON[struct {
		Status string `json:"status"`
	}](t, body)
	if payload.Status != "ok" {
		t.Errorf("post-panic healthz status = %q, want ok (server must keep serving)", payload.Status)
	}
}

// --- caller-supplied middleware ---------------------------------------------

// middlewareTrace records the order in which middleware entries ran and what
// they observed; its mutex covers the server goroutine writing and the test
// goroutine reading.
type middlewareTrace struct {
	mu     sync.Mutex
	events []string
}

func (tr *middlewareTrace) record(event string) {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	tr.events = append(tr.events, event)
}

func (tr *middlewareTrace) eventsCopy() []string {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return append([]string(nil), tr.events...)
}

// waitForEvents waits until the trace holds exactly want. The server may
// still be finishing the response after the client has read the body, so the
// middleware exit events can lag the HTTP response slightly.
func waitForEvents(t *testing.T, tr *middlewareTrace, want []string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		if got := tr.eventsCopy(); slices.Equal(got, want) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("middleware events = %v, want %v", tr.eventsCopy(), want)
		}
		time.Sleep(time.Millisecond)
	}
}

// observedWriter captures the status code written through it — including the
// 500 that panic recovery writes — and passes Unwrap through so the
// middleware behaves like a real ResponseWriter.
type observedWriter struct {
	http.ResponseWriter
	status int
}

func (w *observedWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *observedWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

func (w *observedWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// tracingMiddleware records entry/exit order plus the status and content type
// it observed, and sets a response header so tests can prove it ran on the
// way in.
func tracingMiddleware(tr *middlewareTrace, name string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tr.record(name + ":enter")
			w.Header().Set("X-Middleware-"+name, "set")
			observed := &observedWriter{ResponseWriter: w}
			next.ServeHTTP(observed, r)
			tr.record(fmt.Sprintf("%s:exit status=%d content-type=%s",
				name, observed.status, observed.Header().Get("Content-Type")))
		})
	}
}

// newTestEnvWithMiddleware starts an httptest server around srv.Handler() with
// the given Options.Middleware entries, mirroring newTestEnv.
func newTestEnvWithMiddleware(t *testing.T, router server.Router, middleware ...func(http.Handler) http.Handler) *httptest.Server {
	t.Helper()
	srv, err := server.New(server.Options{
		Router:     router,
		Logger:     slog.New(slog.NewTextHandler(io.Discard, nil)),
		Middleware: middleware,
	})
	if err != nil {
		t.Fatalf("server.New() error = %v", err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

// TestOptionsMiddlewareWrapsWholeChain pins the ordering contract:
// Options.Middleware runs outside the built-in chain (first entry outermost),
// its headers reach the response, and it observes the final status — the
// handler's 200, and the 500 that panic recovery produces without the panic
// reaching the instrumentation layer.
func TestOptionsMiddlewareWrapsWholeChain(t *testing.T) {
	t.Run("order and observed response", func(t *testing.T) {
		tr := &middlewareTrace{}
		ts := newTestEnvWithMiddleware(t, &fakeRouter{},
			tracingMiddleware(tr, "A"), tracingMiddleware(tr, "B"))

		resp, _ := do(t, ts, http.MethodGet, "/healthz", "", nil)
		requireStatus(t, resp, http.StatusOK)

		for _, name := range []string{"A", "B"} {
			if got := resp.Header.Get("X-Middleware-" + name); got != "set" {
				t.Errorf("X-Middleware-%s = %q, want set", name, got)
			}
		}
		// A wraps B, B wraps the built-in chain: entry order A -> B, exit
		// order B -> A, and both observe the handler's JSON 200.
		waitForEvents(t, tr, []string{
			"A:enter",
			"B:enter",
			"B:exit status=200 content-type=application/json",
			"A:exit status=200 content-type=application/json",
		})
	})

	t.Run("observes recovery status after a panic", func(t *testing.T) {
		tr := &middlewareTrace{}
		ts := newTestEnvWithMiddleware(t, &fakeRouter{routePanic: "kaboom"},
			tracingMiddleware(tr, "A"), tracingMiddleware(tr, "B"))

		resp, body := do(t, ts, http.MethodPost, "/v1/route", `{"prompt":"hi"}`, nil)
		requireStatus(t, resp, http.StatusInternalServerError)
		if errBody := decodeJSON[errorBody](t, body); errBody.Code != "internal" {
			t.Errorf("code = %q, want internal", errBody.Code)
		}

		// Recovery runs inside the user middleware, so the panic is caught
		// before either entry sees it — and both finish normally, observing
		// the 500 recovery produced.
		waitForEvents(t, tr, []string{
			"A:enter",
			"B:enter",
			"B:exit status=500 content-type=application/json",
			"A:exit status=500 content-type=application/json",
		})
	})
}

// --- lifecycle --------------------------------------------------------------

func TestNewRequiresRouter(t *testing.T) {
	if _, err := server.New(server.Options{}); err == nil {
		t.Fatal("server.New(Options{}) error = nil, want Router-required error")
	}
}

func TestServeListenerServesAndShutsDownGracefully(t *testing.T) {
	srv, err := server.New(server.Options{
		Router: &fakeRouter{},
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("server.New() error = %v", err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- srv.ServeListener(ctx, ln) }()

	url := "http://" + ln.Addr().String() + "/healthz"
	client := &http.Client{Timeout: time.Second}
	var lastErr error
	for attempt := 0; attempt < 100; attempt++ {
		var resp *http.Response
		resp, lastErr = client.Get(url)
		if lastErr == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("GET /healthz over real listener status = %d, want 200", resp.StatusCode)
			}
			lastErr = nil
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if lastErr != nil {
		t.Fatalf("server never became reachable on %s: %v", url, lastErr)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("ServeListener() error = %v, want nil after graceful shutdown", err)
		}
	case <-time.After(time.Second):
		t.Fatal("ServeListener() did not return within 1s of context cancellation")
	}
}
