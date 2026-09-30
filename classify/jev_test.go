package classify_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/classify"
)

const validAnswersJSON = `{
	"domain": {"choice": "code", "probabilities": {"code": 0.9}, "confidence": 0.9},
	"complexity": {"score": 1.4, "legend": {}, "probabilities": {}, "confidence": 0.75},
	"needs_long_context": {"noul": 0.1},
	"needs_vision": {"noul": 0.05},
	"latency_sensitive": {"noul": 0.6}
}`

const fakeJevResponseJSON = `{"answers": ` + validAnswersJSON + `}`

const wantQuestionsJSON = `{
	"domain": {
		"type": "choice",
		"instructions": "What is the primary domain of this request",
		"criteria": {
			"code": "Programming, debugging, code review, or software engineering",
			"math_reasoning": "Mathematics, logic, multi-step reasoning, or proofs",
			"creative": "Creative writing, copywriting, or brainstorming",
			"factual_lookup": "Simple factual question, summarization, or extraction",
			"other": "Does not clearly fit any of the above"
		}
	},
	"complexity": {
		"type": "score",
		"instructions": "How much reasoning depth does this request require",
		"criteria": [
			"Trivial, direct answer with no reasoning needed",
			"Requires a few steps of reasoning or domain knowledge",
			"Requires deep, multi-step reasoning or extensive domain expertise"
		]
	},
	"needs_long_context": {
		"type": "noul",
		"instructions": "This request references or requires a large amount of context (a long document, large codebase, or extensive history)"
	},
	"needs_vision": {
		"type": "noul",
		"instructions": "This request requires understanding or generating image content"
	},
	"latency_sensitive": {
		"type": "noul",
		"instructions": "This request implies the user needs a fast/immediate response (e.g. interactive, real-time)"
	}
}`

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func jsonHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}
}

func newTestJev(t *testing.T, handler http.HandlerFunc, opts ...classify.Option) *classify.Jev {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	options := []classify.Option{
		classify.WithBaseURL(server.URL),
		classify.WithPath("/api/v1/systemone"),
	}
	return classify.New("test-key", append(options, opts...)...)
}

func assertClassifierUnavailable(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("Classify() error = nil, want ClassifierUnavailableError")
	}
	if !errors.Is(err, modelrouter.ErrClassifierUnavailable) {
		t.Errorf("errors.Is(err, ErrClassifierUnavailable) = false, err = %v", err)
	}
	var target *modelrouter.ClassifierUnavailableError
	if !errors.As(err, &target) {
		t.Errorf("errors.As(err, **ClassifierUnavailableError) = false, err = %v", err)
	}
}

func TestClassifyParsesAWellFormedJevResponse(t *testing.T) {
	jev := newTestJev(t, jsonHandler(fakeJevResponseJSON))

	classification, err := jev.Classify(context.Background(), "Fix this Python function")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}

	profile := classification.Profile
	if profile.Domain != modelrouter.DomainCode {
		t.Errorf("Domain = %q, want %q", profile.Domain, modelrouter.DomainCode)
	}
	if !almostEqual(profile.DomainConfidence, 0.9) {
		t.Errorf("DomainConfidence = %v, want 0.9", profile.DomainConfidence)
	}
	if !almostEqual(profile.ComplexityScore, 1.4) {
		t.Errorf("ComplexityScore = %v, want 1.4", profile.ComplexityScore)
	}
	if !almostEqual(profile.ComplexityConfidence, 0.75) {
		t.Errorf("ComplexityConfidence = %v, want 0.75", profile.ComplexityConfidence)
	}
	if !almostEqual(profile.NeedsLongContext, 0.1) {
		t.Errorf("NeedsLongContext = %v, want 0.1", profile.NeedsLongContext)
	}
	if !almostEqual(profile.NeedsVision, 0.05) {
		t.Errorf("NeedsVision = %v, want 0.05", profile.NeedsVision)
	}
	if !almostEqual(profile.LatencySensitive, 0.6) {
		t.Errorf("LatencySensitive = %v, want 0.6", profile.LatencySensitive)
	}

	// Upstream: assert raw == FAKE_JEV_RESPONSE (both decoded).
	var want, got any
	if err := json.Unmarshal([]byte(fakeJevResponseJSON), &want); err != nil {
		t.Fatalf("decoding expected response: %v", err)
	}
	if err := json.Unmarshal(classification.Raw, &got); err != nil {
		t.Fatalf("decoding Classification.Raw: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Classification.Raw = %s, want %s", classification.Raw, fakeJevResponseJSON)
	}

	// This fixture carries no usage block.
	if classification.Usage != nil {
		t.Errorf("Classification.Usage = %+v, want nil", classification.Usage)
	}
}

func TestClassifyReturnsClassifierUnavailableOnHTTPError(t *testing.T) {
	jev := newTestJev(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})

	_, err := jev.Classify(context.Background(), "anything")
	assertClassifierUnavailable(t, err)
}

func TestClassifyReturnsClassifierUnavailableOnMissingAnswersKey(t *testing.T) {
	jev := newTestJev(t, jsonHandler(`{}`))

	_, err := jev.Classify(context.Background(), "anything")
	assertClassifierUnavailable(t, err)
}

func TestClassifyReturnsClassifierUnavailableOnIncompleteResponse(t *testing.T) {
	jev := newTestJev(t, jsonHandler(`{"answers": {}}`))

	_, err := jev.Classify(context.Background(), "anything")
	assertClassifierUnavailable(t, err)
}

func TestClassifyReturnsClassifierUnavailableOnNonNumericConfidence(t *testing.T) {
	const malformed = `{
		"answers": {
			"domain": {"choice": "code", "confidence": "not_a_number"},
			"complexity": {"score": 1.4, "confidence": 0.75},
			"needs_long_context": {"noul": 0.1},
			"needs_vision": {"noul": 0.05},
			"latency_sensitive": {"noul": 0.6}
		}
	}`
	jev := newTestJev(t, jsonHandler(malformed))

	_, err := jev.Classify(context.Background(), "anything")
	assertClassifierUnavailable(t, err)
}

func TestClassifyAcceptsAnyDomainChoice(t *testing.T) {
	const unusual = `{
		"answers": {
			"domain": {"choice": "sports", "confidence": 0.4},
			"complexity": {"score": 0.5, "confidence": 0.5},
			"needs_long_context": {"noul": 0.1},
			"needs_vision": {"noul": 0.05},
			"latency_sensitive": {"noul": 0.6}
		}
	}`
	jev := newTestJev(t, jsonHandler(unusual))

	classification, err := jev.Classify(context.Background(), "anything")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if got := classification.Profile.Domain; got != modelrouter.Domain("sports") {
		t.Errorf("Domain = %q, want the choice passed through as %q", got, "sports")
	}
}

func TestClassifyRequestShape(t *testing.T) {
	const prompt = "Fix this Python function"

	type requestPayload struct {
		State     string          `json:"state"`
		Model     string          `json:"model"`
		Questions json.RawMessage `json:"questions"`
	}
	type capturedRequest struct {
		method        string
		path          string
		authorization string
		contentType   string
		payload       requestPayload
	}
	captured := make(chan capturedRequest, 1)

	handler := func(w http.ResponseWriter, r *http.Request) {
		cr := capturedRequest{
			method:        r.Method,
			path:          r.URL.Path,
			authorization: r.Header.Get("Authorization"),
			contentType:   r.Header.Get("Content-Type"),
		}
		if body, err := io.ReadAll(r.Body); err != nil {
			t.Errorf("reading request body: %v", err)
		} else if err := json.Unmarshal(body, &cr.payload); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		captured <- cr
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(fakeJevResponseJSON))
	}

	jev := newTestJev(t, handler)
	if _, err := jev.Classify(context.Background(), prompt); err != nil {
		t.Fatalf("Classify() error = %v", err)
	}

	cr := <-captured
	if cr.method != http.MethodPost {
		t.Errorf("request method = %q, want POST", cr.method)
	}
	if cr.path != "/api/v1/systemone" {
		t.Errorf("request path = %q, want /api/v1/systemone", cr.path)
	}
	if cr.authorization != "Bearer test-key" {
		t.Errorf("Authorization header = %q, want %q", cr.authorization, "Bearer test-key")
	}
	if cr.contentType != "application/json" {
		t.Errorf("Content-Type header = %q, want application/json", cr.contentType)
	}
	if cr.payload.State != prompt {
		t.Errorf("request state = %q, want %q", cr.payload.State, prompt)
	}
	if cr.payload.Model != "~typesafe/jev-latest" {
		t.Errorf("request model = %q, want %q", cr.payload.Model, "~typesafe/jev-latest")
	}

	var questions map[string]struct {
		Type     string          `json:"type"`
		Criteria json.RawMessage `json:"criteria"`
	}
	if err := json.Unmarshal(cr.payload.Questions, &questions); err != nil {
		t.Fatalf("decoding questions: %v", err)
	}
	wantTypes := map[string]string{
		"domain":             "choice",
		"complexity":         "score",
		"needs_long_context": "noul",
		"needs_vision":       "noul",
		"latency_sensitive":  "noul",
	}
	if len(questions) != len(wantTypes) {
		t.Errorf("len(questions) = %d, want %d", len(questions), len(wantTypes))
	}
	for name, want := range wantTypes {
		if got := questions[name].Type; got != want {
			t.Errorf("questions[%q].type = %q, want %q", name, got, want)
		}
	}

	var domainCriteria map[string]string
	if err := json.Unmarshal(questions["domain"].Criteria, &domainCriteria); err != nil {
		t.Fatalf("decoding domain criteria: %v", err)
	}
	if len(domainCriteria) != 5 {
		t.Errorf("len(domain criteria) = %d, want 5", len(domainCriteria))
	}
	var complexityCriteria []string
	if err := json.Unmarshal(questions["complexity"].Criteria, &complexityCriteria); err != nil {
		t.Fatalf("decoding complexity criteria: %v", err)
	}
	if len(complexityCriteria) != 3 {
		t.Errorf("len(complexity criteria) = %d, want 3", len(complexityCriteria))
	}

	// The whole payload must equal the fixed questions wire format exactly.
	var gotQuestions, wantQuestions any
	if err := json.Unmarshal(cr.payload.Questions, &gotQuestions); err != nil {
		t.Fatalf("decoding questions payload: %v", err)
	}
	if err := json.Unmarshal([]byte(wantQuestionsJSON), &wantQuestions); err != nil {
		t.Fatalf("decoding expected questions: %v", err)
	}
	if !reflect.DeepEqual(gotQuestions, wantQuestions) {
		t.Errorf("questions payload = %s, want %s", cr.payload.Questions, wantQuestionsJSON)
	}
}

func TestClassifyParsesUsage(t *testing.T) {
	body := `{"answers": ` + validAnswersJSON + `, "usage": {"input_tokens": 12, "output_tokens": 3, "cost": 0.0005}}`
	jev := newTestJev(t, jsonHandler(body))

	classification, err := jev.Classify(context.Background(), "anything")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}

	if classification.Usage == nil {
		t.Fatal("Classification.Usage = nil, want parsed usage")
	}
	if got := classification.Usage.InputTokens; got != 12 {
		t.Errorf("Usage.InputTokens = %d, want 12", got)
	}
	if got := classification.Usage.OutputTokens; got != 3 {
		t.Errorf("Usage.OutputTokens = %d, want 3", got)
	}
	if got := classification.Usage.TotalTokens; got != 0 {
		t.Errorf("Usage.TotalTokens = %d, want 0 (absent from the response)", got)
	}
	if classification.Usage.CostUSD == nil {
		t.Fatal("Usage.CostUSD = nil, want 0.0005")
	}
	if got := *classification.Usage.CostUSD; !almostEqual(got, 0.0005) {
		t.Errorf("Usage.CostUSD = %v, want 0.0005", got)
	}
}

func TestClassifyParsesOpenRouterUsageAliases(t *testing.T) {
	body := `{"answers": ` + validAnswersJSON + `, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": "0.0005"}}`
	jev := newTestJev(t, jsonHandler(body))

	classification, err := jev.Classify(context.Background(), "anything")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if classification.Usage == nil {
		t.Fatal("Classification.Usage = nil, want parsed usage")
	}
	if got := classification.Usage.InputTokens; got != 10 {
		t.Errorf("Usage.InputTokens = %d, want 10", got)
	}
	if got := classification.Usage.OutputTokens; got != 5 {
		t.Errorf("Usage.OutputTokens = %d, want 5", got)
	}
	if got := classification.Usage.TotalTokens; got != 15 {
		t.Errorf("Usage.TotalTokens = %d, want 15", got)
	}
	if classification.Usage.CostUSD == nil {
		t.Fatal("Usage.CostUSD = nil, want 0.0005")
	}
	if got := *classification.Usage.CostUSD; !almostEqual(got, 0.0005) {
		t.Errorf("Usage.CostUSD = %v, want 0.0005", got)
	}
}

func TestClassifyMalformedUsageDoesNotFailClassification(t *testing.T) {
	cases := map[string]string{
		"usage is a string":      `"oops"`,
		"usage is an array":      `[12, 3]`,
		"usage is null":          `null`,
		"usage is empty object":  `{}`,
		"unknown fields only":    `{"foo": 1}`,
		"input tokens malformed": `{"input_tokens": "abc"}`,
		"tokens of wrong type":   `{"input_tokens": true}`,
		"null fields only":       `{"input_tokens": null, "cost": null}`,
	}
	for name, usage := range cases {
		t.Run(name, func(t *testing.T) {
			jev := newTestJev(t, jsonHandler(`{"answers": `+validAnswersJSON+`, "usage": `+usage+`}`))
			classification, err := jev.Classify(context.Background(), "anything")
			if err != nil {
				t.Fatalf("Classify() error = %v, want success despite malformed usage", err)
			}
			if classification.Usage != nil {
				t.Errorf("Classification.Usage = %+v, want nil for malformed usage", classification.Usage)
			}
		})
	}
}

func TestClassifyTimeout(t *testing.T) {
	handler := func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(fakeJevResponseJSON))
	}
	jev := newTestJev(t, handler, classify.WithTimeout(20*time.Millisecond))

	_, err := jev.Classify(context.Background(), "anything")
	assertClassifierUnavailable(t, err)
}

func TestClassifyCustomBaseURLAndPath(t *testing.T) {
	paths := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths <- r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(fakeJevResponseJSON))
	}))
	t.Cleanup(server.Close)

	jev := classify.New("test-key",
		classify.WithBaseURL(server.URL+"/"),
		classify.WithPath("custom/systemone"),
	)
	if _, err := jev.Classify(context.Background(), "anything"); err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if got := <-paths; got != "/custom/systemone" {
		t.Errorf("request path = %q, want /custom/systemone", got)
	}
}

type trackingTransport struct {
	base   http.RoundTripper
	closes int32
}

func (t *trackingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.base.RoundTrip(req)
}

func (t *trackingTransport) CloseIdleConnections() {
	atomic.AddInt32(&t.closes, 1)
}

func TestCloseDoesNotCloseInjectedClient(t *testing.T) {
	server := httptest.NewServer(jsonHandler(fakeJevResponseJSON))
	t.Cleanup(server.Close)

	transport := &trackingTransport{base: server.Client().Transport}
	client := &http.Client{Transport: transport}
	jev := classify.New("test-key",
		classify.WithBaseURL(server.URL),
		classify.WithPath("/api/v1/systemone"),
		classify.WithHTTPClient(client),
	)

	if _, err := jev.Classify(context.Background(), "anything"); err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if err := jev.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := jev.Classify(context.Background(), "anything"); err != nil {
		t.Errorf("Classify() after Close() error = %v; an injected client must stay usable", err)
	}

	if got := atomic.LoadInt32(&transport.closes); got != 0 {
		t.Errorf("injected client CloseIdleConnections calls = %d, want 0", got)
	}
}

func TestCloseClosesSelfConstructedClient(t *testing.T) {
	const attempts = 10
	for attempt := 0; attempt < attempts; attempt++ {
		var newConns int32
		idle := make(chan struct{}, 4)
		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(fakeJevResponseJSON))
		}))
		server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
			switch state {
			case http.StateNew:
				atomic.AddInt32(&newConns, 1)
			case http.StateIdle:
				select {
				case idle <- struct{}{}:
				default:
				}
			}
		}
		server.Start()

		jev := classify.New("test-key", classify.WithBaseURL(server.URL))
		if _, err := jev.Classify(context.Background(), "anything"); err != nil {
			server.Close()
			t.Fatalf("Classify() error = %v", err)
		}
		select {
		case <-idle:
		case <-time.After(5 * time.Second):
			server.Close()
			t.Fatal("server never observed an idle connection")
		}
		if err := jev.Close(); err != nil {
			server.Close()
			t.Fatalf("Close() error = %v", err)
		}
		_, err := jev.Classify(context.Background(), "anything")
		_ = jev.Close()
		closed := err == nil && atomic.LoadInt32(&newConns) >= 2
		server.Close()

		if closed {
			return
		}
	}
	t.Errorf("after Close, later requests kept reusing the old connection (%d attempts): Close must close the owned client", attempts)
}

func TestCloseIsIdempotentAndNilSafe(t *testing.T) {
	jev := classify.New("test-key")
	for attempt := 1; attempt <= 2; attempt++ {
		if err := jev.Close(); err != nil {
			t.Fatalf("Close() #%d error = %v", attempt, err)
		}
	}

	var nilJev *classify.Jev
	if err := nilJev.Close(); err != nil {
		t.Errorf("nil *Jev Close() error = %v, want nil", err)
	}
}

func TestNilOptionsAreIgnored(t *testing.T) {
	server := httptest.NewServer(jsonHandler(fakeJevResponseJSON))
	t.Cleanup(server.Close)

	jev := classify.New("test-key",
		classify.WithBaseURL(server.URL),
		classify.WithHTTPClient(nil),
		classify.WithLogger(nil),
	)
	if _, err := jev.Classify(context.Background(), "anything"); err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if err := jev.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestClassifyAgainstRealJevAPI(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("requires OPENROUTER_API_KEY")
	}

	jev := classify.New(apiKey)
	defer func() { _ = jev.Close() }()

	classification, err := jev.Classify(
		context.Background(),
		"Write a Python function to check if a binary tree is balanced, with tests.",
	)
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}

	validDomains := map[modelrouter.Domain]bool{
		modelrouter.DomainCode:          true,
		modelrouter.DomainMathReasoning: true,
		modelrouter.DomainCreative:      true,
		modelrouter.DomainFactualLookup: true,
		modelrouter.DomainOther:         true,
	}
	if !validDomains[classification.Profile.Domain] {
		t.Errorf("Domain = %q, want one of code, math_reasoning, creative, factual_lookup, other", classification.Profile.Domain)
	}
	if c := classification.Profile.DomainConfidence; c < 0 || c > 1 {
		t.Errorf("DomainConfidence = %v, want 0 <= c <= 1", c)
	}
	if s := classification.Profile.ComplexityScore; s < 0 || s > 2 {
		t.Errorf("ComplexityScore = %v, want 0 <= s <= 2", s)
	}
	if len(classification.Raw) == 0 {
		t.Error("Classification.Raw is empty, want the raw Jev response")
	}
}
