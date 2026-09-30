// Package classify provides request classifiers for the router.
//
// Jev is one narrow JSON POST carrying five fixed questions (a structured
// choice/score/noul questionnaire) to OpenRouter's systemone route by
// default; base URL, path and model stay configurable, so pointing at
// TypeSafe directly is a configuration change, not a code change. Every
// failure — transport error, timeout, non-2xx status, invalid JSON, missing
// or non-numeric answer fields — maps to a
// modelrouter.ClassifierUnavailableError, so routing can fail open or closed
// uniformly. The endpoint's optional usage block is parsed leniently: missing
// or malformed usage never fails a classification.
package classify

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
)

const (
	// DefaultBaseURL is the default Jev API base URL: OpenRouter, which
	// fronts TypeSafe's Jev model.
	DefaultBaseURL = "https://openrouter.ai"
	// DefaultPath is the systemone endpoint appended to the base URL.
	DefaultPath = "/api/v1/systemone"
	// DefaultModel is the OpenRouter model slug used to reach Jev.
	DefaultModel = "~typesafe/jev-latest"
	// defaultTimeout is the classifier call timeout: it must not outlive
	// the request it guards.
	defaultTimeout = 5 * time.Second
)

type Option func(*Jev)

func WithHTTPClient(client *http.Client) Option {
	return func(j *Jev) {
		if client == nil {
			return
		}
		j.client = client
	}
}

func WithBaseURL(baseURL string) Option {
	return func(j *Jev) {
		j.baseURL = strings.TrimRight(baseURL, "/")
	}
}

func WithPath(path string) Option {
	return func(j *Jev) {
		if path != "" && !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		j.path = path
	}
}

func WithModel(model string) Option {
	return func(j *Jev) {
		j.model = model
	}
}

func WithTimeout(timeout time.Duration) Option {
	return func(j *Jev) {
		j.timeout = timeout
	}
}

func WithLogger(logger *slog.Logger) Option {
	return func(j *Jev) {
		if logger == nil {
			return
		}
		j.logger = logger
	}
}

// Jev classifies requests with TypeSafe's Jev model through OpenRouter's
// systemone route (configurable via WithBaseURL/WithPath/WithModel).
type Jev struct {
	apiKey  string
	baseURL string
	path    string
	model   string
	timeout time.Duration

	client     *http.Client
	ownsClient bool
	logger     *slog.Logger
}

var _ modelrouter.Classifier = (*Jev)(nil)

// New returns a Jev classifier authenticating with apiKey.
//
// The returned Jev owns its HTTP client (5-second timeout) unless one is
// injected with WithHTTPClient, in which case the caller keeps ownership and
// Close will not touch it.
func New(apiKey string, opts ...Option) *Jev {
	j := &Jev{
		apiKey:  apiKey,
		baseURL: DefaultBaseURL,
		path:    DefaultPath,
		model:   DefaultModel,
		timeout: defaultTimeout,
		logger:  slog.Default(),
	}
	for _, opt := range opts {
		opt(j)
	}
	if j.client == nil {
		j.client = &http.Client{Timeout: j.timeout}
		j.ownsClient = true
	}
	return j
}

func (j *Jev) Close() error {
	if j == nil {
		return nil
	}
	if j.ownsClient && j.client != nil {
		j.client.CloseIdleConnections()
	}
	return nil
}

// classifyRequest is the systemone request body:
type classifyRequest struct {
	State     string              `json:"state"`
	Model     string              `json:"model"`
	Questions map[string]question `json:"questions"`
}

type question struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

var questions = map[string]question{
	"domain": {
		Type:         "choice",
		Instructions: "What is the primary domain of this request",
		Criteria: map[string]string{
			"code":           "Programming, debugging, code review, or software engineering",
			"math_reasoning": "Mathematics, logic, multi-step reasoning, or proofs",
			"creative":       "Creative writing, copywriting, or brainstorming",
			"factual_lookup": "Simple factual question, summarization, or extraction",
			"other":          "Does not clearly fit any of the above",
		},
	},
	"complexity": {
		Type:         "score",
		Instructions: "How much reasoning depth does this request require",
		Criteria: []string{
			"Trivial, direct answer with no reasoning needed",
			"Requires a few steps of reasoning or domain knowledge",
			"Requires deep, multi-step reasoning or extensive domain expertise",
		},
	},
	"needs_long_context": {
		Type:         "noul",
		Instructions: "This request references or requires a large amount of context (a long document, large codebase, or extensive history)",
	},
	"needs_vision": {
		Type:         "noul",
		Instructions: "This request requires understanding or generating image content",
	},
	"latency_sensitive": {
		Type:         "noul",
		Instructions: "This request implies the user needs a fast/immediate response (e.g. interactive, real-time)",
	},
}

// rawResponse is the wire envelope. Answers and Usage stay raw so each field
// is validated individually: any missing or malformed answer field fails the
// whole classification, while a missing or malformed usage block does not.
type rawResponse struct {
	Answers json.RawMessage `json:"answers"`
	Usage   json.RawMessage `json:"usage"`
}

func (j *Jev) Classify(ctx context.Context, prompt string) (modelrouter.Classification, error) {
	payload, err := json.Marshal(classifyRequest{
		State:     prompt,
		Model:     j.model,
		Questions: questions,
	})
	if err != nil {
		return j.fail(fmt.Errorf("encoding request: %w", err))
	}

	url := j.baseURL + j.path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return j.fail(fmt.Errorf("building request for %s: %w", url, err))
	}
	req.Header.Set("Authorization", "Bearer "+j.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := j.client.Do(req)
	if err != nil {
		return j.fail(fmt.Errorf("posting to %s: %w", url, err))
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return j.fail(fmt.Errorf("unexpected HTTP status %d from %s", resp.StatusCode, url))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return j.fail(fmt.Errorf("reading response body: %w", err))
	}
	classification, err := parseResponse(body)
	if err != nil {
		return j.fail(err)
	}
	return classification, nil
}

func (j *Jev) fail(cause error) (modelrouter.Classification, error) {
	j.logger.Warn("classify/jev: Jev classification call failed", "error", cause)
	return modelrouter.Classification{}, &modelrouter.ClassifierUnavailableError{
		Err: fmt.Errorf("Jev classification call failed: %w", cause),
	}
}

func parseResponse(body []byte) (modelrouter.Classification, error) {
	var envelope rawResponse
	if err := json.Unmarshal(body, &envelope); err != nil {
		return modelrouter.Classification{}, fmt.Errorf("invalid JSON response: %w", err)
	}

	answers, err := parseAnswers(envelope.Answers)
	if err != nil {
		return modelrouter.Classification{}, err
	}
	profile, err := parseProfile(answers)
	if err != nil {
		return modelrouter.Classification{}, err
	}
	return modelrouter.Classification{
		Profile: profile,
		Raw:     body,
		Usage:   parseUsage(envelope.Usage),
	}, nil
}

func parseAnswers(raw json.RawMessage) (map[string]json.RawMessage, error) {
	if len(raw) == 0 || isJSONNull(raw) {
		return nil, errors.New(`response is missing the "answers" field`)
	}
	var answers map[string]json.RawMessage
	if err := json.Unmarshal(raw, &answers); err != nil {
		return nil, errors.New(`"answers" is not an object`)
	}
	return answers, nil
}

func parseProfile(answers map[string]json.RawMessage) (modelrouter.RequestProfile, error) {
	domain, err := answerObject(answers, "domain")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	complexity, err := answerObject(answers, "complexity")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	needsLongContext, err := answerObject(answers, "needs_long_context")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	needsVision, err := answerObject(answers, "needs_vision")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	latencySensitive, err := answerObject(answers, "latency_sensitive")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}

	choice, err := stringField(domain, "choice", "answers.domain.choice")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	domainConfidence, err := floatField(domain, "confidence", "answers.domain.confidence")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	complexityScore, err := floatField(complexity, "score", "answers.complexity.score")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	complexityConfidence, err := floatField(complexity, "confidence", "answers.complexity.confidence")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	longContext, err := floatField(needsLongContext, "noul", "answers.needs_long_context.noul")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	vision, err := floatField(needsVision, "noul", "answers.needs_vision.noul")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}
	latency, err := floatField(latencySensitive, "noul", "answers.latency_sensitive.noul")
	if err != nil {
		return modelrouter.RequestProfile{}, err
	}

	// The domain choice is accepted as-is, with no enum validation: an
	// unexpected value just fails to match any configured chain and falls
	// back to the default model.
	return modelrouter.RequestProfile{
		Domain:               modelrouter.Domain(choice),
		DomainConfidence:     domainConfidence,
		ComplexityScore:      complexityScore,
		ComplexityConfidence: complexityConfidence,
		NeedsLongContext:     longContext,
		NeedsVision:          vision,
		LatencySensitive:     latency,
	}, nil
}

func answerObject(answers map[string]json.RawMessage, name string) (map[string]json.RawMessage, error) {
	raw, ok := answers[name]
	if !ok || len(raw) == 0 || isJSONNull(raw) {
		return nil, fmt.Errorf("response is missing the %q answer", name)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, fmt.Errorf("%q answer is not an object", name)
	}
	return fields, nil
}

func stringField(fields map[string]json.RawMessage, field, path string) (string, error) {
	raw, ok := fields[field]
	if !ok || len(raw) == 0 || isJSONNull(raw) {
		return "", fmt.Errorf("response is missing %s", path)
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", fmt.Errorf("%s is not a string", path)
	}
	return value, nil
}

func floatField(fields map[string]json.RawMessage, field, path string) (float64, error) {
	raw, ok := fields[field]
	if !ok || len(raw) == 0 || isJSONNull(raw) {
		return 0, fmt.Errorf("response is missing %s", path)
	}
	value, err := numberValue(raw)
	if err != nil {
		return 0, fmt.Errorf("%s is not numeric: %w", path, err)
	}
	return value, nil
}

func numberValue(raw json.RawMessage) (float64, error) {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, err
	}
	switch v := value.(type) {
	case float64:
		return v, nil
	case string:
		number, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return 0, fmt.Errorf("%q is not numeric", v)
		}
		return number, nil
	default:
		return 0, fmt.Errorf("unexpected type %T", value)
	}
}

func parseUsage(raw json.RawMessage) *modelrouter.Usage {
	if len(raw) == 0 || isJSONNull(raw) {
		return nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nil
	}

	var usage modelrouter.Usage
	found := false
	type tokenField struct {
		names []string
		dst   *int
	}
	tokenFields := []tokenField{
		{names: []string{"input_tokens", "prompt_tokens"}, dst: &usage.InputTokens},
		{names: []string{"output_tokens", "completion_tokens"}, dst: &usage.OutputTokens},
		{names: []string{"total_tokens", "total"}, dst: &usage.TotalTokens},
	}
	for _, spec := range tokenFields {
		tokenRaw, ok := firstPresent(fields, spec.names...)
		if !ok {
			continue
		}
		value, err := numberValue(tokenRaw)
		if err != nil {
			return nil
		}
		*spec.dst = int(value)
		found = true
	}
	if costRaw, ok := firstPresent(fields, "cost", "cost_usd"); ok {
		value, err := numberValue(costRaw)
		if err != nil {
			return nil
		}
		usage.CostUSD = &value
		found = true
	}
	if !found {
		return nil
	}
	return &usage
}

func firstPresent(fields map[string]json.RawMessage, names ...string) (json.RawMessage, bool) {
	for _, name := range names {
		raw, ok := fields[name]
		if ok && len(raw) > 0 && !isJSONNull(raw) {
			return raw, true
		}
	}
	return nil, false
}

func isJSONNull(raw json.RawMessage) bool {
	return string(bytes.TrimSpace(raw)) == "null"
}
