// Package openrouter provides the live OpenRouter model catalog:
// GET {base}/api/v1/models, decoded {"data": [...]}, with an in-memory TTL
// cache (default 600s) and stale-cache fallback on a failed refetch.
//
// Parsing is per-entry and tolerant: malformed entries are logged and
// skipped so one bad row cannot take down routing, and OpenRouter's "-1"
// sentinel for variable/unknown pricing is skipped silently — a
// cost-optimizing router must not route to a model with unknown pricing.
// Entries decode one at a time over json.RawMessage rather than a single
// typed decode, since that tolerance needs per-entry error handling.
package openrouter

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
	"sync"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// ProviderName is the provider stamped on every model from this catalog.
const ProviderName = "openrouter"

const (
	// DefaultBaseURL is the production OpenRouter API base URL.
	DefaultBaseURL = "https://openrouter.ai"
	// DefaultCacheTTL is the default catalog cache TTL (600 seconds).
	DefaultCacheTTL = 600 * time.Second
	// defaultHTTPTimeout is the default HTTP client timeout (10 seconds).
	defaultHTTPTimeout = 10 * time.Second
	// modelsPath is the catalog endpoint appended to the configured base URL.
	modelsPath = "/api/v1/models"
)

// errNegativePricing marks an entry skipped silently because OpenRouter
// reports negative sentinel prices for variable/unknown pricing.
var errNegativePricing = errors.New("negative sentinel pricing")

// Option configures a Catalog created by New.
type Option func(*Catalog)

// WithHTTPClient injects the HTTP client used for catalog requests.
//
// An injected client stays the caller's property: Close never touches it.
// Passing nil keeps the default client (10-second timeout).
func WithHTTPClient(client *http.Client) Option {
	return func(c *Catalog) {
		if client == nil {
			return
		}
		c.client = client
		c.ownsClient = false
	}
}

// WithBaseURL overrides the API base URL (default DefaultBaseURL). Trailing
// slashes are ignored.
func WithBaseURL(baseURL string) Option {
	return func(c *Catalog) {
		c.baseURL = strings.TrimRight(baseURL, "/")
	}
}

// WithCacheTTL sets how long a fetched catalog is served from memory before
// it is refetched (default DefaultCacheTTL). A TTL of 0 (or less) refetches on
// every Models call.
func WithCacheTTL(ttl time.Duration) Option {
	return func(c *Catalog) {
		c.cacheTTL = ttl
	}
}

// WithLogger sets the slog logger for malformed-entry and stale-cache
// warnings (default slog.Default()). Passing nil keeps the default.
func WithLogger(logger *slog.Logger) Option {
	return func(c *Catalog) {
		if logger == nil {
			return
		}
		c.logger = logger
	}
}

// Catalog fetches and caches OpenRouter's model catalog. It is safe for
// concurrent use: a mutex serializes refreshes, so concurrent cold Models
// calls share a single fetch.
type Catalog struct {
	apiKey   string
	baseURL  string
	cacheTTL time.Duration

	client     *http.Client
	ownsClient bool
	logger     *slog.Logger

	mu        sync.Mutex
	models    []modelrouter.ModelInfo
	fetchedAt time.Time
}

var _ modelrouter.Catalog = (*Catalog)(nil)

// New returns an OpenRouter catalog client authenticating with apiKey.
//
// The returned Catalog owns its HTTP client (10-second timeout) unless one is
// injected with WithHTTPClient, in which case the caller keeps ownership and
// Close will not touch it.
func New(apiKey string, opts ...Option) *Catalog {
	c := &Catalog{
		apiKey:     apiKey,
		baseURL:    DefaultBaseURL,
		cacheTTL:   DefaultCacheTTL,
		client:     &http.Client{Timeout: defaultHTTPTimeout},
		ownsClient: true,
		logger:     slog.Default(),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Close releases resources owned by the Catalog: it closes the idle
// connections of the HTTP client this instance constructed itself. An
// injected client is never touched. Close is idempotent and safe on a nil
// *Catalog.
func (c *Catalog) Close() error {
	if c == nil {
		return nil
	}
	if c.ownsClient && c.client != nil {
		c.client.CloseIdleConnections()
	}
	return nil
}

// Models returns the catalog, refetching when the cache is empty or older
// than the configured TTL. The result is a defensive copy, so mutating it
// cannot corrupt the cache.
//
// On refetch failure:
//
//   - with any cached copy (even stale) it logs a warning containing
//     "falling back to stale cache" and returns the cached copy;
//   - with no cached copy it returns a *modelrouter.CatalogUnavailableError
//     wrapping the underlying cause.
func (c *Catalog) Models(ctx context.Context) ([]modelrouter.ModelInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.models != nil && c.cacheTTL > 0 && time.Since(c.fetchedAt) < c.cacheTTL {
		return copyModels(c.models), nil
	}

	// The cache timestamp is the request start time, not the response time.
	now := time.Now()
	models, err := c.fetch(ctx)
	if err != nil {
		if c.models != nil {
			c.logger.Warn(fmt.Sprintf(
				"openrouter: failed to fetch catalog (falling back to stale cache): %v", err))
			return copyModels(c.models), nil
		}
		return nil, &modelrouter.CatalogUnavailableError{Err: err}
	}
	c.models = models
	c.fetchedAt = now
	return copyModels(models), nil
}

// fetch performs one HTTP request and decode; cache policy is Models'
// responsibility so callers cannot observe partial refreshes.
func (c *Catalog) fetch(ctx context.Context) ([]modelrouter.ModelInfo, error) {
	url := c.baseURL + modelsPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("building catalog request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("fetching %s: unexpected HTTP status %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: reading response body: %w", url, err)
	}
	models, err := parseCatalog(body, c.logger)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", url, err)
	}
	return models, nil
}

// copyModels returns a shallow copy of models. ModelInfo is a value type, so
// the result shares no mutable state with the cache. The result is always
// non-nil.
func copyModels(models []modelrouter.ModelInfo) []modelrouter.ModelInfo {
	out := make([]modelrouter.ModelInfo, len(models))
	copy(out, models)
	return out
}

// rawCatalogResponse is the wire envelope: {"data": [...]}.
type rawCatalogResponse struct {
	Data json.RawMessage `json:"data"`
}

// parseCatalog tolerantly parses {"data": [...]}. A missing, null or non-list
// "data" field is an error; individual malformed entries are logged and
// skipped; entries present but all skipped (malformed or negative-sentinel)
// is an error ("no valid entries found in catalog response").
func parseCatalog(body []byte, logger *slog.Logger) ([]modelrouter.ModelInfo, error) {
	var envelope rawCatalogResponse
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}
	if len(envelope.Data) == 0 || isJSONNull(envelope.Data) {
		return nil, errors.New("catalog response is missing the 'data' field")
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(envelope.Data, &entries); err != nil {
		return nil, fmt.Errorf("catalog response 'data' field is not a list: %w", err)
	}

	models := make([]modelrouter.ModelInfo, 0, len(entries))
	for i, entry := range entries {
		model, err := parseModel(entry)
		switch {
		case errors.Is(err, errNegativePricing):
			// Silent skip: OpenRouter's variable/unknown pricing sentinel.
			continue
		case err != nil:
			logger.Warn("openrouter: skipping malformed catalog entry", "index", i, "error", err)
			continue
		}
		models = append(models, model)
	}
	if len(entries) > 0 && len(models) == 0 {
		return nil, errors.New("no valid entries found in catalog response")
	}
	return models, nil
}

// rawCatalogEntry carries one entry with raw sub-objects, so type problems in
// a single field become malformed-entry skips instead of failing the whole
// decode.
type rawCatalogEntry struct {
	ID            json.RawMessage `json:"id"`
	ContextLength json.RawMessage `json:"context_length"`
	Pricing       json.RawMessage `json:"pricing"`
	Architecture  json.RawMessage `json:"architecture"`
}

// parseModel parses one catalog entry, tolerating malformed sub-fields:
//
//   - a missing or empty id is malformed;
//   - context_length missing/null/empty is 0, non-numeric is malformed;
//   - pricing.prompt/completion missing/null/empty is 0, non-numeric is
//     malformed;
//   - negative prices return errNegativePricing, which parseCatalog skips
//     silently;
//   - vision comes from architecture.input_modalities when present, else the
//     legacy architecture.modality string ("text+image->text"), else no image
//     support.
func parseModel(entry json.RawMessage) (modelrouter.ModelInfo, error) {
	var raw rawCatalogEntry
	if err := json.Unmarshal(entry, &raw); err != nil {
		return modelrouter.ModelInfo{}, fmt.Errorf("entry is not an object: %w", err)
	}

	// Architecture is resolved before prices: a null architecture is
	// malformed, not text-only.
	supportsVision, err := parseVision(raw.Architecture)
	if err != nil {
		return modelrouter.ModelInfo{}, err
	}
	prompt, completion, err := parsePrices(raw.Pricing)
	if err != nil {
		return modelrouter.ModelInfo{}, err
	}
	if prompt < 0 || completion < 0 {
		return modelrouter.ModelInfo{}, errNegativePricing
	}
	id, err := parseID(raw.ID)
	if err != nil {
		return modelrouter.ModelInfo{}, err
	}
	contextLength, err := parseContextLength(raw.ContextLength)
	if err != nil {
		return modelrouter.ModelInfo{}, err
	}

	return modelrouter.ModelInfo{
		Provider:                   ProviderName,
		ID:                         id,
		PricePer1KPromptTokens:     prompt,
		PricePer1KCompletionTokens: completion,
		ContextLength:              contextLength,
		SupportsVision:             supportsVision,
	}, nil
}

// parseID requires a non-empty string id.
func parseID(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || isJSONNull(raw) {
		return "", errors.New("missing id")
	}
	var id string
	if err := json.Unmarshal(raw, &id); err != nil {
		return "", errors.New("id is not a string")
	}
	if id == "" {
		return "", errors.New("empty id")
	}
	return id, nil
}

// parseContextLength: missing, null or empty is 0; JSON numbers and numeric
// strings are accepted; anything else is malformed.
func parseContextLength(raw json.RawMessage) (int, error) {
	if len(raw) == 0 || isJSONNull(raw) {
		return 0, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, fmt.Errorf("context_length: %w", err)
	}
	switch v := value.(type) {
	case float64:
		return int(v), nil
	case string:
		if v == "" {
			return 0, nil
		}
		length, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return 0, fmt.Errorf("context_length %q is not numeric", v)
		}
		return int(length), nil
	default:
		return 0, fmt.Errorf("context_length has unexpected type %T", value)
	}
}

// parsePrices converts OpenRouter's per-token price strings to per-1k-token
// values. Missing/null/empty prices are 0; non-numeric values are malformed;
// null pricing itself is malformed.
func parsePrices(raw json.RawMessage) (prompt, completion float64, err error) {
	if len(raw) == 0 {
		return 0, 0, nil
	}
	if isJSONNull(raw) {
		return 0, 0, errors.New("pricing is null")
	}
	var pricing map[string]json.RawMessage
	if err := json.Unmarshal(raw, &pricing); err != nil {
		return 0, 0, errors.New("pricing is not an object")
	}
	if prompt, err = parsePrice(pricing["prompt"]); err != nil {
		return 0, 0, fmt.Errorf("pricing.prompt: %w", err)
	}
	if completion, err = parsePrice(pricing["completion"]); err != nil {
		return 0, 0, fmt.Errorf("pricing.completion: %w", err)
	}
	return prompt * 1000, completion * 1000, nil
}

// parsePrice: missing/null/empty is 0, JSON numbers and numeric strings are
// accepted, anything else is malformed.
func parsePrice(raw json.RawMessage) (float64, error) {
	if len(raw) == 0 || isJSONNull(raw) {
		return 0, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, err
	}
	switch v := value.(type) {
	case float64:
		return v, nil
	case string:
		if v == "" {
			return 0, nil
		}
		price, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return 0, fmt.Errorf("%q is not numeric", v)
		}
		return price, nil
	default:
		return 0, fmt.Errorf("unexpected type %T for price", value)
	}
}

// parseVision prefers architecture.input_modalities, falls back to the
// legacy "modality" string ("text+image->text"), and treats a missing
// architecture as text-only; a present-but-null architecture is malformed.
func parseVision(raw json.RawMessage) (bool, error) {
	if len(raw) == 0 {
		return false, nil
	}
	if isJSONNull(raw) {
		return false, errors.New("architecture is null")
	}
	var architecture map[string]json.RawMessage
	if err := json.Unmarshal(raw, &architecture); err != nil {
		return false, errors.New("architecture is not an object")
	}

	rawInputs, hasInputs := architecture["input_modalities"]
	if hasInputs && !isJSONNull(rawInputs) {
		var modalities []string
		if err := json.Unmarshal(rawInputs, &modalities); err != nil {
			return false, errors.New("architecture.input_modalities is not a list of strings")
		}
		return containsImage(modalities), nil
	}

	modality := "text->text"
	if rawModality, ok := architecture["modality"]; ok && !isJSONNull(rawModality) {
		if err := json.Unmarshal(rawModality, &modality); err != nil {
			return false, errors.New("architecture.modality is not a string")
		}
	}
	inputs := strings.Split(strings.Split(modality, "->")[0], "+")
	return containsImage(inputs), nil
}

// containsImage reports whether the modality list names image input.
func containsImage(modalities []string) bool {
	for _, modality := range modalities {
		if modality == "image" {
			return true
		}
	}
	return false
}

// isJSONNull reports whether raw holds the JSON literal null.
func isJSONNull(raw json.RawMessage) bool {
	return string(bytes.TrimSpace(raw)) == "null"
}
