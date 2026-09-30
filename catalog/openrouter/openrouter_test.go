package openrouter_test

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/catalog/openrouter"
)

// loadFixture reads the sample OpenRouter catalog response used by the
// fetch/parse tests.
func loadFixture(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "openrouter_catalog_sample.json"))
	if err != nil {
		t.Fatalf("reading catalog fixture: %v", err)
	}
	return data
}

func newServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

func newCatalog(t *testing.T, server *httptest.Server, opts ...openrouter.Option) *openrouter.Catalog {
	t.Helper()
	base := []openrouter.Option{
		openrouter.WithBaseURL(server.URL),
		openrouter.WithHTTPClient(server.Client()),
	}
	catalog := openrouter.New("test-key", append(base, opts...)...)
	t.Cleanup(func() { _ = catalog.Close() })
	return catalog
}

func jsonHandler(status int, body []byte) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}
}

func fixtureWithExtraEntries(t *testing.T, extraJSON ...string) []byte {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal(loadFixture(t), &payload); err != nil {
		t.Fatalf("unmarshaling fixture: %v", err)
	}
	entries, ok := payload["data"].([]any)
	if !ok {
		t.Fatalf("fixture data is %T, want []any", payload["data"])
	}
	for _, raw := range extraJSON {
		var entry any
		if err := json.Unmarshal([]byte(raw), &entry); err != nil {
			t.Fatalf("unmarshaling extra entry %s: %v", raw, err)
		}
		entries = append(entries, entry)
	}
	payload["data"] = entries
	out, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshaling payload: %v", err)
	}
	return out
}

func modelIDSet(models []modelrouter.ModelInfo) map[string]bool {
	ids := make(map[string]bool, len(models))
	for _, model := range models {
		ids[model.ID] = true
	}
	return ids
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

type recordingHandler struct {
	mu       sync.Mutex
	messages []string
}

func (h *recordingHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *recordingHandler) Handle(_ context.Context, record slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = append(h.messages, record.Message)
	return nil
}

func (h *recordingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *recordingHandler) WithGroup(string) slog.Handler { return h }

func (h *recordingHandler) contains(substr string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, message := range h.messages {
		if strings.Contains(message, substr) {
			return true
		}
	}
	return false
}

func (h *recordingHandler) all() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.messages...)
}

type requestInfo struct {
	method string
	path   string
	auth   string
}

func TestFetchParsesBothArchitectureShapes(t *testing.T) {
	fixture := loadFixture(t)
	got := make(chan requestInfo, 1)
	server := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		got <- requestInfo{method: r.Method, path: r.URL.Path, auth: r.Header.Get("Authorization")}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	})
	catalog := newCatalog(t, server)

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if len(models) != 4 {
		t.Fatalf("len(models) = %d, want 4", len(models))
	}

	byID := make(map[string]modelrouter.ModelInfo, len(models))
	for _, model := range models {
		byID[model.ID] = model
		if model.Provider != openrouter.ProviderName {
			t.Errorf("model %q Provider = %q, want %q", model.ID, model.Provider, openrouter.ProviderName)
		}
	}

	// Modern shape: architecture.input_modalities.
	gpt := byID["openai/gpt-4o"]
	if gpt.ContextLength != 128_000 {
		t.Errorf("gpt-4o ContextLength = %d, want 128000", gpt.ContextLength)
	}
	if !gpt.SupportsVision {
		t.Error("gpt-4o SupportsVision = false, want true")
	}
	if !almostEqual(gpt.PricePer1KPromptTokens, 2.5) {
		t.Errorf("gpt-4o PricePer1KPromptTokens = %v, want 2.5", gpt.PricePer1KPromptTokens)
	}
	if !almostEqual(gpt.PricePer1KCompletionTokens, 10.0) {
		t.Errorf("gpt-4o PricePer1KCompletionTokens = %v, want 10.0", gpt.PricePer1KCompletionTokens)
	}

	// Legacy shape: architecture.modality "text->text".
	deepseek := byID["deepseek/deepseek-r1"]
	if deepseek.SupportsVision {
		t.Error("deepseek-r1 SupportsVision = true, want false")
	}
	if !almostEqual(deepseek.PricePer1KPromptTokens, 0.5) {
		t.Errorf("deepseek-r1 PricePer1KPromptTokens = %v, want 0.5", deepseek.PricePer1KPromptTokens)
	}

	info := <-got
	if info.method != http.MethodGet {
		t.Errorf("request method = %q, want GET", info.method)
	}
	if info.path != "/api/v1/models" {
		t.Errorf("request path = %q, want /api/v1/models", info.path)
	}
	if info.auth != "Bearer test-key" {
		t.Errorf("Authorization header = %q, want %q", info.auth, "Bearer test-key")
	}
}

func TestFetchParsesLegacyModalityImageSupport(t *testing.T) {
	payload := fixtureWithExtraEntries(t, `{
		"id": "legacy/vision-model",
		"context_length": 32000,
		"pricing": {"prompt": "0.001", "completion": "0.002"},
		"architecture": {"modality": "text+image->text"}
	}`)
	server := newServer(t, jsonHandler(http.StatusOK, payload))
	catalog := newCatalog(t, server)

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	for _, model := range models {
		if model.ID == "legacy/vision-model" {
			if !model.SupportsVision {
				t.Error("legacy/vision-model SupportsVision = false, want true from text+image->text")
			}
			return
		}
	}
	t.Fatalf("legacy/vision-model missing from parsed catalog: %v", modelIDSet(models))
}

func TestFetchTreatsMissingArchitectureAsTextOnly(t *testing.T) {
	payload := fixtureWithExtraEntries(t,
		`{"id": "bare/model", "context_length": 1000, "pricing": {"prompt": "0.001", "completion": "0.001"}}`)
	server := newServer(t, jsonHandler(http.StatusOK, payload))
	catalog := newCatalog(t, server)

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if len(models) != 5 {
		t.Fatalf("len(models) = %d, want 5", len(models))
	}
	for _, model := range models {
		if model.ID == "bare/model" && model.SupportsVision {
			t.Error("bare/model SupportsVision = true, want false with no architecture")
		}
	}
}

func TestFetchCachesResultWithinTTL(t *testing.T) {
	fixture := loadFixture(t)
	var requests int32
	server := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&requests, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	})
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(600*time.Second))

	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("second Models() error = %v", err)
	}

	if got := atomic.LoadInt32(&requests); got != 1 {
		t.Errorf("HTTP requests = %d, want 1 (second call must hit the TTL cache)", got)
	}
}

func TestFetchWithZeroTTLRefetches(t *testing.T) {
	fixture := loadFixture(t)
	var requests int32
	server := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&requests, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	})
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(0))

	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("second Models() error = %v", err)
	}

	if got := atomic.LoadInt32(&requests); got != 2 {
		t.Errorf("HTTP requests = %d, want 2 (TTL 0 must refetch every call)", got)
	}
}

func TestFetchFallsBackToStaleCacheOnFailure(t *testing.T) {
	fixture := loadFixture(t)
	var requests int32
	server := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&requests, 1) == 1 {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(fixture)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	})
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(0))

	first, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	second, err := catalog.Models(context.Background()) // TTL 0: refetch hits the 500 -> stale cache
	if err != nil {
		t.Fatalf("second Models() error = %v, want stale-cache fallback", err)
	}

	if !reflect.DeepEqual(second, first) {
		t.Errorf("stale fallback result differs from first result:\n second = %v\n first  = %v", second, first)
	}
	if got := atomic.LoadInt32(&requests); got != 2 {
		t.Errorf("HTTP requests = %d, want 2", got)
	}
}

func TestFetchErrorsWhenNoCacheAndRequestFails(t *testing.T) {
	server := newServer(t, jsonHandler(http.StatusInternalServerError, nil))
	catalog := newCatalog(t, server)

	_, err := catalog.Models(context.Background())
	if !errors.Is(err, modelrouter.ErrCatalogUnavailable) {
		t.Fatalf("Models() error = %v, want errors.Is(..., ErrCatalogUnavailable)", err)
	}
	var unavailable *modelrouter.CatalogUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("Models() error = %v, want *CatalogUnavailableError", err)
	}
	if unavailable.Err == nil {
		t.Error("CatalogUnavailableError.Err = nil, want wrapped cause")
	}
}

func TestFetchLogsWarningOnStaleCacheFallback(t *testing.T) {
	fixture := loadFixture(t)
	var requests int32
	server := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&requests, 1) == 1 {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(fixture)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	})
	logs := &recordingHandler{}
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(0), openrouter.WithLogger(slog.New(logs)))

	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("second Models() error = %v", err)
	}

	if !logs.contains("falling back to stale cache") {
		t.Errorf("log messages = %v, want a warning containing %q", logs.all(), "falling back to stale cache")
	}
	if !logs.contains("500") {
		t.Errorf("log messages = %v, want the warning to include the underlying error (status 500)", logs.all())
	}
}

func TestFetchHandlesMalformed200ResponseWithCache(t *testing.T) {
	fixture := loadFixture(t)
	var requests int32
	server := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if atomic.AddInt32(&requests, 1) == 1 {
			_, _ = w.Write(fixture)
			return
		}
		_, _ = w.Write([]byte(`{}`)) // malformed 200: missing "data"
	})
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(0))

	first, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	second, err := catalog.Models(context.Background()) // re-fetches, gets malformed 200 -> stale cache
	if err != nil {
		t.Fatalf("second Models() error = %v, want stale-cache fallback", err)
	}
	if !reflect.DeepEqual(second, first) {
		t.Errorf("stale fallback result differs from first result:\n second = %v\n first  = %v", second, first)
	}
}

func TestFetchErrorsOnMalformed200ResponseWithoutCache(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"missing data key", `{}`},
		{"null data", `{"data": null}`},
		{"data not a list", `{"data": "nope"}`},
		{"invalid json", `not json`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newServer(t, jsonHandler(http.StatusOK, []byte(tt.body)))
			catalog := newCatalog(t, server)

			_, err := catalog.Models(context.Background())
			if !errors.Is(err, modelrouter.ErrCatalogUnavailable) {
				t.Fatalf("Models() error = %v, want errors.Is(..., ErrCatalogUnavailable)", err)
			}
		})
	}
}

func TestFetchSkipsEntriesWithNegativeSentinelPricing(t *testing.T) {
	payload := fixtureWithExtraEntries(t, `{
		"id": "openrouter/auto",
		"context_length": 128000,
		"pricing": {"prompt": "-1", "completion": "-1"},
		"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}
	}`)
	logs := &recordingHandler{}
	server := newServer(t, jsonHandler(http.StatusOK, payload))
	catalog := newCatalog(t, server, openrouter.WithLogger(slog.New(logs)))

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}

	if modelIDSet(models)["openrouter/auto"] {
		t.Error("openrouter/auto kept despite negative sentinel pricing")
	}
	for _, model := range models {
		if model.PricePer1KPromptTokens < 0 || model.PricePer1KCompletionTokens < 0 {
			t.Errorf("model %q kept with negative price: %v/%v",
				model.ID, model.PricePer1KPromptTokens, model.PricePer1KCompletionTokens)
		}
	}
	// Negative sentinel pricing is a silent skip (no warning), unlike
	// malformed entries.
	if messages := logs.all(); len(messages) != 0 {
		t.Errorf("log messages = %v, want none (negative sentinel skip is silent)", messages)
	}
}

func TestFetchSkipsASingleMalformedEntryAndReturnsTheRest(t *testing.T) {
	// Missing "id" -> malformed inside _parse_model, should be skipped not fatal.
	payload := fixtureWithExtraEntries(t, `{
		"context_length": 32000,
		"pricing": {"prompt": "0.001", "completion": "0.002"},
		"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}
	}`)
	server := newServer(t, jsonHandler(http.StatusOK, payload))
	catalog := newCatalog(t, server)

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if len(models) != 4 {
		t.Fatalf("len(models) = %d, want 4 (malformed entry skipped)", len(models))
	}
	want := map[string]bool{
		"openai/gpt-4o":                    true,
		"anthropic/claude-haiku-4-5":       true,
		"deepseek/deepseek-r1":             true,
		"meta-llama/llama-3.1-8b-instruct": true,
	}
	if got := modelIDSet(models); !reflect.DeepEqual(got, want) {
		t.Errorf("model IDs = %v, want %v", got, want)
	}
}

func TestFetchSkipsEntryWithNullArchitecture(t *testing.T) {
	payload := fixtureWithExtraEntries(t, `{
		"id": "vendor/broken-architecture",
		"context_length": 32000,
		"pricing": {"prompt": "0.001", "completion": "0.002"},
		"architecture": null
	}`)
	server := newServer(t, jsonHandler(http.StatusOK, payload))
	catalog := newCatalog(t, server)

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if modelIDSet(models)["vendor/broken-architecture"] {
		t.Error("vendor/broken-architecture kept despite null architecture")
	}
	if len(models) != 4 {
		t.Errorf("len(models) = %d, want 4", len(models))
	}
}

func TestFetchLogsAndSkipsMalformedEntry(t *testing.T) {
	payload := fixtureWithExtraEntries(t, `{"context_length": 32000}`) // missing "id"
	logs := &recordingHandler{}
	server := newServer(t, jsonHandler(http.StatusOK, payload))
	catalog := newCatalog(t, server, openrouter.WithLogger(slog.New(logs)))

	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("Models() error = %v", err)
	}

	if !logs.contains("malformed") {
		t.Errorf("log messages = %v, want a warning containing %q", logs.all(), "malformed")
	}
}

func TestFetchErrorsWhenAllEntriesAreSkipped(t *testing.T) {
	tests := []struct {
		name    string
		payload string
	}{
		{
			name:    "all malformed",
			payload: `{"data": [{"context_length": 32000}, {"id": ""}]}`,
		},
		{
			name:    "all negative sentinel",
			payload: `{"data": [{"id": "a", "pricing": {"prompt": "-1", "completion": "-1"}}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newServer(t, jsonHandler(http.StatusOK, []byte(tt.payload)))
			catalog := newCatalog(t, server)

			_, err := catalog.Models(context.Background())
			if !errors.Is(err, modelrouter.ErrCatalogUnavailable) {
				t.Fatalf("Models() error = %v, want errors.Is(..., ErrCatalogUnavailable)", err)
			}
			if !strings.Contains(err.Error(), "no valid entries found in catalog response") {
				t.Errorf("Models() error = %q, want it to mention no valid entries", err)
			}
		})
	}
}

func TestFetchFallsBackToStaleCacheWhenAllEntriesMalformed(t *testing.T) {
	fixture := loadFixture(t)
	var requests int32
	server := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if atomic.AddInt32(&requests, 1) == 1 {
			_, _ = w.Write(fixture)
			return
		}
		_, _ = w.Write([]byte(`{"data": [{"context_length": 32000}]}`))
	})
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(0))

	first, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	second, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("second Models() error = %v, want stale-cache fallback", err)
	}
	if !reflect.DeepEqual(second, first) {
		t.Errorf("stale fallback result differs from first result:\n second = %v\n first  = %v", second, first)
	}
}

func TestFetchEmptyDataReturnsEmptyCatalog(t *testing.T) {
	server := newServer(t, jsonHandler(http.StatusOK, []byte(`{"data": []}`)))
	catalog := newCatalog(t, server)

	models, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if models == nil {
		t.Error("Models() = nil, want non-nil empty slice")
	}
	if len(models) != 0 {
		t.Errorf("len(models) = %d, want 0", len(models))
	}
}

func TestModelsReturnsDefensiveCopyOfCache(t *testing.T) {
	fixture := loadFixture(t)
	server := newServer(t, jsonHandler(http.StatusOK, fixture))
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(time.Minute))

	first, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("first Models() error = %v", err)
	}
	first[0].ID = "mutated"
	first[0].PricePer1KPromptTokens = 999

	second, err := catalog.Models(context.Background())
	if err != nil {
		t.Fatalf("second Models() error = %v", err)
	}
	if second[0].ID != "openai/gpt-4o" {
		t.Errorf("cached model ID = %q, want %q (cache was mutated)", second[0].ID, "openai/gpt-4o")
	}
	if !almostEqual(second[0].PricePer1KPromptTokens, 2.5) {
		t.Errorf("cached prompt price = %v, want 2.5 (cache was mutated)", second[0].PricePer1KPromptTokens)
	}
}

func TestModelsIsConcurrentSafe(t *testing.T) {
	fixture := loadFixture(t)
	var requests int32
	server := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&requests, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	})
	catalog := newCatalog(t, server, openrouter.WithCacheTTL(time.Minute))

	const goroutines = 16
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			models, err := catalog.Models(context.Background())
			if err != nil {
				t.Errorf("Models() error = %v", err)
				return
			}
			if len(models) != 4 {
				t.Errorf("len(models) = %d, want 4", len(models))
			}
		}()
	}
	wg.Wait()

	if got := atomic.LoadInt32(&requests); got != 1 {
		t.Errorf("HTTP requests = %d, want 1 (concurrent cold calls must share one fetch)", got)
	}
}

func TestCloseIsIdempotentAndNilSafe(t *testing.T) {
	catalog := openrouter.New("test-key")
	for attempt := 1; attempt <= 2; attempt++ {
		if err := catalog.Close(); err != nil {
			t.Fatalf("Close() #%d error = %v", attempt, err)
		}
	}

	var nilCatalog *openrouter.Catalog
	if err := nilCatalog.Close(); err != nil {
		t.Errorf("nil *Catalog Close() error = %v, want nil", err)
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
	fixture := loadFixture(t)
	server := newServer(t, jsonHandler(http.StatusOK, fixture))
	transport := &trackingTransport{base: server.Client().Transport}
	client := &http.Client{Transport: transport}
	catalog := openrouter.New("test-key", openrouter.WithBaseURL(server.URL), openrouter.WithHTTPClient(client))

	if _, err := catalog.Models(context.Background()); err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if err := catalog.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := catalog.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}

	if got := atomic.LoadInt32(&transport.closes); got != 0 {
		t.Errorf("injected client CloseIdleConnections calls = %d, want 0", got)
	}
}

// Upstream asserts `catalog._client.is_closed` on the self-constructed httpx
// client; Go's http.Client is not introspectable, so closure is observed
// through connection reuse: after Close, the next request must dial a fresh
// connection. net/http hands a connection back to the idle pool
// asynchronously after a response, so the scenario is retried a few times to
// avoid a scheduling race — a no-op Close fails every attempt.
func TestCloseClosesSelfConstructedClient(t *testing.T) {
	const attempts = 10
	fixture := loadFixture(t)
	for attempt := 0; attempt < attempts; attempt++ {
		var newConns int32
		idle := make(chan struct{}, 4)
		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(fixture)
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

		catalog := openrouter.New("test-key",
			openrouter.WithBaseURL(server.URL),
			openrouter.WithCacheTTL(0),
		)
		if _, err := catalog.Models(context.Background()); err != nil {
			server.Close()
			t.Fatalf("Models() error = %v", err)
		}
		select {
		case <-idle:
		case <-time.After(5 * time.Second):
			server.Close()
			t.Fatal("server never observed an idle connection")
		}
		if err := catalog.Close(); err != nil {
			server.Close()
			t.Fatalf("Close() error = %v", err)
		}
		_, err := catalog.Models(context.Background())
		_ = catalog.Close()
		closed := err == nil && atomic.LoadInt32(&newConns) >= 2
		server.Close()

		if closed {
			return
		}
	}
	t.Errorf("after Close, later requests kept reusing the old connection (%d attempts): Close must close the owned client", attempts)
}
