# Design notes

This document records deliberate design decisions in jev-router and maps
documented behavior to the tests that pin it, so a change that touches one of
these contracts has a clear place to check for regressions.

---

## 1. Behavior reference

### 1.1 Catalog fetch (`catalog/openrouter`)

| Behavior | Test(s) |
|---|---|
| `GET {base}/api/v1/models` with `Authorization: Bearer <key>`, decode `{"data":[...]}` | all fetch tests in `catalog/openrouter/openrouter_test.go`; live: `integration_test.go::TestIntegrationFetchLiveCatalog` |
| Modern architecture shape: `architecture.input_modalities` containing `"image"` → vision | `TestFetchParsesBothArchitectureShapes` |
| Legacy modality string (`"text+image->text"`) as fallback | `TestFetchParsesLegacyModalityImageSupport` |
| Missing/`null` architecture → text-only | `TestFetchTreatsMissingArchitectureAsTextOnly`, `TestFetchSkipsEntryWithNullArchitecture` |
| `pricing.prompt` / `pricing.completion` are **per-token** USD → scaled ×1000 to per-1k | every pricing assertion |
| Negative sentinel pricing (OpenRouter `"-1"` = variable/unknown) → skip silently | `TestFetchSkipsEntriesWithNegativeSentinelPricing` |
| Malformed single entry → warn + skip, keep the rest (per-entry decode over `json.RawMessage`) | `TestFetchSkipsASingleMalformedEntryAndReturnsTheRest`, `TestFetchLogsAndSkipsMalformedEntry` |
| All entries malformed → error (not an empty catalog) | `TestFetchErrorsWhenAllEntriesAreSkipped`; with a warm cache: `TestFetchFallsBackToStaleCacheWhenAllEntriesMalformed` |
| In-memory TTL cache (default 600s) | `TestFetchCachesResultWithinTTL`, `TestFetchWithZeroTTLRefetches` |
| Refetch failure with warm cache → serve stale + warning log (no error) | `TestFetchFallsBackToStaleCacheOnFailure`, `TestFetchLogsWarningOnStaleCacheFallback`, `TestFetchHandlesMalformed200ResponseWithCache` |
| Failure with no cache → `CatalogUnavailableError` | `TestFetchErrorsWhenNoCacheAndRequestFails`, `TestFetchErrorsOnMalformed200ResponseWithoutCache` |
| Cache serves defensive copies; usable concurrently | `TestModelsReturnsDefensiveCopyOfCache`, `TestModelsIsConcurrentSafe` |
| Close ownership: self-constructed HTTP client is closed, injected one never | `TestCloseIsIdempotentAndNilSafe`, `TestCloseDoesNotCloseInjectedClient`, `TestCloseClosesSelfConstructedClient` |
| Every model is stamped `Provider: "openrouter"` | all fetch tests via fixture assertions |

### 1.2 Filtering (`catalog.go::FilterModels`)

| Behavior | Test(s) |
|---|---|
| Drop models without vision when `needs_vision` | `catalog_test.go::TestFilterModelsDropsModelsMissingVisionSupport` |
| Drop models below `min_context_tokens` / above `max_price_per_1k_tokens` | `TestFilterModelsDropsModelsBelowMinContextOrAboveMaxPrice` |
| Price constraint checked against the **blended** price (prompt + completion per 1k) | `TestFilterModelsMaxPriceUsesBlendedPromptPlusCompletionPrice` |
| No constraints → keep everything | `TestFilterModelsDefaultConstraintsKeepEverything` |
| Boundaries are inclusive (`>=` context, `<=` price) | `TestFilterModelsContextAndPriceBoundariesAreInclusive` |
| Preserve order; never mutate input; empty input → non-nil empty result | `TestFilterModelsPreservesOrderAndDoesNotMutateInput`, `TestFilterModelsEmptyInputReturnsNonNilEmpty` |
| Provider allow/exclude lists | `RoutingConstraints.AllowsProvider`; `TestFilterModelsProviderAllowAndExclude` |

### 1.3 Classification (`classify/jev.go`)

| Behavior | Test(s) |
|---|---|
| One narrow JSON POST per routing call; request carries the **five fixed questions** (domain choice; complexity score; `needs_long_context`/`needs_vision`/`latency_sensitive` noul) | `classify/jev_test.go::TestClassifyRequestShape` (full `DeepEqual` against a pinned wire payload) |
| Parse `answers.domain.choice/confidence`, `answers.complexity.score/confidence`, `answers.*.noul` | `TestClassifyParsesAWellFormedJevResponse` |
| Any domain string is passed through (no enum check) | `TestClassifyAcceptsAnyDomainChoice` |
| Transport error, non-2xx, malformed JSON, missing/non-numeric fields → `ClassifierUnavailableError` (wrapped cause, `errors.Is`/`errors.As`) | `TestClassifyReturnsClassifierUnavailableOnHTTPError`, `…OnMissingAnswersKey`, `…OnIncompleteResponse`, `…OnNonNumericConfidence`, `TestClassifyTimeout` |
| Default 5s timeout on a self-owned client; injected client never closed | `TestClassifyTimeout`, `TestCloseDoesNotCloseInjectedClient`, `TestCloseClosesSelfConstructedClient`, `TestCloseIsIdempotentAndNilSafe` |
| Optional `usage` block, parsed leniently (never fails a classification) | `TestClassifyParsesUsage`, `TestClassifyParsesOpenRouterUsageAliases`, `TestClassifyMalformedUsageDoesNotFailClassification` |
| Configurable base URL/path/model | `WithBaseURL`/`WithPath`/`WithModel`; `TestClassifyCustomBaseURLAndPath`; live: `config/integration_test.go::TestIntegrationRouteWithLiveJev` |

### 1.4 Configuration (`config/config.go`)

| Behavior | Test(s) |
|---|---|
| Defaults: cache TTL 600s, `fail_open`, standard Jev endpoint at 5s | `config/config.go::Default`; `TestLoadEnvOnlyDefaults` |
| Environment-only mode: OpenRouter key read from the environment; error **names the missing variable** | `Load("")`; `TestLoadEnvOnlyMissingKeyNamesVar`, `TestLoadEnvOnlyEmptyKeyCountsAsMissing` |
| Invalid `on_classifier_error` values rejected (typos, wrong case) | `Validate`; `TestValidateRejectsTypoedOnClassifierError`, `TestValidateRejectsWrongCaseOnClassifierError` |
| Explicit values honored, defaults preserved for absent fields | strict `KnownFields` decode on top of `Default()`; `TestLoadExplicitValuesHonored`, `TestLoadPartialFileKeepsDefaults` |
| Duration strings or integer seconds; strict unknown-field rejection; `JEV_ROUTER_CONFIG`/`JEV_ROUTER_SERVER_ADDR`; catalog paths relative to the config file; per-provider validation | `Duration`, `Load`, `ResolvePath`, `Validate`, `validateProviders`; `TestDurationUnmarshal`, `TestDurationMarshalEmitsString`, `TestLoadRejectsUnknownFields`, `TestLoadUsesConfigPathEnv`, `TestLoadServerAddrEnvOverride`, `TestDirAndResolvePath`, `TestValidateTable`, `config/assembly_test.go` |
| `routing.models` maps each of the five classified domains to an ordered chain of `{provider, model}` candidates; `routing.default_model` covers unclassified requests and unmapped domains, and is appended to every domain chain (when not already present) as the last-resort candidate | `config/config.go::Routing`; `router_test.go` (mapped selection and fallback) |

### 1.4a OpenAI-compatible API (`server/openai.go`)

| Behavior | Test(s) |
|---|---|
| Real `openai-go` client round-trips `/v1/chat/completions` (content, usage, model, finish reason, developer→system, `max_completion_tokens` precedence) and `/v1/models` | `TestOpenAISDKRoundTrip`, `TestOpenAISDKErrorAndModels` |
| Unsupported inputs (stream, `n>1`, tools, tool role, image parts, bad content) are 400s in OpenAI's envelope with `code`/`param`; the router is never called | `TestChatCompletionsRejections` |
| Empty `tools`, text-part arrays, null assistant content, extra fields and `Authorization` are accepted | `TestChatCompletionsAcceptsOpenAIQuirks` |
| Provider finish reasons normalize to `stop`/`length`/`content_filter` | `TestNormalizedFinishReasonsOnTheWire` |
| Transport errors (405) on compat paths use OpenAI's envelope | `TestCompatPathsUseOpenAIEnvelopeForTransportErrors` |
| Request bodies are capped (413) on every POST endpoint | `TestRequestBodyLimit` |
| A Messages-only completion classifies the last user message, not an empty string | `TestCompleteClassifiesMessagesOnlyRequests` |

### 1.5 Diagnostics (`diagnostics.go`)

| Behavior | Test(s) |
|---|---|
| `DescribeEliminatingConstraints`: per-constraint survivor counts; names the constraint(s) responsible, including the "combination of constraints" case | `diagnostics_test.go` (`NeedsVision`, `MinContextTokens`, `MaxPricePer1KTokens`, `Combination`, `EmptyCatalog`) |
| Per-provider suffix (`per provider: X N/M`) when more than one provider contributed | `TestDescribeEliminatingConstraintsPerProviderSuffix`, `TestRouteNoCandidateErrorIncludesPerProviderCounts` |

---

## 2. Mapped routing

`routing.models` maps each of the five classified domains (`code`,
`math_reasoning`, `creative`, `factual_lookup`, `other`) to an ordered chain
of `{provider, model}` candidates. `routing.default_model` is the chain for
unclassified requests (nil profile under `fail_open`) and unmapped domains,
and is appended to every domain chain — when not already present — as the
last-resort candidate.

- `RoutingDecision` carries `Source` (`"classification"` when a domain chain
  was selected, `"default"` otherwise) and `Candidates` (the ordered attempt
  chain after constraint filtering).
- Catalogs are consulted only when a constraint needs their metadata
  (`needs_vision`, `min_context_tokens`, `max_price_per_1k_tokens`); provider
  allow/exclude and mapped selection work without them. When they are
  fetched, the all-must-succeed rule applies — any catalog failure fails the
  route with every error joined — and a candidate missing from the catalogs
  fails the metadata checks. An entirely filtered chain surfaces as
  `*NoCandidateModelError` naming the configured candidates (plus catalog
  diagnostics when they were fetched).
- Fail-open classifier behavior routes on the default model with a nil
  profile — never a fabricated one.
- With no explicit model, `Complete` tries `decision.Candidates` in order; a
  missing provider registration or any `*ProviderError` advances to the next
  candidate. A single-candidate chain returns that error unchanged; with
  multiple attempts the errors join as
  `modelrouter: all N candidate models failed: …`, and `errors.Is`/`errors.As`
  still reach the inner errors.
- Explicit-model requests never fall back — `Provider` is required, the
  classifier is skipped, and `RoutingDecision` stays zero.
- The eval harness measures classification-domain correctness (`Matched =
  DomainOK`); price fields (`chosen_blended_price`, `median_blended_price`,
  `frontier_blended_price`, `savings_vs_frontier`, `mean_savings`) are
  computed only when the caller supplies a `ListModels` price pool
  (`Options.Models`, which the CLI fetches best-effort) and read 0 without
  one.
- Completions are synchronous only: every layer exposes exactly one
  completion call, `POST /v1/complete` always returns the complete result as
  plain JSON, and a legacy `"stream": true` body field is ignored like any
  other unknown field. `POST /v1/chat/completions` is also single-shot, but
  rejects `"stream": true` with a 400 because OpenAI SDKs would otherwise try
  to parse the JSON body as SSE.

Test mapping: `config/config_test.go` (mapping/env-override cases),
`router_test.go` (mapped selection and fallback), `eval/eval_test.go`
(`Matched = DomainOK`, pool-relative prices), plus the CLI tests.

---

## 3. Design decisions

1. **Jev is called through OpenRouter's systemone endpoint with a single
   key.** Default: `POST https://openrouter.ai/api/v1/systemone`, model
   `~typesafe/jev-latest`, authenticated with `OPENROUTER_API_KEY`.
   Rationale: one key covers classifier + catalog + completions for the
   common case; `base_url`/`path`/`model` remain configurable, so pointing
   at TypeSafe directly is a configuration change.
2. **The OpenRouter catalog uses stdlib HTTP with tolerant per-entry parsing
   instead of the OpenRouter Go SDK** (the SDK's all-or-nothing decode
   cannot skip malformed entries, and the catalog needs to keep serving
   despite them). Completions **do** use the official `OpenRouterTeam/go-sdk`.
3. **Non-OpenRouter providers require user-supplied catalogs** (JSON/YAML
   files with prices you maintain). Rationale: those APIs expose no portable
   public price data; shipping stale snapshots would silently misroute.
4. **The Router owns nothing.** It is pure injection with no `Close`;
   assembly code (`config.App`) owns construction and lifecycle. Rationale:
   ownership lives in one place.
5. **Go error model: sentinels + typed carriers** (`ErrCatalogUnavailable`,
   `ErrClassifierUnavailable`, `ErrNoCandidate`, `ErrProvider`,
   `ErrProviderNotConfigured`) with `errors.Is`/`errors.As` and wrapped
   causes. Rationale: idiomatic Go, and the HTTP layer maps sentinels to
   stable codes.
6. **The eval harness checks domain accuracy** (plus optional pool-relative
   savings), and excludes skipped (profile-less) decisions from every
   evaluated aggregate. Rationale: savings are meaningful only relative to a
   known pool, and a skipped decision was never checked by the classifier,
   so counting it would bias accuracy.
7. **Diagnostics wording is Go-style**: booleans render as `true`/`false`,
   the constraint combination renders as a Go struct literal, and
   multi-provider results append a per-provider survivor suffix.
8. **Telemetry scope**: spans `modelrouter.{route,catalog.fetch,classify,
   models,complete}` and exactly three metrics (`modelrouter.decisions`,
   `modelrouter.classify.duration`, `modelrouter.upstream.errors`). No
   provider-internal spans and no token/cost counters yet. Rationale: keep
   the observable contract stable and provider-agnostic; deeper
   instrumentation is on the roadmap.
9. **The HTTP server has no authentication.** Rationale: scope control; it's
   intended for localhost/internal use behind a reverse proxy, and auth is a
   roadmap item that belongs in front of the router.
10. **Bedrock offline tests use an interface seam** over the SDK's Bedrock
    Runtime client (`*bedrockruntime.Client` satisfies it) so white-box tests
    can inject a fake without HTTP. Rationale: production uses the SDK
    directly while keeping the provider unit-testable.
