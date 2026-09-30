# Architecture notes

Status: implemented. This is a condensed record of the original design plan
and the phase-by-phase build history, kept for anyone digging into why the
codebase is shaped the way it is. For current behavior and design rationale,
see [docs/design-notes.md](../design-notes.md); for the HTTP/CLI/config
surface, see the [README](../../README.md).

Two decisions made after the initial build are worth knowing up front, since
the rest of this document predates them:

- **v2 (mapped routing):** the original price-percentile scorer was replaced
  with explicit, per-domain candidate chains and a configured default model
  (see [docs/design-notes.md](../design-notes.md) §2). `scorer.go` and its
  config knobs (`confidence_threshold`, `weight_complexity_match`,
  `weight_domain`) no longer exist.
- **Streaming removed:** SSE/streaming support was removed entirely in favor
  of a synchronous-only product. `Router.Stream`, every provider's `Stream`
  method, `server/sse.go` and the streaming CLI flags no longer exist.

---

## 1. Goal

A multi-provider LLM router with:

- **Multi-provider routing** (factory-style providers: OpenRouter, Bedrock,
  OpenAI, Anthropic, Gemini, generic OpenAI-compatible).
- **Jev classification via OpenRouter** (`POST /api/v1/systemone`, model
  `~typesafe/jev-latest`) — one `OPENROUTER_API_KEY` covers classifier,
  catalog and completions by default.
- **CLI + HTTP server**, with OpenTelemetry traces and metrics.
- **A layered offline/integration/eval test strategy.**

## 2. Confirmed decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Deliverable shape | Library + CLI (`jev-router`) + HTTP server |
| 2 | Provider architecture | Multi-provider routing: each provider contributes its own catalog; the winning candidate's provider executes the completion |
| 3 | Jev surface | `POST https://openrouter.ai/api/v1/systemone`, `model: ~typesafe/jev-latest` (base URL / path / model configurable) |
| 4 | Module identity | `github.com/vprprudhvi/jev-router`, package `modelrouter`, binary `jev-router` |
| 5 | HTTP clients | Official/third-party SDKs everywhere; a small custom client only for the Jev endpoint |
| 6 | Provider matrix | AWS Bedrock, OpenAI, Anthropic, Google Gemini, OpenAI-compatible (custom base URL) |
| 7 | Non-OpenRouter price sources | User-supplied catalogs only (JSON/YAML files); no shipped defaults |
| 8 | HTTP server scope | `POST /v1/route`, `POST /v1/complete`, `GET /healthz`; request IDs; structured `slog`; no auth (localhost/internal); OTel instead of Prometheus |
| 9 | Telemetry | OTel traces + metrics via the API; OTLP exporter via standard env vars; no-op when unconfigured |
| 10 | Tests & evals | (1) offline unit tests; (2) env-gated integration tests (skip without `OPENROUTER_API_KEY`); (3) `jev-router eval` CLI: 8 labeled prompts, JSON output, non-zero exit on threshold breach |
| 11 | Config | YAML file (`--config`) + env overrides; catalogs as file paths |

Assumed: Go 1.25+; default catalog cache TTL 600s, `on_classifier_error
fail_open`.

## 3. Package layout

```
jev-router/
├── go.mod                          # module github.com/vprprudhvi/jev-router
├── router.go, types.go, errors.go, classifier.go, completion.go,
│   catalog.go, diagnostics.go      # package modelrouter: Router + core types
├── config/                         # YAML schema, Load/Validate, Build -> App
├── catalog/                        # Catalog interface: live OpenRouter catalog, static user catalogs
├── classify/                       # Classifier interface + Jev systemone client
├── provider/                       # registry/factory + provider implementations
│   ├── registry.go                 # Register/New; database/sql style
│   ├── openrouter/, openai/, anthropic/, gemini/, bedrock/
│   └── builtin/                    # wires the built-ins into the registry
├── server/                         # HTTP API
├── internal/cli/                   # the jev-router CLI
├── internal/telemetry/             # OTel SDK + OTLP setup
├── internal/testutil/              # shared test fakes/fixtures
├── eval/                           # labeled prompts + metrics + JSON report
├── cmd/jev-router/                 # main
├── docs/
├── .github/workflows/ci.yml
├── Makefile
└── README.md
```

Import rule: `cmd/`, `eval/` may import everything; `provider/*` imports the
root package's `Catalog`/`Classifier`/`Completion` interfaces; the root
package never imports a provider implementation.

## 4. Provider layer

```go
type Catalog interface { Models(ctx context.Context) ([]modelrouter.ModelInfo, error) }
type Completion interface {
    Complete(ctx context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error)
}
```

- **Factory/registry**: `provider.Register(name, factory)` +
  `provider.New(name, cfg)` (database/sql style), so third parties can add
  providers; built-ins self-register via `provider/builtin`.
- **Model identity**: `ModelInfo{Provider, ID (native), DisplayID}`.
  Completions always execute on the winning candidate's own provider — no
  cross-provider ID translation.
- **Unit-testability**: every SDK client is constructed behind a small
  constructor hook; tests point SDKs at `httptest.Server` via base-URL
  overrides (Bedrock uses an interface seam instead — see design-notes.md).
- **Retries**: SDK defaults only; no custom retry layer.
- **Close semantics**: anything a component constructed itself, it closes;
  injected clients are never closed.

## 5. Implementation phases

| Phase | Content | Commit |
|-------|---------|--------|
| P0 | Scaffold: `go.mod`, LICENSE, README stub, Makefile, CI | `23a044c` |
| P1 | Core types + error model + exports | `cbcbbf4` |
| P2 | Scorer (rank percentile, per-provider merge + round-robin) — **later replaced by mapped routing (v2)** | `2b776fa` |
| P3 | Catalogs: OpenRouter live client + static user catalogs | `2ab3345` |
| P4 | Jev classifier via OpenRouter systemone | `ebca483` |
| P5 | Router orchestration, completion interfaces, diagnostics | `d00bfc9` |
| P6a | Provider registry + OpenRouter/OpenAI completions | `1adabf8` |
| P6b | Anthropic, Gemini, Bedrock completions + builtin registration | `79523b7` |
| P7a | YAML config schema (`Load`/`Validate`) + example configs | `7076175` |
| P7b | Config assembly (`Build`, `App` lifecycle, compatible providers) | `2149614` |
| P8a | HTTP server + `Router.ListModels` (originally shipped with SSE streaming — **later removed**) | `3c532e5` |
| P8b | OpenTelemetry traces + metrics (OTLP, no-op default) | `42e89ae` |
| P9 | CLI: route/complete/models/serve/validate/version | `82ef302` |
| P10 | Eval harness (`eval` package + `jev-router eval`) | `a1a575b` |
| P11 | Env-gated live integration tests | `496e562` |
| P12 | Documentation | — |

Testing strategy: every offline unit test runs with `go test ./...` (no
keys, no network); env-gated integration tests run with
`go test -run Integration ./...` behind `OPENROUTER_API_KEY`; the eval
harness runs manually (or in CI with secrets) since it makes real, billed
calls. Also: `go vet ./...`, `go test -race ./...`, table-driven style, no
third-party test framework (stdlib `testing` + `internal/testutil` helpers).
