# AGENTS.md

Reference for coding agents working in this repository. For product/user docs, see [README.md](README.md).

## What this is

`jev-router` is a Go multi-provider LLM router. Pipeline: one Jev classification call (via OpenRouter) → select a configured, ordered chain of provider/model candidates for the classified domain → filter by hard constraints → `Complete` tries the chain in order, falling back to `routing.default_model` on provider failure. See [docs/design-notes.md](docs/design-notes.md) and [docs/plans/](docs/plans/) for design rationale and history.

Go 1.26+. Module: `github.com/vprprudhvi/jev-router`.

## Build, test, lint

```bash
go build ./...
go vet ./...
go test ./...                       # offline, no keys, no network
go test -race -count=1 ./...
golangci-lint run                   # config: .golangci.yml — errcheck, govet, ineffassign, staticcheck, unused, misspell
make build|test|test-race|cover|vet|lint
```

Env-gated integration tests (skipped without keys, never run in CI):

```bash
OPENROUTER_API_KEY=... go test -run Integration ./catalog/openrouter ./config
```

HTTP smoke test (builds the binary, starts `serve`, curls every endpoint): `./scripts/api-smoke.sh` or `make smoke`.

Always run `go build ./...`, `go vet ./...` and `go test ./...` after any change, before considering it done. `gofmt -l .` must report nothing.

## Layout

| Package | Contents |
|---|---|
| `modelrouter` (root: `router.go`, `types.go`, `errors.go`, `classifier.go`, `completion.go`, `catalog.go`, `diagnostics.go`) | Public types, error model, `FilterModels`, mapped-chain selection, diagnostics, `Router` orchestration (`Route`/`Complete`/`ListModels`) |
| `config` | YAML schema, `Load`/`Validate`, env overrides, `Build` → `App` (assembled router + owned closers) |
| `catalog/openrouter` | Live OpenRouter catalog: TTL cache, stale fallback, tolerant per-entry parsing |
| `catalog/static` | User JSON/YAML catalogs, fail-fast validation |
| `classify` | Jev client (`POST /api/v1/systemone`, 5s timeout, five fixed questions) |
| `provider` | Completion factory registry (`database/sql`-style: `Register`/`New`), `Config`, `WrapError` |
| `provider/builtin` | Wires the built-in factories (openrouter, openai, anthropic, gemini, bedrock) into the registry |
| `provider/{anthropic,openai,gemini,bedrock,openrouter}` | One `modelrouter.Completion` implementation per SDK |
| `server` | HTTP API, request IDs, recovery, logging, graceful shutdown |
| `internal/cli` | The `jev-router` CLI (commands, flags, exit codes) |
| `internal/telemetry` | OTel SDK + OTLP setup — the only package importing the SDK |
| `internal/testutil` | Shared fakes/fixtures for Router tests |
| `eval` | Labeled prompts + metrics/report |
| `cmd/jev-router` | `main` — a thin wrapper over `internal/cli` |

## Invariants agents must not break

These are load-bearing design decisions, not incidental style. Changing behavior here needs a deliberate decision, not a drive-by refactor.

- **The root package (`modelrouter`) imports no provider, no `classify/`, no OTel SDK.** It depends on the OTel *API* only (tracer/meter providers default to the otel globals, which are no-ops unless `internal/telemetry` installed SDK providers) and on the `Catalog`/`Classifier`/`Completion` interfaces. Provider and classifier implementations live in their own packages and are wired in by `config` / `provider/builtin`.
- **The `Router` constructs nothing and closes nothing.** Catalogs, the classifier and completions are injected via `RouterOptions` and owned by the caller (`config.App.Close`). Do not add a `Router.Close`.
- **Providers never set `CompletionResult.RoutingDecision`.** The `Router` always overwrites it after a successful attempt.
- **Classification failure never fabricates a `RequestProfile`.** `fail_open` routes on `DefaultModel` with a nil `Profile`; `fail_closed` returns `*ClassifierUnavailableError` unchanged. Don't invent a placeholder profile.
- **Catalogs are fetched only when a constraint needs their metadata** (`needs_vision`, `min_context_tokens`, `max_price_per_1k_tokens`). Provider allow/exclude filtering and chain selection never need a catalog fetch. When catalogs are fetched, a failing one is never silently skipped: `Route`/`ListModels` return every catalog error joined (`errors.Join`), not a partial pool.
- **Every provider error is `*modelrouter.ProviderError`**, built through `provider.WrapError(name, op, statusCode, err)` — don't hand-construct the struct literal in a new provider; add to the shared helper in `provider/registry.go` instead.
- **Sentinel errors + typed errors, always with `Is`/`Unwrap`.** New failure modes should follow the `errors.go` pattern (a sentinel `Err*` value plus a `*XError` type implementing `Is`/`Unwrap`), so callers can keep using a single `errors.Is` check.
- **Config decodes strictly.** Unknown YAML fields are errors (`decoder.KnownFields(true)`); don't silently ignore new fields.
- **Catalog paths in a config file resolve against that file's directory** (`config.File.ResolvePath`), not the process cwd.
- **HTTP clients/loggers passed through `provider.Config` are caller-owned.** A provider never closes an injected `HTTPClient`; when nil, it constructs its own with no timeout (completions are context-first — bound calls with `ctx`, not a client timeout).
- **The HTTP server has no authentication.** It's meant for localhost/internal use behind a reverse proxy. Don't add auth without being asked; do keep every response (including 404/405) as JSON.
- **Completions are synchronous only.** There is no streaming/SSE. On the native `POST /v1/complete`, a legacy `"stream": true` field is ignored via normal "unknown JSON fields are ignored" behavior. On the OpenAI-compatible `POST /v1/chat/completions`, `"stream": true` is a deliberate 400 (`streaming_not_supported`), because an SDK expecting SSE cannot parse a JSON body.
- **OpenAI-compat paths use OpenAI's error envelope; native paths keep `{"error","code"}`.** `server.Server.fail` picks by path (`isCompatPath`). OpenAI types live only in `server/openai.go` — never in the root package. The compat surface is text-only, single-choice, tool-free; unsupported inputs are a 400, not silently dropped. The `model` field is accepted but ignored (the router always routes).
- **CLI exit codes are part of the contract:** 0 success, 1 runtime/API failure, 2 usage error, 3 eval accuracy below `--min-accuracy`. `cli.Run` never calls `os.Exit` (so it stays testable in-process); `cmd/jev-router/main.go` does that.
- **Telemetry is a true no-op when disabled.** `internal/telemetry` is the only package touching the OTel SDK/exporters/globals; span and metric names (`modelrouter.route`, `modelrouter.decisions`, ...) are a stable observable contract — don't rename them casually.

## Testing conventions

- Unit tests are fully offline (no keys, no network) and must stay that way; anything needing real credentials goes behind `go test -run Integration` in `catalog/openrouter` or `config`, gated on `OPENROUTER_API_KEY` being set.
- HTTP-facing behavior is tested against `httptest.Server`; SDK-backed providers are tested against fakes/`httptest` (bedrock uses a white-box `converseAPI` test seam — see `newWithAPI` in `provider/bedrock/bedrock.go`).
- `internal/testutil` holds the shared Router test doubles (`FakeCatalog`, `FakeClassifier`, `FakeCompletion`) and fixture models/profiles (`CHEAP`, `EXPENSIVE`, `CONFIDENT_TRIVIAL_PROFILE`) — reuse these instead of redefining fakes in a new test file.
- `docs/design-notes.md` §1 maps documented behavior to the tests that pin it; when touching that behavior, check whether a row there needs updating.

## Comment style in this codebase

Prefer short doc comments that state the current contract and the non-obvious "why". Avoid:
- narrating change history ("this used to work differently", "deliberate deviation from an earlier design") — that belongs in a commit message or `docs/design-notes.md`, not in code;
- restating what the code already says;
- referencing removed functionality (e.g. the retired price scorer) as if it still exists.

## Where to look for more

- [README.md](README.md) — product-facing usage (quickstart, configuration reference, HTTP API, CLI).
- [docs/catalog-format.md](docs/catalog-format.md) — user catalog file schema.
- [docs/design-notes.md](docs/design-notes.md) — behavior-to-test mapping and design rationale.
- [docs/plans/](docs/plans/) — architecture history.
