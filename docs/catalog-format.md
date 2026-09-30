# User catalog format

OpenRouter
is the only provider served by a live catalog (`GET /api/v1/models`, with a TTL
cache). Every other provider — `openai`, `anthropic`, `gemini`, `bedrock` and
any `providers.compatible[]` endpoint — reads its model list, context windows,
vision support and prices from a **user-supplied catalog file** referenced by
its `catalog:` config field:

```yaml
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    catalog: catalogs/openai.yaml   # path resolved against the config file's directory
```

The loader lives in [`catalog/static`](../catalog/static/static.go); the
schema below is complete, with example values you can adapt.

## Schema

YAML (preferred) or JSON; both forms are accepted:

```yaml
provider: openai            # required; see "Provider name rule" below
models:                     # required; at least one entry
  - id: gpt-5.2                              # required; provider-native model id
    display_id: openai/gpt-5.2               # optional; defaults to the id for display
    context_length: 400000                   # tokens, >= 0
    supports_vision: true
    price_per_1k_prompt_tokens: 0.00125      # USD per 1,000 tokens, >= 0
    price_per_1k_completion_tokens: 0.01     # USD per 1,000 tokens, >= 0
```

```json
{
  "provider": "groq",
  "models": [
    {
      "id": "llama-3.1-8b-instant",
      "display_id": "groq/llama-3.1-8b-instant",
      "context_length": 131072,
      "supports_vision": false,
      "price_per_1k_prompt_tokens": 0.00005,
      "price_per_1k_completion_tokens": 0.00008
    }
  ]
}
```

| Field | Required | Type | Rules | Meaning |
|---|---|---|---|---|
| `provider` | yes | string | non-empty | Provider name stamped on every model (see the rule below). |
| `models` | yes | list | at least one entry | The provider's routable models. |
| `models[].id` | yes | string | non-empty; unique within the file | Provider-native model id — exactly what completion requests send to the provider API. |
| `models[].display_id` | no | string | — | Human-readable label (e.g. `openai/gpt-5.2`); when empty the `id` is used (`ModelInfo.Label`). Never sent to the provider. |
| `models[].context_length` | no (0) | integer | `>= 0` | Maximum context window in tokens; drives `min_context_tokens` filtering. |
| `models[].supports_vision` | no (`false`) | boolean | — | Whether the model accepts image input; drives `needs_vision` filtering. |
| `models[].price_per_1k_prompt_tokens` | no (0) | number | `>= 0` | USD per 1,000 prompt tokens. |
| `models[].price_per_1k_completion_tokens` | no (0) | number | `>= 0` | USD per 1,000 completion tokens. |

Notes:

- **Prices are USD per 1,000 tokens** (the same unit as OpenRouter's catalog
  after normalization). Filtering uses the *blended* price =
  `price_per_1k_prompt_tokens + price_per_1k_completion_tokens`; there is no
  separate "blended" field.
- Prices of `0` are valid (e.g. self-hosted endpoints) and make the model the
  cheapest candidate; **negative prices are rejected** — OpenRouter's `-1`
  "variable pricing" sentinel has no meaning in a user catalog.
- Unknown fields are ignored by the YAML/JSON decoders, so extra annotations
  are harmless (the OpenRouter live parser is the strictness-sensitive one).
- The file is parsed **eagerly** when the configuration is assembled
  (`config.Build`) and the models are kept in memory; edits require a restart.

## Validation rules (fail-fast)

Unlike the tolerant OpenRouter parser — which skips malformed *entries* because
it cannot control the upstream data — a hand-authored catalog fails **fast**,
naming the file, the offending model index, its id and the field:

| Problem | Example error |
|---|---|
| empty `provider` | `catalog file a.yaml: provider is required and must be non-empty` |
| empty `models` | `catalog file a.yaml: models must contain at least one model` |
| empty id | `catalog file a.yaml: model[0] (""): id is required and must be non-empty` |
| duplicate id | `catalog file a.yaml: model[1] ("x"): duplicate id (first defined at model[0])` |
| negative values | `catalog file a.yaml: model[2] ("x"): negative price_per_1k_prompt_tokens` (also `negative context_length`, `negative price_per_1k_completion_tokens`) |

`jev-router validate --config …` loads and assembles the configuration, so it
parses every catalog and reports these errors before any traffic is served.

## Parsing and path resolution

- File extension selects the parser: `.json` is decoded with `encoding/json`;
  everything else (`.yaml`, `.yml`, or unknown) is decoded as YAML. YAML is a
  JSON superset, so either parser accepts well-formed JSON.
- Catalog paths written in the config file resolve **relative to that file's
  directory** (`config.File.ResolvePath`): absolute paths are used unchanged,
  and in environment-only mode (no config file) relative paths are
  cwd-relative. Moving a config file together with its `catalogs/` directory
  keeps every reference valid.

## Provider name rule

The `provider` value must have a **matching completion factory**, because
completion dispatch looks providers up by this name:

- built-in factories (registered by `provider/builtin`): `openrouter`,
  `openai`, `anthropic`, `gemini`, `bedrock`; or
- an exact `providers.compatible[].name` from your config (any unique name —
  `groq`, `ollama`, `vllm`, … — bound to a `base_url` and an OpenAI-wire
  completion).

Concretely: the loader stamps every model with the catalog's `provider` string,
and `Router.Complete` resolves the winner's provider against the
configured completions map. A catalog whose `provider` disagrees with the name
it is configured under (e.g. a file declaring `provider: foo` attached to
`providers.openai`) will parse and route successfully, but any completion for
one of its models fails with **`provider not configured`**
(`modelrouter.ErrProviderNotConfigured`) — keep the two names identical.

## See also

- [README Configuration](../README.md#configuration) — a complete config
  wiring each provider to its catalog.
- [`docs/design-notes.md`](design-notes.md) — how filtering consumes these prices.
