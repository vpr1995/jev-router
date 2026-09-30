// Package static provides user-supplied model catalogs loaded from JSON or
// YAML files (see docs/catalog-format.md). Unlike the tolerant OpenRouter
// parser, validation here fails fast, naming the offending model index, id
// and field — a hand-authored catalog with a typo should be fixed, not
// silently skipped.
package static

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// catalogFile is the on-disk schema shared by JSON and YAML catalogs.
type catalogFile struct {
	Provider string         `json:"provider" yaml:"provider"`
	Models   []catalogModel `json:"models" yaml:"models"`
}

// catalogModel is one entry of the on-disk schema.
type catalogModel struct {
	ID                         string  `json:"id" yaml:"id"`
	DisplayID                  string  `json:"display_id" yaml:"display_id"`
	ContextLength              int     `json:"context_length" yaml:"context_length"`
	SupportsVision             bool    `json:"supports_vision" yaml:"supports_vision"`
	PricePer1KPromptTokens     float64 `json:"price_per_1k_prompt_tokens" yaml:"price_per_1k_prompt_tokens"`
	PricePer1KCompletionTokens float64 `json:"price_per_1k_completion_tokens" yaml:"price_per_1k_completion_tokens"`
}

// Catalog is a validated, immutable in-memory catalog loaded from a user
// file. It implements modelrouter.Catalog.
type Catalog struct {
	provider string
	models   []modelrouter.ModelInfo
}

var _ modelrouter.Catalog = (*Catalog)(nil)

// Load reads and validates a catalog file.
//
// The extension selects the parser: ".json" is decoded with encoding/json,
// and everything else — including ".yaml"/".yml" and unknown extensions, as a
// fallback — is decoded as YAML (JSON is a YAML subset, so the YAML parser
// accepts either format too). Failures name the file and, for validation
// problems, the offending model, e.g.:
//
//	catalog file a.yaml: model[2] ("x"): negative price_per_1k_prompt_tokens
func Load(path string) (*Catalog, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading catalog file %s: %w", path, err)
	}

	var doc catalogFile
	if strings.EqualFold(filepath.Ext(path), ".json") {
		if err := json.Unmarshal(data, &doc); err != nil {
			return nil, fmt.Errorf("catalog file %s: invalid JSON: %w", path, err)
		}
	} else if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("catalog file %s: invalid YAML: %w", path, err)
	}

	catalog, err := validate(doc)
	if err != nil {
		return nil, fmt.Errorf("catalog file %s: %w", path, err)
	}
	return catalog, nil
}

// Parse validates an in-memory catalog document, accepting both JSON and YAML
// (the YAML parser is a superset of JSON).
func Parse(data []byte) (*Catalog, error) {
	var doc catalogFile
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("catalog: invalid YAML: %w", err)
	}
	catalog, err := validate(doc)
	if err != nil {
		return nil, fmt.Errorf("catalog: %w", err)
	}
	return catalog, nil
}

// Models returns a defensive copy of the catalog's models, each stamped with
// the catalog's provider. The context exists only to satisfy
// modelrouter.Catalog; an in-memory catalog has nothing to cancel.
func (c *Catalog) Models(_ context.Context) ([]modelrouter.ModelInfo, error) {
	models := make([]modelrouter.ModelInfo, len(c.models))
	copy(models, c.models)
	return models, nil
}

// Provider returns the provider name declared by the catalog file.
func (c *Catalog) Provider() string { return c.provider }

// validate applies the catalog schema checks, naming the offending model
// index, id and field on failure.
func validate(doc catalogFile) (*Catalog, error) {
	if doc.Provider == "" {
		return nil, errors.New("provider is required and must be non-empty")
	}
	if len(doc.Models) == 0 {
		return nil, errors.New("models must contain at least one model")
	}

	firstIndex := make(map[string]int, len(doc.Models))
	models := make([]modelrouter.ModelInfo, 0, len(doc.Models))
	for i, entry := range doc.Models {
		if entry.ID == "" {
			return nil, fmt.Errorf("model[%d] (\"\"): id is required and must be non-empty", i)
		}
		if first, ok := firstIndex[entry.ID]; ok {
			return nil, fmt.Errorf("model[%d] (%q): duplicate id (first defined at model[%d])", i, entry.ID, first)
		}
		firstIndex[entry.ID] = i
		if entry.ContextLength < 0 {
			return nil, fmt.Errorf("model[%d] (%q): negative context_length", i, entry.ID)
		}
		if entry.PricePer1KPromptTokens < 0 {
			return nil, fmt.Errorf("model[%d] (%q): negative price_per_1k_prompt_tokens", i, entry.ID)
		}
		if entry.PricePer1KCompletionTokens < 0 {
			return nil, fmt.Errorf("model[%d] (%q): negative price_per_1k_completion_tokens", i, entry.ID)
		}

		models = append(models, modelrouter.ModelInfo{
			Provider:                   doc.Provider,
			ID:                         entry.ID,
			DisplayID:                  entry.DisplayID,
			ContextLength:              entry.ContextLength,
			SupportsVision:             entry.SupportsVision,
			PricePer1KPromptTokens:     entry.PricePer1KPromptTokens,
			PricePer1KCompletionTokens: entry.PricePer1KCompletionTokens,
		})
	}
	return &Catalog{provider: doc.Provider, models: models}, nil
}
