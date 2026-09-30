package modelrouter

import (
	"context"
	"encoding/json"
)

// Classifier turns a request into a RequestProfile.
//
// Implementations must map every failure (transport, timeout, non-2xx status,
// malformed response) to a *ClassifierUnavailableError, so callers decide
// between fail-open and fail-closed routing with a single errors.Is check.
type Classifier interface {
	Classify(ctx context.Context, prompt string) (Classification, error)
}

// Classification is the outcome of one classifier call.
type Classification struct {
	Profile RequestProfile
	Raw     json.RawMessage // raw classifier response, for debugging
	Usage   *Usage          // optional; set when the endpoint reports usage
}
