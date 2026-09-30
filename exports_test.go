package modelrouter_test

import (
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// TestTopLevelPackageExportsThePublicAPI pins the public API surface:
// referencing every exported symbol here fails compilation if anything
// stops being exported.
func TestTopLevelPackageExportsThePublicAPI(t *testing.T) {
	var (
		_ = modelrouter.DomainCode
		_ = modelrouter.ModelInfo{}
		_ = modelrouter.RoutingConstraints{}
		_ = modelrouter.RequestProfile{}
		_ = modelrouter.ModelChoice{}
		_ = modelrouter.RoutingDecision{}
		_ = modelrouter.CompletionResult{}
		_ = modelrouter.Usage{}

		_ = &modelrouter.CatalogUnavailableError{}
		_ = &modelrouter.ClassifierUnavailableError{}
		_ = &modelrouter.NoCandidateModelError{}
		_ = &modelrouter.ProviderError{}
	)

	var (
		_ = modelrouter.ErrCatalogUnavailable
		_ = modelrouter.ErrClassifierUnavailable
		_ = modelrouter.ErrNoCandidate
		_ = modelrouter.ErrProvider
	)

	// v2 mapped-routing surface: domain validation and the decision source
	// values are part of the pinned API (the retired scorer's symbols are
	// gone).
	var (
		_ = modelrouter.IsValidDomain
		_ = modelrouter.SourceClassification
		_ = modelrouter.SourceDefault
	)
}
