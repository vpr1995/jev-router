// Tests for Router.ListModels: the catalog-stage-only listing API. It shares
// Route's fetch/join stage, so the failure and ordering rules asserted here
// also pin behavior Route relies on.
package modelrouter_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	modelrouter "github.com/vprprudhvi/jev-router"
	"github.com/vprprudhvi/jev-router/internal/testutil"
)

// listModelIDs formats model IDs for compact order-sensitive assertions.
func listModelIDs(models []modelrouter.ModelInfo) []string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

// listRouterOptions builds RouterOptions for listing-only tests: ListModels
// never classifies or selects, but New still requires a routing config, so
// the CHEAP model becomes the default.
func listRouterOptions(catalogs ...modelrouter.Catalog) modelrouter.RouterOptions {
	return modelrouter.RouterOptions{Catalogs: catalogs, DefaultModel: defaultChoicePtr()}
}

func TestListModelsConcatenatesCatalogsInOrder(t *testing.T) {
	other := modelrouter.ModelInfo{
		Provider:                   "other",
		ID:                         "other/only",
		PricePer1KPromptTokens:     0.5,
		PricePer1KCompletionTokens: 0.5,
		ContextLength:              4_000,
	}
	first := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	second := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{other}}
	router := mustNewRouter(t, listRouterOptions(first, second))

	models, err := router.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListModels() error = %v", err)
	}

	got := listModelIDs(models)
	want := []string{testutil.CHEAP.ID, testutil.EXPENSIVE.ID, other.ID}
	if len(got) != len(want) {
		t.Fatalf("ListModels() returned %d models (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("ListModels()[%d].ID = %q, want %q (order must follow catalog order)", i, got[i], want[i])
		}
	}
	if first.Calls() != 1 || second.Calls() != 1 {
		t.Errorf("catalog calls = (%d, %d), want (1, 1)", first.Calls(), second.Calls())
	}
}

func TestListModelsFiltersAllowedAndExcludedProviders(t *testing.T) {
	alphaOne := modelrouter.ModelInfo{Provider: "alpha", ID: "alpha/one", ContextLength: 1_000}
	betaOne := modelrouter.ModelInfo{Provider: "beta", ID: "beta/one", ContextLength: 1_000}
	alphaTwo := modelrouter.ModelInfo{Provider: "alpha", ID: "alpha/two", ContextLength: 1_000}
	betaTwo := modelrouter.ModelInfo{Provider: "beta", ID: "beta/two", ContextLength: 1_000}

	first := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{alphaOne, betaOne}}
	second := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{alphaTwo, betaTwo}}
	router := mustNewRouter(t, listRouterOptions(first, second))

	cases := []struct {
		name        string
		constraints modelrouter.RoutingConstraints
		want        []string
	}{
		{
			name:        "allowed providers keep only those providers",
			constraints: modelrouter.RoutingConstraints{AllowedProviders: []string{"alpha"}},
			want:        []string{alphaOne.ID, alphaTwo.ID},
		},
		{
			name:        "excluded providers remove only those providers",
			constraints: modelrouter.RoutingConstraints{ExcludedProviders: []string{"alpha"}},
			want:        []string{betaOne.ID, betaTwo.ID},
		},
		{
			name: "exclusion wins over inclusion",
			constraints: modelrouter.RoutingConstraints{
				AllowedProviders:  []string{"alpha"},
				ExcludedProviders: []string{"alpha"},
			},
			want: []string{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			models, err := router.ListModels(context.Background(), &tc.constraints)
			if err != nil {
				t.Fatalf("ListModels() error = %v", err)
			}
			got := listModelIDs(models)
			if fmt.Sprint(got) != fmt.Sprint(tc.want) {
				t.Errorf("ListModels() IDs = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestListModelsJoinsEveryCatalogError(t *testing.T) {
	errFirst := errors.New("first catalog down")
	errSecond := errors.New("second catalog down")
	first := &testutil.FakeCatalog{Err: errFirst}
	second := &testutil.FakeCatalog{Err: errSecond}
	classifier := &testutil.FakeClassifier{}
	opts := listRouterOptions(first, second)
	opts.Classifier = classifier
	router := mustNewRouter(t, opts)

	models, err := router.ListModels(context.Background(), nil)
	if err == nil {
		t.Fatal("ListModels() error = nil, want joined catalog failures")
	}
	if models != nil {
		t.Errorf("ListModels() models = %v, want nil alongside the error", models)
	}
	if !errors.Is(err, errFirst) || !errors.Is(err, errSecond) {
		t.Errorf("errors.Is did not match both catalog failures; err = %v", err)
	}

	var joined interface{ Unwrap() []error }
	if !errors.As(err, &joined) {
		t.Fatalf("ListModels() error = %v (%T), want a joining error", err, err)
	}
	failures := joined.Unwrap()
	if len(failures) != 2 {
		t.Fatalf("joined failures = %d, want 2 (%v)", len(failures), failures)
	}
	if !errors.Is(failures[0], errFirst) || !errors.Is(failures[1], errSecond) {
		t.Errorf("joined failures = %v, want [%v %v] in catalog order", failures, errFirst, errSecond)
	}
	if classifier.Calls() != 0 {
		t.Errorf("classifier calls = %d, want 0 (listing never classifies)", classifier.Calls())
	}
}

func TestListModelsReturnsEmptySliceWithoutError(t *testing.T) {
	cheapOnly := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP}}
	router := mustNewRouter(t, listRouterOptions(cheapOnly))

	models, err := router.ListModels(context.Background(), &modelrouter.RoutingConstraints{NeedsVision: true})
	if err != nil {
		t.Fatalf("ListModels() error = %v, want nil for a filtered-to-empty listing", err)
	}
	if models == nil {
		t.Error("ListModels() = nil, want a non-nil empty slice")
	}
	if len(models) != 0 {
		t.Errorf("ListModels() = %v, want no models (vision constraint drops the only model)", listModelIDs(models))
	}

	empty := &testutil.FakeCatalog{}
	emptyRouter := mustNewRouter(t, listRouterOptions(empty))
	models, err = emptyRouter.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListModels() over empty catalog error = %v, want nil", err)
	}
	if models == nil || len(models) != 0 {
		t.Errorf("ListModels() over empty catalog = %v, want a non-nil empty slice", models)
	}
}

func TestListModelsNilConstraintsUsesDefaults(t *testing.T) {
	catalog := &testutil.FakeCatalog{ModelsList: []modelrouter.ModelInfo{testutil.CHEAP, testutil.EXPENSIVE}}
	router := mustNewRouter(t, listRouterOptions(catalog))

	nilModels, err := router.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListModels(nil) error = %v", err)
	}
	zeroModels, err := router.ListModels(context.Background(), &modelrouter.RoutingConstraints{})
	if err != nil {
		t.Fatalf("ListModels(zero constraints) error = %v", err)
	}
	want := []string{testutil.CHEAP.ID, testutil.EXPENSIVE.ID}
	if got := fmt.Sprint(listModelIDs(nilModels)); got != fmt.Sprint(want) {
		t.Errorf("ListModels(nil) IDs = %v, want %v (nil must mean the zero constraints)", got, want)
	}
	if got := fmt.Sprint(listModelIDs(zeroModels)); got != fmt.Sprint(want) {
		t.Errorf("ListModels(zero constraints) IDs = %v, want %v", got, want)
	}

	filtered, err := router.ListModels(context.Background(), &modelrouter.RoutingConstraints{
		MaxPricePer1KTokens: float64Ptr(0.5),
	})
	if err != nil {
		t.Fatalf("ListModels(max price) error = %v", err)
	}
	if got := listModelIDs(filtered); len(got) != 1 || got[0] != testutil.CHEAP.ID {
		t.Errorf("ListModels(max price 0.5) IDs = %v, want [%s]", got, testutil.CHEAP.ID)
	}
}
