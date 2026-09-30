GO ?= go

.PHONY: build test test-race cover vet lint eval serve smoke clean

build:
	$(GO) build ./...

test:
	$(GO) test ./...

test-race:
	$(GO) test -race ./...

cover:
	$(GO) test -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out | tail -1

vet:
	$(GO) vet ./...

lint:
	golangci-lint run

# Billed live run: needs OPENROUTER_API_KEY.
eval:
	$(GO) run ./cmd/jev-router eval --min-accuracy 0.75

# Local server: defaults to the git-ignored jev-router.local.yaml (see the
# README quickstart for environment-only mode). Override with
# `make serve CONFIG=path/to/config.yaml`.
CONFIG ?= jev-router.local.yaml

serve:
	$(GO) run ./cmd/jev-router serve --config $(CONFIG)

# HTTP smoke test: starts serve and exercises every endpoint (one small billed
# completion unless --no-complete). Needs a usable config or env-only defaults.
smoke:
	./scripts/api-smoke.sh

clean:
	rm -rf bin dist coverage.out coverage.html
