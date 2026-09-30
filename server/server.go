// Package server exposes a modelrouter.Router over HTTP: POST /v1/route,
// POST /v1/complete (plain JSON), the OpenAI-compatible POST
// /v1/chat/completions and GET /v1/models, and GET /healthz. Every completion
// is a single synchronous JSON response; there is no streaming.
//
// The server is transport-only: routing, classification, catalog access and
// completions stay behind the injected Router interface, and this package
// owns HTTP concerns only — request decoding/validation, status/code
// mapping, request IDs, panic recovery and slog request logging, assembled
// as a middleware chain by buildHandler. Options.Middleware entries fold
// outside that built-in chain (around panic recovery too), so instrumentation
// layers observe the final status of every request.
//
// There is no authentication: the server is intended for localhost/internal
// use behind a reverse proxy.
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"runtime/debug"
	"strings"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
)

const (
	// defaultAddr is the listen address used when Options.Addr is empty.
	defaultAddr = ":8080"
	// defaultReadTimeout bounds reading a full request when
	// Options.ReadTimeout is zero.
	defaultReadTimeout = 30 * time.Second
	// defaultMaxBodyBytes caps a request body when Options.MaxBodyBytes is zero.
	defaultMaxBodyBytes = 4 << 20
	// shutdownTimeout is how long ServeListener waits for in-flight requests
	// after its context is cancelled.
	shutdownTimeout = 10 * time.Second

	// requestIDHeader carries the request ID in both directions.
	requestIDHeader = "X-Request-ID"
	// contentTypeJSON is the Content-Type of every response.
	contentTypeJSON = "application/json"
)

// Stable error codes for the {"error","code"} bodies. not_found and
// method_not_allowed cover the transport-level 404/405 responses, which are
// JSON like every other response.
const (
	codeInvalidRequest        = "invalid_request"
	codeNoCandidate           = "no_candidate"
	codeCatalogUnavailable    = "catalog_unavailable"
	codeClassifierUnavailable = "classifier_unavailable"
	codeProviderError         = "provider_error"
	codeInternal              = "internal"
	codeNotFound              = "not_found"
	codeMethodNotAllowed      = "method_not_allowed"
)

// Endpoint paths. They are matched exactly (no trailing-slash forms).
const (
	pathRoute    = "/v1/route"
	pathComplete = "/v1/complete"
	pathHealthz  = "/healthz"
)

// Router is the subset of *modelrouter.Router the HTTP API needs. It is an
// interface so handlers can be tested with fakes and so Router-like
// implementations (wrappers, decorators) stay usable.
type Router interface {
	Route(ctx context.Context, prompt string, constraints *modelrouter.RoutingConstraints) (modelrouter.RoutingDecision, error)
	Complete(ctx context.Context, req modelrouter.CompletionRequest) (modelrouter.CompletionResult, error)
}

// Options configures New.
type Options struct {
	// Router serves every endpoint. Required.
	Router Router
	// Logger receives request logs and panic reports. Nil means
	// slog.Default().
	Logger *slog.Logger
	// Addr is the TCP listen address used by Serve. Empty means ":8080".
	Addr string
	// ReadTimeout bounds reading a full request (headers and body). Zero
	// means 30s.
	ReadTimeout time.Duration
	// WriteTimeout bounds writing a full response. Zero means no timeout.
	WriteTimeout time.Duration
	// MaxBodyBytes caps the size of a request body; larger bodies get a 413.
	// Zero means 4 MiB.
	MaxBodyBytes int64
	// Middleware are extra wrapping layers applied outermost — around the
	// whole built-in chain (panic recovery, request ID, logging), with
	// Middleware[0] as the single outermost wrapper. This is the hook for
	// instrumentation such as otelhttp: because recovery runs inside it, it
	// observes the final status code, including recovered-panic 500s.
	// Entries must be non-nil and should stay panic-free themselves, since
	// the built-in recovery only wraps what is inside it.
	Middleware []func(http.Handler) http.Handler
}

// Server serves the HTTP API over an injected Router. Create it with New;
// the zero value is not usable.
type Server struct {
	router          Router
	logger          *slog.Logger
	addr            string
	readTimeout     time.Duration
	writeTimeout    time.Duration
	maxBodyBytes    int64
	started         time.Time
	extraMiddleware []middleware
	handler         http.Handler
}

// New validates opts and builds a Server.
//
// Router is required, everything else has defaults: Logger slog.Default(),
// Addr ":8080", ReadTimeout 30s, WriteTimeout 0 (no timeout — see
// Options.WriteTimeout).
func New(opts Options) (*Server, error) {
	if opts.Router == nil {
		return nil, errors.New("server: Router is required")
	}

	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}
	addr := opts.Addr
	if addr == "" {
		addr = defaultAddr
	}
	readTimeout := opts.ReadTimeout
	if readTimeout == 0 {
		readTimeout = defaultReadTimeout
	}

	maxBodyBytes := opts.MaxBodyBytes
	if maxBodyBytes <= 0 {
		maxBodyBytes = defaultMaxBodyBytes
	}

	// Copy the middleware list so later caller mutations cannot affect the
	// Server (same defensive rule as the Router's catalogs).
	extraMiddleware := make([]middleware, len(opts.Middleware))
	for i, wrap := range opts.Middleware {
		extraMiddleware[i] = wrap
	}

	s := &Server{
		router:          opts.Router,
		logger:          logger,
		addr:            addr,
		readTimeout:     readTimeout,
		writeTimeout:    opts.WriteTimeout,
		maxBodyBytes:    maxBodyBytes,
		started:         time.Now(),
		extraMiddleware: extraMiddleware,
	}
	s.handler = s.buildHandler()
	return s, nil
}

// Handler returns the fully wrapped handler: the endpoints, the built-in
// recovery, request-ID and logging middleware, and any Options.Middleware
// entries outside them. It may be served directly, mounted under a prefix by
// an outer mux, or wrapped further by instrumentation. The returned handler
// is immutable and safe for concurrent use.
func (s *Server) Handler() http.Handler { return s.handler }

// ServeListener serves on ln until ctx is cancelled, then shuts down
// gracefully: Shutdown gets 10 seconds for in-flight requests (forced Close
// after that), and the http.ErrServerClosed that Shutdown makes Serve return
// is reported as nil. A listener error ends the call with that error.
func (s *Server) ServeListener(ctx context.Context, ln net.Listener) error {
	srv := &http.Server{
		Handler:      s.handler,
		ReadTimeout:  s.readTimeout,
		WriteTimeout: s.writeTimeout,
	}

	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()

	select {
	case err := <-serveErr:
		// Serve never returns nil; without a Shutdown the only clean stop is
		// a pre-closed listener.
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		shutdownErr := srv.Shutdown(shutdownCtx)
		if shutdownErr != nil {
			// Timed out (or the shutdown context failed): force the
			// remaining connections closed so Serve returns and no
			// goroutine is left behind.
			_ = srv.Close()
		}
		err := <-serveErr
		if shutdownErr != nil {
			return shutdownErr
		}
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	}
}

// Serve listens on s.Addr and serves until ctx is cancelled; see
// ServeListener for the shutdown contract.
func (s *Server) Serve(ctx context.Context) error {
	ln, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("server: listen on %s: %w", s.addr, err)
	}
	return s.ServeListener(ctx, ln)
}

// --- routing and middleware -------------------------------------------------

// middleware wraps a handler with one cross-cutting concern. buildHandler
// applies the list in order, so the first entry is the outermost wrapper.
type middleware func(http.Handler) http.Handler

// middlewares returns the standard chain. It is a method (rather than a
// literal inside buildHandler) so later phases can insert or reorder
// instrumentation centrally. Options.Middleware entries are not part of this
// list: buildHandler folds them outside it.
func (s *Server) middlewares() []middleware {
	return []middleware{
		s.withRecovery,  // outermost of the standard chain: catches panics from every layer below
		s.withRequestID, // before logging, so log lines carry the ID
		s.withLogging,
	}
}

func (s *Server) buildHandler() http.Handler {
	var h http.Handler = http.HandlerFunc(s.dispatch)
	// Caller-supplied middleware wraps the whole built-in chain — recovery
	// included — so Options.Middleware[0] is the single outermost wrapper.
	chain := append(append([]middleware{}, s.extraMiddleware...), s.middlewares()...)
	// Fold from the inside out: the first middleware ends up outermost.
	for i := len(chain) - 1; i >= 0; i-- {
		h = chain[i](h)
	}
	return h
}

// dispatch matches exact paths. Unknown paths 404; a known path with the
// wrong method 405 with an Allow header. Both are JSON.
func (s *Server) dispatch(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case pathRoute:
		if s.requireMethod(w, r, http.MethodPost) {
			s.handleRoute(w, r)
		}
	case pathComplete:
		if s.requireMethod(w, r, http.MethodPost) {
			s.handleComplete(w, r)
		}
	case pathChatCompletions:
		if s.requireMethod(w, r, http.MethodPost) {
			s.handleChatCompletions(w, r)
		}
	case pathModels:
		if s.requireMethod(w, r, http.MethodGet) {
			s.handleModels(w, r)
		}
	case pathHealthz:
		if s.requireMethod(w, r, http.MethodGet) {
			writeJSON(w, http.StatusOK, statusPayload{Status: "ok"})
		}
	default:
		if strings.HasPrefix(r.URL.Path, pathModels+"/") {
			if s.requireMethod(w, r, http.MethodGet) {
				s.handleModel(w, r)
			}
			return
		}
		s.fail(w, r, http.StatusNotFound, codeNotFound, "unknown path "+r.URL.Path)
	}
}

func (s *Server) requireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r.Method == method {
		return true
	}
	w.Header().Set("Allow", method)
	s.fail(w, r, http.StatusMethodNotAllowed, codeMethodNotAllowed,
		fmt.Sprintf("method %s not allowed (want %s)", r.Method, method))
	return false
}

// --- handlers ---------------------------------------------------------------

// routeRequest is the POST /v1/route body.
type routeRequest struct {
	Prompt      string                          `json:"prompt"`
	Constraints *modelrouter.RoutingConstraints `json:"constraints,omitempty"`
}

func (s *Server) handleRoute(w http.ResponseWriter, r *http.Request) {
	var req routeRequest
	if err := s.decodeBody(w, r, &req); err != nil {
		status, msg := bodyError(err)
		s.fail(w, r, status, codeInvalidRequest, msg)
		return
	}
	if req.Prompt == "" {
		s.fail(w, r, http.StatusBadRequest, codeInvalidRequest, "prompt is required")
		return
	}

	decision, err := s.router.Route(r.Context(), req.Prompt, req.Constraints)
	if err != nil {
		s.writeRouterError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, decision)
}

// completeRequest is the POST /v1/complete body. Model/Provider are
// deliberately absent: the HTTP API always routes (pinning a model is a
// library-level use case, not part of the v1 wire contract).
type completeRequest struct {
	Prompt      string                          `json:"prompt"`
	Messages    []modelrouter.Message           `json:"messages"`
	Constraints *modelrouter.RoutingConstraints `json:"constraints,omitempty"`
	MaxTokens   int                             `json:"max_tokens,omitempty"`
	Temperature *float64                        `json:"temperature,omitempty"`
}

func (s *Server) handleComplete(w http.ResponseWriter, r *http.Request) {
	var req completeRequest
	if err := s.decodeBody(w, r, &req); err != nil {
		status, msg := bodyError(err)
		s.fail(w, r, status, codeInvalidRequest, msg)
		return
	}
	if req.Prompt == "" && len(req.Messages) == 0 {
		s.fail(w, r, http.StatusBadRequest, codeInvalidRequest, "prompt or messages is required")
		return
	}

	completionReq := modelrouter.CompletionRequest{
		Prompt:      req.Prompt,
		Messages:    req.Messages,
		Constraints: req.Constraints,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}
	result, err := s.router.Complete(r.Context(), completionReq)
	if err != nil {
		s.writeRouterError(w, r, err)
		return
	}
	setRoutingHeaders(w, result)
	writeJSON(w, http.StatusOK, result)
}

// decodeBody reads one JSON value from the request body, capped at the
// server's body limit. Unknown fields are ignored.
func (s *Server) decodeBody(w http.ResponseWriter, r *http.Request, v any) error {
	return json.NewDecoder(http.MaxBytesReader(w, r.Body, s.maxBodyBytes)).Decode(v)
}

// bodyError maps a decodeBody failure onto a status and message.
func bodyError(err error) (int, string) {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		return http.StatusRequestEntityTooLarge, fmt.Sprintf("request body exceeds %d bytes", tooLarge.Limit)
	}
	return http.StatusBadRequest, "invalid JSON body: " + err.Error()
}

// --- payloads and error mapping ---------------------------------------------

// errorPayload is the body of every JSON error response.
type errorPayload struct {
	Error string `json:"error"`
	Code  string `json:"code"`
}

// statusPayload is the /healthz body.
type statusPayload struct {
	Status string `json:"status"`
}

// errorStatus maps a Router error onto the HTTP status and stable code:
// no candidate -> 422; catalog/classifier/provider unavailable -> 502;
// anything else -> 500.
func errorStatus(err error) (int, string) {
	switch {
	case errors.Is(err, modelrouter.ErrNoCandidate):
		return http.StatusUnprocessableEntity, codeNoCandidate
	case errors.Is(err, modelrouter.ErrCatalogUnavailable):
		return http.StatusBadGateway, codeCatalogUnavailable
	case errors.Is(err, modelrouter.ErrClassifierUnavailable):
		return http.StatusBadGateway, codeClassifierUnavailable
	case errors.Is(err, modelrouter.ErrProvider):
		return http.StatusBadGateway, codeProviderError
	default:
		return http.StatusInternalServerError, codeInternal
	}
}

func (s *Server) writeRouterError(w http.ResponseWriter, r *http.Request, err error) {
	status, code := errorStatus(err)
	s.fail(w, r, status, code, err.Error())
}

// fail writes an error in the envelope of the endpoint family r addressed:
// OpenAI's for the compatible paths, {"error","code"} for the rest.
func (s *Server) fail(w http.ResponseWriter, r *http.Request, status int, code, message string) {
	if isCompatPath(r.URL.Path) {
		writeOAIError(w, status, code, "", message)
		return
	}
	s.writeError(w, status, code, message)
}

func (s *Server) writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, errorPayload{Error: message, Code: code})
}

// writeJSON marshals payload before touching the response, so an encoding
// failure cannot produce a half-written success body.
func writeJSON(w http.ResponseWriter, status int, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		// This package's payloads always marshal; keep the response
		// well-formed if a future payload ever does not.
		status = http.StatusInternalServerError
		data = []byte(`{"error":"failed to encode response","code":"internal"}`)
	}
	w.Header().Set("Content-Type", contentTypeJSON)
	w.WriteHeader(status)
	_, _ = w.Write(append(data, '\n'))
}

// --- middleware -------------------------------------------------------------

// responseRecorder remembers the status code (for request logging and for
// deciding whether recovery may still write a 500).
type responseRecorder struct {
	http.ResponseWriter
	code int
}

func (r *responseRecorder) WriteHeader(code int) {
	if r.code != 0 {
		return // first WriteHeader wins, mirroring net/http
	}
	r.code = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(b []byte) (int, error) {
	if r.code == 0 {
		r.code = http.StatusOK
	}
	return r.ResponseWriter.Write(b)
}

// Unwrap lets http.ResponseController reach the underlying writer.
func (r *responseRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func (r *responseRecorder) status() int {
	if r.code == 0 {
		return http.StatusOK
	}
	return r.code
}

func (r *responseRecorder) wroteHeader() bool { return r.code != 0 }

// requestIDKey is the context key type for the request ID; an unexported
// struct type cannot collide with other packages' keys.
type requestIDKey struct{}

func requestIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey{}).(string)
	return id
}

// withRequestID passes a non-empty X-Request-ID through or generates 16
// random bytes (32 hex characters when hex-encoded), exposes it on the
// request context for logging and echoes it on the response.
func (s *Server) withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get(requestIDHeader)
		if id == "" {
			id = newRequestID(s.logger)
		}
		w.Header().Set(requestIDHeader, id)
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func newRequestID(logger *slog.Logger) string {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		// crypto/rand.Read does not fail in practice; stay operational
		// rather than failing every request if it ever does.
		logger.Error("generate request id", "error", err)
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf[:])
}

// withLogging emits one structured line per request with method, path,
// status, duration and request ID. Panicking requests skip this line — the
// recovery middleware logs them with the stack instead.
func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		recorder := &responseRecorder{ResponseWriter: w}
		next.ServeHTTP(recorder, r)
		s.logger.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", recorder.status(),
			"duration_ms", float64(time.Since(start).Microseconds())/1000,
			"request_id", requestIDFrom(r.Context()),
		)
	})
}

// withRecovery is the outermost middleware: a panic anywhere below is logged
// with its stack, and — when nothing has been written yet — answered with a
// 500 JSON body. If headers (or a body) are already on the wire, the
// status cannot be changed anymore, so recovery only logs and stops the
// request.
func (s *Server) withRecovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recorder := &responseRecorder{ResponseWriter: w}
		defer func() {
			recovered := recover()
			if recovered == nil {
				return
			}
			s.logger.Error("panic recovered",
				"panic", recovered,
				"stack", string(debug.Stack()),
				"method", r.Method,
				"path", r.URL.Path,
				"request_id", requestIDFrom(r.Context()),
			)
			if recorder.wroteHeader() {
				return
			}
			s.fail(recorder, r, http.StatusInternalServerError, codeInternal, "internal server error")
		}()
		next.ServeHTTP(recorder, r)
	})
}
