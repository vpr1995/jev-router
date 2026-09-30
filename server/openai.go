package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	modelrouter "github.com/vprprudhvi/jev-router"
)

// OpenAI-compatible surface. These endpoints let stock OpenAI clients and SDKs
// (base_url = http://host:8080/v1) use the router as a drop-in chat backend.
// The wire model is intentionally narrower than OpenAI's: text-only messages,
// one choice, no streaming, no tools. Anything unsupported is rejected with a
// 400 in OpenAI's error envelope instead of being silently ignored, so an SDK
// never mistakes a degraded answer for the one it asked for.

const (
	pathChatCompletions = "/v1/chat/completions"
	pathModels          = "/v1/models"

	// autoModelID is the single virtual model advertised by /v1/models. The
	// "model" field of a chat request is accepted with any value: the router
	// always chooses the model itself.
	autoModelID = "auto"

	// Response headers describing the routing outcome.
	headerModel    = "X-Jev-Router-Model"
	headerProvider = "X-Jev-Router-Provider"
	headerSource   = "X-Jev-Router-Source"
)

// OpenAI error types.
const (
	oaiInvalidRequest = "invalid_request_error"
	oaiServerError    = "server_error"
)

// oaiErrorBody is OpenAI's {"error": {...}} envelope. Param and Code are
// null when unset, as in OpenAI's own responses.
type oaiErrorBody struct {
	Error oaiError `json:"error"`
}

type oaiError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// isCompatPath reports whether path belongs to the OpenAI-compatible surface,
// whose errors use OpenAI's envelope rather than {"error","code"}.
func isCompatPath(path string) bool {
	return path == pathChatCompletions || path == pathModels || strings.HasPrefix(path, pathModels+"/")
}

func oaiErrorType(status int) string {
	if status >= 500 {
		return oaiServerError
	}
	return oaiInvalidRequest
}

func writeOAIError(w http.ResponseWriter, status int, code, param, message string) {
	e := oaiError{Message: message, Type: oaiErrorType(status)}
	if code != "" {
		e.Code = &code
	}
	if param != "" {
		e.Param = &param
	}
	if status >= 500 {
		// The router already walked its whole candidate chain (and made a
		// billed classifier call); an SDK retry would repeat all of it.
		w.Header().Set("X-Should-Retry", "false")
	}
	writeJSON(w, status, oaiErrorBody{Error: e})
}

// --- chat completions -------------------------------------------------------

// chatRequest is the subset of POST /v1/chat/completions that is honored.
// Fields not listed (top_p, stop, seed, response_format, user, ...) are
// ignored, as are all unknown fields. Constraints is a router extension.
type chatRequest struct {
	Model               string            `json:"model"`
	Messages            []chatMessage     `json:"messages"`
	Stream              bool              `json:"stream"`
	N                   *int              `json:"n"`
	MaxTokens           int               `json:"max_tokens"`
	MaxCompletionTokens int               `json:"max_completion_tokens"`
	Temperature         *float64          `json:"temperature"`
	Tools               []json.RawMessage `json:"tools"`
	Functions           []json.RawMessage `json:"functions"`
	ResponseFormat      *struct {
		Type string `json:"type"`
	} `json:"response_format"`
	Logprobs    bool                            `json:"logprobs"`
	TopLogprobs int                             `json:"top_logprobs"`
	Stop        json.RawMessage                 `json:"stop"`
	Constraints *modelrouter.RoutingConstraints `json:"constraints"`
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// chatRequestError is a request rejected with a 400; Param names the offending
// request field.
type chatRequestError struct {
	Code    string
	Param   string
	Message string
}

func (e *chatRequestError) Error() string { return e.Message }

func badChat(code, param, format string, args ...any) *chatRequestError {
	return &chatRequestError{Code: code, Param: param, Message: fmt.Sprintf(format, args...)}
}

// toCompletionRequest validates req and maps it onto a router request.
func (req chatRequest) toCompletionRequest() (modelrouter.CompletionRequest, *chatRequestError) {
	if req.Stream {
		return modelrouter.CompletionRequest{}, badChat("streaming_not_supported", "stream",
			"streaming is not supported; omit \"stream\" or set it to false")
	}
	if req.N != nil && *req.N != 1 {
		return modelrouter.CompletionRequest{}, badChat("unsupported_parameter", "n",
			"only n=1 is supported")
	}
	if len(req.Tools) > 0 || len(req.Functions) > 0 {
		return modelrouter.CompletionRequest{}, badChat("unsupported_parameter", "tools",
			"tools and functions are not supported")
	}
	if rf := req.ResponseFormat; rf != nil && rf.Type != "" && rf.Type != "text" {
		return modelrouter.CompletionRequest{}, badChat("unsupported_parameter", "response_format",
			"response_format %q is not supported; only text output is available", rf.Type)
	}
	if req.Logprobs || req.TopLogprobs > 0 {
		return modelrouter.CompletionRequest{}, badChat("unsupported_parameter", "logprobs",
			"logprobs are not supported")
	}
	if stop := bytes.TrimSpace(req.Stop); len(stop) > 0 && string(stop) != "null" && string(stop) != "[]" && string(stop) != `""` {
		return modelrouter.CompletionRequest{}, badChat("unsupported_parameter", "stop",
			"stop sequences are not supported")
	}
	if len(req.Messages) == 0 {
		return modelrouter.CompletionRequest{}, badChat("invalid_request", "messages",
			"messages is required and must not be empty")
	}

	messages := make([]modelrouter.Message, 0, len(req.Messages))
	for i, m := range req.Messages {
		param := fmt.Sprintf("messages[%d]", i)
		role, ok := normalizeRole(m.Role)
		if !ok {
			return modelrouter.CompletionRequest{}, badChat("unsupported_parameter", param+".role",
				"unsupported message role %q (want system, developer, user or assistant)", m.Role)
		}
		text, err := flattenContent(m.Content)
		if err != nil {
			return modelrouter.CompletionRequest{}, badChat("unsupported_parameter", param+".content", "%s", err.Error())
		}
		if text == "" {
			return modelrouter.CompletionRequest{}, badChat("invalid_request", param+".content",
				"message content must not be empty")
		}
		messages = append(messages, modelrouter.Message{Role: role, Content: text})
	}

	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens < 0 {
		return modelrouter.CompletionRequest{}, badChat("invalid_request", "max_tokens", "max_tokens must not be negative")
	}
	return modelrouter.CompletionRequest{
		Messages:    messages,
		Constraints: req.Constraints,
		MaxTokens:   maxTokens,
		Temperature: req.Temperature,
	}, nil
}

// normalizeRole maps OpenAI roles onto the router's: developer is the newer
// name for system. tool and function roles need tool support and are rejected.
func normalizeRole(role string) (string, bool) {
	switch role {
	case "system", "developer":
		return "system", true
	case "user", "assistant":
		return role, true
	}
	return "", false
}

// flattenContent turns a message content value into text. OpenAI allows a
// string, null (assistant messages), or an array of typed parts; only text
// parts are supported and they are concatenated.
func flattenContent(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", fmt.Errorf("invalid content: %v", err)
		}
		return s, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("content must be a string or an array of content parts")
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type != "text" {
			return "", fmt.Errorf("unsupported content part type %q (only text is supported)", p.Type)
		}
		b.WriteString(p.Text)
	}
	return b.String(), nil
}

type chatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

type chatChoice struct {
	Index        int              `json:"index"`
	Message      chatReplyMessage `json:"message"`
	FinishReason string           `json:"finish_reason"`
}

type chatReplyMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := s.decodeBody(w, r, &req); err != nil {
		status, msg := bodyError(err)
		writeOAIError(w, status, codeInvalidRequest, "", msg)
		return
	}
	completionReq, reqErr := req.toCompletionRequest()
	if reqErr != nil {
		writeOAIError(w, http.StatusBadRequest, reqErr.Code, reqErr.Param, reqErr.Message)
		return
	}

	result, err := s.router.Complete(r.Context(), completionReq)
	if err != nil {
		s.writeRouterError(w, r, err)
		return
	}
	setRoutingHeaders(w, result)
	writeJSON(w, http.StatusOK, newChatResponse(requestIDFrom(r.Context()), time.Now(), result))
}

func newChatResponse(requestID string, now time.Time, result modelrouter.CompletionResult) chatCompletionResponse {
	model := result.Model
	if model == "" {
		model = result.RoutingDecision.ModelID
	}
	var usage chatUsage
	if u := result.Usage; u != nil {
		usage = chatUsage{PromptTokens: u.InputTokens, CompletionTokens: u.OutputTokens, TotalTokens: u.TotalTokens}
		if usage.TotalTokens == 0 {
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		}
	}
	return chatCompletionResponse{
		ID:      "chatcmpl-" + requestID,
		Object:  "chat.completion",
		Created: now.Unix(),
		Model:   model,
		Choices: []chatChoice{{
			Index:        0,
			Message:      chatReplyMessage{Role: "assistant", Content: result.Content},
			FinishReason: normalizeFinishReason(result.FinishReason),
		}},
		Usage: usage,
	}
}

// normalizeFinishReason maps provider-native stop reasons (OpenAI, Anthropic,
// Gemini, Bedrock) onto OpenAI's stop | length | content_filter. Unknown and
// empty values become "stop"; tool-use reasons too, since tool calls are never
// surfaced.
func normalizeFinishReason(reason string) string {
	switch strings.ToLower(reason) {
	case "length", "max_tokens", "max_output_tokens", "model_context_window_exceeded":
		return "length"
	case "content_filter", "content_filtered", "guardrail_intervened", "refusal",
		"safety", "recitation", "blocklist", "prohibited_content", "spii", "image_safety":
		return "content_filter"
	}
	return "stop"
}

// setRoutingHeaders reports the served model on the response so callers can
// see the routing outcome without parsing a body.
func setRoutingHeaders(w http.ResponseWriter, result modelrouter.CompletionResult) {
	d := result.RoutingDecision
	if d.ModelID == "" {
		return
	}
	w.Header().Set(headerModel, d.ModelID)
	w.Header().Set(headerProvider, d.Provider)
	w.Header().Set(headerSource, d.Source)
}

// --- models -----------------------------------------------------------------

type modelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type modelList struct {
	Object string        `json:"object"`
	Data   []modelObject `json:"data"`
}

// autoModel is the one advertised model. It never depends on a catalog fetch,
// so a flaky catalog cannot break client startup.
func (s *Server) autoModel() modelObject {
	return modelObject{ID: autoModelID, Object: "model", Created: s.started.Unix(), OwnedBy: "jev-router"}
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, modelList{Object: "list", Data: []modelObject{s.autoModel()}})
}

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, pathModels+"/")
	if id != autoModelID {
		writeOAIError(w, http.StatusNotFound, "model_not_found", "model",
			fmt.Sprintf("the model %q does not exist; use %q", id, autoModelID))
		return
	}
	writeJSON(w, http.StatusOK, s.autoModel())
}
