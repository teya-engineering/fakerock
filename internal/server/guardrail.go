package server

import (
	"net/http"

	"github.com/saltpay/fakerock/internal/bedrock"
)

// Always passes content through untouched. The endpoint exists so callers that wrap
// their requests in guardrails keep working; it enforces nothing.
func handleApplyGuardrail(w http.ResponseWriter) {
	writeJSON(w, bedrock.ApplyGuardrailResponse{
		Action:      bedrock.GuardrailActionNone,
		Outputs:     []bedrock.GuardrailOutput{},
		Assessments: nil,
	})
}
