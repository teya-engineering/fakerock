package translate

import (
	"errors"

	"github.com/saltpay/fakerock/internal/bedrock"
	"github.com/saltpay/fakerock/internal/openai"
)

var ErrNoEmbedding = errors.New("backend returned no embedding")

// dimensions is the resolved width (0 means "leave it to the backend"). Passing it on lets a
// backend that honours the field (Matryoshka models) return the right width directly, rather than
// relying on the resize below.
func ToOpenAIEmbedding(model, input string, dimensions int) openai.EmbeddingRequest {
	req := openai.EmbeddingRequest{Model: model, Input: input}
	if dimensions > 0 {
		req.Dimensions = &dimensions
	}
	return req
}

// The store the caller writes to has a fixed width, so when a width was requested the vector must be
// exactly that. A backend that ignored the request rarely matches, so trim or zero-pad to fit. This
// is a test stand-in: the vector is the right shape, not semantically equivalent to Titan, and it is
// not re-normalised to unit length.
func FromOpenAIEmbedding(resp openai.EmbeddingResponse, dimensions int) (bedrock.TitanEmbeddingResponse, error) {
	if len(resp.Data) == 0 {
		return bedrock.TitanEmbeddingResponse{}, ErrNoEmbedding
	}

	embedding := resp.Data[0].Embedding
	if dimensions > 0 {
		embedding = resize(embedding, dimensions)
	}

	return bedrock.TitanEmbeddingResponse{
		Embedding:           embedding,
		InputTextTokenCount: resp.Usage.PromptTokens,
	}, nil
}

// size is always positive here; callers reject non-positive dimensions.
func resize(vector []float32, size int) []float32 {
	if len(vector) == size {
		return vector
	}
	out := make([]float32, size)
	copy(out, vector)
	return out
}
