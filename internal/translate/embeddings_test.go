package translate

import (
	"testing"

	"github.com/saltpay/fakerock/internal/openai"
)

func TestToOpenAIEmbeddingSetsDimensionsOnlyWhenPositive(t *testing.T) {
	withDims := ToOpenAIEmbedding("model", "hi", 512)
	if withDims.Dimensions == nil || *withDims.Dimensions != 512 {
		t.Errorf("dimensions = %v, want 512 forwarded to the backend", withDims.Dimensions)
	}
	if withDims.Model != "model" || withDims.Input != "hi" {
		t.Errorf("request = %+v", withDims)
	}

	unset := ToOpenAIEmbedding("model", "hi", 0)
	if unset.Dimensions != nil {
		t.Errorf("dimensions = %v, want nil when unset", unset.Dimensions)
	}
}

func TestFromOpenAIEmbeddingResizesToRequestedWidth(t *testing.T) {
	tests := []struct {
		name     string
		backend  []float32
		want     int
		wantHead []float32
	}{
		{"truncates longer", []float32{1, 2, 3, 4}, 2, []float32{1, 2}},
		{"pads shorter", []float32{1, 2}, 4, []float32{1, 2, 0, 0}},
		{"leaves equal", []float32{1, 2, 3}, 3, []float32{1, 2, 3}},
		{"no resize when zero", []float32{1, 2, 3}, 0, []float32{1, 2, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := openai.EmbeddingResponse{
				Data:  []openai.EmbeddingData{{Embedding: tt.backend}},
				Usage: openai.Usage{PromptTokens: 5},
			}
			got, err := FromOpenAIEmbedding(resp, tt.want)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(tt.wantHead) != len(got.Embedding) {
				t.Fatalf("embedding = %v, want %v", got.Embedding, tt.wantHead)
			}
			for i, v := range tt.wantHead {
				if got.Embedding[i] != v {
					t.Errorf("embedding[%d] = %v, want %v", i, got.Embedding[i], v)
				}
			}
			if got.InputTextTokenCount != 5 {
				t.Errorf("inputTextTokenCount = %d, want 5", got.InputTextTokenCount)
			}
		})
	}
}

func TestFromOpenAIEmbeddingRejectsEmptyResponse(t *testing.T) {
	_, err := FromOpenAIEmbedding(openai.EmbeddingResponse{}, 1024)
	if err != ErrNoEmbedding {
		t.Errorf("err = %v, want ErrNoEmbedding", err)
	}
}
