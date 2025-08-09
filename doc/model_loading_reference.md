Mapping Table: llama.cpp to candle-rs for SmolLM3-3B Q4_K_M GGUF
The following tensor structure, MODEL_ARCH.SMOLLM3, is defined in the source file llama.h from the repository github.com/ggerganov/llama.cpp. It specifies the tensors required for the SmolLM3-3B model in Q4_K_M quantization and is used by llama.cpp to load the model from a GGUF file by mapping tensor names (e.g., token_embd.weight, layers.*.attn_q.weight) to their roles in the transformer architecture (e.g., token embeddings, attention weights). This table maps these llama.cpp components to their equivalents in candle-rs (version 0.9.1) for loading and running the SmolLM3-3B Q4_K_M GGUF model, prioritizing quantized operations (via candle_core::quantized::QMatMul) for performance-critical tensors to fully utilize Q4_K_M quantization benefits.
MODEL_ARCH.SMOLLM3: [
    MODEL_TENSOR.TOKEN_EMBD,
    MODEL_TENSOR.OUTPUT_NORM,
    MODEL_TENSOR.OUTPUT,
    MODEL_TENSOR.ROPE_FREQS,
    MODEL_TENSOR.ATTN_NORM,
    MODEL_TENSOR.ATTN_Q,
    MODEL_TENSOR.ATTN_K,
    MODEL_TENSOR.ATTN_V,
    MODEL_TENSOR.ATTN_OUT,
    MODEL_TENSOR.ATTN_ROT_EMBD,
    MODEL_TENSOR.FFN_NORM,
    MODEL_TENSOR.FFN_GATE,
    MODEL_TENSOR.FFN_DOWN,
    MODEL_TENSOR.FFN_UP,
]




Component
llama.cpp
candle-rs (0.9.1)
Description



GGUF File Parsing
gguf.c:- gguf_init_from_file- gguf_get_tensor_info- gguf_load_tensor
Custom parser or gguf-rs:- Read GGUF header, metadata, and tensors- Convert to candle_core::quantized::QTensor or candle_core::Tensor
llama.cpp reads the GGUF file to extract metadata (e.g., architecture=SMOLLM3, context_length=65536) and Q4_K_M tensors listed in MODEL_ARCH.SMOLLM3. In candle-rs, implement a parser to load Q4_K_M tensors into candle_core::quantized::QTensor and non-quantized tensors into candle_core::Tensor.


Tensor Management
ggml.c:- ggml_tensor- ggml_init- ggml_map_by_name
candle_core:- quantized::QTensor- Tensor- Device::Cpu or Device::Cuda
llama.cpp manages Q4_K_M tensors (e.g., GGML_TYPE_Q4_K) and f32 tensors from MODEL_ARCH.SMOLLM3. candle-rs uses candle_core::quantized::QTensor for Q4_K_M tensors and candle_core::Tensor for others, with device placement (CPU/GPU).


Model Loading
llama.cpp:- llama_model_loader- llama_load_tensors- llama_model_init_from_file
Custom logic in candle-transformers:- Load tensors into a custom SmolLM3 struct
llama.cpp maps Q4_K_M tensors to a llama_model struct using MODEL_ARCH.SMOLLM3. In candle-rs, create a SmolLM3 struct, mapping Q4_K_M tensors to QMatMul and others to Embedding, Linear, or LayerNorm.


Tokenizer
common.cpp:- llama_tokenize- Loads Llama 3.2 tokenizer
tokenizers (0.21):- Tokenizer::from_file
llama.cpp loads the Llama 3.2 tokenizer (32,000 tokens) for TOKEN_EMBD. In candle-rs, use tokenizers::Tokenizer to load tokenizer.json and encode input text.


Token Embeddings
MODEL_TENSOR.TOKEN_EMBD- Tensor: token_embd.weight
candle_nn::Embedding
Maps token IDs to vectors (shape: [32000, 3072]). llama.cpp loads token_embd.weight in f32/f16. In candle-rs, load into candle_nn::Embedding (not quantized).


Output Normalization
MODEL_TENSOR.OUTPUT_NORM- Tensor: output_norm.weight
candle_nn::LayerNorm
Applies layer normalization. llama.cpp loads output_norm.weight in f32. In candle-rs, load into candle_nn::LayerNorm (not quantized).


Output Layer
MODEL_TENSOR.OUTPUT- Tensor: output.weight
candle_core::quantized::QMatMul
Projects hidden states to logits (shape: [3072, 32000]). llama.cpp loads output.weight in Q4_K_M. In candle-rs, load into QMatMul for quantized computation.


RoPE Frequencies
MODEL_TENSOR.ROPE_FREQS- Tensor: rope_freqs
Custom RoPE logic in candle-transformers
Stores RoPE frequencies. llama.cpp uses rope_freqs in f32 with NoRope. In candle-rs, implement custom RoPE (not quantized).


Attention Normalization
MODEL_TENSOR.ATTN_NORM- Tensor: layers.*.attn_norm.weight
candle_nn::LayerNorm
Normalizes attention inputs. llama.cpp loads attn_norm.weight in f32. In candle-rs, load into candle_nn::LayerNorm (not quantized).


Attention Query (Q)
MODEL_TENSOR.ATTN_Q- Tensor: layers.*.attn_q.weight
candle_core::quantized::QMatMul
Query weights for GQA (4 groups, 16 heads). llama.cpp loads attn_q.weight in Q4_K_M with ggml_matmul_q4_k. In candle-rs, load into QMatMul for quantized computation.


Attention Key (K)
MODEL_TENSOR.ATTN_K- Tensor: layers.*.attn_k.weight
candle_core::quantized::QMatMul
Key weights for GQA. llama.cpp loads attn_k.weight in Q4_K_M. In candle-rs, load into QMatMul.


Attention Value (V)
MODEL_TENSOR.ATTN_V- Tensor: layers.*.attn_v.weight
candle_core::quantized::QMatMul
Value weights for GQA. llama.cpp loads attn_v.weight in Q4_K_M. In candle-rs, load into QMatMul.


Attention Output
MODEL_TENSOR.ATTN_OUT- Tensor: layers.*.attn_out.weight
candle_core::quantized::QMatMul
Output projection for attention. llama.cpp loads attn_out.weight in Q4_K_M. In candle-rs, load into QMatMul.


Attention Rotary Embeddings
MODEL_TENSOR.ATTN_ROT_EMBD- Tensor: layers.*.attn_rot_embd
Custom RoPE logic in candle-transformers
Handles rotary embeddings with NoRope. llama.cpp uses attn_rot_embd in f32. In candle-rs, implement custom RoPE (not quantized).


FFN Normalization
MODEL_TENSOR.FFN_NORM- Tensor: layers.*.ffn_norm.weight
candle_nn::LayerNorm
Normalizes feed-forward inputs. llama.cpp loads ffn_norm.weight in f32. In candle-rs, load into candle_nn::LayerNorm (not quantized).


FFN Gate
MODEL_TENSOR.FFN_GATE- Tensor: layers.*.ffn_gate.weight
candle_core::quantized::QMatMul
Gate weights for FFN. llama.cpp loads ffn_gate.weight in Q4_K_M. In candle-rs, load into QMatMul.


FFN Down
MODEL_TENSOR.FFN_DOWN- Tensor: layers.*.ffn_down.weight
candle_core::quantized::QMatMul
Down-projection weights for FFN. llama.cpp loads ffn_down.weight in Q4_K_M. In candle-rs, load into QMatMul.


FFN Up
MODEL_TENSOR.FFN_UP- Tensor: layers.*.ffn_up.weight
candle_core::quantized::QMatMul
Up-projection weights for FFN. llama.cpp loads ffn_up.weight in Q4_K_M. In candle-rs, load into QMatMul.


Inference
llama.cpp:- llama_generate
candle-transformers:- Custom forward method
llama.cpp runs the forward pass with Q4_K_M tensors using ggml_matmul_q4_k, guided by MODEL_ARCH.SMOLLM3. In candle-rs, implement a forward method in a custom SmolLM3 struct, using QMatMul for quantized tensors.


How llama.cpp Uses MODEL_ARCH.SMOLLM3 to Load SmolLM3-3B Q4_K_M
The MODEL_ARCH.SMOLLM3 array, defined in llama.h, specifies the tensor types for the SmolLM3-3B model. For the Q4_K_M GGUF model, llama.cpp uses this array as follows:

GGUF File Parsing (gguf.c):

gguf_init_from_file reads the GGUF header, metadata (e.g., architecture=SMOLLM3, num_layers=32, context_length=65536), and tensor list.
gguf_get_tensor_info verifies that all tensors in MODEL_ARCH.SMOLLM3 (e.g., layers.*.attn_q.weight) are present, with weight tensors in GGML_TYPE_Q4_K and normalization/RoPE in f32.
gguf_load_tensor loads Q4_K_M tensors (e.g., ATTN_Q, FFN_GATE) into memory without dequantization.


Tensor Mapping (llama.cpp):

llama_model_loader maps tensor names to roles in MODEL_ARCH.SMOLLM3 (e.g., MODEL_TENSOR.ATTN_Q to layers.*.attn_q.weight).
llama_load_tensors assigns Q4_K_M tensors to the llama_model struct for 32 layers, using optimized kernels for quantized operations.


Model Initialization:

llama_model_init_from_file sets up the transformer (32 layers, 16 heads with 4 GQA groups, 3072 hidden size) and tokenizer.
MODEL_TENSOR.ROPE_FREQS and MODEL_TENSOR.ATTN_ROT_EMBD configure RoPE with NoRope logic (skipping every fourth layer).


Inference Setup:

llama_generate processes tokens through embeddings (TOKEN_EMBD), quantized attention (ATTN_Q, ATTN_K, ATTN_V, ATTN_OUT via ggml_matmul_q4_k), quantized FFN (FFN_GATE, FFN_DOWN, FFN_UP), and normalization (ATTN_NORM, FFN_NORM, OUTPUT_NORM).
Output is projected to logits via MODEL_TENSOR.OUTPUT using Q4_K_M.



Notes

SmolLM3-3B Q4_K_M: Decoder-only transformer with 32 layers, 16 attention heads (4 GQA groups), 3072 hidden size, 64k context length (extensible to 128k via YaRN), and NoRope. Q4_K_M reduces memory to ~1.9GB.
Quantized Operations: Use candle_core::quantized::QMatMul for ATTN_Q, ATTN_K, ATTN_V, ATTN_OUT, FFN_GATE, FFN_DOWN, FFN_UP, and OUTPUT to match Q4_K_M. Extend candle-transformers::models::llama::LlamaAttention and FFN layers.
GGUF Support: candle-rs 0.9.1 lacks native GGUF support. Implement a parser based on llama.cpp’s gguf.c or use gguf-rs.
Dependencies:[dependencies]
candle-core = "0.9.1"
candle-nn = "0.9.1"
candle-transformers = "0.9.1"
tokenizers = "0.21"


NoRope: Implement custom logic in candle-transformers to skip RoPE every fourth layer, using f32 tensors for ROPE_FREQS and ATTN_ROT_EMBD.
Quantization Check: Parser must verify Q4_K type for weight tensors, using QMatMul for Q4_K_M and Linear/Embedding/LayerNorm for f32 tensors.


### Example Implementation for Quantized QKV in `candle-rs`
To illustrate how to use `QMatMul` for QKV tensors, here’s a simplified example of extending the attention mechanism:

```rust
use candle_core::{Device, Tensor};
use candle_core::quantized::{QTensor, QMatMul};
use candle_transformers::models::llama::LlamaAttention;

struct SmolLM3Attention {
    q_matmul: QMatMul, // Quantized Q projection
    k_matmul: QMatMul, // Quantized K projection
    v_matmul: QMatMul, // Quantized V projection
    // Other fields (e.g., attn_out, rope)
}

impl SmolLM3Attention {
    fn new(q_weight: QTensor, k_weight: QTensor, v_weight: QTensor, device: &Device) -> Self {
        Self {
            q_matmul: QMatMul::from_qtensor(q_weight).unwrap(),
            k_matmul: QMatMul::from_qtensor(k_weight).unwrap(),
            v_matmul: QMatMul::from_qtensor(v_weight).unwrap(),
            // Initialize other fields
        }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Perform quantized matrix multiplications for Q, K, V
        let q = self.q_matmul.forward(x)?;
        let k = self.k_matmul.forward(x)?;
        let v = self.v_matmul.forward(x)?;
        // Implement GQA and NoRope logic
        // ...
        Ok(v) // Simplified; actual attention logic follows
    }
}
```

This approach ensures QKV computations remain quantized, mirroring `llama.cpp`’s efficiency.

If you need further details on implementing the GGUF parser or customizing `LlamaAttention` for GQA and NoRope, let me know!