================================================================================
                    GGUF INSPECTION REPORT
================================================================================

📁 FILE INFORMATION
----------------------------------------
Path: models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf
Size: 1.78 GB
Total tensors: 326
Total metadata: 32

🏗️  ARCHITECTURE
----------------------------------------
Architecture: smollm3
Vocab size: Some(128256)
Hidden size: Some(2048)
Number of layers: Some(36)

📊 TENSOR QUANTIZATION REPORT
----------------------------------------
Q4_K_M tensors (quantized): 216
F32 tensors (not quantized): 73
Other tensor types: 37

  Q4_K_M Tensors (MUST use QMatMul):
    - blk.0.attn_k.weight: shape [512, 2048]
    - blk.0.attn_output.weight: shape [2048, 2048]
    - blk.0.attn_q.weight: shape [2048, 2048]
    - blk.0.ffn_gate.weight: shape [11008, 2048]
    - blk.0.ffn_up.weight: shape [11008, 2048]
    - blk.1.attn_k.weight: shape [512, 2048]
    - blk.1.attn_output.weight: shape [2048, 2048]
    - blk.1.attn_q.weight: shape [2048, 2048]
    - blk.1.ffn_gate.weight: shape [11008, 2048]
    - blk.1.ffn_up.weight: shape [11008, 2048]
    - blk.10.attn_k.weight: shape [512, 2048]
    - blk.10.attn_output.weight: shape [2048, 2048]
    - blk.10.attn_q.weight: shape [2048, 2048]
    - blk.10.attn_v.weight: shape [512, 2048]
    - blk.10.ffn_down.weight: shape [2048, 11008]
    - blk.10.ffn_gate.weight: shape [11008, 2048]
    - blk.10.ffn_up.weight: shape [11008, 2048]
    - blk.11.attn_k.weight: shape [512, 2048]
    - blk.11.attn_output.weight: shape [2048, 2048]
    - blk.11.attn_q.weight: shape [2048, 2048]
    - blk.11.attn_v.weight: shape [512, 2048]
    - blk.11.ffn_down.weight: shape [2048, 11008]
    - blk.11.ffn_gate.weight: shape [11008, 2048]
    - blk.11.ffn_up.weight: shape [11008, 2048]
    - blk.12.attn_k.weight: shape [512, 2048]
    - blk.12.attn_output.weight: shape [2048, 2048]
    - blk.12.attn_q.weight: shape [2048, 2048]
    - blk.12.ffn_gate.weight: shape [11008, 2048]
    - blk.12.ffn_up.weight: shape [11008, 2048]
    - blk.13.attn_k.weight: shape [512, 2048]
    - blk.13.attn_output.weight: shape [2048, 2048]
    - blk.13.attn_q.weight: shape [2048, 2048]
    - blk.13.attn_v.weight: shape [512, 2048]
    - blk.13.ffn_down.weight: shape [2048, 11008]
    - blk.13.ffn_gate.weight: shape [11008, 2048]
    - blk.13.ffn_up.weight: shape [11008, 2048]
    - blk.14.attn_k.weight: shape [512, 2048]
    - blk.14.attn_output.weight: shape [2048, 2048]
    - blk.14.attn_q.weight: shape [2048, 2048]
    - blk.14.attn_v.weight: shape [512, 2048]
    - blk.14.ffn_down.weight: shape [2048, 11008]
    - blk.14.ffn_gate.weight: shape [11008, 2048]
    - blk.14.ffn_up.weight: shape [11008, 2048]
    - blk.15.attn_k.weight: shape [512, 2048]
    - blk.15.attn_output.weight: shape [2048, 2048]
    - blk.15.attn_q.weight: shape [2048, 2048]
    - blk.15.ffn_gate.weight: shape [11008, 2048]
    - blk.15.ffn_up.weight: shape [11008, 2048]
    - blk.16.attn_k.weight: shape [512, 2048]
    - blk.16.attn_output.weight: shape [2048, 2048]
    - blk.16.attn_q.weight: shape [2048, 2048]
    - blk.16.attn_v.weight: shape [512, 2048]
    - blk.16.ffn_down.weight: shape [2048, 11008]
    - blk.16.ffn_gate.weight: shape [11008, 2048]
    - blk.16.ffn_up.weight: shape [11008, 2048]
    - blk.17.attn_k.weight: shape [512, 2048]
    - blk.17.attn_output.weight: shape [2048, 2048]
    - blk.17.attn_q.weight: shape [2048, 2048]
    - blk.17.attn_v.weight: shape [512, 2048]
    - blk.17.ffn_down.weight: shape [2048, 11008]
    - blk.17.ffn_gate.weight: shape [11008, 2048]
    - blk.17.ffn_up.weight: shape [11008, 2048]
    - blk.18.attn_k.weight: shape [512, 2048]
    - blk.18.attn_output.weight: shape [2048, 2048]
    - blk.18.attn_q.weight: shape [2048, 2048]
    - blk.18.ffn_gate.weight: shape [11008, 2048]
    - blk.18.ffn_up.weight: shape [11008, 2048]
    - blk.19.attn_k.weight: shape [512, 2048]
    - blk.19.attn_output.weight: shape [2048, 2048]
    - blk.19.attn_q.weight: shape [2048, 2048]
    - blk.19.attn_v.weight: shape [512, 2048]
    - blk.19.ffn_down.weight: shape [2048, 11008]
    - blk.19.ffn_gate.weight: shape [11008, 2048]
    - blk.19.ffn_up.weight: shape [11008, 2048]
    - blk.2.attn_k.weight: shape [512, 2048]
    - blk.2.attn_output.weight: shape [2048, 2048]
    - blk.2.attn_q.weight: shape [2048, 2048]
    - blk.2.ffn_gate.weight: shape [11008, 2048]
    - blk.2.ffn_up.weight: shape [11008, 2048]
    - blk.20.attn_k.weight: shape [512, 2048]
    - blk.20.attn_output.weight: shape [2048, 2048]
    - blk.20.attn_q.weight: shape [2048, 2048]
    - blk.20.attn_v.weight: shape [512, 2048]
    - blk.20.ffn_down.weight: shape [2048, 11008]
    - blk.20.ffn_gate.weight: shape [11008, 2048]
    - blk.20.ffn_up.weight: shape [11008, 2048]
    - blk.21.attn_k.weight: shape [512, 2048]
    - blk.21.attn_output.weight: shape [2048, 2048]
    - blk.21.attn_q.weight: shape [2048, 2048]
    - blk.21.ffn_gate.weight: shape [11008, 2048]
    - blk.21.ffn_up.weight: shape [11008, 2048]
    - blk.22.attn_k.weight: shape [512, 2048]
    - blk.22.attn_output.weight: shape [2048, 2048]
    - blk.22.attn_q.weight: shape [2048, 2048]
    - blk.22.attn_v.weight: shape [512, 2048]
    - blk.22.ffn_down.weight: shape [2048, 11008]
    - blk.22.ffn_gate.weight: shape [11008, 2048]
    - blk.22.ffn_up.weight: shape [11008, 2048]
    - blk.23.attn_k.weight: shape [512, 2048]
    - blk.23.attn_output.weight: shape [2048, 2048]
    - blk.23.attn_q.weight: shape [2048, 2048]
    - blk.23.attn_v.weight: shape [512, 2048]
    - blk.23.ffn_down.weight: shape [2048, 11008]
    - blk.23.ffn_gate.weight: shape [11008, 2048]
    - blk.23.ffn_up.weight: shape [11008, 2048]
    - blk.24.attn_k.weight: shape [512, 2048]
    - blk.24.attn_output.weight: shape [2048, 2048]
    - blk.24.attn_q.weight: shape [2048, 2048]
    - blk.24.ffn_gate.weight: shape [11008, 2048]
    - blk.24.ffn_up.weight: shape [11008, 2048]
    - blk.25.attn_k.weight: shape [512, 2048]
    - blk.25.attn_output.weight: shape [2048, 2048]
    - blk.25.attn_q.weight: shape [2048, 2048]
    - blk.25.attn_v.weight: shape [512, 2048]
    - blk.25.ffn_down.weight: shape [2048, 11008]
    - blk.25.ffn_gate.weight: shape [11008, 2048]
    - blk.25.ffn_up.weight: shape [11008, 2048]
    - blk.26.attn_k.weight: shape [512, 2048]
    - blk.26.attn_output.weight: shape [2048, 2048]
    - blk.26.attn_q.weight: shape [2048, 2048]
    - blk.26.attn_v.weight: shape [512, 2048]
    - blk.26.ffn_down.weight: shape [2048, 11008]
    - blk.26.ffn_gate.weight: shape [11008, 2048]
    - blk.26.ffn_up.weight: shape [11008, 2048]
    - blk.27.attn_k.weight: shape [512, 2048]
    - blk.27.attn_output.weight: shape [2048, 2048]
    - blk.27.attn_q.weight: shape [2048, 2048]
    - blk.27.ffn_gate.weight: shape [11008, 2048]
    - blk.27.ffn_up.weight: shape [11008, 2048]
    - blk.28.attn_k.weight: shape [512, 2048]
    - blk.28.attn_output.weight: shape [2048, 2048]
    - blk.28.attn_q.weight: shape [2048, 2048]
    - blk.28.attn_v.weight: shape [512, 2048]
    - blk.28.ffn_down.weight: shape [2048, 11008]
    - blk.28.ffn_gate.weight: shape [11008, 2048]
    - blk.28.ffn_up.weight: shape [11008, 2048]
    - blk.29.attn_k.weight: shape [512, 2048]
    - blk.29.attn_output.weight: shape [2048, 2048]
    - blk.29.attn_q.weight: shape [2048, 2048]
    - blk.29.attn_v.weight: shape [512, 2048]
    - blk.29.ffn_down.weight: shape [2048, 11008]
    - blk.29.ffn_gate.weight: shape [11008, 2048]
    - blk.29.ffn_up.weight: shape [11008, 2048]
    - blk.3.attn_k.weight: shape [512, 2048]
    - blk.3.attn_output.weight: shape [2048, 2048]
    - blk.3.attn_q.weight: shape [2048, 2048]
    - blk.3.ffn_gate.weight: shape [11008, 2048]
    - blk.3.ffn_up.weight: shape [11008, 2048]
    - blk.30.attn_k.weight: shape [512, 2048]
    - blk.30.attn_output.weight: shape [2048, 2048]
    - blk.30.attn_q.weight: shape [2048, 2048]
    - blk.30.ffn_gate.weight: shape [11008, 2048]
    - blk.30.ffn_up.weight: shape [11008, 2048]
    - blk.31.attn_k.weight: shape [512, 2048]
    - blk.31.attn_output.weight: shape [2048, 2048]
    - blk.31.attn_q.weight: shape [2048, 2048]
    - blk.31.ffn_gate.weight: shape [11008, 2048]
    - blk.31.ffn_up.weight: shape [11008, 2048]
    - blk.32.attn_k.weight: shape [512, 2048]
    - blk.32.attn_output.weight: shape [2048, 2048]
    - blk.32.attn_q.weight: shape [2048, 2048]
    - blk.32.ffn_gate.weight: shape [11008, 2048]
    - blk.32.ffn_up.weight: shape [11008, 2048]
    - blk.33.attn_k.weight: shape [512, 2048]
    - blk.33.attn_output.weight: shape [2048, 2048]
    - blk.33.attn_q.weight: shape [2048, 2048]
    - blk.33.ffn_gate.weight: shape [11008, 2048]
    - blk.33.ffn_up.weight: shape [11008, 2048]
    - blk.34.attn_k.weight: shape [512, 2048]
    - blk.34.attn_output.weight: shape [2048, 2048]
    - blk.34.attn_q.weight: shape [2048, 2048]
    - blk.34.ffn_gate.weight: shape [11008, 2048]
    - blk.34.ffn_up.weight: shape [11008, 2048]
    - blk.35.attn_k.weight: shape [512, 2048]
    - blk.35.attn_output.weight: shape [2048, 2048]
    - blk.35.attn_q.weight: shape [2048, 2048]
    - blk.35.ffn_gate.weight: shape [11008, 2048]
    - blk.35.ffn_up.weight: shape [11008, 2048]
    - blk.4.attn_k.weight: shape [512, 2048]
    - blk.4.attn_output.weight: shape [2048, 2048]
    - blk.4.attn_q.weight: shape [2048, 2048]
    - blk.4.attn_v.weight: shape [512, 2048]
    - blk.4.ffn_down.weight: shape [2048, 11008]
    - blk.4.ffn_gate.weight: shape [11008, 2048]
    - blk.4.ffn_up.weight: shape [11008, 2048]
    - blk.5.attn_k.weight: shape [512, 2048]
    - blk.5.attn_output.weight: shape [2048, 2048]
    - blk.5.attn_q.weight: shape [2048, 2048]
    - blk.5.attn_v.weight: shape [512, 2048]
    - blk.5.ffn_down.weight: shape [2048, 11008]
    - blk.5.ffn_gate.weight: shape [11008, 2048]
    - blk.5.ffn_up.weight: shape [11008, 2048]
    - blk.6.attn_k.weight: shape [512, 2048]
    - blk.6.attn_output.weight: shape [2048, 2048]
    - blk.6.attn_q.weight: shape [2048, 2048]
    - blk.6.ffn_gate.weight: shape [11008, 2048]
    - blk.6.ffn_up.weight: shape [11008, 2048]
    - blk.7.attn_k.weight: shape [512, 2048]
    - blk.7.attn_output.weight: shape [2048, 2048]
    - blk.7.attn_q.weight: shape [2048, 2048]
    - blk.7.attn_v.weight: shape [512, 2048]
    - blk.7.ffn_down.weight: shape [2048, 11008]
    - blk.7.ffn_gate.weight: shape [11008, 2048]
    - blk.7.ffn_up.weight: shape [11008, 2048]
    - blk.8.attn_k.weight: shape [512, 2048]
    - blk.8.attn_output.weight: shape [2048, 2048]
    - blk.8.attn_q.weight: shape [2048, 2048]
    - blk.8.attn_v.weight: shape [512, 2048]
    - blk.8.ffn_down.weight: shape [2048, 11008]
    - blk.8.ffn_gate.weight: shape [11008, 2048]
    - blk.8.ffn_up.weight: shape [11008, 2048]
    - blk.9.attn_k.weight: shape [512, 2048]
    - blk.9.attn_output.weight: shape [2048, 2048]
    - blk.9.attn_q.weight: shape [2048, 2048]
    - blk.9.ffn_gate.weight: shape [11008, 2048]
    - blk.9.ffn_up.weight: shape [11008, 2048]

  F32 Tensors (not quantized):
    - blk.0.attn_norm.weight: shape [2048]
    - blk.0.ffn_norm.weight: shape [2048]
    - blk.1.attn_norm.weight: shape [2048]
    - blk.1.ffn_norm.weight: shape [2048]
    - blk.10.attn_norm.weight: shape [2048]
    - blk.10.ffn_norm.weight: shape [2048]
    - blk.11.attn_norm.weight: shape [2048]
    - blk.11.ffn_norm.weight: shape [2048]
    - blk.12.attn_norm.weight: shape [2048]
    - blk.12.ffn_norm.weight: shape [2048]
    - blk.13.attn_norm.weight: shape [2048]
    - blk.13.ffn_norm.weight: shape [2048]
    - blk.14.attn_norm.weight: shape [2048]
    - blk.14.ffn_norm.weight: shape [2048]
    - blk.15.attn_norm.weight: shape [2048]
    - blk.15.ffn_norm.weight: shape [2048]
    - blk.16.attn_norm.weight: shape [2048]
    - blk.16.ffn_norm.weight: shape [2048]
    - blk.17.attn_norm.weight: shape [2048]
    - blk.17.ffn_norm.weight: shape [2048]
    - blk.18.attn_norm.weight: shape [2048]
    - blk.18.ffn_norm.weight: shape [2048]
    - blk.19.attn_norm.weight: shape [2048]
    - blk.19.ffn_norm.weight: shape [2048]
    - blk.2.attn_norm.weight: shape [2048]
    - blk.2.ffn_norm.weight: shape [2048]
    - blk.20.attn_norm.weight: shape [2048]
    - blk.20.ffn_norm.weight: shape [2048]
    - blk.21.attn_norm.weight: shape [2048]
    - blk.21.ffn_norm.weight: shape [2048]
    - blk.22.attn_norm.weight: shape [2048]
    - blk.22.ffn_norm.weight: shape [2048]
    - blk.23.attn_norm.weight: shape [2048]
    - blk.23.ffn_norm.weight: shape [2048]
    - blk.24.attn_norm.weight: shape [2048]
    - blk.24.ffn_norm.weight: shape [2048]
    - blk.25.attn_norm.weight: shape [2048]
    - blk.25.ffn_norm.weight: shape [2048]
    - blk.26.attn_norm.weight: shape [2048]
    - blk.26.ffn_norm.weight: shape [2048]
    - blk.27.attn_norm.weight: shape [2048]
    - blk.27.ffn_norm.weight: shape [2048]
    - blk.28.attn_norm.weight: shape [2048]
    - blk.28.ffn_norm.weight: shape [2048]
    - blk.29.attn_norm.weight: shape [2048]
    - blk.29.ffn_norm.weight: shape [2048]
    - blk.3.attn_norm.weight: shape [2048]
    - blk.3.ffn_norm.weight: shape [2048]
    - blk.30.attn_norm.weight: shape [2048]
    - blk.30.ffn_norm.weight: shape [2048]
    - blk.31.attn_norm.weight: shape [2048]
    - blk.31.ffn_norm.weight: shape [2048]
    - blk.32.attn_norm.weight: shape [2048]
    - blk.32.ffn_norm.weight: shape [2048]
    - blk.33.attn_norm.weight: shape [2048]
    - blk.33.ffn_norm.weight: shape [2048]
    - blk.34.attn_norm.weight: shape [2048]
    - blk.34.ffn_norm.weight: shape [2048]
    - blk.35.attn_norm.weight: shape [2048]
    - blk.35.ffn_norm.weight: shape [2048]
    - blk.4.attn_norm.weight: shape [2048]
    - blk.4.ffn_norm.weight: shape [2048]
    - blk.5.attn_norm.weight: shape [2048]
    - blk.5.ffn_norm.weight: shape [2048]
    - blk.6.attn_norm.weight: shape [2048]
    - blk.6.ffn_norm.weight: shape [2048]
    - blk.7.attn_norm.weight: shape [2048]
    - blk.7.ffn_norm.weight: shape [2048]
    - blk.8.attn_norm.weight: shape [2048]
    - blk.8.ffn_norm.weight: shape [2048]
    - blk.9.attn_norm.weight: shape [2048]
    - blk.9.ffn_norm.weight: shape [2048]
    - output_norm.weight: shape [2048]

  Other Tensor Types:
    - blk.0.attn_v.weight: Q6K shape [512, 2048]
    - blk.0.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.1.attn_v.weight: Q6K shape [512, 2048]
    - blk.1.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.12.attn_v.weight: Q6K shape [512, 2048]
    - blk.12.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.15.attn_v.weight: Q6K shape [512, 2048]
    - blk.15.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.18.attn_v.weight: Q6K shape [512, 2048]
    - blk.18.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.2.attn_v.weight: Q6K shape [512, 2048]
    - blk.2.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.21.attn_v.weight: Q6K shape [512, 2048]
    - blk.21.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.24.attn_v.weight: Q6K shape [512, 2048]
    - blk.24.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.27.attn_v.weight: Q6K shape [512, 2048]
    - blk.27.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.3.attn_v.weight: Q6K shape [512, 2048]
    - blk.3.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.30.attn_v.weight: Q6K shape [512, 2048]
    - blk.30.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.31.attn_v.weight: Q6K shape [512, 2048]
    - blk.31.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.32.attn_v.weight: Q6K shape [512, 2048]
    - blk.32.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.33.attn_v.weight: Q6K shape [512, 2048]
    - blk.33.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.34.attn_v.weight: Q6K shape [512, 2048]
    - blk.34.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.35.attn_v.weight: Q6K shape [512, 2048]
    - blk.35.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.6.attn_v.weight: Q6K shape [512, 2048]
    - blk.6.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.9.attn_v.weight: Q6K shape [512, 2048]
    - blk.9.ffn_down.weight: Q6K shape [2048, 11008]
    - token_embd.weight: Q6K shape [128256, 2048]

🦙 SMOLLM3 METADATA
----------------------------------------
  smollm3.vocab_size: 128256
  smollm3.attention.head_count: 16
  smollm3.rope.dimension_count: 128
  smollm3.block_count: 36
  smollm3.rope.freq_base: 5000000
  smollm3.context_length: 65536
  smollm3.feed_forward_length: 11008
  smollm3.attention.layer_norm_rms_epsilon: 0.000001
  smollm3.embedding_length: 2048
  smollm3.attention.head_count_kv: 4

🦙 LLAMA METADATA STATUS
----------------------------------------

❌ Missing Llama keys:
  - llama.attention.head_count
  - llama.attention.head_count_kv
  - llama.block_count
  - llama.context_length
  - llama.embedding_length
  - llama.feed_forward_length
  - llama.vocab_size
  - llama.rope.freq_base
  - llama.rope.dimension_count
  - llama.attention.layer_norm_rms_epsilon

🔄 SUGGESTED MAPPINGS
----------------------------------------
  smollm3.attention.head_count → llama.attention.head_count
  smollm3.attention.head_count_kv → llama.attention.head_count_kv
  smollm3.block_count → llama.block_count
  smollm3.context_length → llama.context_length
  smollm3.embedding_length → llama.embedding_length
  smollm3.feed_forward_length → llama.feed_forward_length
  smollm3.vocab_size → llama.vocab_size
  smollm3.rope.dimension_count → llama.rope.dimension_count
  smollm3.attention.layer_norm_rms_epsilon → llama.attention.layer_norm_rms_epsilon

📝 SUMMARY
----------------------------------------
✅ Valid GGUF file
✅ 326 tensors found (216 quantized, 73 F32)
⚠️  10 Llama metadata keys need mapping

================================================================================

⚠️  Action required: Metadata mapping needed for Llama compatibility

Key Findings
Model Architecture

Architecture: SmolLM3 (needs mapping to Llama format)
Hidden size: 2048 (not 3072 as initially thought)
Intermediate size: 11008 (not 8192)
Layers: 36 ✓
Attention heads: 16 ✓
KV heads: 4 (GQA 4:1 ratio) ✓
Vocab size: 128256 ✓
Context length: 65536 (not 131072, but still substantial)
RoPE freq base: 5000000 (not 1000000)

Quantization Distribution

216 Q4_K tensors: Most attention and FFN weights (use QMatMul)
37 Q6K tensors: Some V weights, FFN down weights, and embeddings
73 F32 tensors: All normalization weights (not quantized)

Important Observations

Mixed quantization: The model uses both Q4_K and Q6K quantization, not just Q4_K_M
Pattern in layers: Some layers have Q6K for attn_v and ffn_down weights (layers 0,1,2,3,6,9,12,15,18,21,24,27,30-35)
Token embeddings: Quantized with Q6K, not F32
All metadata needs mapping: SmolLM3 format → Llama format

Next Steps for Implementation
The metadata mapping in your gguf_loader.rs needs updating with the correct values:
rust// Correct values from the inspection:
("smollm3.embedding_length", "llama.embedding_length", Value::U32(2048))  // not 3072
("smollm3.feed_forward_length", "llama.feed_forward_length", Value::U32(11008))  // not 8192
("smollm3.context_length", "llama.context_length", Value::U32(65536))  // not 131072
("smollm3.rope.freq_base", "llama.rope.freq_base", Value::F32(5000000.0))  // not 1000000.0
The model is ready to load once the metadata mapping is updated with these correct values!