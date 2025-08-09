//! SmolLM3-specific configuration extending Llama config

use candle_core::Result;
use candle_core::quantized::gguf_file::Content;

/// SmolLM3-specific configuration extending Llama config
#[derive(Clone, Debug)]
pub struct SmolLM3Config {
    /// Base Llama configuration
    pub base: candle_transformers::models::quantized_llama::Config,
    /// Enable thinking mode
    pub enable_thinking: bool,
    /// Thinking token IDs
    pub think_token_id: u32,
    pub think_end_token_id: u32,
    /// NoPE layer indices (every 4th layer starting from 3)
    pub nope_layer_indices: Vec<usize>,
}

impl SmolLM3Config {
    /// Create SmolLM3-3B configuration
    pub fn smollm3_3b() -> Self {
        let base = candle_transformers::models::quantized_llama::Config {
            hidden_size: 3072,
            intermediate_size: 8192,
            vocab_size: 128256,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8, // GQA with 4:1 ratio
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0, // Extended context RoPE
            use_flash_attn: false, // Not yet supported in Candle
            head_dim: 96, // 3072 / 32
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: 131072,
        };
        
        // NoPE layers: indices 3, 7, 11, 15, 19, 23, 27, 31, 35
        let nope_layer_indices = (0..36).filter(|i| i % 4 == 3).collect();
        
        Self {
            base,
            enable_thinking: true,
            think_token_id: 128002,
            think_end_token_id: 128003,
            nope_layer_indices,
        }
    }
    
    /// Create config from GGUF metadata
    pub fn from_gguf(content: &Content) -> Result<Self> {
        use super::gguf_loader::{get_metadata_u32, get_metadata_f32};
        
        let hidden_size = get_metadata_u32(content, &["llama.embedding_length", "hidden_size"])
            .unwrap_or(3072) as usize;
        let intermediate_size = get_metadata_u32(content, &["llama.feed_forward_length", "intermediate_size"])
            .unwrap_or(8192) as usize;
        let vocab_size = get_metadata_u32(content, &["llama.vocab_size", "vocab_size"])
            .unwrap_or(128256) as usize;
        let num_hidden_layers = get_metadata_u32(content, &["llama.block_count", "n_layer"])
            .unwrap_or(36) as usize;
        let num_attention_heads = get_metadata_u32(content, &["llama.attention.head_count", "n_head"])
            .unwrap_or(32) as usize;
        let num_key_value_heads = get_metadata_u32(content, &["llama.attention.head_count_kv", "n_kv_head"])
            .unwrap_or(8) as usize;
        let rms_norm_eps = get_metadata_f32(content, &["llama.attention.layer_norm_rms_epsilon", "rms_norm_eps"])
            .unwrap_or(1e-5) as f64;
        let rope_theta = get_metadata_f32(content, &["llama.rope.freq_base", "rope_theta"])
            .unwrap_or(1000000.0) as f32;
        
        let base = candle_transformers::models::quantized_llama::Config {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta,
            use_flash_attn: false,
            head_dim: hidden_size / num_attention_heads,
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: 131072,
        };
        
        // NoPE layers for SmolLM3
        let nope_layer_indices = (0..num_hidden_layers).filter(|i| i % 4 == 3).collect();
        
        Ok(Self {
            base,
            enable_thinking: true,
            think_token_id: 128002,
            think_end_token_id: 128003,
            nope_layer_indices,
        })
    }
}

impl Default for SmolLM3Config {
    fn default() -> Self {
        Self::smollm3_3b()
    }
}
