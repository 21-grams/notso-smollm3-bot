//! SmolLM3-specific configuration extending Llama config

use candle_core::Result;
use candle_core::quantized::gguf_file::Content;
use super::ThinkingTokens;

/// Base Llama-style configuration for SmolLM3
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub use_flash_attn: bool,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<()>,  // Simplified for now
    pub max_position_embeddings: usize,
}

/// SmolLM3-specific configuration extending Llama config
#[derive(Clone, Debug)]
pub struct SmolLM3Config {
    /// Base Llama configuration
    pub base: LlamaConfig,
    /// Enable thinking mode
    pub enable_thinking: bool,
    /// Thinking token IDs
    pub thinking_tokens: ThinkingTokens,
    /// Thinking token IDs (legacy compatibility)
    pub think_token_id: u32,
    pub think_end_token_id: u32,
    /// NoPE layer indices (every 4th layer starting from 3)
    pub nope_layer_indices: Vec<usize>,
    /// Alias for compatibility
    pub nope_layers: Vec<usize>,
}

impl SmolLM3Config {
    /// Create SmolLM3-3B configuration
    pub fn smollm3_3b() -> Self {
        let base = LlamaConfig {
            hidden_size: 2048,
            intermediate_size: 11008,
            vocab_size: 128256,
            num_hidden_layers: 36,
            num_attention_heads: 16,
            num_key_value_heads: 4, // GQA with 4:1 ratio
            rms_norm_eps: 1e-5,
            rope_theta: 5000000.0, // Extended context RoPE
            use_flash_attn: false, // Not yet supported in Candle
            head_dim: 128, // 2048 / 16
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: 65536,
        };
        
        // NoPE layers: indices 3, 7, 11, 15, 19, 23, 27, 31, 35
        let nope_layer_indices: Vec<usize> = (0..36).filter(|i| i % 4 == 3).collect();
        let thinking_tokens = ThinkingTokens::default_smollm3();
        
        Self {
            base,
            enable_thinking: true,
            thinking_tokens: thinking_tokens.clone(),
            think_token_id: thinking_tokens.start,
            think_end_token_id: thinking_tokens.end,
            nope_layer_indices: nope_layer_indices.clone(),
            nope_layers: nope_layer_indices,
        }
    }
    
    /// Create config from GGUF metadata
    pub fn from_gguf(content: &Content) -> Result<Self> {
        use super::gguf_loader::{get_metadata_u32, get_metadata_f32};
        
        let hidden_size = get_metadata_u32(content, &["llama.embedding_length", "hidden_size"])
            .unwrap_or(2048) as usize;
        let intermediate_size = get_metadata_u32(content, &["llama.feed_forward_length", "intermediate_size"])
            .unwrap_or(11008) as usize;
        let vocab_size = get_metadata_u32(content, &["llama.vocab_size", "vocab_size"])
            .unwrap_or(128256) as usize;
        let num_hidden_layers = get_metadata_u32(content, &["llama.block_count", "n_layer"])
            .unwrap_or(36) as usize;
        let num_attention_heads = get_metadata_u32(content, &["llama.attention.head_count", "n_head"])
            .unwrap_or(16) as usize;
        let num_key_value_heads = get_metadata_u32(content, &["llama.attention.head_count_kv", "n_kv_head"])
            .unwrap_or(4) as usize;
        let rms_norm_eps = get_metadata_f32(content, &["llama.attention.layer_norm_rms_epsilon", "rms_norm_eps"])
            .unwrap_or(1e-5) as f64;
        let rope_theta = get_metadata_f32(content, &["llama.rope.freq_base", "rope_theta"])
            .unwrap_or(5000000.0) as f32;
        
        let base = LlamaConfig {
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
            max_position_embeddings: 65536,
        };
        
        // NoPE layers for SmolLM3
        let nope_layer_indices: Vec<usize> = (0..num_hidden_layers).filter(|i| i % 4 == 3).collect();
        let thinking_tokens = ThinkingTokens::default_smollm3();
        
        Ok(Self {
            base,
            enable_thinking: true,
            thinking_tokens: thinking_tokens.clone(),
            think_token_id: thinking_tokens.start,
            think_end_token_id: thinking_tokens.end,
            nope_layer_indices: nope_layer_indices.clone(),
            nope_layers: nope_layer_indices,
        })
    }
}

impl Default for SmolLM3Config {
    fn default() -> Self {
        Self::smollm3_3b()
    }
}
