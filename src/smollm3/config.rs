//! SmolLM3 model configuration

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmolLM3Config {
    // Model architecture
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    
    // Context and RoPE
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
    
    // SmolLM3 specific
    pub nope_layers: Vec<usize>,
    pub thinking_tokens: ThinkingTokens,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTokens {
    pub start_id: u32,
    pub end_id: u32,
}

impl Default for SmolLM3Config {
    fn default() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 2048,
            num_layers: 36,
            num_heads: 16,
            num_kv_heads: 4,  // GQA 4:1 ratio
            intermediate_size: 11008,
            max_position_embeddings: 65536,
            rope_theta: 2_000_000.0,
            rms_norm_eps: 1e-5,
            nope_layers: vec![3, 7, 11, 15, 19, 23, 27, 31, 35],
            thinking_tokens: ThinkingTokens {
                start_id: 128002,
                end_id: 128003,
            },
        }
    }
}

impl SmolLM3Config {
    pub fn is_nope_layer(&self, layer_idx: usize) -> bool {
        self.nope_layers.contains(&layer_idx)
    }
}
