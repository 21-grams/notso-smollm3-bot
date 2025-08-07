use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmolLM3Config {
    // Model architecture
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,  // For GQA
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    
    // SmolLM3 specific features
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub nope_layers: Vec<usize>,  // Layers without position encoding
    pub thinking_tokens: ThinkingTokens,
    pub tool_calling_enabled: bool,
    
    // Generation parameters
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,  // "yarn"
    pub factor: f32,           // 2.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTokens {
    pub start_id: u32,  // 128002 for <think>
    pub end_id: u32,    // 128003 for </think>
}

impl Default for SmolLM3Config {
    fn default() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 2048,
            n_layers: 36,
            n_heads: 16,
            n_kv_heads: 4,  // GQA 4:1 ratio
            intermediate_size: 11008,
            max_position_embeddings: 65536,
            
            rope_theta: 2_000_000.0,
            rope_scaling: RopeScaling {
                scaling_type: "yarn".to_string(),
                factor: 2.0,
            },
            nope_layers: vec![3, 7, 11, 15, 19, 23, 27, 31, 35],
            thinking_tokens: ThinkingTokens {
                start_id: 128002,
                end_id: 128003,
            },
            tool_calling_enabled: true,
            
            temperature: 0.7,
            top_p: 0.9,
            max_new_tokens: 512,
        }
    }
}

impl SmolLM3Config {
    pub fn is_nope_layer(&self, layer_idx: usize) -> bool {
        self.nope_layers.contains(&layer_idx)
    }
    
    pub fn gqa_ratio(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}
