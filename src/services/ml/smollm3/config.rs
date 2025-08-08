//! SmolLM3 configuration with complete architectural parameters

use serde::{Deserialize, Serialize};
use candle_core::Device;

fn default_device() -> Device {
    Device::Cpu
}

/// Complete SmolLM3 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmolLM3FullConfig {
    // Model architecture
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    
    // RoPE configuration
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    
    // Model parameters
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    
    // Generation parameters
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    
    // Paths
    pub model_path: String,
    pub tokenizer_path: String,
    pub tokenizer_config_path: String,
    
    // Device configuration
    #[serde(skip, default = "default_device")]
    pub device: Device,
    
    // Advanced features
    pub enable_thinking: bool,
    pub enable_tools: bool,
    pub enable_streaming: bool,
    
    // Performance settings
    pub use_flash_attention: bool,
    pub memory_efficient: bool,
    pub kv_cache_size: usize,
    pub top_k: Option<usize>,
    pub max_context_tokens: usize,
    pub use_kv_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub factor: f64,
    #[serde(rename = "type")]
    pub scaling_type: String,
    pub original_max_position_embeddings: Option<usize>,
}

impl Default for SmolLM3FullConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["SmolLM3ForCausalLM".to_string()],
            vocab_size: 128256,
            hidden_size: 2048,
            intermediate_size: 11008,
            num_hidden_layers: 36,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 128,
            max_position_embeddings: 32768,
            
            rope_theta: 2000000.0,
            rope_scaling: Some(RopeScaling {
                factor: 2.0,
                scaling_type: "yarn".to_string(),
                original_max_position_embeddings: Some(32768),
            }),
            
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            attention_bias: false,
            attention_dropout: 0.0,
            
            temperature: 0.8,
            top_p: 0.9,
            max_tokens: 256,
            
            model_path: "models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf".to_string(),
            tokenizer_path: "models/tokenizer.json".to_string(),
            tokenizer_config_path: "models/tokenizer_config.json".to_string(),
            
            device: Device::Cpu,
            
            enable_thinking: true,
            enable_tools: false,
            enable_streaming: true,
            
            use_flash_attention: false,
            memory_efficient: true,
            kv_cache_size: 4096,
            top_k: Some(50),
            max_context_tokens: 32768,
            use_kv_cache: true,
        }
    }
}

// Also export simplified config for compatibility
pub use SmolLM3FullConfig as SmolLM3Config;
