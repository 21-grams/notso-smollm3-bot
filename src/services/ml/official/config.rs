use candle_transformers::models::llama::{Config as LlamaConfig, LlamaEosToks};
use serde::{Deserialize, Serialize};

/// Minimal config that matches what LlamaConfig actually has
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomLlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,  // Actually f32, not f64
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
}

impl From<CustomLlamaConfig> for LlamaConfig {
    fn from(custom: CustomLlamaConfig) -> Self {
        // Only set fields that actually exist in LlamaConfig
        LlamaConfig {
            vocab_size: custom.vocab_size,
            hidden_size: custom.hidden_size,
            num_hidden_layers: custom.num_hidden_layers,
            num_attention_heads: custom.num_attention_heads,
            num_key_value_heads: custom.num_key_value_heads,
            intermediate_size: custom.intermediate_size,
            rope_theta: custom.rope_theta,
            rms_norm_eps: custom.rms_norm_eps,
            max_position_embeddings: custom.max_position_embeddings,
            bos_token_id: Some(0),
            eos_token_id: Some(LlamaEosToks::Single(2)),
            use_flash_attn: false,
            tie_word_embeddings: false,
            rope_scaling: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmolLM3Config {
    /// Base config that can be serialized
    pub base: CustomLlamaConfig,
    
    /// SmolLM3-specific features (kept separate)
    pub nope_layers: Vec<usize>,
    pub thinking_tokens: ThinkingTokens,
    pub reasoning_mode: ReasoningMode,
    pub tool_calling: ToolCallingConfig,
    
    // Model behavior settings that aren't in LlamaConfig
    pub activation: ActivationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    Silu,
    Gelu,
    Relu,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::Silu
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTokens {
    pub start: u32,
    pub end: u32,
}

impl Default for ThinkingTokens {
    fn default() -> Self {
        Self {
            start: 128002,
            end: 128003,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReasoningMode {
    Think,
    NoThink,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallingConfig {
    pub enabled: bool,
    pub format: ToolFormat,
    pub available_tools: Vec<String>,
}

impl Default for ToolCallingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            format: ToolFormat::XML,
            available_tools: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ToolFormat {
    XML,
    Python,
    JSON,
}

impl Default for SmolLM3Config {
    fn default() -> Self {
        let base = CustomLlamaConfig {
            vocab_size: 128256,
            hidden_size: 2048,
            num_hidden_layers: 36,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            intermediate_size: 11008,
            rope_theta: 2000000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 65536,
        };
        
        Self {
            base,
            nope_layers: vec![3, 7, 11, 15, 19, 23, 27, 31, 35],
            thinking_tokens: ThinkingTokens::default(),
            reasoning_mode: ReasoningMode::Think,
            tool_calling: ToolCallingConfig::default(),
            activation: ActivationType::default(),
        }
    }
}

impl SmolLM3Config {
    pub fn to_llama_config(&self) -> LlamaConfig {
        self.base.clone().into()
    }
    
    pub fn is_nope_layer(&self, layer_idx: usize) -> bool {
        self.nope_layers.contains(&layer_idx)
    }
}
