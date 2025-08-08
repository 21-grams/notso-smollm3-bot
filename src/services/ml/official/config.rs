//! SmolLM3 configuration extending official LlamaConfig

use candle_transformers::models::quantized_llama::LlamaConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmolLM3Config {
    /// Base LlamaConfig for official compatibility
    pub base: LlamaConfig,
    
    /// SmolLM3-specific features
    pub nope_layers: Vec<usize>,              // [3,7,11,15,19,23,27,31,35]
    pub thinking_tokens: ThinkingTokens,      // <think>, </think>
    pub reasoning_mode: ReasoningMode,        // think/no_think
    pub tool_calling: ToolCallingConfig,      // Tool execution settings
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTokens {
    pub start: u32,        // <think> = 128002
    pub end: u32,          // </think> = 128003
    pub start_id: u32,     // Alias for compatibility
    pub end_id: u32,       // Alias for compatibility
}

impl ThinkingTokens {
    pub fn new() -> Self {
        Self {
            start: 128002,
            end: 128003,
            start_id: 128002,
            end_id: 128003,
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
        // Create LlamaConfig manually since it doesn't have Default
        let base = LlamaConfig {
            vocab_size: 128256,
            hidden_size: 2048,
            n_layer: 36,
            n_head: 16,
            n_kv_head: 4,           // GQA 4:1 ratio
            intermediate_size: 11008,
            max_seq_len: 65536,     // Extended context
            rope_theta: 2000000.0,  // 2M theta
            rms_norm_eps: 1e-5,
        };
        
        Self {
            base,
            nope_layers: vec![3, 7, 11, 15, 19, 23, 27, 31, 35],
            thinking_tokens: ThinkingTokens::new(),
            reasoning_mode: ReasoningMode::Think,
            tool_calling: ToolCallingConfig::default(),
        }
    }
}

impl SmolLM3Config {
    /// Convert to official LlamaConfig
    pub fn to_llama_config(&self) -> LlamaConfig {
        self.base.clone()
    }
    
    /// Check if layer should skip RoPE (NoPE layer)
    pub fn is_nope_layer(&self, layer_idx: usize) -> bool {
        self.nope_layers.contains(&layer_idx)
    }
    
    /// Create config for SmolLM3-3B
    pub fn smollm3_3b() -> Self {
        Self::default()
    }
}
