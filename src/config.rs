use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Server configuration
    pub host: String,
    pub port: u16,
    
    // Model configuration
    pub model_path: String,
    pub tokenizer_path: String,
    pub device: DeviceConfig,
    
    // SmolLM3 specific
    pub max_context_length: usize,
    pub thinking_mode_default: bool,
    pub temperature: f32,
    pub top_p: f32,
    
    // Performance
    pub batch_size: usize,
    pub max_concurrent_sessions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),
    Metal,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Config {
            host: env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: env::var("PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()?,
            
            model_path: env::var("MODEL_PATH")
                .unwrap_or_else(|_| "models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf".to_string()),
            tokenizer_path: env::var("TOKENIZER_PATH")
                .unwrap_or_else(|_| "models/tokenizer.json".to_string()),
            device: DeviceConfig::Cpu,
            
            max_context_length: 65536,
            thinking_mode_default: true,
            temperature: 0.7,
            top_p: 0.9,
            
            batch_size: 1,
            max_concurrent_sessions: 10,
        })
    }
    
    pub fn to_candle_device(&self) -> candle_core::Device {
        match &self.device {
            DeviceConfig::Cpu => candle_core::Device::Cpu,
            DeviceConfig::Cuda(idx) => {
                candle_core::Device::new_cuda(*idx).unwrap_or(candle_core::Device::Cpu)
            }
            DeviceConfig::Metal => {
                candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
            }
        }
    }
}
