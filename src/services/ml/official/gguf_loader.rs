//! Custom GGUF loader for SmolLM3 that handles missing metadata gracefully

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{Device, Result};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::collections::HashMap;
use std::path::Path;
use std::io::{Read, Seek, SeekFrom};

/// Inspect GGUF file metadata and return a detailed report
pub fn inspect_gguf<P: AsRef<Path>>(path: P) -> Result<GgufInspectionReport> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let content = gguf_file::Content::read(&mut file)?;
    
    let mut report = GgufInspectionReport {
        valid: true,
        tensor_count: content.tensor_infos.len(),
        metadata_count: content.metadata.len(),
        architecture: None,
        has_llama_metadata: false,
        missing_keys: Vec::new(),
        found_keys: HashMap::new(),
        all_metadata_keys: Vec::new(),
    };
    
    // Check architecture
    if let Some(arch) = content.metadata.get("general.architecture") {
        report.architecture = Some(format!("{:?}", arch));
    }
    
    // Check for required Llama keys
    let required_keys = [
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.block_count",
        "llama.context_length",
        "llama.embedding_length",
    ];
    
    for key in &required_keys {
        if let Some(value) = content.metadata.get(*key) {
            report.found_keys.insert(key.to_string(), format!("{:?}", value));
        } else {
            report.missing_keys.push(key.to_string());
        }
    }
    
    report.has_llama_metadata = report.missing_keys.is_empty();
    
    // Collect all metadata keys
    report.all_metadata_keys = content.metadata.keys().cloned().collect();
    report.all_metadata_keys.sort();
    
    Ok(report)
}

/// Report from GGUF inspection
#[derive(Debug, Clone)]
pub struct GgufInspectionReport {
    pub valid: bool,
    pub tensor_count: usize,
    pub metadata_count: usize,
    pub architecture: Option<String>,
    pub has_llama_metadata: bool,
    pub missing_keys: Vec<String>,
    pub found_keys: HashMap<String, String>,
    pub all_metadata_keys: Vec<String>,
}

impl GgufInspectionReport {
    /// Print a formatted report
    pub fn print_report(&self) {
        println!("🔍 GGUF Inspection Report");
        println!("=".repeat(50));
        println!("✅ Valid GGUF: {}", self.valid);
        println!("📊 Tensors: {}", self.tensor_count);
        println!("📋 Metadata entries: {}", self.metadata_count);
        
        if let Some(arch) = &self.architecture {
            println!("🏗️ Architecture: {}", arch);
        }
        
        println!("\n🦙 Llama Metadata: {}", 
                 if self.has_llama_metadata { "✅ Complete" } else { "❌ Missing" });
        
        if !self.missing_keys.is_empty() {
            println!("\n❌ Missing required keys:");
            for key in &self.missing_keys {
                println!("  - {}", key);
            }
        }
        
        if !self.found_keys.is_empty() {
            println!("\n✅ Found keys:");
            for (key, value) in &self.found_keys {
                println!("  - {}: {}", key, value);
            }
        }
        
        if !self.all_metadata_keys.is_empty() {
            println!("\n📋 Available metadata keys (first 20):");
            for key in self.all_metadata_keys.iter().take(20) {
                println!("  - {}", key);
            }
        }
    }
}

/// Custom loader that works with SmolLM3 GGUF files
pub struct SmolLM3GgufLoader;

impl SmolLM3GgufLoader {
    /// Load GGUF file with SmolLM3-specific handling
    pub fn load_gguf<P: AsRef<Path>>(
        path: P,
        device: &Device,
    ) -> Result<(gguf_file::Content, HashMap<String, QTensor>)> {
        tracing::info!("📦 Loading SmolLM3 GGUF file");
        
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)?;
        
        // Log what we found
        tracing::info!("📊 GGUF content: {} tensors, {} metadata entries",
                      content.tensor_infos.len(),
                      content.metadata.len());
        
        // Check architecture
        if let Some(arch) = content.metadata.get("general.architecture") {
            tracing::info!("🏗️ Architecture: {:?}", arch);
        }
        
        // Extract model dimensions from metadata with fallbacks
        let n_heads = Self::get_metadata_u32(&content, &[
            "llama.attention.head_count",
            "llama.attention_head_count",  // Alternative key
            "attention.head_count",
        ]).unwrap_or(32);  // SmolLM3-3B default
        
        let n_kv_heads = Self::get_metadata_u32(&content, &[
            "llama.attention.head_count_kv",
            "llama.attention_head_count_kv",
            "attention.head_count_kv",
        ]).unwrap_or(8);  // SmolLM3-3B uses GQA with 8 KV heads
        
        let n_layers = Self::get_metadata_u32(&content, &[
            "llama.block_count",
            "llama.layer_count",
            "block_count",
        ]).unwrap_or(36);  // SmolLM3-3B has 36 layers
        
        let hidden_size = Self::get_metadata_u32(&content, &[
            "llama.embedding_length",
            "llama.hidden_size",
            "hidden_size",
        ]).unwrap_or(3072);  // SmolLM3-3B hidden size
        
        let vocab_size = Self::get_metadata_u32(&content, &[
            "llama.vocab_size",
            "tokenizer.ggml.model.vocab_size",
            "vocab_size",
        ]).unwrap_or(50304);  // SmolLM3 vocab size
        
        tracing::info!("📐 Model dimensions: heads={}, kv_heads={}, layers={}, hidden={}, vocab={}",
                      n_heads, n_kv_heads, n_layers, hidden_size, vocab_size);
        
        // Load tensors
        let mut tensors = HashMap::new();
        let tensor_start_offset = content.tensor_data_offset;
        file.seek(SeekFrom::Start(tensor_start_offset))?;
        
        // Note: QTensor loading from GGUF requires the correct implementation
        // For now, we'll return empty tensors map and focus on metadata
        // Real implementation would iterate through tensor_infos and load each
        
        tracing::info!("✅ Loaded {} tensors", tensors.len());
        
        Ok((content, tensors))
    }
    
    /// Try to get metadata with multiple possible keys
    fn get_metadata_u32(content: &gguf_file::Content, keys: &[&str]) -> Option<u32> {
        for key in keys {
            if let Some(gguf_file::Value::U32(val)) = content.metadata.get(*key) {
                return Some(*val);
            }
            if let Some(gguf_file::Value::U64(val)) = content.metadata.get(*key) {
                return Some(*val as u32);
            }
            if let Some(gguf_file::Value::I32(val)) = content.metadata.get(*key) {
                return Some(*val as u32);
            }
        }
        None
    }
    
    /// Create a config from GGUF metadata
    pub fn config_from_gguf(content: &gguf_file::Content) -> super::super::config::SmolLM3Config {
        use super::super::config::SmolLM3Config;
        
        let mut config = SmolLM3Config::default();
        
        // Update config from metadata if available
        if let Some(val) = Self::get_metadata_u32(content, &["llama.attention.head_count", "attention.head_count"]) {
            config.n_heads = val as usize;
        }
        if let Some(val) = Self::get_metadata_u32(content, &["llama.attention.head_count_kv", "attention.head_count_kv"]) {
            config.n_kv_heads = val as usize;
        }
        if let Some(val) = Self::get_metadata_u32(content, &["llama.block_count", "block_count"]) {
            config.n_layers = val as usize;
        }
        if let Some(val) = Self::get_metadata_u32(content, &["llama.embedding_length", "hidden_size"]) {
            config.hidden_size = val as usize;
        }
        if let Some(val) = Self::get_metadata_u32(content, &["llama.vocab_size", "vocab_size"]) {
            config.vocab_size = val as usize;
        }
        if let Some(val) = Self::get_metadata_u32(content, &["llama.context_length", "context_length"]) {
            config.max_position_embeddings = val as usize;
        }
        
        config
    }
}

/// Alternative: Try to use the candle-transformers quantized model directly
pub fn try_load_as_quantized_llama<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<ModelWeights> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let content = gguf_file::Content::read(&mut file)?;
    
    // Try to create ModelWeights, but handle missing metadata
    // This might fail if required keys are missing
    ModelWeights::from_gguf(content, &mut file, device)
}
