//! Enhanced GGUF loader that maps SmolLM3 metadata to Llama format
//! This allows us to use candle_transformers::models::quantized_llama directly

use candle_core::quantized::gguf_file;
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{Device, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Comprehensive metadata mapping for SmolLM3 to Llama compatibility
pub fn map_smollm3_to_llama_metadata(content: &mut Content) {
    tracing::info!("🔄 Mapping SmolLM3 metadata to Llama format");
    
    // Define all possible SmolLM3 key variations and their Llama equivalents
    let key_mappings = [
        // Attention configuration
        (vec!["smollm3.attention.head_count", "smollm.attention_head_count", "attention.n_heads"], 
         "llama.attention.head_count", Value::U32(16)),
        (vec!["smollm3.attention.head_count_kv", "smollm.attention_head_count_kv", "attention.n_kv_heads"], 
         "llama.attention.head_count_kv", Value::U32(4)),
        
        // Model architecture
        (vec!["smollm3.block_count", "smollm.n_layer", "n_layers", "block_count"], 
         "llama.block_count", Value::U32(36)),
        (vec!["smollm3.context_length", "smollm.max_seq_len", "max_position_embeddings"], 
         "llama.context_length", Value::U32(65536)),
        (vec!["smollm3.embedding_length", "smollm.hidden_size", "hidden_size", "d_model"], 
         "llama.embedding_length", Value::U32(2048)),
        (vec!["smollm3.feed_forward_length", "smollm.intermediate_size", "intermediate_size"], 
         "llama.feed_forward_length", Value::U32(11008)),
        
        // Vocabulary
        (vec!["smollm3.vocab_size", "tokenizer.ggml.model.vocab_size", "vocab_size"], 
         "llama.vocab_size", Value::U32(128256)),
        
        // RoPE configuration
        (vec!["smollm3.rope.freq_base", "smollm3.rope.theta", "rope.theta", "rope_theta"], 
         "llama.rope.freq_base", Value::F32(5000000.0)),
        (vec!["smollm3.rope.dimension_count", "rope.dim", "rope_dim"], 
         "llama.rope.dimension_count", Value::U32(128)),
        
        // Normalization
        (vec!["smollm3.attention.layer_norm_rms_epsilon", "rms_norm_eps", "norm_eps"], 
         "llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
    ];
    
    // Track what we find and map
    let mut mapped_keys = Vec::new();
    let mut found_keys = HashMap::new();
    
    for (possible_keys, llama_key, default_value) in &key_mappings {
        // Check if Llama key already exists
        if content.metadata.contains_key(*llama_key) {
            continue;
        }
        
        // Try to find value from SmolLM3 keys
        let mut value_found = None;
        for smollm_key in possible_keys {
            if let Some(value) = content.metadata.get(*smollm_key) {
                value_found = Some(value.clone());
                found_keys.insert(smollm_key.to_string(), llama_key.to_string());
                break;
            }
        }
        
        // Insert found value or default
        if let Some(value) = value_found {
            content.metadata.insert(llama_key.to_string(), value);
            mapped_keys.push(format!("{} (from SmolLM3)", llama_key));
        } else {
            content.metadata.insert(llama_key.to_string(), default_value.clone());
            mapped_keys.push(format!("{} (default)", llama_key));
        }
    }
    
    // Special handling for architecture type
    if !content.metadata.contains_key("general.architecture") {
        content.metadata.insert("general.architecture".to_string(), Value::String("llama".to_string()));
    }
    
    // Add rope scaling if not present (SmolLM3 uses extended context)
    if !content.metadata.contains_key("llama.rope.scaling.type") {
        content.metadata.insert("llama.rope.scaling.type".to_string(), Value::String("linear".to_string()));
        content.metadata.insert("llama.rope.scaling.factor".to_string(), Value::F32(2.0));
    }
    
    tracing::info!("✅ Mapped {} metadata keys", mapped_keys.len());
    for key in &mapped_keys {
        tracing::debug!("  - {}", key);
    }
    
    if !found_keys.is_empty() {
        tracing::info!("📋 SmolLM3 to Llama key mappings:");
        for (from, to) in &found_keys {
            tracing::debug!("  {} -> {}", from, to);
        }
    }
}

/// Load GGUF file with SmolLM3 compatibility
pub fn load_smollm3_gguf<P: AsRef<Path>>(path: P, _device: &Device) -> Result<Content> {
    let path = path.as_ref();
    tracing::info!("📂 Loading GGUF from: {:?}", path);
    
    // Validate file exists and has reasonable size
    if !path.exists() {
        candle_core::bail!("GGUF file not found: {:?}", path);
    }
    
    let metadata = std::fs::metadata(path)?;
    let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    tracing::info!("📦 GGUF file size: {:.2} GB", size_gb);
    
    if size_gb < 0.5 {
        tracing::warn!("⚠️ GGUF file seems small for a 3B model: {:.2} GB", size_gb);
    }
    
    // Read GGUF content
    let mut file = File::open(path)?;
    let mut content = Content::read(&mut file)?;
    
    tracing::info!("📊 GGUF content: {} tensors, {} metadata entries",
                  content.tensor_infos.len(),
                  content.metadata.len());
    
    // Log original architecture if present
    if let Some(arch) = content.metadata.get("general.architecture") {
        tracing::info!("🏗️ Original architecture: {:?}", arch);
    }
    
    // Apply SmolLM3 to Llama metadata mapping
    map_smollm3_to_llama_metadata(&mut content);
    
    // Validate critical metadata after mapping
    validate_llama_metadata(&content)?;
    
    Ok(content)
}

/// Validate that all required Llama metadata is present
fn validate_llama_metadata(content: &Content) -> Result<()> {
    let required_keys = [
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.block_count",
        "llama.embedding_length",
        "llama.vocab_size",
    ];
    
    let mut missing = Vec::new();
    for key in &required_keys {
        if !content.metadata.contains_key(*key) {
            missing.push(*key);
        }
    }
    
    if !missing.is_empty() {
        candle_core::bail!("Missing required Llama metadata after mapping: {:?}", missing);
    }
    
    tracing::info!("✅ All required Llama metadata present");
    Ok(())
}

/// Get a metadata value as u32, trying multiple key variations
pub fn get_metadata_u32(content: &Content, keys: &[&str]) -> Option<u32> {
    for key in keys {
        match content.metadata.get(*key) {
            Some(Value::U32(val)) => return Some(*val),
            Some(Value::U64(val)) => return Some(*val as u32),
            Some(Value::I32(val)) => return Some(*val as u32),
            Some(Value::U16(val)) => return Some(*val as u32),
            Some(Value::U8(val)) => return Some(*val as u32),
            _ => continue,
        }
    }
    None
}

/// Get a metadata value as f32, trying multiple key variations  
pub fn get_metadata_f32(content: &Content, keys: &[&str]) -> Option<f32> {
    for key in keys {
        match content.metadata.get(*key) {
            Some(Value::F32(val)) => return Some(*val),
            Some(Value::F64(val)) => return Some(*val as f32),
            _ => continue,
        }
    }
    None
}

// Legacy functions for backward compatibility

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
        println!("{}", "=".repeat(50));
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



/// Custom loader that works with SmolLM3 GGUF files (legacy)
pub struct SmolLM3GgufLoader;

impl SmolLM3GgufLoader {
    /// Load GGUF file with SmolLM3-specific handling
    pub fn load_gguf<P: AsRef<Path>>(
        path: P,
        device: &Device,
    ) -> Result<(gguf_file::Content, HashMap<String, candle_core::quantized::QTensor>)> {
        let content = load_smollm3_gguf(path, device)?;
        let tensors = HashMap::new(); // Tensors will be loaded by ModelWeights::from_gguf
        Ok((content, tensors))
    }
    
    /// Try to get metadata with multiple possible keys
    fn get_metadata_u32(content: &gguf_file::Content, keys: &[&str]) -> Option<u32> {
        get_metadata_u32(content, keys)
    }
    
    /// Create a config from GGUF metadata
    pub fn config_from_gguf(content: &gguf_file::Content) -> super::config::SmolLM3Config {
        super::config::SmolLM3Config::from_gguf(content).unwrap_or_else(|_| {
            super::config::SmolLM3Config::default()
        })
    }
}

/// Alternative: Try to use the candle-transformers quantized model directly
pub fn try_load_as_quantized_llama<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<candle_transformers::models::quantized_llama::ModelWeights> {
    let content = load_smollm3_gguf(&path, device)?;
    let mut file = std::fs::File::open(path)?;
    candle_transformers::models::quantized_llama::ModelWeights::from_gguf(content, &mut file, device)
}
