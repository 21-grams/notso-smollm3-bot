/// GGUF Inspector Tool for SmolLM3
/// 
/// This tool reads the GGUF file and provides a comprehensive report of:
/// - All tensor names and their quantization types
/// - All metadata keys and values
/// - Missing Llama metadata keys that need mapping
/// - Tensor shape verification

use anyhow::{Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType};
use candle_core::quantized::gguf_file::{Content, Value};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Comprehensive GGUF inspection report
#[derive(Debug)]
struct GgufReport {
    // File info
    file_path: String,
    file_size_gb: f64,
    
    // Basic counts
    tensor_count: usize,
    metadata_count: usize,
    
    // Tensors grouped by quantization type
    q4_k_m_tensors: Vec<(String, Vec<u64>)>,
    f32_tensors: Vec<(String, Vec<u64>)>,
    other_tensors: Vec<(String, String, Vec<u64>)>, // (name, type, shape)
    
    // Metadata analysis
    architecture: Option<String>,
    vocab_size: Option<u32>,
    hidden_size: Option<u32>,
    n_layers: Option<u32>,
    
    // SmolLM3 specific metadata
    smollm3_metadata: HashMap<String, String>,
    
    // Llama metadata status
    llama_metadata_present: HashMap<String, String>,
    llama_metadata_missing: Vec<String>,
    
    // Key mapping suggestions
    mapping_suggestions: Vec<(String, String)>, // (smollm3_key, llama_key)
}

impl GgufReport {
    fn new(path: &str, size_gb: f64) -> Self {
        Self {
            file_path: path.to_string(),
            file_size_gb: size_gb,
            tensor_count: 0,
            metadata_count: 0,
            q4_k_m_tensors: Vec::new(),
            f32_tensors: Vec::new(),
            other_tensors: Vec::new(),
            architecture: None,
            vocab_size: None,
            hidden_size: None,
            n_layers: None,
            smollm3_metadata: HashMap::new(),
            llama_metadata_present: HashMap::new(),
            llama_metadata_missing: Vec::new(),
            mapping_suggestions: Vec::new(),
        }
    }
    
    fn print(&self) {
        println!("\n{}", "=".repeat(80));
        println!("                    GGUF INSPECTION REPORT");
        println!("{}", "=".repeat(80));
        
        // File info
        println!("\n📁 FILE INFORMATION");
        println!("{}", "-".repeat(40));
        println!("Path: {}", self.file_path);
        println!("Size: {:.2} GB", self.file_size_gb);
        println!("Total tensors: {}", self.tensor_count);
        println!("Total metadata: {}", self.metadata_count);
        
        // Architecture info
        println!("\n🏗️  ARCHITECTURE");
        println!("{}", "-".repeat(40));
        println!("Architecture: {}", self.architecture.as_ref().unwrap_or(&"Unknown".to_string()));
        println!("Vocab size: {:?}", self.vocab_size);
        println!("Hidden size: {:?}", self.hidden_size);
        println!("Number of layers: {:?}", self.n_layers);
        
        // Tensor report
        println!("\n📊 TENSOR QUANTIZATION REPORT");
        println!("{}", "-".repeat(40));
        println!("Q4_K_M tensors (quantized): {}", self.q4_k_m_tensors.len());
        println!("F32 tensors (not quantized): {}", self.f32_tensors.len());
        println!("Other tensor types: {}", self.other_tensors.len());
        
        // Q4_K_M tensors
        if !self.q4_k_m_tensors.is_empty() {
            println!("\n  Q4_K_M Tensors (MUST use QMatMul):");
            for (name, shape) in &self.q4_k_m_tensors {
                println!("    - {}: shape {:?}", name, shape);
            }
        }
        
        // F32 tensors
        if !self.f32_tensors.is_empty() {
            println!("\n  F32 Tensors (not quantized):");
            for (name, shape) in &self.f32_tensors {
                println!("    - {}: shape {:?}", name, shape);
            }
        }
        
        // Other tensors
        if !self.other_tensors.is_empty() {
            println!("\n  Other Tensor Types:");
            for (name, dtype, shape) in &self.other_tensors {
                println!("    - {}: {} shape {:?}", name, dtype, shape);
            }
        }
        
        // SmolLM3 metadata
        if !self.smollm3_metadata.is_empty() {
            println!("\n🦙 SMOLLM3 METADATA");
            println!("{}", "-".repeat(40));
            for (key, value) in &self.smollm3_metadata {
                println!("  {}: {}", key, value);
            }
        }
        
        // Llama metadata status
        println!("\n🦙 LLAMA METADATA STATUS");
        println!("{}", "-".repeat(40));
        
        if !self.llama_metadata_present.is_empty() {
            println!("✅ Present Llama keys:");
            for (key, value) in &self.llama_metadata_present {
                println!("  {}: {}", key, value);
            }
        }
        
        if !self.llama_metadata_missing.is_empty() {
            println!("\n❌ Missing Llama keys:");
            for key in &self.llama_metadata_missing {
                println!("  - {}", key);
            }
        }
        
        // Mapping suggestions
        if !self.mapping_suggestions.is_empty() {
            println!("\n🔄 SUGGESTED MAPPINGS");
            println!("{}", "-".repeat(40));
            for (from, to) in &self.mapping_suggestions {
                println!("  {} → {}", from, to);
            }
        }
        
        // Summary
        println!("\n📝 SUMMARY");
        println!("{}", "-".repeat(40));
        println!("✅ Valid GGUF file");
        println!("✅ {} tensors found ({} quantized, {} F32)", 
                self.tensor_count, 
                self.q4_k_m_tensors.len(), 
                self.f32_tensors.len());
        
        if self.llama_metadata_missing.is_empty() {
            println!("✅ All Llama metadata present");
        } else {
            println!("⚠️  {} Llama metadata keys need mapping", self.llama_metadata_missing.len());
        }
        
        println!("\n{}", "=".repeat(80));
    }
}

/// Extract value as string for display
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Bool(v) => v.to_string(),
        Value::U8(v) => v.to_string(),
        Value::U16(v) => v.to_string(),
        Value::U32(v) => v.to_string(),
        Value::U64(v) => v.to_string(),
        Value::I8(v) => v.to_string(),
        Value::I16(v) => v.to_string(),
        Value::I32(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
        Value::F32(v) => v.to_string(),
        Value::F64(v) => v.to_string(),
        Value::String(v) => v.clone(),
        Value::Array(arr) => format!("Array[{}]", arr.len()),
    }
}

/// Inspect GGUF file and generate comprehensive report
fn inspect_gguf(path: &Path) -> Result<GgufReport> {
    // Check file exists and get size
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Failed to read file metadata: {:?}", path))?;
    let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    
    // Read GGUF content
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open GGUF file: {:?}", path))?;
    let content = Content::read(&mut file)
        .with_context(|| "Failed to parse GGUF file")?;
    
    let mut report = GgufReport::new(path.to_str().unwrap_or("unknown"), size_gb);
    report.tensor_count = content.tensor_infos.len();
    report.metadata_count = content.metadata.len();
    
    // Analyze tensors by quantization type
    for (name, info) in &content.tensor_infos {
        let shape: Vec<u64> = info.shape.dims().iter().map(|&x| x as u64).collect();
        match info.ggml_dtype {
            GgmlDType::Q4K => {
                report.q4_k_m_tensors.push((name.clone(), shape));
            }
            GgmlDType::F32 => {
                report.f32_tensors.push((name.clone(), shape));
            }
            dtype => {
                report.other_tensors.push((name.clone(), format!("{:?}", dtype), shape));
            }
        }
    }
    
    // Sort tensors by name for better readability
    report.q4_k_m_tensors.sort_by(|a, b| a.0.cmp(&b.0));
    report.f32_tensors.sort_by(|a, b| a.0.cmp(&b.0));
    report.other_tensors.sort_by(|a, b| a.0.cmp(&b.0));
    
    // Extract key metadata
    if let Some(arch) = content.metadata.get("general.architecture") {
        report.architecture = Some(value_to_string(arch));
    }
    
    // Look for SmolLM3 specific metadata
    for (key, value) in &content.metadata {
        if key.starts_with("smollm") || key.contains("smollm") {
            report.smollm3_metadata.insert(key.clone(), value_to_string(value));
        }
        
        // Extract common values
        match key.as_str() {
            "tokenizer.ggml.model.vocab_size" | "vocab_size" | "smollm3.vocab_size" => {
                if let Value::U32(v) = value {
                    report.vocab_size = Some(*v);
                }
            }
            "hidden_size" | "smollm3.embedding_length" | "smollm3.hidden_size" => {
                if let Value::U32(v) = value {
                    report.hidden_size = Some(*v);
                }
            }
            "n_layers" | "smollm3.block_count" | "block_count" => {
                if let Value::U32(v) = value {
                    report.n_layers = Some(*v);
                }
            }
            _ => {}
        }
    }
    
    // Check for required Llama metadata
    let required_llama_keys = vec![
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.block_count",
        "llama.context_length",
        "llama.embedding_length",
        "llama.feed_forward_length",
        "llama.vocab_size",
        "llama.rope.freq_base",
        "llama.rope.dimension_count",
        "llama.attention.layer_norm_rms_epsilon",
    ];
    
    for key in &required_llama_keys {
        if let Some(value) = content.metadata.get(*key) {
            report.llama_metadata_present.insert(key.to_string(), value_to_string(value));
        } else {
            report.llama_metadata_missing.push(key.to_string());
        }
    }
    
    // Generate mapping suggestions based on SmolLM3 keys found
    let mapping_patterns = vec![
        (vec!["smollm3.attention.head_count", "attention.n_heads"], "llama.attention.head_count"),
        (vec!["smollm3.attention.head_count_kv", "attention.n_kv_heads"], "llama.attention.head_count_kv"),
        (vec!["smollm3.block_count", "n_layers", "block_count"], "llama.block_count"),
        (vec!["smollm3.context_length", "max_position_embeddings"], "llama.context_length"),
        (vec!["smollm3.embedding_length", "hidden_size"], "llama.embedding_length"),
        (vec!["smollm3.feed_forward_length", "intermediate_size"], "llama.feed_forward_length"),
        (vec!["smollm3.vocab_size", "vocab_size"], "llama.vocab_size"),
        (vec!["smollm3.rope.theta", "rope_theta"], "llama.rope.freq_base"),
        (vec!["smollm3.rope.dimension_count", "rope_dim"], "llama.rope.dimension_count"),
        (vec!["smollm3.attention.layer_norm_rms_epsilon", "rms_norm_eps"], "llama.attention.layer_norm_rms_epsilon"),
    ];
    
    for (possible_keys, llama_key) in mapping_patterns {
        if report.llama_metadata_missing.contains(&llama_key.to_string()) {
            for smollm_key in possible_keys {
                if content.metadata.contains_key(smollm_key) {
                    report.mapping_suggestions.push((smollm_key.to_string(), llama_key.to_string()));
                    break;
                }
            }
        }
    }
    
    Ok(report)
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive(tracing::Level::INFO.into()))
        .init();
    
    println!("\n🔍 SmolLM3 GGUF Inspector v1.0");
    println!("================================\n");
    
    // Default path to the GGUF file
    let gguf_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf".to_string());
    
    let path = Path::new(&gguf_path);
    
    if !path.exists() {
        eprintln!("❌ Error: GGUF file not found at: {}", gguf_path);
        eprintln!("\nUsage: cargo run --bin inspect_gguf [path_to_gguf]");
        eprintln!("Default: models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf");
        std::process::exit(1);
    }
    
    println!("Reading GGUF file: {}", gguf_path);
    
    match inspect_gguf(path) {
        Ok(report) => {
            report.print();
            
            // Exit with non-zero if mapping is needed
            if !report.llama_metadata_missing.is_empty() {
                println!("\n⚠️  Action required: Metadata mapping needed for Llama compatibility");
                std::process::exit(2);
            }
        }
        Err(e) => {
            eprintln!("\n❌ Error inspecting GGUF file: {:#}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}
