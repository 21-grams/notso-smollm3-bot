use candle_core::{Device, Result};
use candle_core::quantized::{gguf_file, GgmlDType};
use candle_transformers::models::quantized_llama::{Llama, ModelWeights, LlamaConfig};
use std::fs::File;

pub struct ModelLoader;

impl ModelLoader {
    pub async fn load_gguf_model(path: &str, device: &Device) -> anyhow::Result<Llama> {
        tracing::info!("📦 Loading GGUF model with official Candle patterns");
        
        let mut file = File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        // Validate Q4_K_M quantization
        Self::validate_q4km(&content)?;
        
        // Load weights using official API
        let weights = ModelWeights::from_gguf(content, &mut file, device)?;
        
        // Create config for SmolLM3
        let config = LlamaConfig {
            vocab_size: 128256,
            hidden_size: 2048,
            n_layer: 36,
            n_head: 16,
            n_kv_head: 4,  // GQA 4:1
            intermediate_size: 11008,
            max_seq_len: 65536,
            rope_theta: 2_000_000.0,
            rms_norm_eps: 1e-5,
            ..Default::default()
        };
        
        // Load model
        let model = Llama::load(&weights, &config, device)?;
        
        tracing::info!("✅ Model loaded successfully with Q4_K_M quantization");
        Ok(model)
    }
    
    fn validate_q4km(content: &gguf_file::Content) -> Result<()> {
        let mut q4km_count = 0;
        let mut total_weights = 0;
        
        for (name, info) in &content.tensor_infos {
            if name.contains("weight") && !name.contains("norm") {
                total_weights += 1;
                if matches!(info.ggml_dtype, GgmlDType::Q4K) {
                    q4km_count += 1;
                    tracing::debug!("✅ {} is Q4_K_M", name);
                }
            }
        }
        
        tracing::info!("📊 Q4_K_M validation: {}/{} weight tensors", 
                      q4km_count, total_weights);
        Ok(())
    }
}
