//! Official GGUF loading using Candle patterns

use candle_transformers::models::quantized_llama::ModelWeights;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Result};
use std::path::Path;

pub struct OfficialLoader;

impl OfficialLoader {
    /// Load GGUF using official Candle patterns
    pub async fn load_gguf<P: AsRef<Path>>(
        path: P,
        device: &Device,
    ) -> Result<ModelWeights> {
        tracing::info!("🚀 Loading GGUF with official Candle patterns");
        
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        tracing::info!("📊 GGUF loaded: {} tensors, {} metadata entries",
                      content.tensor_infos.len(),
                      content.metadata.len());
        
        let weights = ModelWeights::from_gguf(content, &mut file, device)?;
        
        tracing::info!("✅ Official model weights loaded successfully");
        Ok(weights)
    }
    
    /// Validate GGUF file before loading
    pub fn validate_gguf<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            candle_core::bail!("GGUF file not found: {:?}", path);
        }
        
        let metadata = std::fs::metadata(path)?;
        let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
        
        if size_gb < 0.1 {
            candle_core::bail!("GGUF file too small: {:.2} GB", size_gb);
        }
        
        tracing::info!("✅ GGUF validation passed: {:.2} GB", size_gb);
        Ok(())
    }
    
    /// Check if file is GGUF format
    pub fn is_gguf<P: AsRef<Path>>(path: P) -> bool {
        if let Ok(mut file) = std::fs::File::open(path) {
            let mut magic = [0u8; 4];
            if std::io::Read::read_exact(&mut file, &mut magic).is_ok() {
                // GGUF magic number: GGUF (0x47475546)
                return magic == [0x47, 0x47, 0x55, 0x46];
            }
        }
        false
    }
}
