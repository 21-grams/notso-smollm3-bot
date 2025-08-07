use crate::config::Config;
use candle_core::{Device, Tensor, Result as CandleResult};
use candle_transformers::models::quantized_llama::{Llama, ModelWeights, LlamaConfig};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct InferenceEngine {
    model: Option<Arc<Mutex<Llama>>>,
    device: Device,
    config: LlamaConfig,
    is_stub_mode: bool,
}

impl InferenceEngine {
    pub async fn new(config: &Config) -> anyhow::Result<Self> {
        let device = config.to_candle_device();
        
        // Try to load model, fall back to stub if not found
        let (model, is_stub_mode) = match Self::load_model(&config.model_path, &device).await {
            Ok(m) => (Some(Arc::new(Mutex::new(m))), false),
            Err(e) => {
                tracing::warn!("⚠️ Could not load model: {}. Running in stub mode.", e);
                (None, true)
            }
        };
        
        let llama_config = LlamaConfig {
            vocab_size: 128256,
            hidden_size: 2048,
            n_layer: 36,
            n_head: 16,
            n_kv_head: 4,  // GQA 4:1 ratio
            intermediate_size: 11008,
            max_seq_len: 65536,
            rope_theta: 2_000_000.0,
            rms_norm_eps: 1e-5,
            ..Default::default()
        };
        
        Ok(Self {
            model,
            device,
            config: llama_config,
            is_stub_mode,
        })
    }
    
    async fn load_model(path: &str, device: &Device) -> anyhow::Result<Llama> {
        use crate::inference::candle::model_loader::ModelLoader;
        
        tracing::info!("🔧 Loading model from: {}", path);
        ModelLoader::load_gguf_model(path, device).await
    }
    
    pub fn is_stub(&self) -> bool {
        self.is_stub_mode
    }
    
    pub async fn forward(&self, input: &Tensor, position: usize) -> CandleResult<Tensor> {
        if let Some(model) = &self.model {
            model.lock().await.forward(input, position)
        } else {
            // Stub mode - return random tensor
            Tensor::randn(
                0f32,
                1f32,
                input.shape().dims(),
                &self.device
            )
        }
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
}
