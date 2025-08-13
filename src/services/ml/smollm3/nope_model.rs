//! NoPE-aware SmolLM3 model implementation using Candle's optimized components
//!
//! This implementation leverages Candle's built-in RmsNorm, rotary embeddings,
//! and repeat_kv functions for optimal performance while maintaining NoPE layer support.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_core::quantized::{QTensor, gguf_file, QMatMul};
use candle_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::sync::Arc;

/// NoPE-aware SmolLM3 model with selective RoPE application
pub struct NopeModel {
    embed_tokens: Arc<QTensor>,
    layers: Vec<NopeLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    config: crate::services::ml::official::config::SmolLM3Config,
    kv_cache: Vec<Option<(Tensor, Tensor)>>,
}

impl NopeModel {
    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        tracing::info!("Loading NoPE-aware SmolLM3 model from GGUF");

        let mut file = File::open(&path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open GGUF file: {}", e)))?;
        let content = gguf_file::Content::read(&mut file)?;
        let config = crate::services::ml::official::config::SmolLM3Config::from_gguf(&content)?;

        let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::new();
        tracing::info!("Loading {} tensors from GGUF", content.tensor_infos.len());

        for (tensor_name, tensor_info) in content.tensor_infos.iter() {
            tracing::trace!("Loading tensor: {} (type: {:?})", tensor_name, tensor_info.ggml_dtype);
            let tensor = content.tensor(&mut file, tensor_name, device)?;
            
            // Log critical tensor types
            if tensor_name == "token_embd.weight" {
                tracing::info!("Embeddings tensor type: {:?}", tensor_info.ggml_dtype);
            }
            
            tensors.insert(tensor_name.clone(), Arc::new(tensor));
        }

        // Load embeddings (tied for input and output)
        let embed_tokens = tensors.get("token_embd.weight")
            .ok_or_else(|| candle_core::Error::Msg("Missing token_embd.weight".to_string()))?
            .clone();

        tracing::info!("Using tied embeddings for output projection");
        let lm_head = QMatMul::from_arc(embed_tokens.clone())?;

        // Use Candle's RmsNorm
        let norm_weight = tensors.get("output_norm.weight")
            .ok_or_else(|| candle_core::Error::Msg("Missing output_norm.weight".to_string()))?
            .dequantize(device)?;
        let norm = RmsNorm::new(norm_weight, config.base.rms_norm_eps);

        // Load layers
        let mut layers = Vec::with_capacity(config.base.num_hidden_layers);
        for i in 0..config.base.num_hidden_layers {
            let layer = NopeLayer::load(&tensors, i, &config, device)?;
            layers.push(layer);
        }

        // Pre-allocate KV cache
        let kv_cache = vec![None; config.base.num_hidden_layers];

        tracing::info!("✅ NoPE model loaded with {} layers", layers.len());
        tracing::info!("   NoPE layers: {:?}", config.nope_layer_indices);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config,
            kv_cache,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        tracing::debug!("[NOPE_MODEL] Starting forward pass at position {}", position);
        
        let (batch_size, seq_len, flat_input) = match input_ids.dims() {
            &[seq] => (1, seq, input_ids.clone()),
            &[batch, seq] => (batch, seq, input_ids.flatten(0, 1)?),
            _ => return Err(candle_core::Error::Msg(
                format!("Invalid input dimensions: {:?}", input_ids.dims())
            )),
        };

        // Embeddings
        let embed_weight = self.embed_tokens.dequantize(&self.device)?;
        let mut hidden_states = embed_weight.index_select(&flat_input, 0)?
            .reshape((batch_size, seq_len, self.config.base.hidden_size))?;

        // Process through layers
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let cache = self.kv_cache.get_mut(layer_idx);
            hidden_states = layer.forward(&hidden_states, cache, position, layer_idx, &self.config)?;
        }
        
        // Final norm and projection
        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        
        tracing::debug!("[NOPE_MODEL] Forward pass complete in {:?}", start_time.elapsed());
        Ok(logits)
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache = vec![None; self.config.base.num_hidden_layers];
    }

    pub fn config(&self) -> &crate::services::ml::official::config::SmolLM3Config {
        &self.config
    }
}

/// Transformer layer with NoPE support
pub struct NopeLayer {
    self_attn: NopeAttention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl NopeLayer {
    fn load(
        tensors: &HashMap<String, Arc<QTensor>>,
        layer_idx: usize,
        config: &crate::services::ml::official::config::SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);
        let self_attn = NopeAttention::load(tensors, &prefix, layer_idx, config, device)?;
        let mlp = Mlp::load(tensors, &prefix, device)?;

        // Use Candle's RmsNorm
        let input_norm_weight = tensors.get(&format!("{}.attn_norm.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.attn_norm.weight", prefix)))?
            .dequantize(device)?;
        let input_layernorm = RmsNorm::new(input_norm_weight, config.base.rms_norm_eps);

        let post_norm_weight = tensors.get(&format!("{}.ffn_norm.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.ffn_norm.weight", prefix)))?
            .dequantize(device)?;
        let post_attention_layernorm = RmsNorm::new(post_norm_weight, config.base.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        position: usize,
        layer_idx: usize,
        config: &crate::services::ml::official::config::SmolLM3Config,
    ) -> Result<Tensor> {
        // Pre-norm architecture
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&hidden_states, cache, position, layer_idx, config)?;
        let hidden_states = (residual + attn_output)?;
        
        // MLP block
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        
        Ok((residual + mlp_output)?)
    }
}

/// Attention module with NoPE and GQA support
pub struct NopeAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    rotary_emb: Option<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_nope: bool,
    #[cfg(feature = "flash-attn")]
    use_flash_attn: bool,
}

impl NopeAttention {
    fn load(
        tensors: &HashMap<String, Arc<QTensor>>,
        prefix: &str,
        layer_idx: usize,
        config: &crate::services::ml::official::config::SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        // Load projection matrices
        let q_proj = QMatMul::from_arc(tensors.get(&format!("{}.attn_q.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.attn_q.weight", prefix)))?
            .clone())?;
        let k_proj = QMatMul::from_arc(tensors.get(&format!("{}.attn_k.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.attn_k.weight", prefix)))?
            .clone())?;
        let v_proj = QMatMul::from_arc(tensors.get(&format!("{}.attn_v.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.attn_v.weight", prefix)))?
            .clone())?;
        let o_proj = QMatMul::from_arc(tensors.get(&format!("{}.attn_output.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.attn_output.weight", prefix)))?
            .clone())?;

        // Check if this is a NoPE layer
        let is_nope = config.nope_layer_indices.contains(&layer_idx);
        
        // Only create rotary embeddings if not a NoPE layer
        let rotary_emb = if !is_nope {
            Some(RotaryEmbedding::new(DType::F32, config, device)?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.base.num_attention_heads,
            num_kv_heads: config.base.num_key_value_heads,
            head_dim: config.base.head_dim,
            is_nope,
            #[cfg(feature = "flash-attn")]
            use_flash_attn: config.use_flash_attention && device.is_cuda(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        position: usize,
        _layer_idx: usize,
        _config: &crate::services::ml::official::config::SmolLM3Config,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings (skip for NoPE layers)
        let (q, k) = if !self.is_nope {
            if let Some(ref rope) = self.rotary_emb {
                let (cos, sin) = rope.get_cos_sin(position, seq_len)?;
                let q_rot = apply_rotary_pos_emb(&q, &cos, &sin)?;
                let k_rot = apply_rotary_pos_emb(&k, &cos, &sin)?;
                (q_rot, k_rot)
            } else {
                (q, k)
            }
        } else {
            (q, k)
        };

        // KV cache update
        let (k, v) = if let Some(cache_ref) = cache {
            match cache_ref {
                Some((prev_k, prev_v)) => {
                    let k = Tensor::cat(&[prev_k.as_ref(), &k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[prev_v.as_ref(), &v], 2)?.contiguous()?;
                    *cache_ref = Some((k.clone(), v.clone()));
                    (k, v)
                }
                None => {
                    *cache_ref = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        } else {
            (k, v)
        };

        // Use Candle's repeat_kv for GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        // Compute attention
        #[cfg(feature = "flash-attn")]
        let attn_output = if self.use_flash_attn {
            use candle_flash_attn::flash_attn;
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, scale, true)?
        } else {
            self.standard_attention(&q, &k, &v, seq_len)?
        };

        #[cfg(not(feature = "flash-attn"))]
        let attn_output = self.standard_attention(&q, &k, &v, seq_len)?;

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn standard_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, seq_len: usize) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        
        // Create causal mask
        let mask = self.make_causal_mask(seq_len, q.device())?;
        let scores = scores.broadcast_add(&mask)?;
        
        // Use explicit dimension for softmax (fixes CUDA issue)
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        attn_weights.matmul(&v)
    }

    fn make_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Create causal mask efficiently
        let mask = Tensor::tril2(seq_len, DType::F32, device)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
        let ones = Tensor::ones((seq_len, seq_len), DType::F32, device)?;
        let mask = ((ones - mask)? * neg_inf)?;
        Ok(mask)
    }
}

/// MLP module using SwiGLU activation
pub struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Mlp {
    fn load(
        tensors: &HashMap<String, Arc<QTensor>>,
        prefix: &str,
        _device: &Device,
    ) -> Result<Self> {
        let gate_proj = QMatMul::from_arc(tensors.get(&format!("{}.ffn_gate.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.ffn_gate.weight", prefix)))?
            .clone())?;
        let up_proj = QMatMul::from_arc(tensors.get(&format!("{}.ffn_up.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.ffn_up.weight", prefix)))?
            .clone())?;
        let down_proj = QMatMul::from_arc(tensors.get(&format!("{}.ffn_down.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.ffn_down.weight", prefix)))?
            .clone())?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // SwiGLU activation: gate(x) * up(x)
        let gate = self.gate_proj.forward(hidden_states)?.silu()?;
        let up = self.up_proj.forward(hidden_states)?;
        let intermediate = (gate * up)?;
        self.down_proj.forward(&intermediate)
    }
}

/// Optimized Rotary Embedding implementation
pub struct RotaryEmbedding {
    cos_cached: Tensor,
    sin_cached: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        config: &crate::services::ml::official::config::SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = config.base.head_dim;
        let max_seq_len = config.base.max_position_embeddings;
        
        // Apply YaRN scaling if configured
        let theta = if let Some(ref scaling) = config.rope_scaling {
            if scaling.scaling_type == "yarn" {
                // YaRN scaling formula
                let scale_factor = scaling.factor.powf(
                    (head_dim as f64) / (head_dim as f64 - 2.0)
                );
                (config.base.rope_theta as f64 * scale_factor) as f32
            } else {
                config.base.rope_theta
            }
        } else {
            config.base.rope_theta
        };

        // Pre-compute cos and sin for all positions
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / (2 * half_dim) as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?
            .to_dtype(dtype)?
            .unsqueeze(0)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .unsqueeze(1)?;
        let freqs = t.matmul(&inv_freq)?;
        let cos_cached = freqs.cos()?;
        let sin_cached = freqs.sin()?;

        Ok(Self {
            cos_cached,
            sin_cached,
        })
    }

    pub fn get_cos_sin(&self, position: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_cached.narrow(0, position, seq_len)?;
        let sin = self.sin_cached.narrow(0, position, seq_len)?;
        Ok((cos, sin))
    }
}

/// Apply rotary position embeddings using Candle's optimized rope function
fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Fallback implementation since candle doesn't export rope directly
    {
        let (_batch_size, _num_heads, _seq_len, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;
        
        // Split x into two halves
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
        
        // Broadcast cos and sin
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        
        // Apply rotation
        let rot_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rot_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        
        Tensor::cat(&[&rot_x1, &rot_x2], D::Minus1)
    }
}
