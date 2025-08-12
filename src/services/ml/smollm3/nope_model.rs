//! NoPE-aware SmolLM3 model implementation using candle-nn's rotary embeddings
//!
//! This implementation provides full control over RoPE application, allowing
//! certain layers to skip position encoding (NoPE layers) while maintaining
//! GPU acceleration through candle-nn's optimized kernels.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_core::quantized::{QTensor, gguf_file, QMatMul};
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
    rotary_emb: RotaryEmbedding,
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
        let tensor_names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        tracing::debug!("Available tensors: {:?}", tensor_names.iter().filter(|n|
            n.contains("token") || n.contains("embed") || n.contains("output")
        ).collect::<Vec<_>>());

        for (tensor_name, _tensor_info) in content.tensor_infos.iter() {
            tracing::trace!("Loading tensor: {}", tensor_name);
            let tensor = content.tensor(&mut file, tensor_name, device)?;
            tensors.insert(tensor_name.clone(), Arc::new(tensor));
        }

        let embed_tokens = tensors.get("token_embd.weight")
            .ok_or_else(|| {
                tracing::error!("Could not find token_embd.weight. Available: {:?}", 
                    tensors.keys().filter(|k| k.contains("token") || k.contains("emb")).collect::<Vec<_>>());
                candle_core::Error::Msg("Missing token_embd.weight".to_string())
            })?
            .clone();

        tracing::info!("Using tied embeddings for output projection");
        let lm_head = QMatMul::from_arc(embed_tokens.clone())?;

        let norm_weight = tensors.get("output_norm.weight")
            .ok_or_else(|| candle_core::Error::Msg("Missing output_norm.weight".to_string()))?
            .dequantize(device)?;
        let norm = RmsNorm::new(norm_weight, 1e-5);

        let mut layers = Vec::with_capacity(config.base.num_hidden_layers);
        for i in 0..config.base.num_hidden_layers {
            let layer = NopeLayer::load(&tensors, i, &config, device)?;
            layers.push(layer);
        }

        let rotary_emb = RotaryEmbedding::new(DType::F32, &config, device)?;
        let kv_cache = vec![None; config.base.num_hidden_layers];

        tracing::info!("✅ NoPE model loaded with {} layers", layers.len());
        tracing::info!("   NoPE layers: {:?}", config.nope_layer_indices);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            device: device.clone(),
            config,
            kv_cache,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        let (batch_size, seq_len, flat_input) = match input_ids.dims() {
            &[seq] => (1, seq, input_ids.clone()),
            &[batch, seq] => (batch, seq, input_ids.flatten(0, 1)?),
            _ => return Err(candle_core::Error::Msg(
                format!("Invalid input dimensions: {:?}", input_ids.dims())
            )),
        };

        tracing::debug!("NoPE forward: position={}, batch={}, seq_len={}", position, batch_size, seq_len);

        let embed_weight = self.embed_tokens.dequantize(&self.device)?;
        let mut hidden_states = embed_weight.index_select(&flat_input, 0)?
            .reshape((batch_size, seq_len, self.config.base.hidden_size))?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if self.config.nope_layer_indices.contains(&layer_idx) {
                tracing::trace!("Processing NoPE layer {}", layer_idx);
            }
            let cache = self.kv_cache.get_mut(layer_idx);
            hidden_states = layer.forward(&hidden_states, &self.rotary_emb, cache, position, layer_idx)?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        tracing::debug!("Logits shape: {:?}", logits.dims());
        Ok(logits)
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache = vec![None; self.config.base.num_hidden_layers];
    }

    pub fn config(&self) -> &crate::services::ml::official::config::SmolLM3Config {
        &self.config
    }
}

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
        let self_attn = NopeAttention::load(tensors, &prefix, config, device)?;
        let mlp = Mlp::load(tensors, &prefix, device)?;

        let input_norm_weight = tensors.get(&format!("{}.attn_norm.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.attn_norm.weight", prefix)))?
            .dequantize(device)?;
        let input_layernorm = RmsNorm::new(input_norm_weight, 1e-5);

        let post_norm_weight = tensors.get(&format!("{}.ffn_norm.weight", prefix))
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.ffn_norm.weight", prefix)))?
            .dequantize(device)?;
        let post_attention_layernorm = RmsNorm::new(post_norm_weight, 1e-5);

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
        rotary_emb: &RotaryEmbedding,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        position: usize,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&hidden_states, rotary_emb, cache, position, layer_idx)?;
        let hidden_states = (residual + attn_output)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        Ok((residual + mlp_output)?)
    }
}

pub struct NopeAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl NopeAttention {
    fn load(
        tensors: &HashMap<String, Arc<QTensor>>,
        prefix: &str,
        config: &crate::services::ml::official::config::SmolLM3Config,
        _device: &Device,
    ) -> Result<Self> {
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

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.base.num_attention_heads,
            num_kv_heads: config.base.num_key_value_heads,
            head_dim: config.base.head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        rotary_emb: &RotaryEmbedding,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        position: usize,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = hidden_states.dims3()?;
        tracing::trace!("Attention forward: batch={}, seq={}, layer={}", batch_size, seq_len, layer_idx);

        // Project Q, K, V using QMatMul
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary_emb.apply(&q, &k, position, layer_idx)?;

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

        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let mask = self.make_causal_mask(seq_len, hidden_states.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let output = self.o_proj.forward(&attn_output)?;
        Ok(output)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let repeat_count = self.num_heads / self.num_kv_heads;
        if repeat_count == 1 {
            return Ok(x.clone());
        }
        let (batch_size, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((batch_size, num_kv_heads, repeat_count, seq_len, head_dim))?
            .reshape((batch_size, num_kv_heads * repeat_count, seq_len, head_dim))
    }

    fn make_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask = Tensor::tril2(seq_len, DType::F32, device)?;
        let zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
        let causal_mask = mask.where_cond(&zeros, &neg_inf)?;
        Ok(causal_mask)
    }
}

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
        let gate = self.gate_proj.forward(hidden_states)?.silu()?;
        let up = self.up_proj.forward(hidden_states)?;
        let intermediate = (gate * up)?;
        let output = self.down_proj.forward(&intermediate)?;
        Ok(output)
    }
}

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean_x2 = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(mean_x2 + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        Ok(x_normed.broadcast_mul(&self.weight)?)
    }
}

pub struct RotaryEmbedding {
    cos_cached: Tensor,
    sin_cached: Tensor,
    nope_layers: Vec<usize>,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        config: &crate::services::ml::official::config::SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = config.base.head_dim;
        let max_seq_len = config.base.max_position_embeddings;
        let theta = config.base.rope_theta;

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
            nope_layers: config.nope_layer_indices.clone(),
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, position: usize, layer_idx: usize) -> Result<(Tensor, Tensor)> {
        if self.nope_layers.contains(&layer_idx) {
            tracing::debug!("Layer {} is NoPE - skipping RoPE", layer_idx);
            return Ok((q.clone(), k.clone()));
        }
        let seq_len = q.dim(2)?;
        let cos = self.cos_cached.narrow(0, position, seq_len)?;
        let sin = self.sin_cached.narrow(0, position, seq_len)?;
        let q_rot = self.apply_rotary(q, &cos, &sin)?;
        let k_rot = self.apply_rotary(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_batch_size, _num_heads, _seq_len, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        let rot_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rot_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&rot_x1, &rot_x2], D::Minus1)
    }
}
