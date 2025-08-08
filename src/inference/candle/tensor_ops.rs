//! Common tensor operations and utilities for inference
//! 
//! This module provides low-level tensor operations that are used across
//! the inference pipeline, including sampling, normalization, and utilities
//! for working with quantized tensors.

use candle_core::{Tensor, Device, Result, DType, IndexOp};
use candle_nn::ops::softmax;
use rand::{thread_rng, Rng};

/// Temperature-based sampling operations
pub struct SamplingOps;

impl SamplingOps {
    /// Sample from logits using temperature
    pub fn sample_with_temperature(
        logits: &Tensor,
        temperature: f64,
        top_p: Option<f64>,
    ) -> Result<u32> {
        let logits = if temperature != 1.0 {
            (logits / temperature)?
        } else {
            logits.clone()
        };
        
        // Apply softmax to get probabilities
        let probs = softmax(&logits, logits.rank() - 1)?;
        
        // Apply top-p (nucleus) sampling if specified
        let probs = if let Some(p) = top_p {
            Self::apply_top_p(&probs, p)?
        } else {
            probs
        };
        
        // Sample from distribution
        Self::sample_from_probs(&probs)
    }
    
    /// Apply top-p (nucleus) sampling
    fn apply_top_p(probs: &Tensor, p: f64) -> Result<Tensor> {
        let probs_vec = probs.to_vec1::<f32>()?;
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Sort by probability (descending)
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed_probs.len();
        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Zero out probabilities below cutoff
        let mut new_probs = vec![0.0f32; probs_vec.len()];
        let mut sum = 0.0;
        for i in 0..cutoff_idx {
            let (idx, prob) = indexed_probs[i];
            new_probs[idx] = prob;
            sum += prob;
        }
        
        // Renormalize
        for p in &mut new_probs {
            *p /= sum;
        }
        
        Tensor::from_vec(new_probs, probs.shape(), probs.device())
    }
    
    /// Sample from probability distribution
    fn sample_from_probs(probs: &Tensor) -> Result<u32> {
        let probs_vec = probs.to_vec1::<f32>()?;
        let mut rng = thread_rng();
        let sample = rng.gen::<f32>();
        
        let mut cumsum = 0.0;
        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumsum += prob;
            if cumsum >= sample {
                return Ok(idx as u32);
            }
        }
        
        // Fallback to last token (shouldn't happen with proper normalization)
        Ok((probs_vec.len() - 1) as u32)
    }
    
    /// Greedy sampling (argmax)
    pub fn greedy_sample(logits: &Tensor) -> Result<u32> {
        let logits_vec = logits.to_vec1::<f32>()?;
        let max_idx = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        Ok(max_idx as u32)
    }
    
    /// Top-k sampling
    pub fn sample_top_k(logits: &Tensor, k: usize, temperature: f64) -> Result<u32> {
        let logits_vec = logits.to_vec1::<f32>()?;
        let mut indexed_logits: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l))
            .collect();
        
        // Sort by logit value (descending)
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top k
        indexed_logits.truncate(k);
        
        // Apply temperature and softmax to top-k
        let top_k_logits: Vec<f32> = indexed_logits
            .iter()
            .map(|(_, l)| l / temperature as f32)
            .collect();
        
        // Compute softmax manually
        let max_logit = top_k_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = top_k_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let probs: Vec<f32> = top_k_logits
            .iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect();
        
        // Sample from top-k distribution
        let mut rng = thread_rng();
        let sample = rng.gen::<f32>();
        
        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= sample {
                return Ok(indexed_logits[i].0 as u32);
            }
        }
        
        Ok(indexed_logits[0].0 as u32)
    }
}

/// Tensor normalization operations
pub struct NormalizationOps;

impl NormalizationOps {
    /// RMS normalization (used in LLaMA models)
    pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean_x2 = x.sqr()?.mean_keepdim(x.rank() - 1)?;
        let norm = (mean_x2 + eps)?.sqrt()?;
        let x_normed = x.broadcast_div(&norm)?;
        x_normed.to_dtype(x_dtype)
    }
    
    /// Layer normalization
    pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        
        let mean = x.mean_keepdim(x.rank() - 1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(x.rank() - 1)?;
        let std = (var + eps)?.sqrt()?;
        
        let x_normed = x_centered.broadcast_div(&std)?;
        let x_scaled = x_normed.broadcast_mul(gamma)?;
        let x_shifted = x_scaled.broadcast_add(beta)?;
        
        x_shifted.to_dtype(x_dtype)
    }
}

/// Attention-specific tensor operations
pub struct AttentionOps;

impl AttentionOps {
    /// Apply causal mask to attention scores
    pub fn apply_causal_mask(
        scores: &Tensor,
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mask = Self::create_causal_mask(seq_len, scores.dtype(), device)?;
        scores.broadcast_add(&mask)
    }
    
    /// Create causal attention mask
    fn create_causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let mask_value = match dtype {
            DType::F16 => -65504.0,
            DType::BF16 => f32::NEG_INFINITY,
            DType::F32 => f32::NEG_INFINITY,
            DType::F64 => f64::NEG_INFINITY as f32,
            _ => f32::NEG_INFINITY,
        };
        
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[i * seq_len + j] = mask_value;
            }
        }
        
        Tensor::from_vec(mask, &[seq_len, seq_len], device)?.to_dtype(dtype)
    }
    
    /// Compute attention scores with optional scaling
    pub fn compute_attention_scores(
        query: &Tensor,
        key: &Tensor,
        scale: Option<f64>,
    ) -> Result<Tensor> {
        let d_k = query.dims()[query.rank() - 1] as f64;
        let scale = scale.unwrap_or(1.0 / d_k.sqrt());
        
        let scores = query.matmul(&key.transpose(key.rank() - 2, key.rank() - 1)?)?;
        scores.affine(scale, 0.0)
    }
    
    /// Apply rotary position embeddings
    pub fn apply_rotary_embeddings(
        tensor: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, n_heads, head_dim) = match tensor.dims() {
            &[b, s, h, d] => (b, s, h, d),
            _ => return Ok(tensor.clone()),
        };
        
        let half_dim = head_dim / 2;
        
        // Split tensor
        let x1 = tensor.narrow(3, 0, half_dim)?;
        let x2 = tensor.narrow(3, half_dim, half_dim)?;
        
        // Apply rotation
        let cos_broadcast = cos.broadcast_as(&[batch, seq_len, n_heads, half_dim])?;
        let sin_broadcast = sin.broadcast_as(&[batch, seq_len, n_heads, half_dim])?;
        
        let rotated_x1 = x1.mul(&cos_broadcast)?.sub(&x2.mul(&sin_broadcast)?)?;
        let rotated_x2 = x1.mul(&sin_broadcast)?.add(&x2.mul(&cos_broadcast)?)?;
        
        Tensor::cat(&[&rotated_x1, &rotated_x2], 3)
    }
}

/// Quantization-aware operations
pub struct QuantizedOps;

impl QuantizedOps {
    /// Dequantize tensor for operations
    pub fn dequantize(tensor: &Tensor, scale: f32, zero_point: i32) -> Result<Tensor> {
        let float_tensor = tensor.to_dtype(DType::F32)?;
        float_tensor.affine(scale as f64, -(zero_point as f64 * scale as f64))
    }
    
    /// Quantize tensor
    pub fn quantize(
        tensor: &Tensor,
        scale: f32,
        zero_point: i32,
        dtype: DType,
    ) -> Result<Tensor> {
        let scaled = (tensor / scale as f64)?;
        let shifted = (scaled + zero_point as f64)?;
        
        // Clamp to dtype range
        let (min_val, max_val) = match dtype {
            DType::U8 => (0.0, 255.0),
            DType::I64 => (i8::MIN as f64, i8::MAX as f64),
            _ => return Err(candle_core::Error::Msg("Unsupported quantization dtype".into())),
        };
        
        let clamped = shifted.clamp(min_val, max_val)?;
        clamped.to_dtype(dtype)
    }
    
    /// Compute quantization parameters
    pub fn compute_scale_zero_point(
        tensor: &Tensor,
        dtype: DType,
    ) -> Result<(f32, i32)> {
        let min_val = tensor.min(0)?.to_scalar::<f32>()?;
        let max_val = tensor.max(0)?.to_scalar::<f32>()?;
        
        let (qmin, qmax) = match dtype {
            DType::U8 => (0.0, 255.0),
            DType::I64 => (i8::MIN as f32, i8::MAX as f32),
            _ => return Err(candle_core::Error::Msg("Unsupported quantization dtype".into())),
        };
        
        let scale = (max_val - min_val) / (qmax - qmin);
        let zero_point = qmin - min_val / scale;
        
        Ok((scale, zero_point.round() as i32))
    }
}

/// Utility operations for tensors
pub struct TensorUtils;

impl TensorUtils {
    /// Concatenate tensors along a dimension
    pub fn concat_along(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(candle_core::Error::Msg("Cannot concatenate empty tensor list".into()));
        }
        
        Tensor::cat(tensors, dim)
    }
    
    /// Stack tensors to create a new dimension
    pub fn stack(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(candle_core::Error::Msg("Cannot stack empty tensor list".into()));
        }
        
        let expanded: Result<Vec<_>> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect();
        
        Tensor::cat(&expanded?, dim)
    }
    
    /// Repeat tensor along dimensions
    pub fn repeat_interleave(tensor: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
        let shape = tensor.dims();
        let mut new_shape = shape.to_vec();
        new_shape.insert(dim + 1, repeats);
        
        let expanded = tensor.unsqueeze(dim + 1)?;
        let repeated = expanded.broadcast_as(new_shape.as_slice())?;
        
        // Flatten the repeated dimension
        new_shape[dim] *= repeats;
        new_shape.remove(dim + 1);
        
        repeated.reshape(new_shape.as_slice())
    }
    
    /// Apply a function element-wise
    pub fn apply_elementwise<F>(tensor: &Tensor, f: F) -> Result<Tensor>
    where
        F: Fn(f32) -> f32,
    {
        let values = tensor.to_vec1::<f32>()?;
        let transformed: Vec<f32> = values.iter().map(|&x| f(x)).collect();
        Tensor::from_vec(transformed, tensor.shape(), tensor.device())
    }
    
    /// Check if tensors have compatible shapes for broadcasting
    pub fn are_broadcastable(a: &Tensor, b: &Tensor) -> bool {
        let a_dims = a.dims();
        let b_dims = b.dims();
        
        let min_dims = a_dims.len().min(b_dims.len());
        
        for i in 0..min_dims {
            let a_dim = a_dims[a_dims.len() - 1 - i];
            let b_dim = b_dims[b_dims.len() - 1 - i];
            
            if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
                return false;
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sampling_ops() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            &[5],
            &device,
        )?;
        
        // Test greedy sampling
        let token = SamplingOps::greedy_sample(&logits)?;
        assert_eq!(token, 4); // Index of maximum value
        
        // Test temperature sampling
        let token = SamplingOps::sample_with_temperature(&logits, 0.1, None)?;
        // Should be biased toward higher values
        assert!(token <= 4);
        
        Ok(())
    }
    
    #[test]
    fn test_attention_mask() -> Result<()> {
        let device = Device::Cpu;
        let scores = Tensor::ones(&[4, 4], DType::F32, &device)?;
        let masked = AttentionOps::apply_causal_mask(&scores, 4, &device)?;
        
        // Check that upper triangle is masked
        let masked_vec = masked.to_vec2::<f32>()?;
        assert!(masked_vec[0][1] < -1000.0);
        assert!(masked_vec[0][2] < -1000.0);
        assert!(masked_vec[1][2] < -1000.0);
        
        Ok(())
    }
}
