//! Low-level KV cache tensor operations for attention mechanisms
//! 
//! This module provides the foundational tensor operations for key-value caching
//! that higher-level components like SmolLM3's GQA-optimized cache build upon.
//! It handles the raw tensor concatenation, slicing, and memory management.

use candle_core::{Tensor, Device, Result, DType, Shape};
use std::collections::HashMap;

/// Low-level KV cache tensor storage
pub struct KVCacheTensors {
    /// Layer index -> (accumulated keys, accumulated values)
    tensors: HashMap<usize, (Tensor, Tensor)>,
    /// Maximum sequence length supported
    max_seq_len: usize,
    /// Current position in the sequence
    current_pos: usize,
    /// Device for tensor operations
    device: Device,
    /// Data type for cache tensors
    dtype: DType,
}

impl KVCacheTensors {
    /// Create new KV cache tensor storage
    pub fn new(max_seq_len: usize, dtype: DType, device: Device) -> Self {
        Self {
            tensors: HashMap::new(),
            max_seq_len,
            current_pos: 0,
            device,
            dtype,
        }
    }
    
    /// Append new key-value pairs to cache for a layer
    pub fn append_kv(
        &mut self,
        layer_idx: usize,
        new_keys: &Tensor,
        new_values: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Get dimensions
        let (batch_size, seq_len, n_kv_heads, head_dim) = match new_keys.dims() {
            &[b, s, h, d] => (b, s, h, d),
            _ => return Err(candle_core::Error::Msg(
                format!("Expected 4D tensor for keys, got {:?}", new_keys.dims())
            )),
        };
        
        // Check if we're within limits
        if self.current_pos + seq_len > self.max_seq_len {
            return Err(candle_core::Error::Msg(
                format!("Cache overflow: {} + {} > {}", 
                    self.current_pos, seq_len, self.max_seq_len)
            ));
        }
        
        // Get or create cache tensors for this layer
        let (cached_keys, cached_values) = if let Some((k, v)) = self.tensors.get(&layer_idx) {
            // Concatenate with existing cache
            let new_k = Tensor::cat(&[k, new_keys], 1)?;
            let new_v = Tensor::cat(&[v, new_values], 1)?;
            (new_k, new_v)
        } else {
            // First append - just clone the tensors
            (new_keys.clone(), new_values.clone())
        };
        
        // Update cache
        self.tensors.insert(layer_idx, (cached_keys.clone(), cached_values.clone()));
        self.current_pos += seq_len;
        
        Ok((cached_keys, cached_values))
    }
    
    /// Get cached key-value pairs for a layer
    pub fn get_kv(&self, layer_idx: usize) -> Option<(Tensor, Tensor)> {
        self.tensors.get(&layer_idx).cloned()
    }
    
    /// Update cache with rotated position embeddings (for RoPE)
    pub fn apply_rope(
        &mut self,
        layer_idx: usize,
        rope_freqs: &Tensor,
        start_pos: usize,
    ) -> Result<()> {
        if let Some((keys, values)) = self.tensors.get_mut(&layer_idx) {
            // Apply RoPE to keys only (values don't use positional encoding)
            let rotated_keys = Self::rotate_embeddings(keys, rope_freqs, start_pos)?;
            *keys = rotated_keys;
        }
        Ok(())
    }
    
    /// Rotate embeddings for RoPE
    fn rotate_embeddings(
        tensor: &Tensor,
        freqs: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        // Split tensor into first and second half
        let (_batch, seq_len, _heads, head_dim) = match tensor.dims() {
            &[b, s, h, d] => (b, s, h, d),
            _ => return Ok(tensor.clone()),
        };
        
        let half_dim = head_dim / 2;
        let first_half = tensor.narrow(3, 0, half_dim)?;
        let second_half = tensor.narrow(3, half_dim, half_dim)?;
        
        // Apply rotation
        let cos = freqs.narrow(0, start_pos, seq_len)?.narrow(1, 0, 1)?;
        let sin = freqs.narrow(0, start_pos, seq_len)?.narrow(1, 1, 1)?;
        
        let rotated_first = (&first_half * cos.broadcast_as(first_half.shape())?)?
            .sub(&(&second_half * sin.broadcast_as(second_half.shape())?)?)?;
        let rotated_second = (&first_half * sin.broadcast_as(first_half.shape())?)?
            .add(&(&second_half * cos.broadcast_as(second_half.shape())?)?)?;
        
        // Concatenate back
        Tensor::cat(&[&rotated_first, &rotated_second], 3)
    }
    
    /// Clear cache for specific layer
    pub fn clear_layer(&mut self, layer_idx: usize) {
        self.tensors.remove(&layer_idx);
    }
    
    /// Clear entire cache
    pub fn clear(&mut self) {
        self.tensors.clear();
        self.current_pos = 0;
    }
    
    /// Get current sequence position
    pub fn position(&self) -> usize {
        self.current_pos
    }
    
    /// Reset position without clearing tensors (for reuse)
    pub fn reset_position(&mut self) {
        self.current_pos = 0;
    }
    
    /// Slice cache to specific length
    pub fn slice_to_length(&mut self, length: usize) -> Result<()> {
        if length >= self.current_pos {
            return Ok(());
        }
        
        for (keys, values) in self.tensors.values_mut() {
            *keys = keys.narrow(1, 0, length)?;
            *values = values.narrow(1, 0, length)?;
        }
        
        self.current_pos = length;
        Ok(())
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.tensors.values().map(|(k, v)| {
            let k_size: usize = k.dims().iter().product();
            let v_size: usize = v.dims().iter().product();
            let bytes_per_element = match self.dtype {
                DType::F16 | DType::BF16 => 2,
                DType::F32 => 4,
                DType::F64 => 8,
                _ => 4,
            };
            (k_size + v_size) * bytes_per_element
        }).sum()
    }
}

/// Optimized KV cache for grouped query attention (GQA)
pub struct GQAKVCache {
    /// Base KV cache tensors
    base_cache: KVCacheTensors,
    /// Number of key-value groups
    n_kv_groups: usize,
    /// Number of query heads
    n_query_heads: usize,
    /// Repeat factor for GQA
    repeat_factor: usize,
}

impl GQAKVCache {
    /// Create new GQA-optimized KV cache
    pub fn new(
        max_seq_len: usize,
        n_kv_groups: usize,
        n_query_heads: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        let repeat_factor = n_query_heads / n_kv_groups;
        
        Self {
            base_cache: KVCacheTensors::new(max_seq_len, dtype, device),
            n_kv_groups,
            n_query_heads,
            repeat_factor,
        }
    }
    
    /// Append KV pairs and expand for GQA
    pub fn append_and_expand(
        &mut self,
        layer_idx: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Append to base cache
        let (cached_k, cached_v) = self.base_cache.append_kv(layer_idx, keys, values)?;
        
        // Expand KV groups to match query heads if needed
        if self.repeat_factor > 1 {
            let expanded_k = Self::expand_kv_groups(&cached_k, self.repeat_factor)?;
            let expanded_v = Self::expand_kv_groups(&cached_v, self.repeat_factor)?;
            Ok((expanded_k, expanded_v))
        } else {
            Ok((cached_k, cached_v))
        }
    }
    
    /// Expand KV groups by repeating for GQA
    fn expand_kv_groups(tensor: &Tensor, repeat_factor: usize) -> Result<Tensor> {
        if repeat_factor == 1 {
            return Ok(tensor.clone());
        }
        
        let (batch, seq_len, n_kv_heads, head_dim) = match tensor.dims() {
            &[b, s, h, d] => (b, s, h, d),
            _ => return Ok(tensor.clone()),
        };
        
        // Reshape to add repeat dimension
        let reshaped = tensor.reshape((batch, seq_len, n_kv_heads, 1, head_dim))?;
        
        // Repeat along the new dimension
        let repeated = reshaped.repeat(&[1, 1, 1, repeat_factor, 1])?;
        
        // Reshape back to original dimensions with expanded heads
        repeated.reshape((batch, seq_len, n_kv_heads * repeat_factor, head_dim))
    }
    
    /// Get expanded KV pairs for layer
    pub fn get_expanded_kv(&self, layer_idx: usize) -> Option<Result<(Tensor, Tensor)>> {
        self.base_cache.get_kv(layer_idx).map(|(k, v)| {
            if self.repeat_factor > 1 {
                match (
                    Self::expand_kv_groups(&k, self.repeat_factor),
                    Self::expand_kv_groups(&v, self.repeat_factor),
                ) {
                    (Ok(expanded_k), Ok(expanded_v)) => Ok((expanded_k, expanded_v)),
                    (Err(e), _) | (_, Err(e)) => Err(e),
                }
            } else {
                Ok((k, v))
            }
        })
    }
    
    /// Clear cache
    pub fn clear(&mut self) {
        self.base_cache.clear();
    }
    
    /// Get memory savings ratio compared to non-GQA cache
    pub fn memory_savings_ratio(&self) -> f32 {
        self.repeat_factor as f32
    }
}

/// Sliding window KV cache for limited context
pub struct SlidingWindowCache {
    base_cache: KVCacheTensors,
    window_size: usize,
    stride: usize,
}

impl SlidingWindowCache {
    pub fn new(
        window_size: usize,
        stride: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            base_cache: KVCacheTensors::new(window_size, dtype, device),
            window_size,
            stride,
        }
    }
    
    /// Append with sliding window
    pub fn append_with_window(
        &mut self,
        layer_idx: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = keys.dims()[1];
        
        // If adding would exceed window, slide
        if self.base_cache.position() + seq_len > self.window_size {
            self.slide_window()?;
        }
        
        self.base_cache.append_kv(layer_idx, keys, values)
    }
    
    /// Slide the window by stride
    fn slide_window(&mut self) -> Result<()> {
        for (keys, values) in self.base_cache.tensors.values_mut() {
            let current_len = keys.dims()[1];
            if current_len > self.stride {
                *keys = keys.narrow(1, self.stride, current_len - self.stride)?;
                *values = values.narrow(1, self.stride, current_len - self.stride)?;
            }
        }
        
        let new_pos = self.base_cache.position().saturating_sub(self.stride);
        self.base_cache.current_pos = new_pos;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kv_cache_append() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KVCacheTensors::new(100, DType::F32, device);
        
        // Create test tensors [batch=1, seq=10, heads=4, dim=64]
        let keys = Tensor::randn(0.0f32, 1.0, &[1, 10, 4, 64], &device)?;
        let values = Tensor::randn(0.0f32, 1.0, &[1, 10, 4, 64], &device)?;
        
        // Append to cache
        let (cached_k, cached_v) = cache.append_kv(0, &keys, &values)?;
        
        assert_eq!(cached_k.dims(), &[1, 10, 4, 64]);
        assert_eq!(cached_v.dims(), &[1, 10, 4, 64]);
        assert_eq!(cache.position(), 10);
        
        Ok(())
    }
    
    #[test]
    fn test_gqa_cache() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = GQAKVCache::new(100, 4, 16, DType::F32, device);
        
        // Create test tensors with 4 KV heads
        let keys = Tensor::randn(0.0f32, 1.0, &[1, 10, 4, 64], &device)?;
        let values = Tensor::randn(0.0f32, 1.0, &[1, 10, 4, 64], &device)?;
        
        // Append and expand
        let (expanded_k, expanded_v) = cache.append_and_expand(0, &keys, &values)?;
        
        // Should expand to 16 heads (4x repeat)
        assert_eq!(expanded_k.dims(), &[1, 10, 16, 64]);
        assert_eq!(expanded_v.dims(), &[1, 10, 16, 64]);
        
        Ok(())
    }
}
