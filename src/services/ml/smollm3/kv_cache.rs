//! Pre-allocated KV-cache implementation with slice_assign for zero-copy updates
//!
//! This implementation pre-allocates cache tensors at initialization and uses
//! slice_assign for in-place updates, eliminating allocation overhead during generation.

use candle_core::{Device, DType, Result, Tensor};

/// Pre-allocated KV-cache for efficient generation
pub struct KVCache {
    k_caches: Vec<Option<Tensor>>,
    v_caches: Vec<Option<Tensor>>,
    max_seq_len: usize,
    current_seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: Device,
    dtype: DType,
}

impl KVCache {
    /// Create a new pre-allocated KV-cache
    pub fn new(
        config: &crate::services::ml::official::config::SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        tracing::info!("Creating pre-allocated KV-cache for {} layers", config.base.num_hidden_layers);
        
        // Pre-allocate caches for all layers
        let mut k_caches = Vec::with_capacity(config.base.num_hidden_layers);
        let mut v_caches = Vec::with_capacity(config.base.num_hidden_layers);
        
        // Use F16 for memory efficiency
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        
        for layer_idx in 0..config.base.num_hidden_layers {
            // Pre-allocate with zeros for maximum sequence length
            let k = Tensor::zeros(
                (1, config.base.num_key_value_heads, config.base.max_position_embeddings, config.base.head_dim),
                dtype,
                device
            )?;
            let v = k.clone();
            
            k_caches.push(Some(k));
            v_caches.push(Some(v));
            
            if layer_idx == 0 {
                tracing::debug!(
                    "Layer 0 cache shape: (1, {}, {}, {})",
                    config.base.num_key_value_heads,
                    config.base.max_position_embeddings,
                    config.base.head_dim
                );
            }
        }
        
        tracing::info!(
            "✅ KV-cache initialized: {} layers, max_seq={}, kv_heads={}, head_dim={}",
            config.base.num_hidden_layers,
            config.base.max_position_embeddings,
            config.base.num_key_value_heads,
            config.base.head_dim
        );
        
        Ok(Self {
            k_caches,
            v_caches,
            max_seq_len: config.base.max_position_embeddings,
            current_seq_len: 0,
            num_layers: config.base.num_hidden_layers,
            num_kv_heads: config.base.num_key_value_heads,
            head_dim: config.base.head_dim,
            device: device.clone(),
            dtype,
        })
    }
    
    /// Update cache for a specific layer and return the updated cache
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let start_pos = self.current_seq_len;
        let end_pos = start_pos + seq_len;
        
        if end_pos > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {} exceeds maximum {}",
                end_pos, self.max_seq_len
            )));
        }
        
        // Ensure k and v are in the right dtype
        let k = if k.dtype() != self.dtype {
            k.to_dtype(self.dtype)?
        } else {
            k.clone()
        };
        
        let v = if v.dtype() != self.dtype {
            v.to_dtype(self.dtype)?
        } else {
            v.clone()
        };
        
        // Get mutable references to cache tensors
        let k_cache = self.k_caches[layer_idx]
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg(format!("K-cache not initialized for layer {}", layer_idx)))?;
            
        let v_cache = self.v_caches[layer_idx]
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg(format!("V-cache not initialized for layer {}", layer_idx)))?;
        
        // Use slice_assign for in-place update
        // Shape: (1, num_kv_heads, seq_len, head_dim)
        *k_cache = k_cache.slice_assign(
            &[0..1, 0..self.num_kv_heads, start_pos..end_pos, 0..self.head_dim],
            &k
        )?;
        
        *v_cache = v_cache.slice_assign(
            &[0..1, 0..self.num_kv_heads, start_pos..end_pos, 0..self.head_dim],
            &v
        )?;
        
        // Return only the used portion of the cache
        let k_ret = k_cache.narrow(2, 0, end_pos)?;
        let v_ret = v_cache.narrow(2, 0, end_pos)?;
        
        if layer_idx == 0 {
            tracing::trace!(
                "Layer 0 cache updated: start={}, end={}, returned shape: {:?}",
                start_pos, end_pos, k_ret.dims()
            );
        }
        
        Ok((k_ret, v_ret))
    }
    
    /// Get or update cache for a layer (compatibility method)
    pub fn get_or_update(
        &mut self,
        layer_idx: usize,
        k_new: &Tensor,
        v_new: &Tensor,
        position: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = k_new.dim(2)?;
        
        // If this is the first update at this position, update current_seq_len
        if position == self.current_seq_len {
            self.current_seq_len = position;
        }
        
        self.update(layer_idx, k_new, v_new, seq_len)
    }
    
    /// Update the current sequence position
    pub fn update_position(&mut self, new_position: usize) {
        self.current_seq_len = new_position;
    }
    
    /// Reset cache without deallocation
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        // Don't deallocate, just reset position
        tracing::debug!("KV-cache reset (position set to 0, memory retained)");
    }
    
    /// Clear cache and free memory (for memory-constrained scenarios)
    pub fn clear(&mut self) -> Result<()> {
        self.current_seq_len = 0;
        
        // Re-allocate with zeros
        for i in 0..self.num_layers {
            let k = Tensor::zeros(
                (1, self.num_kv_heads, self.max_seq_len, self.head_dim),
                self.dtype,
                &self.device
            )?;
            let v = k.clone();
            
            self.k_caches[i] = Some(k);
            self.v_caches[i] = Some(v);
        }
        
        tracing::debug!("KV-cache cleared and re-allocated");
        Ok(())
    }
    
    /// Get cache for a specific layer (for compatibility)
    pub fn get(&self, layer_idx: usize) -> Option<(Tensor, Tensor)> {
        match (&self.k_caches[layer_idx], &self.v_caches[layer_idx]) {
            (Some(k), Some(v)) if self.current_seq_len > 0 => {
                // Return only the used portion
                match (k.narrow(2, 0, self.current_seq_len), v.narrow(2, 0, self.current_seq_len)) {
                    (Ok(k_used), Ok(v_used)) => Some((k_used, v_used)),
                    _ => None,
                }
            },
            _ => None,
        }
    }
    
    /// Get current sequence length
    pub fn current_len(&self) -> usize {
        self.current_seq_len
    }
    
    /// Get maximum sequence length
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Batch KV-cache for handling multiple sequences (future extension)
pub struct BatchKVCache {
    caches: Vec<KVCache>,
    batch_size: usize,
}

impl BatchKVCache {
    /// Create a batch of KV-caches
    pub fn new(
        batch_size: usize,
        config: &crate::services::ml::official::config::SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            caches.push(KVCache::new(config, device)?);
        }
        
        Ok(Self {
            caches,
            batch_size,
        })
    }
    
    /// Get cache for a specific batch element
    pub fn get_batch(&mut self, batch_idx: usize) -> Result<&mut KVCache> {
        self.caches.get_mut(batch_idx)
            .ok_or_else(|| candle_core::Error::Msg(format!("Invalid batch index: {}", batch_idx)))
    }
    
    /// Reset all caches in the batch
    pub fn reset_all(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }
}
