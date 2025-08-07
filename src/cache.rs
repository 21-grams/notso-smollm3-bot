use candle_core::{Device, Tensor, Result};

/// KV Cache for efficient autoregressive generation
pub struct KVCache {
    k_cache: Vec<Option<Tensor>>,
    v_cache: Vec<Option<Tensor>>,
    cache_length: usize,
    max_length: usize,
    device: Device,
}

impl KVCache {
    pub fn new(num_layers: usize, max_length: usize, device: &Device) -> Self {
        Self {
            k_cache: vec![None; num_layers],
            v_cache: vec![None; num_layers],
            cache_length: 0,
            max_length,
            device: device.clone(),
        }
    }
    
    /// Get cached K,V for a layer
    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        match (self.k_cache.get(layer_idx)?, self.v_cache.get(layer_idx)?) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
    
    /// Update cache for a layer
    pub fn update(&mut self, layer_idx: usize, k: Tensor, v: Tensor) -> Result<()> {
        // Get sequence length from tensor shape
        let seq_len = k.dim(2)?;
        
        // Update or extend cache
        match self.get(layer_idx) {
            Some((prev_k, prev_v)) if self.cache_length > 0 => {
                // Concatenate with existing cache
                let new_k = Tensor::cat(&[prev_k, &k], 2)?;
                let new_v = Tensor::cat(&[prev_v, &v], 2)?;
                
                self.k_cache[layer_idx] = Some(new_k);
                self.v_cache[layer_idx] = Some(new_v);
            }
            _ => {
                // First update or reset
                self.k_cache[layer_idx] = Some(k);
                self.v_cache[layer_idx] = Some(v);
            }
        }
        
        self.cache_length = self.cache_length.max(seq_len);
        Ok(())
    }
    
    /// Clear the cache
    pub fn clear(&mut self) {
        for k in &mut self.k_cache {
            *k = None;
        }
        for v in &mut self.v_cache {
            *v = None;
        }
        self.cache_length = 0;
    }
    
    /// Get current cache length
    pub fn len(&self) -> usize {
        self.cache_length
    }
    
    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache_length == 0
    }
}
