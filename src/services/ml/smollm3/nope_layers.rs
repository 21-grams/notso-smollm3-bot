//! NoPE (No Position Encoding) layer handling

pub struct NopeHandler {
    nope_layers: Vec<usize>,
}

impl NopeHandler {
    pub fn new(nope_layers: Vec<usize>) -> Self {
        Self { nope_layers }
    }
    
    /// Check if RoPE should be skipped for this layer
    pub fn should_skip_rope(&self, layer_idx: usize) -> bool {
        self.nope_layers.contains(&layer_idx)
    }
    
    /// Get list of NoPE layers
    pub fn nope_layers(&self) -> &[usize] {
        &self.nope_layers
    }
    
    /// Apply NoPE logic to attention scores if needed
    pub fn process_attention(&self, layer_idx: usize, scores: &mut candle_core::Tensor) -> candle_core::Result<()> {
        if self.should_skip_rope(layer_idx) {
            // NoPE layers rely on content-based attention only
            // No additional processing needed here, RoPE is already skipped
            tracing::trace!("NoPE layer {} using content-based attention", layer_idx);
        }
        Ok(())
    }
}

impl Default for NopeHandler {
    fn default() -> Self {
        // Default NoPE layers for SmolLM3
        Self::new(vec![3, 7, 11, 15, 19, 23, 27, 31, 35])
    }
}
