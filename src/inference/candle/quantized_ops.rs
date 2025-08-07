use candle_core::{Tensor, Result};
use candle_core::quantized::{QTensor, QMatMul, GgmlDType};

pub struct QuantizedOps;

impl QuantizedOps {
    /// Direct Q4_K_M matrix multiplication without dequantization
    pub fn q4km_matmul(input: &Tensor, weight: &QTensor) -> Result<Tensor> {
        // Validate Q4_K_M format
        if weight.dtype() != GgmlDType::Q4K {
            candle_core::bail!("Expected Q4_K_M tensor, got {:?}", weight.dtype());
        }
        
        // Use official QMatMul for direct quantized operations
        let qmatmul = QMatMul::from_qtensor(weight)?;
        qmatmul.forward(input)
    }
    
    /// Alternative direct quantized matmul
    pub fn direct_quantized_matmul(input: &Tensor, qtensor: &QTensor) -> Result<Tensor> {
        // Direct operation without QMatMul wrapper
        input.quantized_matmul(qtensor)
    }
    
    /// Check if dequantization is needed (should be avoided)
    pub fn requires_dequantization(operation: &str) -> bool {
        // Most operations can work with quantized tensors
        matches!(operation, "complex_activation" | "custom_op")
    }
}
