//! Test program to understand Candle tensor shape requirements

use candle_core::{Device, DType, Tensor, Result};

fn main() -> Result<()> {
    println!("=== Candle Tensor Shape Research ===\n");
    
    let device = Device::Cpu;
    
    // Test 1: Standard matmul with 2D tensors
    println!("Test 1: Standard 2D matmul");
    let a = Tensor::randn(0f32, 1., (253, 2048), &device)?;
    let b = Tensor::randn(0f32, 1., (2048, 2048), &device)?;
    let c = a.matmul(&b)?;
    println!("  [253, 2048] @ [2048, 2048] = {:?}", c.dims());
    
    // Test 2: 3D x 2D matmul - Does this work?
    println!("\nTest 2: 3D x 2D matmul (will this fail?)");
    let a_3d = Tensor::randn(0f32, 1., (1, 253, 2048), &device)?;
    match a_3d.matmul(&b) {
        Ok(result) => println!("  SUCCESS: [1, 253, 2048] @ [2048, 2048] = {:?}", result.dims()),
        Err(e) => println!("  FAILED: {}", e),
    }
    
    // Test 3: broadcast_matmul if it exists
    println!("\nTest 3: broadcast_matmul (if available)");
    match a_3d.broadcast_matmul(&b) {
        Ok(result) => println!("  SUCCESS: broadcast_matmul works = {:?}", result.dims()),
        Err(e) => println!("  FAILED: {}", e),
    }
    
    // Test 4: Squeeze/Unsqueeze approach
    println!("\nTest 4: Squeeze/Unsqueeze workaround");
    let a_2d = a_3d.squeeze(0)?;
    println!("  After squeeze(0): {:?}", a_2d.dims());
    let result = a_2d.matmul(&b)?;
    println!("  After matmul: {:?}", result.dims());
    let result_3d = result.unsqueeze(0)?;
    println!("  After unsqueeze(0): {:?}", result_3d.dims());
    
    // Test 5: How does embedding work?
    println!("\nTest 5: Embedding lookup");
    let vocab_size = 128256;
    let hidden_size = 2048;
    let embed_weight = Tensor::randn(0f32, 1., (vocab_size, hidden_size), &device)?;
    
    // Input tokens [batch_size=1, seq_len=253]
    let token_ids = Tensor::arange(0u32, 253u32, &device)?;
    println!("  Token IDs shape: {:?}", token_ids.dims());
    
    // Embedding lookup
    let embedded = embed_weight.embedding(&token_ids)?;
    println!("  After embedding: {:?}", embedded.dims());
    
    // With batch dimension
    let token_ids_batch = token_ids.unsqueeze(0)?;
    println!("  Token IDs with batch: {:?}", token_ids_batch.dims());
    
    // Flatten for embedding
    let token_ids_flat = token_ids_batch.flatten(0, 1)?;
    println!("  Flattened: {:?}", token_ids_flat.dims());
    let embedded_flat = embed_weight.embedding(&token_ids_flat)?;
    println!("  Embedded flat: {:?}", embedded_flat.dims());
    let embedded_reshaped = embedded_flat.reshape((1, 253, hidden_size))?;
    println!("  Reshaped to batch: {:?}", embedded_reshaped.dims());
    
    // Test 6: How quantized tensors work
    println!("\nTest 6: Shapes with quantized tensors");
    println!("  Note: Actual QTensor testing requires loading from GGUF");
    println!("  But the principle is: QTensor weights stay same shape");
    println!("  [2048, 2048] weight remains [2048, 2048] when quantized");
    
    Ok(())
}
