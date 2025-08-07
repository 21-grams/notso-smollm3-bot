# Candle.rs Reference

## Official APIs Used

### Model Loading

```rust
use candle_transformers::models::quantized_llama::{Llama, ModelWeights, LlamaConfig};
use candle_core::quantized::gguf_file;

// Load GGUF file
let content = gguf_file::Content::read(&mut file)?;

// Load weights
let weights = ModelWeights::from_gguf(content, &mut file, &device)?;

// Create model
let model = Llama::load(&weights, &config, &device)?;
```

### Quantized Operations (Q4_K_M)

```rust
use candle_core::quantized::{QTensor, QMatMul, GgmlDType};

// Validate Q4_K_M
if qtensor.dtype() == GgmlDType::Q4K {
    // Direct quantized matmul (NO dequantization!)
    let qmatmul = QMatMul::from_qtensor(&qtensor)?;
    let result = qmatmul.forward(&input)?;
}

// Alternative direct operation
let result = input.quantized_matmul(&qtensor)?;
```

### Device Management

```rust
use candle_core::Device;

// CPU
let device = Device::Cpu;

// CUDA
let device = Device::new_cuda(0)?;

// Metal (Apple Silicon)
let device = Device::new_metal(0)?;
```

### Tensor Operations

```rust
use candle_core::{Tensor, DType};

// Create tensor
let tensor = Tensor::new(&[1.0, 2.0, 3.0], &device)?;

// Reshape
let reshaped = tensor.reshape((1, 3))?;

// Matrix multiplication
let result = tensor1.matmul(&tensor2)?;

// Activation
let activated = tensor.silu()?;
```

### Generation Utilities

```rust
use candle_transformers::generation::LogitsProcessor;

let processor = LogitsProcessor::new(
    seed,           // Random seed
    Some(0.7),      // Temperature
    Some(0.9),      // Top-p
);

let next_token = processor.sample(&logits)?;
```

## Key Patterns

### 1. No Dequantization

```rust
// ❌ WRONG - Causes 100x slowdown
let result = qtensor.dequantize(&device)?.matmul(&input)?;

// ✅ CORRECT - Direct quantized operation
let result = QMatMul::from_qtensor(&qtensor)?.forward(&input)?;
```

### 2. Shape Validation

```rust
// Always validate tensor shapes
let shape = tensor.shape();
assert_eq!(shape.dims(), &[batch, seq_len, hidden]);
```

### 3. Error Handling

```rust
use candle_core::Result;

fn operation() -> Result<Tensor> {
    // Candle uses Result<T> for all operations
    let tensor = Tensor::new(&data, &device)?;
    Ok(tensor)
}
```

## Performance Tips

1. **Use quantized operations directly** - Never dequantize unless absolutely necessary
2. **Batch operations** - Process multiple items together
3. **Reuse tensors** - Avoid creating new tensors unnecessarily
4. **Profile with `--release`** - Debug builds are much slower
5. **Enable CPU optimizations** - Use `RUSTFLAGS="-C target-cpu=native"`

## Common Issues

### Shape Mismatch
- GGUF weights may be transposed
- Check actual shapes with `tensor.shape()`

### Device Mismatch
- Ensure all tensors are on same device
- Use `tensor.to_device(&device)?` to move

### Memory Issues
- Q4_K_M uses ~4.5 bits per weight
- Monitor with `tensor.dtype()` and `tensor.nbytes()`
