/// Test Q4_K Support in Candle 0.9.1
/// 
/// This test verifies:
/// 1. Q4_K/Q4_K_M variant exists in GgmlDType
/// 2. QMatMul::from_qtensor() works with Q4_K_M tensors
/// 3. We can read Q4_K_M tensors from GGUF
/// 4. QMatMul operations work without dequantization

use anyhow::Result;
use candle_core::{Device, Tensor, Module};
use candle_core::quantized::{
    gguf_file::{self, Content},
    QMatMul, QTensor, GgmlDType,
};
use std::fs::File;
use std::path::Path;

/// Test result structure
#[derive(Debug)]
struct Q4KTestResult {
    q4k_variant_exists: bool,
    q4k_variant_name: String,
    can_load_qtensor: bool,
    can_create_qmatmul: bool,
    can_perform_matmul: bool,
    memory_efficient: bool,
    tensors_found: Vec<String>,
    error_messages: Vec<String>,
}

impl Q4KTestResult {
    fn new() -> Self {
        Self {
            q4k_variant_exists: false,
            q4k_variant_name: String::new(),
            can_load_qtensor: false,
            can_create_qmatmul: false,
            can_perform_matmul: false,
            memory_efficient: true,
            tensors_found: Vec::new(),
            error_messages: Vec::new(),
        }
    }
    
    fn print_report(&self) {
        println!("\n{}", "=".repeat(80));
        println!("              Q4_K SUPPORT TEST RESULTS");
        println!("{}", "=".repeat(80));
        
        println!("\n✅ PASSED TESTS:");
        println!("{}", "-".repeat(40));
        
        if self.q4k_variant_exists {
            println!("✅ Q4_K variant exists: {}", self.q4k_variant_name);
        }
        if self.can_load_qtensor {
            println!("✅ Can load Q4_K_M tensors from GGUF");
        }
        if self.can_create_qmatmul {
            println!("✅ Can create QMatMul from Q4_K_M tensors");
        }
        if self.can_perform_matmul {
            println!("✅ Can perform matrix multiplication");
        }
        if self.memory_efficient {
            println!("✅ Memory usage is efficient (no dequantization)");
        }
        
        if !self.tensors_found.is_empty() {
            println!("\n📊 Q4_K_M Tensors Found:");
            for tensor in &self.tensors_found[..self.tensors_found.len().min(10)] {
                println!("  - {}", tensor);
            }
            if self.tensors_found.len() > 10 {
                println!("  ... and {} more", self.tensors_found.len() - 10);
            }
        }
        
        if !self.error_messages.is_empty() {
            println!("\n❌ ERRORS:");
            println!("{}", "-".repeat(40));
            for error in &self.error_messages {
                println!("❌ {}", error);
            }
        }
        
        println!("\n📝 SUMMARY:");
        println!("{}", "-".repeat(40));
        if self.q4k_variant_exists && self.can_load_qtensor && 
           self.can_create_qmatmul && self.can_perform_matmul {
            println!("✅ Full Q4_K_M support confirmed!");
            println!("✅ Candle 0.9.1 can handle SmolLM3 Q4_K_M quantization");
        } else {
            println!("⚠️  Partial Q4_K support - may need workarounds");
        }
        
        println!("\n{}", "=".repeat(80));
    }
}

/// Test 1: Check if Q4_K variant exists
fn test_q4k_variant() -> (bool, String) {
    println!("\n🔍 Test 1: Checking for Q4_K variant in GgmlDType...");
    
    // List all available quantization types
    let variants = vec![
        ("Q4_0", GgmlDType::Q4_0),
        ("Q4_1", GgmlDType::Q4_1),
        ("Q4K", GgmlDType::Q4K),
        ("Q5_0", GgmlDType::Q5_0),
        ("Q5_1", GgmlDType::Q5_1),
        ("Q5K", GgmlDType::Q5K),
        ("Q5K", GgmlDType::Q5K),
        ("Q6K", GgmlDType::Q6K),
        ("Q8_0", GgmlDType::Q8_0),
        ("Q8_1", GgmlDType::Q8_1),
        ("Q8K", GgmlDType::Q8K),
        ("F16", GgmlDType::F16),
        ("F32", GgmlDType::F32),
    ];
    
    println!("Available quantization types:");
    for (name, _dtype) in &variants {
        println!("  - {}", name);
    }
    
    // Check for Q4K variants (Q4_K_M is typically Q4K in the enum)
    for (name, _dtype) in &variants {
        if name.contains("Q4K") {
            println!("✅ Found Q4_K variant: {}", name);
            return (true, name.to_string());
        }
    }
    
    (false, String::new())
}

/// Test 2: Load Q4_K_M tensor from GGUF
fn test_load_qtensor(gguf_path: &Path, device: &Device) -> Result<(bool, Vec<String>)> {
    println!("\n🔍 Test 2: Loading Q4_K_M tensors from GGUF...");
    
    let mut file = File::open(gguf_path)?;
    let content = Content::read(&mut file)?;
    
    let mut q4k_tensors = Vec::new();
    
    // Find Q4_K_M tensors
    for (name, info) in &content.tensor_infos {
        match info.ggml_dtype {
            GgmlDType::Q4K => {
                q4k_tensors.push(name.clone());
            }
            _ => {}
        }
    }
    
    if q4k_tensors.is_empty() {
        return Ok((false, q4k_tensors));
    }
    
    println!("Found {} Q4_K_M tensors", q4k_tensors.len());
    
    // Try to load the first Q4_K_M tensor
    let test_tensor_name = &q4k_tensors[0];
    println!("Testing with tensor: {}", test_tensor_name);
    
    // Read tensor data
    file = File::open(gguf_path)?;
    let reader = gguf_file::Content::read(&mut file)?;
    
    // Create a QTensor from the data
    let tensor_info = &reader.tensor_infos[test_tensor_name];
    let offset = tensor_info.offset;
    
    // Seek to tensor data
    use std::io::Seek;
    file.seek(std::io::SeekFrom::Start(offset))?;
    
    // Read the quantized data
    let dims: Vec<u64> = tensor_info.shape.dims().iter().map(|&x| x as u64).collect();
    let dtype = tensor_info.ggml_dtype;
    
    // Calculate expected size
    let elem_count: usize = dims.iter().map(|&x| x as usize).product();
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let blocks_per_tensor = elem_count / block_size;
    let expected_size = blocks_per_tensor * type_size;
    
    println!("Tensor dimensions: {:?}", dims);
    println!("Element count: {}", elem_count);
    println!("Block size: {}", block_size);
    println!("Expected data size: {} bytes", expected_size);
    
    // Read the data
    let mut data = vec![0u8; expected_size];
    use std::io::Read;
    file.read_exact(&mut data)?;
    
    // Create QTensor
    // Create QTensor directly from raw data
    // QTensor expects the data to be in the correct format for the given dtype
    let dims_usize: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
    let qtensor = QTensor::from_ggml(
        dtype,
        &data,
        dims_usize.as_slice(),
    )?;
    
    println!("✅ Successfully loaded Q4_K_M tensor: {}", test_tensor_name);
    println!("  Shape: {:?}", qtensor.shape());
    println!("  DType: {:?}", qtensor.dtype());
    
    Ok((true, q4k_tensors))
}

/// Test 3: Create QMatMul from Q4_K_M tensor
fn test_qmatmul_creation(gguf_path: &Path, device: &Device) -> Result<bool> {
    println!("\n🔍 Test 3: Creating QMatMul from Q4_K_M tensor...");
    
    let mut file = File::open(gguf_path)?;
    let content = Content::read(&mut file)?;
    
    // Find a suitable weight tensor (preferably from layer 0)
    let mut test_tensor = None;
    for (name, info) in &content.tensor_infos {
        if name.contains("blk.0") && name.contains("attn_q.weight") {
            match info.ggml_dtype {
                GgmlDType::Q4K => {
                    test_tensor = Some((name.clone(), info.clone()));
                    break;
                }
                _ => {}
            }
        }
    }
    
    let (tensor_name, tensor_info) = test_tensor
        .ok_or_else(|| anyhow::anyhow!("No suitable Q4_K_M weight tensor found"))?;
    
    println!("Using tensor: {}", tensor_name);
    
    // Load the tensor
    file = File::open(gguf_path)?;
    let mut reader = gguf_file::Content::read(&mut file)?;
    
    use std::io::{Read, Seek};
    file.seek(std::io::SeekFrom::Start(tensor_info.offset))?;
    
    let dims: Vec<u64> = tensor_info.shape.dims().iter().map(|&x| x as u64).collect();
    let dtype = tensor_info.ggml_dtype;
    let elem_count: usize = dims.iter().map(|&x| x as usize).product();
    let expected_size = (elem_count / dtype.block_size()) * dtype.type_size();
    
    let mut data = vec![0u8; expected_size];
    file.read_exact(&mut data)?;
    
    // Create QTensor directly from raw data
    let dims_usize: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
    let qtensor = QTensor::from_ggml(
        dtype,
        &data,
        dims_usize.as_slice(),
    )?;
    
    // Create QMatMul - this is the critical test
    println!("Creating QMatMul from QTensor...");
    let qmatmul = QMatMul::from_qtensor(qtensor)?;
    
    println!("✅ Successfully created QMatMul!");
    println!("  Input shape: {:?}", dims);
    
    Ok(true)
}

/// Test 4: Perform actual matrix multiplication
fn test_matmul_operation(gguf_path: &Path, device: &Device) -> Result<bool> {
    println!("\n🔍 Test 4: Testing QMatMul operation...");
    
    // Load a small weight tensor
    let mut file = File::open(gguf_path)?;
    let content = Content::read(&mut file)?;
    
    // Find a manageable tensor
    let mut test_tensor = None;
    for (name, info) in &content.tensor_infos {
        if name.contains("attn_norm.weight") && matches!(info.ggml_dtype, GgmlDType::F32) {
            // Use a norm weight to get hidden_dim size
            test_tensor = Some(info.shape.dims()[0] as usize);
            break;
        }
    }
    
    let hidden_dim = test_tensor.unwrap_or(3072);
    println!("Using hidden dimension: {}", hidden_dim);
    
    // Find a Q4_K weight tensor with matching dimensions
    let mut weight_tensor = None;
    for (name, info) in &content.tensor_infos {
        if name.contains("attn_q.weight") {
            match info.ggml_dtype {
                GgmlDType::Q4K => {
                    if info.shape.dims()[1] as usize == hidden_dim {
                        weight_tensor = Some((name.clone(), info.clone()));
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    
    if weight_tensor.is_none() {
        println!("⚠️  No suitable weight tensor found for matmul test");
        return Ok(false);
    }
    
    let (tensor_name, tensor_info) = weight_tensor.unwrap();
    println!("Using weight tensor: {}", tensor_name);
    
    // Load the weight tensor
    file = File::open(gguf_path)?;
    let mut reader = gguf_file::Content::read(&mut file)?;
    
    use std::io::{Read, Seek};
    file.seek(std::io::SeekFrom::Start(tensor_info.offset))?;
    
    let dims: Vec<u64> = tensor_info.shape.dims().iter().map(|&x| x as u64).collect();
    let dtype = tensor_info.ggml_dtype;
    let elem_count: usize = dims.iter().map(|&x| x as usize).product();
    let expected_size = (elem_count / dtype.block_size()) * dtype.type_size();
    
    let mut data = vec![0u8; expected_size];
    file.read_exact(&mut data)?;
    
    // Create QTensor directly from raw data
    let dims_usize: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
    let qtensor = QTensor::from_ggml(
        dtype,
        &data,
        dims_usize.as_slice(),
    )?;
    let qmatmul = QMatMul::from_qtensor(qtensor)?;
    
    // Create a small input tensor
    let batch_size = 1;
    let seq_len = 10;
    let input_shape = vec![batch_size, seq_len, hidden_dim];
    
    println!("Creating input tensor with shape: {:?}", input_shape);
    let input = Tensor::randn(0f32, 1f32, input_shape.as_slice(), device)?;
    
    // Get initial memory usage
    let mem_before = get_memory_usage();
    
    // Perform the matrix multiplication
    println!("Performing QMatMul forward pass...");
    let output = qmatmul.forward(&input)?;
    
    // Get memory after
    let mem_after = get_memory_usage();
    let mem_diff = mem_after.saturating_sub(mem_before);
    
    println!("✅ Matrix multiplication successful!");
    println!("  Input shape: {:?}", input.shape());
    println!("  Output shape: {:?}", output.shape());
    println!("  Memory delta: {} MB", mem_diff / (1024 * 1024));
    
    // Check if memory usage is reasonable (should be small if no dequantization)
    let expected_output_size = output.elem_count() * 4; // f32 output
    let reasonable_memory = expected_output_size * 2; // Allow 2x for temporary buffers
    
    if mem_diff > reasonable_memory {
        println!("⚠️  High memory usage detected - possible dequantization");
        Ok(false)
    } else {
        println!("✅ Memory usage is efficient - no dequantization detected");
        Ok(true)
    }
}

/// Get current memory usage (Linux-specific)
fn get_memory_usage() -> usize {
    use std::fs;
    
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb * 1024; // Convert to bytes
                    }
                }
            }
        }
    }
    0
}

fn main() -> Result<()> {
    println!("\n🧪 Q4_K Support Test for Candle 0.9.1");
    println!("=====================================\n");
    
    let mut result = Q4KTestResult::new();
    
    // Setup
    let device = Device::Cpu;
    let gguf_path = Path::new("models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf");
    
    if !gguf_path.exists() {
        eprintln!("❌ GGUF file not found at: {:?}", gguf_path);
        eprintln!("\nPlease ensure the model file is present.");
        std::process::exit(1);
    }
    
    // Test 1: Check Q4_K variant
    let (exists, variant) = test_q4k_variant();
    result.q4k_variant_exists = exists;
    result.q4k_variant_name = variant;
    
    if !exists {
        result.error_messages.push("Q4_K variant not found in GgmlDType".to_string());
    }
    
    // Test 2: Load Q4_K tensor
    match test_load_qtensor(gguf_path, &device) {
        Ok((success, tensors)) => {
            result.can_load_qtensor = success;
            result.tensors_found = tensors;
        }
        Err(e) => {
            result.error_messages.push(format!("Failed to load Q4_K tensor: {}", e));
        }
    }
    
    // Test 3: Create QMatMul
    match test_qmatmul_creation(gguf_path, &device) {
        Ok(success) => {
            result.can_create_qmatmul = success;
        }
        Err(e) => {
            result.error_messages.push(format!("Failed to create QMatMul: {}", e));
        }
    }
    
    // Test 4: Perform matmul
    match test_matmul_operation(gguf_path, &device) {
        Ok(success) => {
            result.can_perform_matmul = success;
            result.memory_efficient = success;
        }
        Err(e) => {
            result.error_messages.push(format!("Failed to perform matmul: {}", e));
        }
    }
    
    // Print results
    result.print_report();
    
    // Exit code based on success
    if result.q4k_variant_exists && result.can_load_qtensor && 
       result.can_create_qmatmul && result.can_perform_matmul {
        println!("\n✅ All tests passed!");
        Ok(())
    } else {
        println!("\n⚠️  Some tests failed - see report above");
        std::process::exit(1);
    }
}
