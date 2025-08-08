//! Low-level Candle device operations and utilities
//! 
//! This module provides the foundational device management layer that
//! the higher-level services build upon. It handles raw device operations,
//! memory management, and device-specific optimizations.

use candle_core::{Device, Result, DType, Tensor};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Low-level device handle with memory tracking
pub struct CandleDevice {
    device: Device,
    memory_used: Arc<AtomicUsize>,
    memory_limit: Option<usize>,
}

impl CandleDevice {
    /// Create a new device handle with optional memory limit
    pub fn new(device: Device, memory_limit: Option<usize>) -> Self {
        Self {
            device,
            memory_used: Arc::new(AtomicUsize::new(0)),
            memory_limit,
        }
    }
    
    /// Create device with automatic selection
    pub fn auto_select() -> Result<Self> {
        let device = Self::detect_best_device()?;
        let memory_limit = Self::get_device_memory_limit(&device);
        Ok(Self::new(device, memory_limit))
    }
    
    /// Detect the best available device
    fn detect_best_device() -> Result<Device> {
        // Check for CUDA
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Self::probe_cuda() {
                tracing::debug!("Detected CUDA device with compute capability");
                return Ok(device);
            }
        }
        
        // Check for Metal on macOS
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Ok(device) = Device::new_metal(0) {
                tracing::debug!("Detected Metal device");
                return Ok(device);
            }
        }
        
        // Fallback to CPU with optimization hints
        tracing::debug!("Using CPU device with optimization hints");
        Ok(Device::Cpu)
    }
    
    #[cfg(feature = "cuda")]
    fn probe_cuda() -> Result<Device> {
        // Try to get device with best compute capability
        let device = Device::new_cuda(0)?;
        
        // Verify CUDA is actually available by creating a small tensor
        let test = Tensor::zeros(&[1], DType::F32, &device)?;
        let _ = test.to_vec1::<f32>()?;
        
        Ok(device)
    }
    
    /// Get memory limit for device type
    fn get_device_memory_limit(device: &Device) -> Option<usize> {
        match device {
            Device::Cuda(_) => {
                // Conservative limit for CUDA - 80% of available
                // For Q4_K_M quantized SmolLM3, we expect ~1.5GB model + 0.5GB overhead
                Some(2 * 1024 * 1024 * 1024) // 2GB limit
            }
            Device::Cpu => {
                // More conservative for CPU to avoid system issues
                Some(4 * 1024 * 1024 * 1024) // 4GB limit
            }
            _ => None,
        }
    }
    
    /// Allocate tensor with memory tracking
    pub fn allocate_tensor(
        &self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        let size = Self::calculate_tensor_size(shape, dtype);
        
        // Check memory limit
        if let Some(limit) = self.memory_limit {
            let current = self.memory_used.load(Ordering::Relaxed);
            if current + size > limit {
                return Err(candle_core::Error::Msg(format!(
                    "Memory limit exceeded: {} + {} > {}",
                    current, size, limit
                )));
            }
        }
        
        let tensor = Tensor::zeros(shape, dtype, &self.device)?;
        self.memory_used.fetch_add(size, Ordering::Relaxed);
        Ok(tensor)
    }
    
    /// Calculate tensor size in bytes
    fn calculate_tensor_size(shape: &[usize], dtype: DType) -> usize {
        let num_elements: usize = shape.iter().product();
        let bytes_per_element = match dtype {
            DType::U8 => 1,
            DType::U32 => 4,
            DType::I64 => 8,
            DType::BF16 => 2,
            DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        };
        num_elements * bytes_per_element
    }
    
    /// Transfer tensor between devices
    pub fn transfer_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device().location() == self.device.location() {
            Ok(tensor.clone())
        } else {
            tensor.to_device(&self.device)
        }
    }
    
    /// Synchronize device operations (for CUDA)
    pub fn synchronize(&self) -> Result<()> {
        match &self.device {
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_device) => {
                cuda_device.synchronize()?;
            }
            _ => {
                // No-op for CPU and other devices
            }
        }
        Ok(())
    }
    
    /// Get device capabilities
    pub fn capabilities(&self) -> DeviceCapabilities {
        match &self.device {
            Device::Cuda(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: true,
                supports_int8: true,
                optimal_batch_size: 8,
                max_threads: 1024,
            },
            Device::Cpu => DeviceCapabilities {
                supports_f16: false,
                supports_bf16: false,
                supports_int8: true,
                optimal_batch_size: 1,
                max_threads: num_cpus::get(),
            },
            _ => DeviceCapabilities::default(),
        }
    }
    
    /// Clear memory tracking
    pub fn clear_memory_tracking(&self) {
        self.memory_used.store(0, Ordering::Relaxed);
    }
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.memory_used.load(Ordering::Relaxed)
    }
    
    /// Get underlying device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Check if device supports operation
    pub fn supports_operation(&self, op: DeviceOperation) -> bool {
        match (&self.device, op) {
            (Device::Cuda(_), DeviceOperation::MatMul) => true,
            (Device::Cuda(_), DeviceOperation::Attention) => true,
            (Device::Cuda(_), DeviceOperation::FlashAttention) => false, // Not yet in Candle
            (Device::Cpu, DeviceOperation::MatMul) => true,
            (Device::Cpu, DeviceOperation::Attention) => true,
            (Device::Cpu, DeviceOperation::FlashAttention) => false,
            _ => false,
        }
    }
    
    /// Optimize tensor layout for device
    pub fn optimize_layout(&self, tensor: &Tensor) -> Result<Tensor> {
        match &self.device {
            Device::Cuda(_) => {
                // Ensure contiguous for CUDA operations
                if !tensor.is_contiguous() {
                    tensor.contiguous()
                } else {
                    Ok(tensor.clone())
                }
            }
            _ => Ok(tensor.clone()),
        }
    }
}

/// Device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub optimal_batch_size: usize,
    pub max_threads: usize,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            supports_f16: false,
            supports_bf16: false,
            supports_int8: false,
            optimal_batch_size: 1,
            max_threads: 1,
        }
    }
}

/// Supported device operations
#[derive(Debug, Clone, Copy)]
pub enum DeviceOperation {
    MatMul,
    Attention,
    FlashAttention,
}

/// Device memory pool for efficient allocation
pub struct DeviceMemoryPool {
    device: Arc<CandleDevice>,
    free_tensors: Vec<(Vec<usize>, DType, Tensor)>,
    max_pool_size: usize,
}

impl DeviceMemoryPool {
    pub fn new(device: Arc<CandleDevice>, max_pool_size: usize) -> Self {
        Self {
            device,
            free_tensors: Vec::new(),
            max_pool_size,
        }
    }
    
    /// Get or allocate tensor from pool
    pub fn get_tensor(
        &mut self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        // Try to find matching tensor in pool
        if let Some(pos) = self.free_tensors.iter().position(|(s, d, _)| {
            s == shape && *d == dtype
        }) {
            let (_, _, tensor) = self.free_tensors.remove(pos);
            return Ok(tensor);
        }
        
        // Allocate new tensor
        self.device.allocate_tensor(shape, dtype)
    }
    
    /// Return tensor to pool
    pub fn return_tensor(&mut self, tensor: Tensor) -> Result<()> {
        if self.free_tensors.len() >= self.max_pool_size {
            // Pool is full, just drop the tensor
            return Ok(());
        }
        
        let shape = tensor.dims().to_vec();
        let dtype = tensor.dtype();
        
        // Clear tensor before pooling
        let zeroed = tensor.zeros_like()?;
        self.free_tensors.push((shape, dtype, zeroed));
        
        Ok(())
    }
    
    /// Clear the pool
    pub fn clear(&mut self) {
        self.free_tensors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_auto_select() {
        let device = CandleDevice::auto_select().unwrap();
        assert!(device.memory_limit.is_some());
    }
    
    #[test]
    fn test_memory_tracking() {
        let device = CandleDevice::auto_select().unwrap();
        let initial = device.memory_usage();
        
        let _tensor = device.allocate_tensor(&[10, 10], DType::F32).unwrap();
        assert!(device.memory_usage() > initial);
        
        device.clear_memory_tracking();
        assert_eq!(device.memory_usage(), 0);
    }
}
