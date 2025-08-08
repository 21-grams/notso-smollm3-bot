//! Device management utilities

use candle_core::{Device, Result};

pub struct DeviceManager;

impl DeviceManager {
    /// Detect optimal device based on availability
    pub fn detect_optimal_device() -> Result<Device> {
        // Try CUDA first
        if let Ok(device) = Device::new_cuda(0) {
            tracing::info!("🎮 Using CUDA device");
            return Ok(device);
        }
        
        // Try Metal on macOS
        #[cfg(target_os = "macos")]
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("🎮 Using Metal device");
            return Ok(device);
        }
        
        // Fallback to CPU
        tracing::info!("💻 Using CPU device");
        Ok(Device::Cpu)
    }
    
    /// Get device info string
    pub fn device_info(device: &Device) -> String {
        match device {
            Device::Cpu => "CPU".to_string(),
            Device::Cuda(_) => "CUDA GPU".to_string(),
            Device::Metal(_) => "Metal GPU".to_string(),
        }
    }
    
    /// Check if device supports operation
    pub fn supports_operation(device: &Device, op: &str) -> bool {
        match (device, op) {
            (Device::Cpu, _) => true,  // CPU supports all ops
            (Device::Cuda(_), "flash_attention") => true,
            (Device::Metal(_), "flash_attention") => false,
            _ => true,
        }
    }
}
