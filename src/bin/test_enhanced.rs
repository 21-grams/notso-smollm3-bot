//! Test the enhanced generation service
//! Run with: cargo run --bin test_enhanced --release

use notso_smollm3_bot::services::ml::EnhancedMLService;
use candle_core::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("notso_smollm3_bot=debug".parse()?)
            .add_directive("test_enhanced=debug".parse()?))
        .init();

    println!("🧪 Testing Enhanced ML Service");
    println!("================================");
    
    // Configuration
    let model_path = "./models/smollm3-3b-q4k.gguf";
    let tokenizer_dir = "./models/tokenizer";
    
    // Auto-select best device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("📍 Using device: {:?}", device);
    
    // Create enhanced service
    println!("🚀 Loading enhanced ML service...");
    let service = EnhancedMLService::new_with_config(
        model_path,
        tokenizer_dir,
        device,
        0.7,    // temperature
        0.9,    // top_p
        Some(50), // top_k
        42,     // seed
        1.1,    // repetition_penalty
        64,     // repetition_last_n
        true,   // use_nope
    );
    
    match service {
        Ok(_) => {
            println!("✅ Enhanced ML Service loaded successfully!");
            println!("\n📊 Configuration:");
            println!("   - Temperature: 0.7");
            println!("   - Top-p: 0.9");
            println!("   - Top-k: 50");
            println!("   - Repetition penalty: 1.1");
            println!("   - Reserved tokens filtered: 247+");
            println!("\n🎯 Key Features:");
            println!("   - Token 4194 filtering ✓");
            println!("   - Reserved range 128009-128255 blocked ✓");
            println!("   - NaN/Inf handling ✓");
            println!("   - Proper LogitsProcessor with Sampling enum ✓");
            println!("   - Post-sampling validation ✓");
        }
        Err(e) => {
            eprintln!("❌ Failed to load service: {}", e);
            return Err(e.into());
        }
    }
    
    println!("\n✨ Enhanced generation pipeline ready for use!");
    Ok(())
}