use notso_smollm3_bot::services::ml::smollm3::SmolLM3Tokenizer;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("Testing tokenizer vocabulary alignment...\n");
    
    // Load tokenizer
    let tokenizer = SmolLM3Tokenizer::from_files("models")?;
    
    // Test 1: Common greetings
    println!("=== Common Greetings ===");
    let greetings = vec!["Hello", "Hi", "Hey", "Good morning", "How are you"];
    for greeting in greetings {
        let encoded = tokenizer.encode(greeting)?;
        let decoded = tokenizer.decode(&encoded)?;
        println!("{:15} -> {:?} -> {}", greeting, encoded, decoded);
    }
    
    // Test 2: Check specific token IDs that model generates
    println!("\n=== Problematic Token IDs ===");
    let problem_ids = vec![11