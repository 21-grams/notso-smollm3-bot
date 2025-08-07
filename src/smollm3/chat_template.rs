pub struct ChatTemplate;

impl ChatTemplate {
    pub fn new() -> Self {
        Self
    }
    
    pub fn format_single_turn(&self, message: &str, thinking_token_id: u32) -> String {
        // SmolLM3 chat template format
        format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            message
        )
    }
    
    pub fn format_with_thinking(&self, message: &str) -> String {
        format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<think>",
            message
        )
    }
    
    pub fn format_multi_turn(&self, messages: &[(String, String)]) -> String {
        let mut result = String::from("<|begin_of_text|>");
        
        for (role, content) in messages {
            result.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                role, content
            ));
        }
        
        result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        result
    }
}
