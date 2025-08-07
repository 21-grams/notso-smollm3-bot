use anyhow::Result;

pub struct SmolLM3Tokenizer {
    inner: Option<tokenizers::Tokenizer>,
    is_stub: bool,
}

impl SmolLM3Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self {
            inner: Some(tokenizer),
            is_stub: false,
        })
    }
    
    pub fn new_stub() -> Self {
        Self {
            inner: None,
            is_stub: true,
        }
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if self.is_stub {
            // Return mock token IDs
            Ok(text.chars().map(|_| 1u32).collect())
        } else if let Some(tokenizer) = &self.inner {
            let encoding = tokenizer.encode(text, false)?;
            Ok(encoding.get_ids().to_vec())
        } else {
            anyhow::bail!("Tokenizer not initialized")
        }
    }
    
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if self.is_stub {
            Ok("Mock decoded text".to_string())
        } else if let Some(tokenizer) = &self.inner {
            Ok(tokenizer.decode(ids, false)?)
        } else {
            anyhow::bail!("Tokenizer not initialized")
        }
    }
}
