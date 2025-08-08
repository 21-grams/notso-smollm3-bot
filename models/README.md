# Model Files

This directory contains the SmolLM3 model and tokenizer files.

## Current Files

- `HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf` - Quantized SmolLM3 3B model (Q4_K_M format)
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Additional tokenizer settings
- `special_tokens_map.json` - Special token mappings

## Notes

- Model files are ignored by git (see .gitignore)
- Q4_K_M quantization reduces model size from ~12GB to ~1.8GB
- Expected memory usage: ~2-3GB GPU RAM or ~4-6GB system RAM

## Model Specifications

- **Parameters**: 3B
- **Quantization**: Q4_K_M (4-bit)
- **Context Length**: 2048 tokens (expandable to 32K)
- **Architecture**: Llama-style with GQA (4:1 ratio)
- **Special Features**: 
  - Thinking tokens (`<think>`, `</think>`)
  - NoPE layers [3,7,11,15,19,23,27,31,35]
  - Tool calling support
