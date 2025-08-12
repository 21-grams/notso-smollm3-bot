# strftime_now Template Function Fix

## Problem
The SmolLM3 Jinja2 template uses a `strftime_now` function to get the current date:
```jinja
{%- set today = strftime_now("%d %B %Y") -%}
```

This function was not registered in the minijinja environment, causing a runtime error:
```
unknown function: strftime_now is unknown (in chat:28)
```

## Solution

### 1. Added Chrono Import
```rust
use chrono::Local;
```

### 2. Registered Custom Function
In `SmolLM3Tokenizer::from_files()`:
```rust
// Register the strftime_now function
chat_template.add_function("strftime_now", |format: String| {
    // Use chrono to get current local time and format it
    let now = Local::now();
    now.format(&format).to_string()
});
```

### 3. Fixed Context Variables
Updated `apply_chat_template()` to use correct variable names:
- Changed `thinking_mode` → `enable_thinking` (matches template)
- Added `xml_tools`, `python_tools`, `tools` (set to false)
- Removed `system_message` (handled by template defaults)

## How It Works

1. **Function Registration**: The `strftime_now` function is registered before loading the template
2. **Chrono Integration**: Uses `chrono::Local::now()` to get current local time
3. **Format String**: The template passes format strings like `"%d %B %Y"` which chrono formats correctly
4. **Result**: Produces dates like "12 August 2025"

## Template Context Flow

The template now receives:
```rust
{
    messages: Vec<ChatMessage>,
    add_generation_prompt: true,
    enable_thinking: bool,
    xml_tools: false,
    python_tools: false,
    tools: false,
}
```

This matches what the template expects and allows it to:
- Format the current date in the metadata section
- Determine reasoning mode based on `enable_thinking`
- Handle tool configurations (currently disabled)

## Testing

The fix can be tested by:
1. Running `cargo build` - should compile without errors
2. Starting the server and sending a message
3. The template should now render without the `strftime_now` error
4. Check logs for the formatted prompt with date metadata

## Benefits

- **Template Compatibility**: Works with the official SmolLM3 template
- **Date Formatting**: Provides accurate current date in responses
- **Extensibility**: Easy to add more template functions if needed
- **Clean Integration**: Uses minijinja's standard function registration
