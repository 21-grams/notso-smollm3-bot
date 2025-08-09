# Claude Session Template

## Session START
```
GitHub: 21-grams/notso-smollm3-bot
Local: \\wsl.localhost\Ubuntu-24.04\root\notso-smollm3-bot
Read: PROJECT_STATUS.md
Focus: [TODAY'S TASK]
```

## Session END Checklist
- [ ] Update PROJECT_STATUS.md - Latest Working Changes
- [ ] Update PROJECT_STATUS.md - Latest Challenges  
- [ ] Update PROJECT_STATUS.md - Next Priority
- [ ] Update technical docs if new patterns found
- [ ] Commit with message: "Session: [description]"
- [ ] Push to GitHub

## Quick Commands
```bash
# Check status
git status

# Commit all changes
git add -A && git commit -m "Session: [description]" && git push

# Run project
cargo run --release

# Test streaming
# Type /quote in chat UI
```

## Critical Reminders
- Axum 0.8: Use `/{param}` not `/:param`
- Candle: Never dequantize, use QMatMul::forward()
- Streaming: Single buffer in services/streaming/
- SmolLM3: All adaptations in services/ml/smollm3/
