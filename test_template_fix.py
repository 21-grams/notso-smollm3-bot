#!/usr/bin/env python3
"""
Test the tokenizer template rendering to verify undefined variable fixes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minijinja import Environment
from datetime import datetime

def test_template_without_system_message():
    """Test that template works when there's no system message"""
    
    # Load template
    with open('templates/smollm3_official.j2', 'r') as f:
        template_str = f.read()
    
    env = Environment()
    env.add_function("strftime_now", lambda fmt: datetime.now().strftime(fmt))
    env.add_template("chat", template_str)
    
    # Test case 1: No system message
    messages = [
        {"role": "user", "content": "Hello!"}
    ]
    
    context = {
        "messages": messages,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "xml_tools": False,
        "python_tools": False,
        "tools": False,
    }
    
    template = env.get_template("chat")
    try:
        result = template.render(context)
        print("✓ Test 1 passed: Template renders without system message")
        print(f"  Output length: {len(result)} chars")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return False
    
    # Test case 2: With system message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    context["messages"] = messages
    
    try:
        result = template.render(context)
        print("✓ Test 2 passed: Template renders with system message")
        print(f"  Output length: {len(result)} chars")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False
    
    # Test case 3: System message with /system_override
    messages = [
        {"role": "system", "content": "/system_override Custom system prompt"},
        {"role": "user", "content": "Hello!"}
    ]
    
    context["messages"] = messages
    
    try:
        result = template.render(context)
        print("✓ Test 3 passed: Template renders with /system_override")
        print(f"  Output length: {len(result)} chars")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False
    
    # Test case 4: With thinking enabled
    messages = [
        {"role": "user", "content": "Hello!"}
    ]
    
    context["messages"] = messages
    context["enable_thinking"] = True
    
    try:
        result = template.render(context)
        print("✓ Test 4 passed: Template renders with thinking enabled")
        print(f"  Output length: {len(result)} chars")
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        return False
    
    print("\n✅ All template tests passed!")
    return True

if __name__ == "__main__":
    success = test_template_without_system_message()
    exit(0 if success else 1)
