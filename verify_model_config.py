#!/usr/bin/env python3
"""
Verify model configuration against hardcoded values
"""

import struct
import json
from pathlib import Path

def read_gguf_metadata(filepath):
    """Read GGUF file metadata"""
    metadata = {}
    
    with open(filepath, 'rb') as f:
        # Read GGUF magic
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"Error: Not a GGUF file (magic: {magic})")
            return None
            
        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        print(f"GGUF Version: {version}")
        
        # Read tensor count and metadata KV count
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Tensor count: {tensor_count}")
        print(f"Metadata KV count: {metadata_kv_count}")
        
        # Read metadata
        for i in range(metadata_kv_count):
            # Read key length and key
            key_len = struct.