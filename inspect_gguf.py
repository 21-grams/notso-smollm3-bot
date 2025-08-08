#!/usr/bin/env python3
"""Inspect GGUF file metadata to understand its structure"""

import struct
import sys
from pathlib import Path

def read_gguf_header(filepath):
    """Read and display GGUF file header and metadata"""
    with open(filepath, 'rb') as f:
        # Read magic number
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"Not a GGUF file. Magic: {magic}")
            return
        
        print(f"✅ Valid GGUF file")
        
        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        print(f"Version: {version}")
        
        # Read tensor count and metadata KV count
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Tensors: {tensor_count}")
        print(f"Metadata entries: {metadata_kv_count}")
        print("\n📋 Metadata Keys:")
        print("-" * 50)
        
        # Read metadata
        for i in range(min(metadata_kv_count, 100)):  # Limit to first 100 entries
            # Read key length and key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            
            # Read value type
            vtype = struct.unpack('<I', f.read(4))[0]
            
            # Read value based on type
            value = None
            if vtype == 0:  # UINT8
                value = struct.unpack('B', f.read(1))[0]
            elif vtype == 1:  # INT8
                value = struct.unpack('b', f.read(1))[0]
            elif vtype == 2:  # UINT16
                value = struct.unpack('<H', f.read(2))[0]
            elif vtype == 3:  # INT16
                value = struct.unpack('<h', f.read(2))[0]
            elif vtype == 4:  # UINT32
                value = struct.unpack('<I', f.read(4))[0]
            elif vtype == 5:  # INT32
                value = struct.unpack('<i', f.read(4))[0]
            elif vtype == 6:  # FLOAT32
                value = struct.unpack('<f', f.read(4))[0]
            elif vtype == 7:  # BOOL
                value = struct.unpack('?', f.read(1))[0]
            elif vtype == 8:  # STRING
                str_len = struct.unpack('<Q', f.read(8))[0]
                value = f.read(str_len).decode('utf-8')
            elif vtype == 9:  # ARRAY
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                value = f"Array[{arr_len}]"
                # Skip array data for now
                if arr_type == 8:  # String array
                    for _ in range(arr_len):
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(str_len)
                else:
                    elem_size = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1}.get(arr_type, 4)
                    f.read(elem_size * arr_len)
            elif vtype == 10:  # UINT64
                value = struct.unpack('<Q', f.read(8))[0]
            elif vtype == 11:  # INT64
                value = struct.unpack('<q', f.read(8))[0]
            elif vtype == 12:  # FLOAT64
                value = struct.unpack('<d', f.read(8))[0]
            
            # Print key-value pair
            if 'attention' in key or 'head' in key or 'layer' in key or 'vocab' in key or 'hidden' in key:
                print(f"{key:40} = {value}")

if __name__ == "__main__":
    gguf_file = Path("models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf")
    if gguf_file.exists():
        read_gguf_header(gguf_file)
    else:
        print(f"File not found: {gguf_file}")
