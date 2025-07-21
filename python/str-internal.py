import ctypes

x = "helloworld"
y = "hello" "world"
a = "hello"
b = "world"
c = a + b
d = "hello" + "world"
e = "helloworld"
f = e + "!"

def get_string_data_ptr(s):
    obj_ptr = id(s)
    
    # Read state field to determine string type
    # PyASCIIObject layout: PyObject_HEAD (16) + length (8) + hash (8) + state (4) = 36 bytes
    # But we need to check the actual structure sizes
    
    # Check if compact by reading state field
    state_offset = 32  # PyObject_HEAD(16) + length(8) + hash(8)
    state = ctypes.c_uint32.from_address(obj_ptr + state_offset).value
    
    compact = bool(state & 0x20)  # compact flag
    ascii_flag = bool(state & 0x40)  # ascii flag
    
    if compact:
        if ascii_flag:
            # PyASCIIObject: data starts right after the structure
            # PyObject_HEAD(16) + length(8) + hash(8) + state(4) + padding(4) = 40 bytes
            return obj_ptr + 40
        else:
            # PyCompactUnicodeObject: PyASCIIObject + utf8_length(8) + utf8(8) = 56 bytes  
            return obj_ptr + 56
    else:
        # Non-compact: read data pointer from data.any field
        # PyCompactUnicodeObject(56) + data union(8) = 64 bytes total
        # data.any is at offset 56
        return ctypes.c_void_p.from_address(obj_ptr + 56).value

def read_string_from_ptr(ptr, length, kind=1):
    """Read string content from raw pointer"""
    if kind == 1:  # 1-byte chars (ASCII/Latin-1)
        return ctypes.string_at(ptr, length).decode('latin-1')
    elif kind == 2:  # 2-byte chars (UCS2)
        data = ctypes.string_at(ptr, length * 2)
        return data.decode('utf-16le')
    elif kind == 4:  # 4-byte chars (UCS4)
        data = ctypes.string_at(ptr, length * 4)
        return data.decode('utf-32le')

def get_string_info(s):
    obj_ptr = id(s)
    
    # Read length (offset 16 after PyObject_HEAD)
    length = ctypes.c_ssize_t.from_address(obj_ptr + 16).value
    
    # Read state (offset 32)
    state = ctypes.c_uint32.from_address(obj_ptr + 32).value
    kind = (state >> 2) & 0x7  # Extract kind bits
    compact = bool(state & 0x20)
    ascii_flag = bool(state & 0x40)
    
    data_ptr = get_string_data_ptr(s)
    content = read_string_from_ptr(data_ptr, length, kind)
    
    return {
        'string': s,
        'length': length,
        'kind': kind,
        'compact': compact,
        'ascii': ascii_flag,
        'data_ptr': hex(data_ptr),
        'read_content': content,
        'matches': content == s
    }

# Test all strings
for name, string in [('x', x), ('y', y), ('a', a), ('b', b), ('c', c), ('d', d), 
                     ('e', e), ('f', f)]:
    info = get_string_info(string)
    print(f"{name} = '{info['string']}'")
    print(f"  Length: {info['length']}, Kind: {info['kind']}")
    print(f"  Compact: {info['compact']}, ASCII: {info['ascii']}")
    print(f"  Data pointer: {info['data_ptr']}")
    print(f"  Read from memory: '{info['read_content']}'")
    print(f"  Content matches: {info['matches']}")
    print()

exit()
s = "hello"
for i in range(10):
    info = get_string_info(s)
    print(f"{name} = '{info['string']}'")
    print(f"  Length: {info['length']}, Kind: {info['kind']}")
    print(f"  Compact: {info['compact']}, ASCII: {info['ascii']}")
    print(f"  Data pointer: {info['data_ptr']}")
    print(f"  Read from memory: '{info['read_content']}'")
    print(f"  Content matches: {info['matches']}")
    print()
    s += "!"
