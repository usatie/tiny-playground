# memoryview is a built-in type
a = memoryview(b"abc")
print(a) # prints: <memory at 0x...>
print(a[0])  # prints: b'a'
print(a[1])  # prints: b'b'
print(a[2])  # prints: b'c'

print(len(a))  # prints: 3
print(a.itemsize)  # prints: 1
print(a.nbytes)  # prints: 3

print(a.format)  # prints: 'B'

import numpy as np

a = np.array([1,2,3,4])
print(a.dtype) # int64
print(a.nbytes) # 32
print(a.itemsize) # 8

# We need to access the pointer to the numpy buffer
ptr = a.data
print(ptr) # <memory at 0x...>
print(ptr.format) # l (l is the format for int64, also known as long)
print(ptr.itemsize) # 8
print(len(ptr)) # 4
print(ptr.nbytes) # 32

print(ptr[0])  # prints: 1
print(ptr[1])  # prints: 2
print(ptr[2])  # prints: 3
print(ptr[3])  # prints: 4

# memoryview contains metadata, and we need plain unsigned char bytes to pass to GPU
casted_ptr = ptr.cast("B")  # Cast to bytes
print(casted_ptr)
print(casted_ptr.itemsize)  # prints: 1
print(len(casted_ptr))  # prints: 32
print(casted_ptr.nbytes)  # prints: 32
print(casted_ptr[0])  # prints: 1
print(casted_ptr[1])  # prints: 0
print(casted_ptr[2])  # prints: 0
print(casted_ptr[3])  # prints: 0
print(casted_ptr[4])  # prints: 0
print(casted_ptr[5])  # prints: 0
print(casted_ptr[6])  # prints: 0
print(casted_ptr[7])  # prints: 0
print(casted_ptr[8])  # prints: 2
print(casted_ptr[9])  # prints: 0
# ...
# This is how little endiann int64 is stored

# We have two pointers: ptr vs casted_ptr
print(np.array(ptr))  # prints: [1 2 3 4]
# Using asarray will not create a separate copy of the underlying memory
print(np.asarray(ptr))  # prints: [1 2 3 4]

# If we use casted_ptr, we get the bytes representation
print(np.array(casted_ptr))
print(np.asarray(casted_ptr))

print(np.array(casted_ptr).view(np.int64))  # prints: [1 2 3 4]
print(np.array(casted_ptr.cast("l")) )  # prints: [1 2 3 4]

