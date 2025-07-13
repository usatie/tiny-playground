from tinygrad.uop.ops import UOp, Ops
from tinygrad import dtypes

print("--- UOp Singleton Example (const) ---")
const1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const2 = UOp(Ops.CONST, dtypes.float, arg=0.5)
print(const1 == const2) # True
# This is a singleton, so the memory address is the same
print(const1 is const2) # True

print("--- UOp Singleton Example (buffer) ---")
buf1 = UOp(Ops.DEFINE_GLOBAL, arg=1)
_buf1 = UOp(Ops.DEFINE_GLOBAL, arg=1)
print(buf1 == _buf1) # True
print(buf1 is _buf1) # True

print("--- UOp Singleton Example (different arg, src) ---")
buf2 = UOp(Ops.DEFINE_GLOBAL, arg=2)

a = UOp(Ops.ADD, src=(const1, buf1))
print(a)

b = UOp(Ops.ADD, src=(const1, buf1))
print(b)

c = UOp(Ops.ADD, src=(const1, buf2))
print(c)

print(a == b) # True, same operation and operands
print(a is b) # True, same UOp instance
print(a == c) # False, different operands
print(a is c) # False, different UOp instances
print(b == c) # False, different operands
print(b is c) # False, different UOp instances

print("--- Checking if almost equal ---")
def remove_buf(uop: UOp):
    src = [remove_buf(_uop) for _uop in uop.src]
    src = tuple([_uop for _uop in src if _uop is not None])
    if uop.op == Ops.DEFINE_GLOBAL: return None
    return uop.replace(src=src)

_a = remove_buf(a)
_c = remove_buf(c)
print(_a)
print(_c)

print(_a == _c) # True, same operation and operands after removing buffers
print(_a is _c) # True, same UOp instance after removing buffers
