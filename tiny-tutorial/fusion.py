from tinygrad import Tensor
"""
a = Tensor.empty(4, 4)
b = a + 4
c = b + 3
c.realize()
"""


# DEBUG=4 python fusion.py
"""
__kernel void E_4_4(__global float* data0, __global float* data1) {
  int lidx0 = get_local_id(0); /* 4 */
  int alu0 = (lidx0<<2);
  float4 val0 = *((__global float4*)((data1+alu0)));
  *((__global float4*)((data0+alu0))) = (float4)((val0.x+7.0f),(val0.y+7.0f),(val0.z+7.0f),(val0.w+7.0f));
}
"""

#a = Tensor.empty(4, 4)
#b = a + 4
#b.realize()
#c = b + 3
#c.realize()

# DEBUG=4 python fusion.py
"""
__kernel void E_4_4n1(__global float* data0, __global float* data1) {
  int lidx0 = get_local_id(0); /* 4 */
  int alu0 = (lidx0<<2);
  float4 val0 = *((__global float4*)((data1+alu0)));
  *((__global float4*)((data0+alu0))) = (float4)((val0.x+4.0f),(val0.y+4.0f),(val0.z+4.0f),(val0.w+4.0f));
}

__kernel void E_4_4n2(__global float* data0, __global float* data1) {
  int lidx0 = get_local_id(0); /* 4 */
  int alu0 = (lidx0<<2);
  float4 val0 = *((__global float4*)((data1+alu0)));
  *((__global float4*)((data0+alu0))) = (float4)((val0.x+3.0f),(val0.y+3.0f),(val0.z+3.0f),(val0.w+3.0f));
}
"""

# DEBUG=4 python fusion.py
a = Tensor.empty(4, 4)
b = a + 4
b = b.contiguous()
c = b + 3
c.realize()

"""
The same output as before, but now the UOp AST is contiguous.
When this AST is lowered and turned into executable kernels, Ops.CONTIGUOUS acts as a breakpoint, such that the tree will be split into two.
"""

# TinyGrad Fusion Breaking with Ops.CONTIGUOUS
"""
def store_or_fuse(ctx: ScheduleContext, b: UOp, x: UOp, st: UOp):
    # b: buffer node to potentially store
    # x: input data/computation result
    # st: store operation node containing shape info

    # If buffer not marked for realization, fuse operations
    if b not in ctx.realizes:
        return x  # Continue fusion - no memory barrier

    # Buffer marked for realization - create breakpoint
    # Store computation result to memory
    ctx.realizes[b] = UOp.store(b, ShapeTracker.from_shape(st.shape).to_uop(), x)

    # Return LOAD operation to read from stored buffer
    # st.st contains shape information from original store operation
    return UOp(Ops.LOAD, x.dtype, (b, unwrap(st.st).to_uop()))

# Pattern matcher that marks Ops.CONTIGUOUS for realization
do_realize = PatternMatcher([
    (UPat(Ops.SINK, name="sink"), sink_outputs),
    # This line forces Ops.CONTIGUOUS to break fusion
    (UPatScheduled({Ops.ASSIGN, Ops.CONTIGUOUS, *GroupOp.Meta}), realize),
])

def realize(ctx: ScheduleContext, b: UOp, to_store: UOp, **kwargs) -> None:
    # Marks buffer for mandatory realization (memory store)
    ctx.realizes[b] = None  # Will be filled later by store_or_fuse
"""
