# https://mesozoic-egg.github.io/tinygrad-notes/20241112_pm.html

from tinygrad import Tensor

a = Tensor.empty(4, 4)
b = a + 1
b.realize()
# NOOPT=1 DEBUG=6 python pm.py
"""
   0 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(16)           []                               0
   1 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(16)           []                               1
   2 Ops.SPECIAL         : dtypes.int                     []                               ('gidx0', 16)
   3 Ops.CONST           : dtypes.float                   []                               1.0
   4 Ops.INDEX           : dtypes.float.ptr(16)           [1, 2]                           None
   5 Ops.LOAD            : dtypes.float                   [4]                              None
   6 Ops.INDEX           : dtypes.float.ptr(16)           [0, 2]                           None
   7 Ops.ADD             : dtypes.float                   [5, '1.0']                       None
   8 Ops.STORE           : dtypes.void                    [6, 7]                           None
   9 Ops.SINK            : dtypes.void                    [8]                              KernelInfo(name='E_\x1b[34m16\x1b[0m\x1b[90m\x1b[0m', global_dims=1, local_dims=0, upcasted=0, dont_use_locals=False, applied_opts=(), opts_to_apply=None)
__kernel void E_16(__global float* data0, __global float* data1) {
  int gidx0 = get_group_id(0); /* 16 */
  float val0 = *(data1+gidx0);
  *(data0+gidx0) = (val0+1.0f);
}
"""

a = Tensor.empty(4, 4)
b = a + 1
schedule = b.schedule()
if schedule:
    print(schedule[0].ast)
"""
UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.VIEW, dtypes.float.ptr(16), arg=ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(16), arg=0, src=()),)),
    UOp(Ops.ADD, dtypes.float, arg=None, src=(
      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(16), arg=ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(16), arg=1, src=()),)),)),
      UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 4), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
"""

a = Tensor.empty(4, 4)
b = a + 1
b.realize()
# DEBUG=6 python pm.py
"""
   0 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(16)           []                               0
   1 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(16)           []                               1
   2 Ops.SPECIAL         : dtypes.int                     []                               ('lidx0', 4)
   3 Ops.CONST           : dtypes.float                   []                               1.0
   4 Ops.CONST           : dtypes.int                     []                               2
   5 Ops.SHL             : dtypes.int                     [2, '2']                         None
   6 Ops.INDEX           : dtypes.float.ptr(16)           [1, 5]                           None
   7 Ops.CAST            : dtypes.float.vec(4).ptr(16)    [6]                              None
   8 Ops.LOAD            : dtypes.float.vec(4)            [7]                              None
   9 Ops.GEP             : dtypes.float                   [8]                              (0,)
  10 Ops.GEP             : dtypes.float                   [8]                              (1,)
  11 Ops.GEP             : dtypes.float                   [8]                              (2,)
  12 Ops.GEP             : dtypes.float                   [8]                              (3,)
  13 Ops.INDEX           : dtypes.float.ptr(16)           [0, 5]                           None
  14 Ops.CAST            : dtypes.float.vec(4).ptr(16)    [13]                             None
  15 Ops.ADD             : dtypes.float                   [9, '1.0']                       None
  16 Ops.ADD             : dtypes.float                   [10, '1.0']                      None
  17 Ops.ADD             : dtypes.float                   [11, '1.0']                      None
  18 Ops.ADD             : dtypes.float                   [12, '1.0']                      None
  19 Ops.VECTORIZE       : dtypes.float.vec(4)            [15, 16, 17, 18]                 None
  20 Ops.STORE           : dtypes.void                    [14, 19]                         None
  21 Ops.SINK            : dtypes.void                    [20]                             KernelInfo(name='
"""
