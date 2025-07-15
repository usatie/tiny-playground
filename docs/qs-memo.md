# Quck Start Memo
https://docs.tinygrad.org/quickstart/

I think numpy should be removed from quick start too.
https://github.com/tinygrad/tinygrad/pull/7789
- His change requires `avg_acc.item()` for it to be actually run on GPU.

## Crash on M3 Mac with `range(512)`
- range(512) on m3 mac does crash, whereas range(511) works.
- minimum repro is `exp1.py`

## Indexing speed comparison numpy vs tinygrad
- benchmark file is `indexing-speed-comp.py`
- I am not sure if it comes from indexing speed or not, but there was some comments and discussion regarding indexing speed.
> yea index is too slow...
https://github.com/tinygrad/tinygrad/pull/7789#issuecomment-2486263110

> there will be a few cleanup PRs after this, but https://github.com/tinygrad/tinygrad/pull/10045 is an actually powerful arange/indexing folder that can handle a huge variety of patterns. it fixed the super slow indexing in torch beautiful_mnist bringing that test speed back to what it used to be with the indexing now in tinygrad, it's needs the FUSE_ARANGE flag though.
> if you find any aranges that aren't fusing, add a test and i'll fix it. 
https://discord.com/channels/1068976834382925865/1068982781490757652/1365677737918070855

> as tinygrad gets to 1.0, it's important we have good canonical usage examples.
> here's beautiful_mnist: https://github.com/tinygrad/tinygrad/pull/2272/files
> what other examples would people like to see?
> (btw that example is currently a bad way to do things since the tensor indexing is absurdly slow)
https://discord.com/channels/1068976834382925865/1068976834928193609/1172936406113452112
```
---numpy indexing---
Test Accuracy: 0.10026296477495107
Time: 3490.89 ms
---tinygrad indexing---
Test Accuracy: 0.1041768590998043
Time: 8487.09 ms
```

- Noete: `FUSE_ARANGE=1 python indexing-speed-comp.py` raised error.
```
---numpy indexing---
Test Accuracy: 0.08549412915851272
Time: 3651.68 ms
---tinygrad indexing---
Time: 9708.79 ms
Traceback (most recent call last):
  File "/Users/shunusami/workspace/tiny/playground/docs/indexing-speed-comp.py", line 65, in <module>
    print(f"Test Accuracy: {avg_acc.item() / num_steps}")
                            ^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4385, in _wrapper
    ret = fn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 328, in item
    return self.data()[(0,) * len(self.shape)]
           ^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
    if _METADATA.get() is not None: return fn(*args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 316, in data
    return self._buffer().as_typed_buffer(self.shape)
           ^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
    if _METADATA.get() is not None: return fn(*args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 302, in _buffer
    def _buffer(self) -> Buffer: return cast(Buffer, self.cast(self.dtype.base).contiguous().to("CPU").realize().uop.base.buffer)
                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
    if _METADATA.get() is not None: return fn(*args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 269, in realize
    run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 192, in run_schedule
    for si, ei in lower_schedule(schedule):
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 185, in lower_schedule
    raise e
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 179, in lower_schedule
    try: yield (si, lower_schedule_item(si))
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 174, in lower_schedule_item
    return ExecItem(*cast(tuple[Runner,list], si_lowerer.rewrite(si.ast, si.bufs)), si.metadata, si.fixedvars)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/uop/ops.py", line 730, in rewrite
    if (ret:=match(uop, ctx)) is not None and ret is not uop: return ret
             ^^^^^^^^^^^^^^^
  File "<string>", line 3, in compiled_match
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 167, in <lambda>
    (UPat(Ops.SINK, name="sink"), lambda ctx,sink: (runner:=get_runner(ctx[0].device, sink), [ctx[x] for x in runner.p.globals])),
                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 135, in get_runner
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 68, in __init__
    self._prg = Device[p.device].runtime(p.function_name, self.lib) if prg is None else prg
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/runtime/ops_gpu.py", line 41, in __init__
    self.kernel = checked(cl.clCreateKernel(self.program, name.encode(), status := ctypes.c_int32()), status)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/runtime/ops_gpu.py", line 15, in checked
    def checked(ret, status): return (check(status.value), ret)[1]
                                      ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/runtime/ops_gpu.py", line 14, in check
    if status != 0: raise RuntimeError(f"OpenCL Error {status}: {cl_errors.get(status, 'Unknown error')}")
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: OpenCL Error -48: CL_INVALID_KERNEL
➜
```

