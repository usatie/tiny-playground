# Abstract
A PyTorch-like <b>frontend</b> for building computation graphs.
A <b>scheduler</b> that partitions the computation graph into kernel-sized subgraphs, each representing a unit of work to be executed together.
A <b>lowering</b> engine that transforms each kernel subgraph (AST) into device-specific code (such as CUDA or OpenCL kernels) that can run on accelerators.
An <b>execution</b> engine that launches the generated code on the target device.
