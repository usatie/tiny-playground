# General Flow
Tensors
-> UOps
(Scheduler) -> ScheduleItem(ast, bufs)
(realize) -> ExecItem(prg, bufs)

# Inside `realize`
(BEAM Search) -> UOps
(Renderer) -> code (CUDA, OpenCL, Metal, etc.)
(Compiler) -> binary

# Exececution
ExecItem can `run()`.
Lists of ExecItem can be condensed into a single ExecItem with the Graph API (rename to Queue?)

# Runtime
Runtimes are responsible for device-specific interactions. They handle tasks such as initializing devices, allocating memory, loading/launching programs, and more.
