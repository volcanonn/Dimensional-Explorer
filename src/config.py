import taichi as ti

# The maximum size the GPU will ever need to prepare memory for
MAX_DIMENSIONS = 6

# The Master Type used for passing fast registers to the GPU
vecMAX = ti.types.vector(MAX_DIMENSIONS, ti.f64)