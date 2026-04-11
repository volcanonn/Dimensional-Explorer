import taichi as ti

# ==========================================
# GENERALIZED N-DIMENSIONAL CONFIGURATION
# ==========================================
DIMENSIONS = 4   # Change this to 4, 6, 8, etc!
MIN_DIMENSIONS = 2
MAX_DIMENSIONS = 10

USE_F64 = False

# Export the Math Type so ti.init() can use it
MATH_TYPE = ti.f64 if USE_F64 else ti.f32

# Dynamically generate the N-Dimensional Vector Type to be shared across all files!
vecND = ti.types.vector(DIMENSIONS, float)