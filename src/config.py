import platform
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, default_fp=ti.f32)

# The maximum size the GPU will ever need to prepare memory for
MAX_DIMENSIONS = 6 

SUPPORT_F64_BASE = True
SUPPORT_F64_TRIG = True

if platform.system() == "Darwin":
    # Apple Metal f64 doesnt exist
    SUPPORT_F64_BASE = False
    SUPPORT_F64_TRIG = False
else:

    ti.set_logging_level(ti.CRITICAL)

    # basic 64-bit allocation
    try:
        @ti.kernel
        def probe_f64_base(val: ti.f64) -> ti.f64:
            return val * ti.cast(2.0, ti.f64)
            
        probe_f64_base(1.0)
    except Exception:
        SUPPORT_F64_BASE = False
        SUPPORT_F64_TRIG = False
        
    # Native 64-bit Trig
    if SUPPORT_F64_BASE:
        try:
            @ti.kernel
            def probe_f64_trig(y: ti.f64, x: ti.f64) -> ti.f64:
                return tm.atan2(y, x) + tm.cos(x) + tm.log(x)
                
            probe_f64_trig(1.0, 2.0)
        except Exception:
            SUPPORT_F64_TRIG = False
    
    ti.set_logging_level(ti.INFO)

# Safe fallback
F64_SAFE = ti.f64 if SUPPORT_F64_BASE else ti.f32

# The Master Types used for passing fast registers to the GPU
vecMAX_f32 = ti.types.vector(MAX_DIMENSIONS, ti.f32)
vecMAX_f64 = ti.types.vector(MAX_DIMENSIONS, F64_SAFE)