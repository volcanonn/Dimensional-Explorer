import taichi as ti
import taichi.math as tm

@ti.func
def psychedelic(t):
    t32 = ti.cast(t, ti.f32)
    half = ti.cast(0.5, ti.f32)
    
    # 2.0 * PI * 1.5 = 3.0 * PI
    three_pi = ti.cast(3.0 * tm.pi, ti.f32)
    
    r = half * tm.sin(t32 * three_pi + ti.cast(0.0, ti.f32)) + half
    g = half * tm.sin(t32 * three_pi + ti.cast(1.5, ti.f32)) + half
    b = half * tm.sin(t32 * three_pi + ti.cast(3.0, ti.f32)) + half
    
    return ti.cast(ti.Vector([r, g, b]), ti.f32)