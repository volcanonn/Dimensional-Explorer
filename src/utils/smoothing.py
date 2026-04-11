import taichi as ti
import taichi.math as tm

@ti.func
def simple_smooth(iterations, final_magnitude_squared, color_freq):
    it = ti.cast(iterations, ti.f32)
    mag = ti.cast(final_magnitude_squared, ti.f32)
    freq = ti.cast(color_freq, ti.f32)
    
    half = ti.cast(0.5, ti.f32)
    
    log_mag = tm.log(mag)
    smooth_val = it - tm.log2(log_mag * half)
    
    return tm.fract(smooth_val * freq)