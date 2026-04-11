import taichi as ti
import utils

@ti.func
def mandelbrot_core(pos_nd: ti.template(), max_iter: int, color_freq: float):
    """
    Takes a raw N-Dimensional vector, extracts the variables it specifically needs,
    runs the iterations, and returns the final RGB color.
    """
    N = ti.static(pos_nd.n)
    
    c = ti.Vector([
        pos_nd[0] if ti.static(N > 0) else 0.0,
        pos_nd[1] if ti.static(N > 1) else 0.0
    ])
    z = ti.Vector([
        pos_nd[2] if ti.static(N > 2) else 0.0,
        pos_nd[3] if ti.static(N > 3) else 0.0
    ])
    e = ti.Vector([
        pos_nd[4] if ti.static(N > 4) else 2.0,
        pos_nd[5] if ti.static(N > 5) else 0.0
    ])
    
    # Run the math iterations specific to this fractal
    iterations = 0
    while z[0]**2 + z[1]**2 < 400.0 and iterations < max_iter:
        z = utils.complex_pow(z, e) + c
        iterations += 1
        
    color = ti.Vector([0.0, 0.0, 0.0]) # Default to Black (inside the set)
    
    if iterations < max_iter:
        final_mag_sqr = z[0]**2 + z[1]**2
        t = utils.smoothing.simple_smooth(iterations, final_mag_sqr, color_freq)
        color = utils.colormap.heledron(t)
        
    return ti.cast(color, ti.f32)