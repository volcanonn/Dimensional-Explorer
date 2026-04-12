import taichi as ti
from funcs import *
import config

@ti.kernel
def nd_slice(
    pixels: ti.template(),
    origin: config.vecMAX,
    right: config.vecMAX,
    up: config.vecMAX,
    zoom: ti.f64,
    max_iter: int,
    color_freq: float,
    func_idx: ti.template(),
    use_f64: ti.template(),
    active_dims: ti.template(),
    colormap_idx: ti.template()
):
    width = pixels.shape[0]
    height = pixels.shape[1]

    float_type = ti.f64 if ti.static(use_f64) else ti.f32

    inv_zoom = ti.cast(1.0 / zoom, float_type)
    half_w = ti.cast(width, float_type) * 0.5
    half_h = ti.cast(height, float_type) * 0.5

    for i, j in pixels:
        screen_x = ti.cast(i, float_type) - half_w
        screen_y = ti.cast(j, float_type) - half_h

        math_x = screen_x * inv_zoom
        math_y = screen_y * inv_zoom

        pos_nd = ti.Vector([
            ti.cast(origin[d], float_type) + 
            (math_x * ti.cast(right[d], float_type)) + 
            (math_y * ti.cast(up[d], float_type))
            for d in ti.static(range(active_dims))
        ])
        
        color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
        
        if ti.static(func_idx == 0):
            color = mandelbrot_core(pos_nd, max_iter, color_freq, use_f64, colormap_idx)
        elif ti.static(func_idx == 1):
            color = conic_core(pos_nd, color_freq, use_f64, colormap_idx)
        elif ti.static(func_idx == 2):
            color = voronoi_core(pos_nd, color_freq, use_f64, colormap_idx)
        elif ti.static(func_idx == 3):
            color = simple_wave_core(pos_nd, color_freq, use_f64, colormap_idx)
        elif ti.static(func_idx == 4):
            color = radial_wave_core(pos_nd, color_freq, use_f64, colormap_idx)
        elif ti.static(func_idx == 5):
            color = paraboloid_core(pos_nd, color_freq, use_f64, colormap_idx)

        is_crosshair = (ti.abs(screen_x) < 6.0 and ti.abs(screen_y) < 1.0 and ti.abs(screen_x) > 2.0) or \
                       (ti.abs(screen_y) < 6.0 and ti.abs(screen_x) < 1.0 and ti.abs(screen_y) > 2.0)
        
        if is_crosshair:
            pixels[i, j] = ti.Vector([1.0, 1.0, 1.0])
        else:
            pixels[i, j] = color