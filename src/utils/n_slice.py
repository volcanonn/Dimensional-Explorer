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
    active_dims: ti.template()
):
    width = pixels.shape[0]
    height = pixels.shape[1]

    if ti.static(use_f64):
        # 64-bit Deep Zoom Loop
        inv_zoom = ti.cast(1.0 / zoom, ti.f64)
        half_w = ti.cast(width, ti.f64) * 0.5
        half_h = ti.cast(height, ti.f64) * 0.5

        for i, j in pixels:
            math_x = (ti.cast(i, ti.f64) - half_w) * inv_zoom
            math_y = (ti.cast(j, ti.f64) - half_h) * inv_zoom

            pos_nd = ti.Vector([
                origin[d] + (math_x * right[d]) + (math_y * up[d])
                for d in ti.static(range(active_dims))
            ])
            
            color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
            
            # THE FIX: ti.static() forces the GPU to delete the unused functions!
            if ti.static(func_idx == 0):
                color = mandelbrot_core(pos_nd, max_iter, color_freq, use_f64)
            elif ti.static(func_idx == 1):
                color = conic_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 2):
                color = voronoi_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 3):
                color = simple_wave_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 4):
                color = radial_wave_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 5):
                color = paraboloid_core(pos_nd, color_freq, use_f64)

            pixels[i, j] = color

    else:
        # 32-bit Speed Loop
        inv_zoom = ti.cast(1.0 / zoom, ti.f32)
        half_w = ti.cast(width, ti.f32) * 0.5
        half_h = ti.cast(height, ti.f32) * 0.5

        for i, j in pixels:
            math_x = (ti.cast(i, ti.f32) - half_w) * inv_zoom
            math_y = (ti.cast(j, ti.f32) - half_h) * inv_zoom

            pos_nd = ti.Vector([
                ti.cast(origin[d], ti.f32) + (math_x * ti.cast(right[d], ti.f32)) + (math_y * ti.cast(up[d], ti.f32))
                for d in ti.static(range(active_dims))
            ])
            
            color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
            
            if ti.static(func_idx == 0):
                color = mandelbrot_core(pos_nd, max_iter, color_freq, use_f64)
            elif ti.static(func_idx == 1):
                color = conic_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 2):
                color = voronoi_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 3):
                color = simple_wave_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 4):
                color = radial_wave_core(pos_nd, color_freq, use_f64)
            elif ti.static(func_idx == 5):
                color = paraboloid_core(pos_nd, color_freq, use_f64)

            pixels[i, j] = color