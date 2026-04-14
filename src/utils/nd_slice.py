import taichi as ti
from funcs import *
import config

@ti.func
def process_pixel(
    i, j, width, height,
    origin, right, up,
    zoom, max_iter, color_freq,
    func_idx: ti.template(),
    use_f64: ti.template(),
    active_dims: ti.template(),
    colormap_idx: ti.template()
):
    float_type = ti.f64 if ti.static(use_f64) else ti.f32

    inv_zoom = ti.cast(1.0 / zoom, float_type)
    half_w = ti.cast(width, float_type) * 0.5
    half_h = ti.cast(height, float_type) * 0.5

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

    # Crosshair rendering
    is_crosshair = (ti.abs(screen_x) < 6.0 and ti.abs(screen_y) < 1.0 and ti.abs(screen_x) > 2.0) or \
                   (ti.abs(screen_y) < 6.0 and ti.abs(screen_x) < 1.0 and ti.abs(screen_y) > 2.0)
    
    if is_crosshair:
        color = ti.Vector([1.0, 1.0, 1.0])

    return color


@ti.kernel
def nd_slice_f32(
    pixels: ti.template(),
    origin: config.vecMAX_f32, right: config.vecMAX_f32, up: config.vecMAX_f32,
    zoom: ti.f32, max_iter: int, color_freq: float,
    func_idx: ti.template(), active_dims: ti.template(), colormap_idx: ti.template()
):
    width = pixels.shape[0]
    height = pixels.shape[1]
    for i, j in pixels:
        pixels[i, j] = process_pixel(
            i, j, width, height, origin, right, up, zoom, 
            max_iter, color_freq, func_idx, False, active_dims, colormap_idx
        )


@ti.kernel
def nd_slice_f64(
    pixels: ti.template(),
    origin: config.vecMAX_f64, right: config.vecMAX_f64, up: config.vecMAX_f64,
    zoom: ti.f64, max_iter: int, color_freq: float,
    func_idx: ti.template(), active_dims: ti.template(), colormap_idx: ti.template()
):
    width = pixels.shape[0]
    height = pixels.shape[1]
    for i, j in pixels:
        pixels[i, j] = process_pixel(
            i, j, width, height, origin, right, up, zoom, 
            max_iter, color_freq, func_idx, True, active_dims, colormap_idx
        )