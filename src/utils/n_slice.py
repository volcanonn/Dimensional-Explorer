import taichi as ti
from funcs import *
import config

@ti.kernel
def nd_slice(
    pixels: ti.template(), 
    origin: config.vecND, right: config.vecND, up: config.vecND,
    zoom: float, max_iter: int, color_freq: float
):
    width = pixels.shape[0]
    height = pixels.shape[1]
    
    inv_zoom = 1.0 / zoom
    half_width = ti.cast(width, float) * 0.5
    half_height = ti.cast(height, float) * 0.5
    
    for i, j in pixels:
        screen_x = ti.cast(i, float) - half_width
        screen_y = ti.cast(j, float) - half_height
        
        math_x = screen_x * inv_zoom
        math_y = screen_y * inv_zoom
        
        pos_nd = origin + (math_x * right) + (math_y * up)
        pixels[i, j] = conic_core(pos_nd, max_iter, color_freq)