import taichi as ti
from .mandelbrots import mandelbrot_core

@ti.kernel
def nd_slice(
    pixels: ti.template(), 
    camera_data: ti.template(),
    zoom: float, pan_x: float, pan_y: float, 
    max_iter: int, color_freq: float
):
    width = pixels.shape[0]
    height = pixels.shape[1]
    
    # Extract the vectors directly from GPU memory
    origin = camera_data[0]
    right = camera_data[1]
    up = camera_data[2]
    
    for i, j in pixels:  
        screen_x = ti.cast(i, float) - ti.cast(width, float) / 2.0
        screen_y = ti.cast(j, float) - ti.cast(height, float) / 2.0
        
        math_x = (screen_x / zoom) - pan_x
        math_y = (screen_y / zoom) - pan_y
        
        pos_nd = origin + (math_x * right) + (math_y * up)
        
        pixels[i, j] = mandelbrot_core(pos_nd, max_iter, color_freq)