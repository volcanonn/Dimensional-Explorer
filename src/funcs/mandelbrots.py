import taichi as ti
import taichi.math as tm
import utils

@ti.kernel
def julia(
    pixels: ti.template(), 
    zoom: float,
    pan_x: float,
    pan_y: float,
    max_iter: int, 
    color_freq: float,
    power: float
):
    width = pixels.shape[0]
    height = pixels.shape[1]
    
    w = ti.Vector([ti.cast(power, float), ti.cast(0.0, float)])

    c = ti.Vector([ti.cast(0.0, float), ti.cast(0.0, float)])
    
    for i, j in pixels:  
        # For a Julia set, the starting Z is the screen's mathematical coordinate
        z = utils.screen_to_math(i, j, width, height, zoom, pan_x, pan_y)
        
        iterations = 0
        while z[0]**2 + z[1]**2 < 400.0 and iterations < max_iter:
            # We add our constant 'c' instead of the screen position
            z = utils.complex_pow(z, w) + c
            iterations += 1
            
        if iterations == max_iter:
            pixels[i, j] = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
        else:
            t = utils.smoothing.simple_smooth(iterations, z[0]**2 + z[1]**2, color_freq)
            pixels[i, j] = utils.colormap.psychedelic(t)

@ti.kernel
def mandelbrot(
    pixels: ti.template(), 
    zoom: float,
    pan_x: float,
    pan_y: float,
    max_iter: int, 
    color_freq: float,
    power: float
):
    width = pixels.shape[0]
    height = pixels.shape[1]
    
    w = ti.Vector([ti.cast(power, float), ti.cast(0.0, float)])
    
    for i, j in pixels:  
        position = utils.screen_to_math(i, j, width, height, zoom, pan_x, pan_y)
        
        # Ensure z starts in the correct precision to stop warnings
        z = ti.Vector([ti.cast(0.0, float), ti.cast(0.0, float)])
        
        iterations = 0
        while z[0]**2 + z[1]**2 < 400.0 and iterations < max_iter:
            z = utils.complex_pow(z, w) + position
            iterations += 1
            
        if iterations == max_iter:
            pixels[i, j] = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
        else:
            t = utils.smoothing.simple_smooth(iterations, z[0]**2 + z[1]**2, color_freq)
            pixels[i, j] = utils.colormap.psychedelic(t)