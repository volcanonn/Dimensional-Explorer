import taichi as ti
import utils

@ti.kernel
def julia(pixels: ti.template(), zoom: float, pan_x: float, pan_y: float):
    width = pixels.shape[0]
    height = pixels.shape[1]
    
    for i, j in pixels:  
        z = utils.screen_to_math(i, j, width, height, zoom, pan_x, pan_y)
        c = ti.Vector([0.0, 0.0])
        
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = utils.complex_sqr(z) + c
            iterations += 1
            
        color = 1.0 - iterations * 0.02
        pixels[i, j] = ti.Vector([color, color, color])


@ti.kernel
def mandelbrot(pixels: ti.template(), zoom: float, pan_x: float, pan_y: float):
    width = pixels.shape[0]
    height = pixels.shape[1]
    
    for i, j in pixels:  
        c = utils.screen_to_math(i, j, width, height, zoom, pan_x, pan_y)
        z = ti.Vector([0.0, 0.0])
        
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = utils.complex_sqr(z) + c
            iterations += 1
            
        color = 1.0 - iterations * 0.02
        pixels[i, j] = ti.Vector([color, color, color])