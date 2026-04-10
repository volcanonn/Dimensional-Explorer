import taichi as ti

@ti.func
def screen_to_math(i, j, width, height, zoom, pan_x, pan_y):
    screen_x = float(i) - float(width) / 2.0
    screen_y = float(j) - float(height) / 2.0
    
    math_x = (screen_x / zoom) + pan_x
    math_y = (screen_y / zoom) + pan_y
    
    return ti.Vector([math_x, math_y])

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])