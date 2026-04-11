import taichi as ti
import utils

@ti.func
def simple_wave_core(pos_nd: ti.template(), max_iter: int, color_freq: float):
    x = pos_nd[0] if ti.static(pos_nd.n > 0) else 0.0
    y = pos_nd[1] if ti.static(pos_nd.n > 1) else 0.0
    
    val1 = utils.smart_sin(x) + utils.smart_cos(y)
    val2 = ti.abs(x * y) + 1e-5 
    
    t = utils.smoothing.simple_smooth(val1, val2, color_freq)
    return utils.colormap.heledron(t)

@ti.func
def radial_wave_core(pos_nd: ti.template(), max_iter: int, color_freq: float):
    x = pos_nd[0] if ti.static(pos_nd.n > 0) else 0.0
    y = pos_nd[1] if ti.static(pos_nd.n > 1) else 0.0
    
    val1 = utils.smart_sin(ti.abs(x - y))
    val2 = ti.abs(x * y) + 1e-5
    
    t = utils.smoothing.simple_smooth(val1, val2, color_freq)
    return utils.colormap.heledron(t)

@ti.func
def paraboloid_core(pos_nd: ti.template(), max_iter: int, color_freq: float):
    x = pos_nd[0] if ti.static(pos_nd.n > 0) else 0.0
    y = pos_nd[1] if ti.static(pos_nd.n > 1) else 0.0
    
    dot_xy = x * y
    val1 = dot_xy
    val2 = ti.abs(dot_xy) + 1e-5
    
    t = utils.smoothing.simple_smooth(val1, val2, color_freq)
    return utils.colormap.heledron(t)