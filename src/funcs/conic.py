import taichi as ti
import utils

@ti.func
def conic_core(pos_nd: ti.template(), max_iter: int, color_freq: float):
    x = pos_nd[0] if ti.static(pos_nd.n > 0) else 0.0
    y = pos_nd[1] if ti.static(pos_nd.n > 1) else 0.0
    z = pos_nd[2] if ti.static(pos_nd.n > 2) else 0.0
    
    # The mathematical distance to the surface of a 3D double-cone
    cone_dist = ti.sqrt(x*x + y*y) - ti.abs(z)
    
    # Creates a sharp, glowing outline along the intersection of the cone!
    glow = utils.smart_exp(-ti.abs(cone_dist) * 10.0 * color_freq)
    
    return utils.colormap.heledron(glow)