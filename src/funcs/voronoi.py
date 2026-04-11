import taichi as ti
import utils

@ti.func
def voronoi_core(pos_nd: ti.template(), max_iter: int, color_freq: float):
    x = pos_nd[0] if ti.static(pos_nd.n > 0) else 0.0
    y = pos_nd[1] if ti.static(pos_nd.n > 1) else 0.0
    uv = ti.Vector([x, y])
    
    # Safe 64-bit manual fract
    grid_uv = (uv * 5.0) - ti.floor(uv * 5.0)
    grid_id = ti.floor(uv * 5.0)
    min_dist = ti.cast(1.0, float)
    
    for y_offset in range(-1, 2):
        for x_offset in range(-1, 2):
            neighbor_id = ti.Vector([float(x_offset), float(y_offset)])
            
            rand_val = utils.random(grid_id + neighbor_id)
            point_pos = neighbor_id + ti.Vector([rand_val, rand_val])
            
            # 64-bit safe length calculation using ti.sqrt
            dx = grid_uv[0] - point_pos[0]
            dy = grid_uv[1] - point_pos[1]
            dist = ti.sqrt(dx*dx + dy*dy)
            
            min_dist = ti.min(min_dist, dist)
            
    t = min_dist * color_freq * 10.0
    return utils.colormap.heledron(t)