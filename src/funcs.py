import taichi as ti
import utils
import config

#Newton Fractals, Mandelbar (Multibrot) Fractal

@ti.func
def mandelbrot_core(pos_nd: ti.template(), max_iter: int, color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero
    z = pos_nd[2] if ti.static(N > 2) else zero
    w = pos_nd[3] if ti.static(N > 3) else zero
    v = pos_nd[4] if ti.static(N > 4) else zero + 2.0
    u = pos_nd[5] if ti.static(N > 5) else zero

    c_vec = ti.Vector([x, y])
    z_vec = ti.Vector([z, w])
    e_vec = ti.Vector([v, u])

    iterations = 0

    while z_vec.norm_sqr() < 400.0 and iterations < max_iter:
        z_vec = utils.complex_pow(z_vec, e_vec, use_f64) + c_vec
        iterations += 1

    color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)

    if iterations < max_iter:
        t = utils.smoothing.simple_smooth(iterations, z_vec[0] ** 2 + z_vec[1] ** 2, color_freq)
        color = utils.apply_colormap(t, colormap_idx)

    return color

@ti.func
def mandelbrot_testing_core(pos_nd: ti.template(), max_iter: int, color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero
    z = pos_nd[2] if ti.static(N > 2) else zero
    w = pos_nd[3] if ti.static(N > 3) else zero
    v = pos_nd[4] if ti.static(N > 4) else zero + 2.0
    u = pos_nd[5] if ti.static(N > 5) else zero

    c_vec = ti.Vector([x, y])
    z_vec = ti.Vector([z, w])
    e_vec = ti.Vector([v, u])

    iterations = 0

    z_old = ti.Vector([z, w])
    check_period = 2
    steps_since_check = 0
    tolerance = 1e-10

    while z_vec.norm_sqr() < 400.0 and iterations < max_iter:
        z_vec_next = utils.complex_pow(z_vec, e_vec, use_f64) + c_vec
        
        # 1. Fast Orbit Trap / Periodicity Check (Distance based)
        diff = z_vec_next - z_old
        if diff.norm_sqr() < tolerance:
            iterations = max_iter
            break
            
        # Brent's Cycle Detection logic
        steps_since_check += 1
        if steps_since_check == check_period:
            z_old = z_vec_next
            check_period *= 2
            steps_since_check = 0
        
        z_vec = z_vec_next
        iterations += 1

    color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)

    if iterations < max_iter:
        t = utils.smoothing.simple_smooth(iterations, z_vec[0] ** 2 + z_vec[1] ** 2, color_freq)
        color = utils.apply_colormap(t, colormap_idx)

    return color

@ti.func
def burningship_core(pos_nd: ti.template(), max_iter: int, color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero
    z = pos_nd[2] if ti.static(N > 2) else zero
    w = pos_nd[3] if ti.static(N > 3) else zero
    v = pos_nd[4] if ti.static(N > 4) else zero + 2.0
    u = pos_nd[5] if ti.static(N > 5) else zero

    c_vec = ti.Vector([x, y])
    z_vec = ti.Vector([z, w])
    e_vec = ti.Vector([v, u])

    iterations = 0

    escape_rad = 100.0

    while z_vec.norm_sqr() < 10000.0 and iterations < max_iter:
        z_vec = utils.complex_pow(ti.abs(z_vec), e_vec, use_f64) + c_vec
        iterations += 1

    color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)

    if iterations < max_iter:
        log_zn = 0.5 * utils.smart_math.smart_log(z_vec.norm_sqr(), use_f64)
        nu = utils.smart_math.smart_log(log_zn / utils.smart_math.smart_log(100.0, use_f64), use_f64) / utils.smart_math.smart_log(v, use_f64)
        t = (ti.cast(iterations, ti.f32) + 1.0 - nu)
        t = t * color_freq
        color = utils.apply_colormap(t, colormap_idx)

    return color


@ti.func
def conic_core(pos_nd: ti.template(), color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero
    z = pos_nd[2] if ti.static(N > 2) else zero

    cone_dist = ti.sqrt(x * x + y * y) - ti.abs(z)
    glow = utils.smart_math.smart_exp(-ti.abs(cone_dist) * (zero + 10.0 * color_freq), use_f64)

    return utils.apply_colormap(glow, colormap_idx)


@ti.func
def voronoi_core(pos_nd: ti.template(), color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero

    uv = ti.Vector([x, y])
    grid_uv = (uv * 5.0) - ti.floor(uv * 5.0)
    grid_id = ti.floor(uv * 5.0)
    
    min_dist = zero + 1.0

    for y_off in range(-1, 2):
        for x_off in range(-1, 2):
            neighbor_id = ti.Vector([zero + float(x_off), zero + float(y_off)])
            rand_val = utils.random(grid_id + neighbor_id, use_f64)
            point_pos = neighbor_id + ti.Vector([rand_val, rand_val])
            
            dx = grid_uv[0] - point_pos[0]
            dy = grid_uv[1] - point_pos[1]
            min_dist = ti.min(min_dist, ti.sqrt(dx * dx + dy * dy))

    t = min_dist * (zero + color_freq * 10.0)
    
    return utils.apply_colormap(t - ti.floor(t), colormap_idx)


@ti.func
def simple_wave_core(pos_nd: ti.template(), color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero

    val = utils.smart_math.smart_sin(x, use_f64) + utils.smart_math.smart_cos(y, use_f64)
    t = ti.abs(val * (zero + color_freq * 2.0))

    return utils.apply_colormap(t - ti.floor(t), colormap_idx)


@ti.func
def radial_wave_core(pos_nd: ti.template(), color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero

    val = utils.smart_math.smart_sin(ti.sqrt(x * x + y * y), use_f64)
    t = ti.abs(val * (zero + color_freq * 5.0))

    return utils.apply_colormap(t - ti.floor(t), colormap_idx)


@ti.func
def paraboloid_core(pos_nd: ti.template(), color_freq: float, use_f64: ti.template(), colormap_idx: ti.template()):
    N = ti.static(pos_nd.n)
    zero = pos_nd[0] * 0.0

    x = pos_nd[0] if ti.static(N > 0) else zero
    y = pos_nd[1] if ti.static(N > 1) else zero

    t = ti.abs((x * y) * (zero + color_freq * 2.0))

    return utils.apply_colormap(t - ti.floor(t), colormap_idx)