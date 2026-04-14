import taichi as ti
import taichi.math as tm
from . import f64_math, colormap
import config

def format_time(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f} s"
    elif ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    elif ns >= 1_000:
        return f"{ns / 1_000:.2f} us"
    else:
        return f"{ns:.2f} ns"

# --- SMART DISPATCHERS ---
@ti.func
def smart_sin(x, use_f64: ti.template()):
    if ti.static(use_f64):
        if ti.static(config.SUPPORT_F64_TRIG):
            return ti.cast(tm.sin(ti.cast(x, config.F64_SAFE)), config.F64_SAFE)
        else:
            return f64_math.f64_sin(ti.cast(x, config.F64_SAFE))
    else:
        return ti.cast(tm.sin(ti.cast(x, ti.f32)), ti.f32)


@ti.func
def smart_cos(x, use_f64: ti.template()):
    if ti.static(use_f64):
        if ti.static(config.SUPPORT_F64_TRIG):
            return ti.cast(tm.cos(ti.cast(x, config.F64_SAFE)), config.F64_SAFE)
        else:
            return f64_math.f64_cos(ti.cast(x, config.F64_SAFE))
    else:
        return ti.cast(tm.cos(ti.cast(x, ti.f32)), ti.f32)


@ti.func
def smart_exp(x, use_f64: ti.template()):
    if ti.static(use_f64):
        if ti.static(config.SUPPORT_F64_TRIG):
            return ti.cast(tm.exp(ti.cast(x, config.F64_SAFE)), config.F64_SAFE)
        else:
            return f64_math.f64_exp(ti.cast(x, config.F64_SAFE))
    else:
        return ti.cast(tm.exp(ti.cast(x, ti.f32)), ti.f32)


@ti.func
def smart_log(x, use_f64: ti.template()):
    if ti.static(use_f64):
        if ti.static(config.SUPPORT_F64_TRIG):
            return ti.cast(tm.log(ti.cast(x, config.F64_SAFE)), config.F64_SAFE)
        else:
            return f64_math.f64_log(ti.cast(x, config.F64_SAFE))
    else:
        return ti.cast(tm.log(ti.cast(x, ti.f32)), ti.f32)


@ti.func
def smart_atan2(y, x, use_f64: ti.template()):
    if ti.static(use_f64):
        if ti.static(config.SUPPORT_F64_TRIG):
            return ti.cast(tm.atan2(ti.cast(y, config.F64_SAFE), ti.cast(x, config.F64_SAFE)), config.F64_SAFE)
        else:
            return f64_math.f64_atan2(ti.cast(y, config.F64_SAFE), ti.cast(x, config.F64_SAFE))
    else:
        return ti.cast(tm.atan2(ti.cast(y, ti.f32), ti.cast(x, ti.f32)), ti.f32)


@ti.func
def random(st, use_f64: ti.template()):
    zero = st[0] * 0.0
    dot_product = st[0] * (zero + 12.9898) + st[1] * (zero + 78.233)
    val = smart_sin(dot_product, use_f64) * (zero + 43758.5453123)
    return val - ti.floor(val)


# --- COMPLEX MATH ---
@ti.func
def complex_mul(a, b):
    return ti.Vector([
        a[0] * b[0] - a[1] * b[1],
        a[0] * b[1] + a[1] * b[0]
    ])


@ti.func
def complex_pow(z, w, use_f64: ti.template()):
    zero = z[0] * 0.0
    result = ti.Vector([zero, zero])
    is_integer = (ti.abs(w[1]) < 1e-4) and (ti.abs(w[0] - ti.round(w[0])) < 1e-4)

    if is_integer:
        n = ti.cast(ti.abs(ti.round(w[0])), ti.i32)
        temp = ti.Vector([zero + 1.0, zero])

        for _ in range(n):
            temp = complex_mul(temp, z)

        if w[0] < 0:
            denom = temp[0] ** 2 + temp[1] ** 2
            temp = ti.Vector([temp[0] / denom, -temp[1] / denom])

        result = temp
    else:
        r = ti.sqrt(z[0] ** 2 + z[1] ** 2)

        if r > 1e-15:
            theta = smart_atan2(z[1], z[0], use_f64)
            ln_r = smart_log(r, use_f64)
            
            A = w[0] * ln_r - w[1] * theta
            B = w[0] * theta + w[1] * ln_r
            
            exp_A = smart_exp(A, use_f64)
            result = ti.Vector([
                exp_A * smart_cos(B, use_f64),
                exp_A * smart_sin(B, use_f64)
            ])

    return result

@ti.kernel
def blit_image(dest: ti.template(), src: ti.template(), offset_x: int, offset_y: int):
    """Copies a mini-viewport directly onto the main screen buffer!"""
    for i, j in src:
        di = i + offset_x
        dj = j + offset_y
        if 0 <= di < dest.shape[0] and 0 <= dj < dest.shape[1]:
            dest[di, dj] = src[i, j]

@ti.func
def apply_colormap(t, colormap_idx: ti.template()):
    """A zero-overhead compile-time dispatcher for colormaps!"""
    color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
    
    if ti.static(colormap_idx == 0):
        color = colormap.heledron(t)
    elif ti.static(colormap_idx == 1):
        color = colormap.psychedelic(t)
        
    return color