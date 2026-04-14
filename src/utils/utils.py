import taichi as ti
import taichi.math as tm
from . import colormap, smart_math

def format_time(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f} s"
    elif ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    elif ns >= 1_000:
        return f"{ns / 1_000:.2f} us"
    else:
        return f"{ns:.2f} ns"

@ti.func
def random(st, use_f64: ti.template()):
    zero = st[0] * 0.0
    dot_product = st[0] * (zero + 12.9898) + st[1] * (zero + 78.233)
    val = smart_math.smart_sin(dot_product, use_f64) * (zero + 43758.5453123)
    return val - ti.floor(val)

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
        r2 = z.norm_sqr()

        if r2 > 1e-15:
            theta = smart_math.smart_atan2(z[1], z[0], use_f64)
            ln_r = 0.5 * smart_math.smart_log(r2, use_f64)
            
            A = w[0] * ln_r - w[1] * theta
            B = w[0] * theta + w[1] * ln_r
            
            exp_A = smart_math.smart_exp(A, use_f64)
            result = ti.Vector([
                exp_A * smart_math.smart_cos(B, use_f64),
                exp_A * smart_math.smart_sin(B, use_f64)
            ])

    return result

@ti.kernel
def blit_image(dest: ti.template(), src: ti.template(), offset_x: int, offset_y: int):
    for i, j in src:
        di = i + offset_x
        dj = j + offset_y
        if 0 <= di < dest.shape[0] and 0 <= dj < dest.shape[1]:
            dest[di, dj] = src[i, j]

@ti.func
def apply_colormap(t, colormap_idx: ti.template()):
    color = ti.cast(ti.Vector([0.0, 0.0, 0.0]), ti.f32)
    
    if ti.static(colormap_idx == 0):
        color = colormap.heledron(t)
    elif ti.static(colormap_idx == 1):
        color = colormap.psychedelic(t)
    elif ti.static(colormap_idx == 2):
        color = colormap.armada(t)
    elif ti.static(colormap_idx == 3):
        color = colormap.ghostly_ice(t)
    elif ti.static(colormap_idx == 4):
        color = colormap.crimson_cyan(t)
    elif ti.static(colormap_idx == 5):
        color = colormap.ocean_purple(t)
    elif ti.static(colormap_idx == 6):
        color = colormap.metallic_silk(t)
    elif ti.static(colormap_idx == 7):
        color = colormap.toxic_green(t)
    elif ti.static(colormap_idx == 8):
        color = colormap.nebula_fake(t)
        
    return color