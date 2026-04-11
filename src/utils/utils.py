import taichi as ti
import taichi.math as tm
from . import f64_math
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
def smart_sin(x):
    res = x
    if ti.static(config.USE_F64):
        res = f64_math.f64_sin(x)
    else:
        res = tm.sin(x)
    return res

@ti.func
def smart_cos(x):
    res = x
    if ti.static(config.USE_F64):
        res = f64_math.f64_cos(x)
    else:
        res = tm.cos(x)
    return res

@ti.func
def smart_exp(x):
    res = x
    if ti.static(config.USE_F64):
        res = f64_math.f64_exp(x)
    else:
        res = tm.exp(x)
    return res

@ti.func
def smart_log(x):
    res = x
    if ti.static(config.USE_F64):
        res = f64_math.f64_log(x)
    else:
        res = tm.log(x)
    return res

@ti.func
def smart_atan2(y, x):
    res = y
    if ti.static(config.USE_F64):
        res = f64_math.f64_atan2(y, x)
    else:
        res = tm.atan2(y, x)
    return res

@ti.func
def random(st):
    """64-bit Safe GLSL pseudo-random generator"""
    dot_product = st[0] * 12.9898 + st[1] * 78.233
    val = smart_sin(dot_product) * 43758.5453123
    return val - ti.floor(val) 

# --- COMPLEX MATH ---
@ti.func
def complex_mul(a, b):
    return ti.Vector([a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]])

@ti.func
def complex_pow(z, w):
    result = ti.Vector([ti.cast(0.0, float), ti.cast(0.0, float)])
    
    # Epsilon Check for Integer Fast-Path
    is_integer = (ti.abs(w[1]) < 1e-4) and (ti.abs(w[0] - ti.round(w[0])) < 1e-4)
    
    if is_integer:
        # Fast-path for integers (Pure multiplication)
        n = ti.cast(ti.abs(ti.round(w[0])), ti.i32)
        temp = ti.Vector([ti.cast(1.0, float), ti.cast(0.0, float)])
        for _ in range(n):
            temp = complex_mul(temp, z)
            
        if w[0] < 0:
            denom = temp[0]**2 + temp[1]**2
            temp = ti.Vector([temp[0]/denom, -temp[1]/denom])
        result = temp
    else:
        # Fractional powers using SMART hardware routing!
        r = ti.sqrt(z[0]**2 + z[1]**2)
        if r > 1e-15:
            # Replaced all f64_math calls with smart_* calls!
            theta = smart_atan2(z[1], z[0])
            ln_r = smart_log(r)
            
            A = w[0] * ln_r - w[1] * theta
            B = w[0] * theta + w[1] * ln_r
            
            exp_A = smart_exp(A)
            result = ti.Vector([
                exp_A * smart_cos(B), 
                exp_A * smart_sin(B)
            ])
            
    return result