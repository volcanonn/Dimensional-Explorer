import taichi as ti
from . import f64_math

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
def screen_to_math(i, j, width, height, zoom, pan_x, pan_y):
    screen_x = ti.cast(i, float) - ti.cast(width, float) / 2.0
    screen_y = ti.cast(j, float) - ti.cast(height, float) / 2.0
    
    math_x = (screen_x / zoom) + pan_x
    math_y = (screen_y / zoom) + pan_y
    
    return ti.Vector([math_x, math_y])

@ti.func
def quat_sqr(q):
    """Squares a 4D Quaternion: (x^2 - y^2 - z^2 - w^2, 2xy, 2xz, 2xw)"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ti.Vector([
        x*x - y*y - z*z - w*w,
        2.0 * x * y,
        2.0 * x * z,
        2.0 * x * w
    ])

@ti.func
def rotate_4d(p, angle, axis1: ti.template(), axis2: ti.template()):
    """Rotates a 4D vector along a specific 2D plane (e.g., XW or YZ)"""
    c = f64_math.f64_cos(angle)
    s = f64_math.f64_sin(angle)
    
    # We do this to avoid modifying the vector while we are reading it
    p_new = p
    p_new[axis1] = p[axis1] * c - p[axis2] * s
    p_new[axis2] = p[axis1] * s + p[axis2] * c
    return p_new

@ti.func
def complex_mul(a, b):
    return ti.Vector([a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]])

@ti.func
def complex_pow(z, w):
    result = ti.Vector([ti.cast(0.0, float), ti.cast(0.0, float)])
    
    is_integer = (ti.abs(w[1]) < 1e-4) and (ti.abs(w[0] - ti.round(w[0])) < 1e-4)
    
    if is_integer:
        n = ti.cast(ti.abs(ti.round(w[0])), ti.i32)
        temp = ti.Vector([ti.cast(1.0, float), ti.cast(0.0, float)])
        for _ in range(n):
            temp = complex_mul(temp, z)
            
        if w[0] < 0:
            denom = temp[0]**2 + temp[1]**2
            temp = ti.Vector([temp[0]/denom, -temp[1]/denom])
        result = temp
    else:
        # Fractional powers
        r = ti.sqrt(z[0]**2 + z[1]**2)
        if r > 1e-15:
            theta = f64_math.f64_atan2(z[1], z[0])
            ln_r = f64_math.f64_log(r)
            
            A = w[0] * ln_r - w[1] * theta
            B = w[0] * theta + w[1] * ln_r
            
            exp_A = f64_math.f64_exp(A)
            result = ti.Vector([
                exp_A * f64_math.f64_cos(B), 
                exp_A * f64_math.f64_sin(B)
            ])
            
    return result