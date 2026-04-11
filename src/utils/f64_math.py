import taichi as ti

@ti.func
def f64_exp(x):
    """Calculates e^x using range reduction and Taylor series."""
    LN2 = 0.6931471805599453
    # Range reduce: x = k * ln(2) + r
    k = ti.floor(x / LN2 + 0.5)
    r = x - k * LN2
    
    # Taylor series for e^r (extremely fast because r is tiny!)
    res = ti.cast(1.0, float)
    term = ti.cast(1.0, float)
    for i in ti.static(range(1, 15)):
        term = term * r / ti.cast(i, float)
        res += term
        
    # Re-apply the 2^k exponent
    twok = ti.cast(1.0, float)
    if k > 0:
        for _ in range(ti.cast(k, ti.i32)): twok *= 2.0
    elif k < 0:
        for _ in range(ti.cast(-k, ti.i32)): twok *= 0.5
        
    return res * twok

@ti.func
def f64_log(x):
    """Calculates ln(x) using Area Hyperbolic Tangent."""
    val = x
    e = ti.cast(0.0, float)
    
    # Scale x to be between ~0.7 and ~1.4 for rapid convergence
    while val > 1.414213562373095:
        val *= 0.5
        e += 1.0
    while val < 0.7071067811865475:
        val *= 2.0
        e -= 1.0

    z = (val - 1.0) / (val + 1.0)
    z2 = z * z
    
    res = z
    term = z
    for i in ti.static(range(1, 15)):
        term *= z2
        res += term / ti.cast(2 * i + 1, float)
        
    return e * 0.6931471805599453 + 2.0 * res

@ti.func
def f64_atan(x):
    """Calculates atan(x) using Euler's accelerated formula."""
    PI_2 = 1.5707963267948966
    inv = False
    sign = ti.cast(1.0, float)
    
    if x < 0.0:
        x = -x
        sign = -1.0
    if x > 1.0:
        x = 1.0 / x
        inv = True
        
    x2 = x * x
    y = x2 / (1.0 + x2)
    term = x / (1.0 + x2)
    res = term
    
    for i in ti.static(range(1, 25)):
        term = term * y * ti.cast(2 * i, float) / ti.cast(2 * i + 1, float)
        res += term
        
    if inv:
        res = PI_2 - res
        
    return res * sign

@ti.func
def f64_atan2(y, x):
    PI = 3.1415926535897932
    PI_2 = 1.5707963267948966
    res = ti.cast(0.0, float)
    
    if x > 0.0: res = f64_atan(y / x)
    elif x < 0.0 and y >= 0.0: res = f64_atan(y / x) + PI
    elif x < 0.0 and y < 0.0: res = f64_atan(y / x) - PI
    elif x == 0.0 and y > 0.0: res = PI_2
    elif x == 0.0 and y < 0.0: res = -PI_2
        
    return res

@ti.func
def f64_sin(x):
    TWO_PI = 6.283185307179586
    PI = 3.1415926535897932
    x_red = x - TWO_PI * ti.floor((x + PI) / TWO_PI)
    
    res = x_red
    term = x_red
    x2 = x_red * x_red
    for i in ti.static(range(1, 15)):
        term = -term * x2 / ti.cast((2 * i) * (2 * i + 1), float)
        res += term
    return res

@ti.func
def f64_cos(x):
    TWO_PI = 6.283185307179586
    PI = 3.1415926535897932
    x_red = x - TWO_PI * ti.floor((x + PI) / TWO_PI)
    
    res = ti.cast(1.0, float)
    term = ti.cast(1.0, float)
    x2 = x_red * x_red
    for i in ti.static(range(1, 15)):
        term = -term * x2 / ti.cast((2 * i - 1) * (2 * i), float)
        res += term
    return res