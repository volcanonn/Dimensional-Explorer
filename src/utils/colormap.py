import taichi as ti
import taichi.math as tm

@ti.func
def psychedelic(t):
    t32 = ti.cast(t, ti.f32)
    half = ti.cast(0.5, ti.f32)
    three_pi = ti.cast(3.0 * tm.pi, ti.f32)

    r = half * tm.sin(t32 * three_pi + ti.cast(0.0, ti.f32)) + half
    g = half * tm.sin(t32 * three_pi + ti.cast(1.5, ti.f32)) + half
    b = half * tm.sin(t32 * three_pi + ti.cast(3.0, ti.f32)) + half

    return ti.cast(ti.Vector([r, g, b]), ti.f32)

@ti.func
def piecewise(t, palette: ti.template()):
    t32 = ti.cast(t, ti.f32)
    n = palette.n
    
    color = ti.Vector([palette[n - 1, 0], palette[n - 1, 1], palette[n - 1, 2]])
    
    if t32 <= palette[0, 3]:
        color = ti.Vector([palette[0, 0], palette[0, 1], palette[0, 2]])
    else:
        for i in ti.static(range(n - 1)):
            if t32 < palette[i + 1, 3]:
                b0 = palette[i, 3]
                b1 = palette[i + 1, 3]
                
                c0 = ti.Vector([palette[i, 0],     palette[i, 1],     palette[i, 2]])
                c1 = ti.Vector([palette[i + 1, 0], palette[i + 1, 1], palette[i + 1, 2]])
                
                color = c0 + (c1 - c0) * (t32 - b0) / (b1 - b0)
                break 
    return color

heledron_palette = ti.Matrix([
    [0.0,           7.0 / 255.0,   100.0 / 255.0,  0.0],
    [32.0 / 255.0,  107.0 / 255.0, 203.0 / 255.0,  0.16],
    [237.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0,  0.42],
    [255.0 / 255.0, 170.0 / 255.0, 0.0,            0.6425],
    [0.0,           0.0,           0.0,            0.8575]
])

@ti.func
def heledron(t):
    t32 = ti.cast(t, ti.f32)

    c0 = ti.Vector([0.0, 7.0 / 255.0, 100.0 / 255.0])
    c1 = ti.Vector([32.0 / 255.0, 107.0 / 255.0, 203.0 / 255.0])
    c2 = ti.Vector([237.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0])
    c3 = ti.Vector([255.0 / 255.0, 170.0 / 255.0, 0.0])
    c4 = ti.Vector([0.0, 0.0, 0.0])

    color = c4

    if t32 < 0.16:
        color = c0 + (c1 - c0) * (t32 - 0.0) / 0.16
    elif t32 < 0.42:
        color = c1 + (c2 - c1) * (t32 - 0.16) / (0.42 - 0.16)
    elif t32 < 0.6425:
        color = c2 + (c3 - c2) * (t32 - 0.42) / (0.6425 - 0.42)
    elif t32 < 0.8575:
        color = c3 + (c4 - c3) * (t32 - 0.6425) / (0.8575 - 0.6425)

    return ti.cast(color, ti.f32)

# I should realy turn these piecewise interpolations into like a easy system cause these functions are the same

@ti.func
def armada(t):
    t32 = ti.cast(t, ti.f32)
    
    t_wrap = t32 - ti.floor(t32)

    #   #Thank you wikipedia
    c0 = ti.Vector([28.0 / 255.0, 43.0 / 255.0, 95.0 / 255.0])   # Deep Navy Blue (Background)
    c1 = ti.Vector([160.0 / 255.0, 15.0 / 255.0, 0.0 / 255.0])   # Dark Crimson Red
    c2 = ti.Vector([255.0 / 255.0, 90.0 / 255.0, 0.0 / 255.0])   # Bright Orange
    c3 = ti.Vector([255.0 / 255.0, 200.0 / 255.0, 0.0 / 255.0])  # Brilliant Yellow
    c4 = c0

    color = c0

    if t_wrap < 0.25:
        color = c0 + (c1 - c0) * (t_wrap - 0.0) / 0.25
    elif t_wrap < 0.60:
        color = c1 + (c2 - c1) * (t_wrap - 0.25) / (0.60 - 0.25)
    elif t_wrap < 0.85:
        color = c2 + (c3 - c2) * (t_wrap - 0.60) / (0.85 - 0.60)
    else:
        color = c3 + (c4 - c3) * (t_wrap - 0.85) / (1.0 - 0.85)

    return ti.cast(color, ti.f32)

@ti.func
def ghostly_ice(t):
    t32 = ti.cast(t, ti.f32)
    t_wrap = t32 - ti.floor(t32)

    c0 = ti.Vector([35.0 / 255.0, 25.0 / 255.0, 45.0 / 255.0])   # Dark Plum / Grey space
    c1 = ti.Vector([40.0 / 255.0, 60.0 / 255.0, 100.0 / 255.0])  # Deep Steel Blue
    c2 = ti.Vector([100.0 / 255.0, 200.0 / 255.0, 255.0 / 255.0])# Bright Cyan
    c3 = ti.Vector([255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0])# Pure White
    c4 = c0

    color = c0
    if t_wrap < 0.4:
        color = c0 + (c1 - c0) * (t_wrap - 0.0) / 0.4
    elif t_wrap < 0.7:
        color = c1 + (c2 - c1) * (t_wrap - 0.4) / (0.7 - 0.4)
    elif t_wrap < 0.9:
        color = c2 + (c3 - c2) * (t_wrap - 0.7) / (0.9 - 0.7)
    else:
        color = c3 + (c4 - c3) * (t_wrap - 0.9) / (1.0 - 0.9)

    return ti.cast(color, ti.f32)

@ti.func
def crimson_cyan(t):
    t32 = ti.cast(t, ti.f32)
    t_wrap = t32 - ti.floor(t32)

    c0 = ti.Vector([60.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0])     # Dark Crimson
    c1 = ti.Vector([240.0 / 255.0, 20.0 / 255.0, 0.0 / 255.0])   # Bright Red
    c2 = ti.Vector([255.0 / 255.0, 200.0 / 255.0, 0.0 / 255.0])  # Yellow
    c3 = ti.Vector([0.0 / 255.0, 255.0 / 255.0, 200.0 / 255.0])  # Sharp Cyan
    c4 = c0

    color = c0
    if t_wrap < 0.5:
        color = c0 + (c1 - c0) * (t_wrap - 0.0) / 0.5
    elif t_wrap < 0.8:
        color = c1 + (c2 - c1) * (t_wrap - 0.5) / (0.8 - 0.5)
    elif t_wrap < 0.95:
        color = c2 + (c3 - c2) * (t_wrap - 0.8) / (0.95 - 0.8)
    else:
        color = c3 + (c4 - c3) * (t_wrap - 0.95) / (1.0 - 0.95)

    return ti.cast(color, ti.f32)

@ti.func
def ocean_purple(t):
    t32 = ti.cast(t, ti.f32)
    t_wrap = t32 - ti.floor(t32)

    c0 = ti.Vector([0.0 / 255.0, 0.0 / 255.0, 150.0 / 255.0])    # Deep Blue
    c1 = ti.Vector([0.0 / 255.0, 180.0 / 255.0, 255.0 / 255.0])  # Cyan
    c2 = ti.Vector([255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0])# White
    c3 = ti.Vector([120.0 / 255.0, 0.0 / 255.0, 180.0 / 255.0])  # Rich Purple
    c4 = c0

    color = c0
    if t_wrap < 0.3:
        color = c0 + (c1 - c0) * (t_wrap - 0.0) / 0.3
    elif t_wrap < 0.5:
        color = c1 + (c2 - c1) * (t_wrap - 0.3) / (0.5 - 0.3)
    elif t_wrap < 0.75:
        color = c2 + (c3 - c2) * (t_wrap - 0.5) / (0.75 - 0.5)
    else:
        color = c3 + (c4 - c3) * (t_wrap - 0.75) / (1.0 - 0.75)

    return ti.cast(color, ti.f32)

@ti.func
def metallic_silk(t):
    t32 = ti.cast(t, ti.f32)
    t_wrap = t32 - ti.floor(t32)

    c0 = ti.Vector([0.0 / 255.0, 100.0 / 255.0, 255.0 / 255.0])  # Bright Blue
    c1 = ti.Vector([255.0 / 255.0, 200.0 / 255.0, 0.0 / 255.0])  # Gold
    c2 = ti.Vector([150.0 / 255.0, 50.0 / 255.0, 0.0 / 255.0])   # Copper
    c3 = ti.Vector([255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0])# White
    c4 = ti.Vector([150.0 / 255.0, 0.0 / 255.0, 150.0 / 255.0])  # Purple
    c5 = c0

    color = c0
    if t_wrap < 0.2:
        color = c0 + (c1 - c0) * (t_wrap - 0.0) / 0.2
    elif t_wrap < 0.4:
        color = c1 + (c2 - c1) * (t_wrap - 0.2) / (0.4 - 0.2)
    elif t_wrap < 0.6:
        color = c2 + (c3 - c2) * (t_wrap - 0.4) / (0.6 - 0.4)
    elif t_wrap < 0.8:
        color = c3 + (c4 - c3) * (t_wrap - 0.6) / (0.8 - 0.6)
    else:
        color = c4 + (c5 - c4) * (t_wrap - 0.8) / (1.0 - 0.8)

    return ti.cast(color, ti.f32)

@ti.func
def toxic_green(t):
    t32 = ti.cast(t, ti.f32)
    t_wrap = t32 - ti.floor(t32)

    c0 = ti.Vector([150.0 / 255.0, 220.0 / 255.0, 0.0 / 255.0])  # Bright Lime
    c1 = ti.Vector([50.0 / 255.0, 160.0 / 255.0, 0.0 / 255.0])   # Mid Green
    c2 = ti.Vector([0.0 / 255.0, 80.0 / 255.0, 0.0 / 255.0])     # Dark Forest Green
    c3 = c0

    color = c0
    if t_wrap < 0.4:
        color = c0 + (c1 - c0) * (t_wrap - 0.0) / 0.4
    elif t_wrap < 0.8:
        color = c1 + (c2 - c1) * (t_wrap - 0.4) / (0.8 - 0.4)
    else:
        color = c2 + (c3 - c2) * (t_wrap - 0.8) / (1.0 - 0.8)

    return ti.cast(color, ti.f32)

@ti.func
def nebula_fake(t):
    t32 = ti.cast(t, ti.f32)
    t_wrap = t32 - ti.floor(t32)

    c0 = ti.Vector([10.0 / 255.0, 20.0 / 255.0, 40.0 / 255.0])   # Deep Space
    c1 = ti.Vector([0.0 / 255.0, 150.0 / 255.0, 150.0 / 255.0])  # Glowing Teal
    c2 = ti.Vector([255.0 / 255.0, 100.0 / 255.0, 0.0 / 255.0])  # Hot Orange
    c3 = ti.Vector([100.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0])  # Ambient Purple
    c4 = c0

    color = c0
    if t_wrap < 0.4:
        color = c0 + (c1 - c0) * (t_wrap - 0.0) / 0.4
    elif t_wrap < 0.6:
        color = c1 + (c2 - c1) * (t_wrap - 0.4) / (0.6 - 0.4)
    elif t_wrap < 0.8:
        color = c2 + (c3 - c2) * (t_wrap - 0.6) / (0.8 - 0.6)
    else:
        color = c3 + (c4 - c3) * (t_wrap - 0.8) / (1.0 - 0.8)

    return ti.cast(color, ti.f32)