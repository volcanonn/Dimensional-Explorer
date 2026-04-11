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