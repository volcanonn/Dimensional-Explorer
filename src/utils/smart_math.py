import taichi as ti
import taichi.math as tm
from . import f64_math
import config

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