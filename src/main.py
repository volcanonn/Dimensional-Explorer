import math
import taichi as ti
import time
import collections

import config
import utils
from numpy import float32 as npfloat32

# performance issues for mandelbrot come from the viewports with exponents mostly
# and how the exponent loop cant be unrolled because it doesnt know if its a integer
# that gets down to 340us with half the screen on the big monitor
# calc times can change if the gpu downclocks

debug = False

DIM_NAMES = []
for i in range(config.MAX_DIMENSIONS):
    if i == 0:
        DIM_NAMES.append("X")
    elif i == 1:
        DIM_NAMES.append("Y")
    elif i == 2:
        DIM_NAMES.append("Z")
    elif i == 3:
        DIM_NAMES.append("W")
    elif i == 4:
        DIM_NAMES.append("V")
    elif i == 5:
        DIM_NAMES.append("U")
    else:
        DIM_NAMES.append(f"D{i}")


class Viewport:
    def __init__(self, idx, dim1, dim2):
        self.idx = idx
        self.dim1 = dim1
        self.dim2 = dim2
        self.zoom = 200.0
        self.visible = True

        self.px_w = 256 if idx > 0 else 0
        self.px_h = 256 if idx > 0 else 0
        self.px_x = 0
        self.px_y = 0

        if idx > 0:
            self.visible = not debug
            self.pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.px_w, self.px_h))
        else:
            self.pixels = None

    def contains_mouse(self, mx_px, my_px):
        in_x = self.px_x <= mx_px <= (self.px_x + self.px_w)
        in_y = self.px_y <= my_px <= (self.px_y + self.px_h)
        return in_x and in_y


class CameraState:
    def __init__(self, name):
        self.zooms = [200.0] * (config.MAX_DIMENSIONS // 2)
        self.vp_axes = [(i * 2, i * 2 + 1) for i in range(config.MAX_DIMENSIONS // 2)]
        self.translations = [0.0] * config.MAX_DIMENSIONS
        self.max_iter = 100
        self.color_freq = 0.05
        self.colormap_idx = 0

        if name == "Mandelbrot":
            self.active_dims = 6
            self.use_f64 = False
            self.translations[4] = 2.0
            self.zooms[0] = 1000
        elif name == "Mandelbrot Testing":
            self.active_dims = 6
            self.use_f64 = False
            self.translations[4] = 2.0
            self.zooms[0] = 1000
        elif name == "Conic Sections":
            self.active_dims = 3
            self.use_f64 = False
            self.color_freq = 0.5
        elif name == "Burning Ship":
            self.active_dims = 6
            self.use_f64 = False
            self.translations[4] = 2.0
            self.colormap_idx = 2
            self.color_freq = 0.01
        else:
            self.active_dims = 2
            self.use_f64 = False
            self.color_freq = 0.1

        self.planes = []
        for i in range(self.active_dims):
            for j in range(i + 1, self.active_dims):
                self.planes.append((i, j))
                
        self.rotations = [0.0] * len(self.planes)

@ti.data_oriented
class App:
    def __init__(self):
        self.window = ti.ui.Window("Dimensional Explorer", (1280, 720), vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        self.frame_times = collections.deque(maxlen=60)
        self.calc_times = collections.deque(maxlen=600 if debug else 120)
        self.last_time = time.perf_counter() 
        self.current_shape = (0, 0)

        # Momentum State
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.momentum_mult = 1.0
        
        self.is_dragging = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        # Function Switcher Setup
        self.functions = [
            "Mandelbrot",
            "Burning Ship",
            "Conic Sections",
            "Voronoi",
            "Simple Wave",
            "Radial Wave",
            "Paraboloid",
            "Mandelbrot Testing"
        ]

        self.func_idx = 0

        self.colormaps = ["Heledron", "Psychedelic", "Armada", "Ghostly Ice", "Crimson Cyan-Fire", "Ocean Purple Waves", "Metallic Silk", "Toxic Green", "Nebula Fake"]
        self.colormap_idx = 0

        self.states = [CameraState(name) for name in self.functions]
        
        self.viewports = []
        self.hovered_vp_idx = 0
        
        num_viewports = config.MAX_DIMENSIONS // 2
        for i in range(num_viewports):
            self.viewports.append(Viewport(i, i * 2, i * 2 + 1))

        self.load_state()


    def run(self):
        while self.window.running:
            self.window.get_events()
            
            actual_shape = self.window.get_window_shape()

            if actual_shape != self.current_shape or self.viewports[0].pixels is None:
                self.current_shape = actual_shape
                self.viewports[0].pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=self.current_shape)


            width, height = self.current_shape
            
            padding = 20
            draw_idx = 0
            for vp in self.viewports[1:]:
                if vp.dim2 < self.active_dims and vp.visible:
                    vp.px_x = width - vp.px_w - padding
                    vp.px_y = height - padding - (draw_idx + 1) * vp.px_h - (draw_idx * padding)
                    draw_idx += 1

            # Generate the Orthogonal N-D Transformation Matrix
            basis = self.get_nd_basis_matrix()

            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            if len(self.frame_times) > 0:
                current_avg_dt = sum(self.frame_times) / len(self.frame_times)
                
                if dt > current_avg_dt * 2.0 or dt < current_avg_dt * 0.5:
                    self.frame_times.clear()
                    
            self.frame_times.append(dt)
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

            self.handle_input(width, height, basis, dt)

            ti.sync()
            starttime = time.perf_counter_ns()

            for vp in self.viewports:
                if vp.dim2 < self.active_dims and vp.visible:
                    if self.use_f64:
                        vec_type = config.vecMAX_f64
                        vp_zoom = float(vp.zoom)
                        
                        utils.nd_slice_f64(
                            vp.pixels,
                            vec_type(self.translations),
                            vec_type(basis[vp.dim1]),
                            vec_type(basis[vp.dim2]),
                            vp_zoom,
                            self.max_iter,
                            self.color_freq,
                            self.func_idx,
                            self.active_dims,
                            self.colormap_idx
                        )
                    else:
                        vec_type = config.vecMAX_f32
                        vp_zoom = npfloat32(vp.zoom)
                        
                        utils.nd_slice_f32(
                            vp.pixels,
                            vec_type(self.translations),
                            vec_type(basis[vp.dim1]),
                            vec_type(basis[vp.dim2]),
                            vp_zoom,
                            self.max_iter,
                            self.color_freq,
                            self.func_idx,
                            self.active_dims,
                            self.colormap_idx
                        )

            ti.sync()
            endtime = time.perf_counter_ns() - starttime

            if not debug:
                if len(self.calc_times) > 0:
                    current_avg = sum(self.calc_times) / len(self.calc_times)
                    
                    if endtime > current_avg * 2.0 or endtime < current_avg * 0.5:
                        self.calc_times.clear()

            self.calc_times.append(endtime)
            avg_calc_time = sum(self.calc_times) / len(self.calc_times)

            for vp in self.viewports[1:]:
                if vp.dim2 < self.active_dims and vp.visible:
                    utils.blit_image(self.viewports[0].pixels, vp.pixels, vp.px_x, vp.px_y)

            self.canvas.set_image(self.viewports[0].pixels)

            with self.gui.sub_window("Engine Stats", 0.02, 0.02, 0.3, 0.47):
                self.gui.text(f"Dimensions: {self.active_dims}D")
                self.gui.text(f"FPS: {fps:.1f}")
                self.gui.text(f"GPU Calc: {utils.format_time(avg_calc_time)}")
                
                self.gui.text("")
                self.gui.text(f"CORE: {self.functions[self.func_idx]}")

                if self.gui.button("Prev Core"):
                    self.save_state()
                    self.func_idx = (self.func_idx - 1) % len(self.functions)
                    self.load_state()

                if self.gui.button("Next Core"):
                    self.save_state()
                    self.func_idx = (self.func_idx + 1) % len(self.functions)
                    self.load_state()

                self.gui.text("")
                self.gui.text(f"Colormap: {self.colormaps[self.colormap_idx]}")
                if self.gui.button("Toggle Colormap"):
                    self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
                    self.save_state()

                self.gui.text("")
                
                if config.SUPPORT_F64_BASE:
                    mode_text = "Native" if config.SUPPORT_F64_TRIG else "Emulated"
                    self.use_f64 = self.gui.checkbox(f"64-bit Precision ({mode_text})", self.use_f64)
                else:
                    self.gui.text("64-bit: HARDWARE UNAVAILABLE")
                    self.use_f64 = False

                if self.func_idx in [0,1,7]:
                    self.max_iter = self.gui.slider_int("Max Iterations", self.max_iter, 10, 1000)

                self.color_freq = self.gui.slider_float("Color Scale", self.color_freq, 0.001, 1.0)

                if self.gui.button("Reset View"):
                    self.states[self.func_idx] = CameraState(self.functions[self.func_idx])
                    self.load_state()
                    for vp in self.viewports:
                        vp.zoom = 200.0
                    
                    self.vel_x = 0.0
                    self.vel_y = 0.0
                    self.momentum_mult = 1.0

            if self.active_dims > 0:
                with self.gui.sub_window("N-D Translations", 0.02, 0.50, 0.25, 0.24):
                    for i in range(self.active_dims):
                        name = ["X", "Y", "Z", "W", "V", "U"][i] if i < 6 else f"D{i}"
                        self.translations[i] = self.smart_slider(f"Pos {name}", self.translations[i], -2.0, 2.0)

            if len(self.planes) > 0:
                with self.gui.sub_window("N-D Rotations", 0.02, 0.75, 0.25, 0.22):
                    for i, (ax1, ax2) in enumerate(self.planes):
                        name1 = DIM_NAMES[ax1]
                        name2 = DIM_NAMES[ax2]
                        
                        self.rotations[i] = self.smart_slider(
                            f"Rot {name1}{name2}", 
                            self.rotations[i], 
                            -180.0, 
                            180.0
                        )

            if self.active_dims >= 2:
                num_mini_vps = (self.active_dims // 2) - 1
                ui_height = 0.04 + 0.08 * num_mini_vps
                
                with self.gui.sub_window("Viewports", 0.02, 0.98, 0.25, ui_height):
                    main_vp = self.viewports[0]
                    m_name1 = DIM_NAMES[main_vp.dim1]
                    m_name2 = DIM_NAMES[main_vp.dim2]
                    self.gui.text(f"Main [{m_name1}-{m_name2}] Zoom: {main_vp.zoom:.1e}")
                    
                    for vp in self.viewports[1:]:
                        if vp.dim2 < self.active_dims:
                            name1 = DIM_NAMES[vp.dim1]
                            name2 = DIM_NAMES[vp.dim2]
                            
                            self.gui.text("")
                            
                            vp.visible = self.gui.checkbox(f"Show [{name1}-{name2}] Plane", vp.visible)
                            
                            # The Swap Button
                            if self.gui.button(f"  Swap [{name1}-{name2}] to Main"):
                                main_vp.dim1, vp.dim1 = vp.dim1, main_vp.dim1
                                main_vp.dim2, vp.dim2 = vp.dim2, main_vp.dim2
                                main_vp.zoom, vp.zoom = vp.zoom, main_vp.zoom
                            
                            self.gui.text(f"  -> Zoom: {vp.zoom:.1e}")

            self.window.show()
    
    def smart_slider(self, label, value, min_val, max_val):
        new_val = self.gui.slider_float(label, value, min_val, max_val)
        if abs(new_val - value) > 1e-5:
            return new_val
        return value
    
    def save_state(self):
        s = self.states[self.func_idx]
        s.zooms = [vp.zoom for vp in self.viewports]
        s.vp_axes = [(vp.dim1, vp.dim2) for vp in self.viewports]
        s.translations = self.translations.copy()
        s.rotations = self.rotations.copy()
        s.max_iter = self.max_iter
        s.color_freq = self.color_freq
        s.use_f64 = self.use_f64
        s.active_dims = self.active_dims
        s.colormap_idx = self.colormap_idx

    def load_state(self):
        s = self.states[self.func_idx]
        for vp, z in zip(self.viewports, s.zooms):
            vp.zoom = z
            
        for vp, axes in zip(self.viewports, s.vp_axes):
            vp.dim1 = axes[0]
            vp.dim2 = axes[1]
        
        self.translations = s.translations.copy()
        self.rotations = s.rotations.copy()
        self.max_iter = s.max_iter
        self.color_freq = s.color_freq
        self.active_dims = s.active_dims
        self.planes = s.planes
        self.use_f64 = s.use_f64
        self.colormap_idx = s.colormap_idx

    def get_nd_basis_matrix(self):
        basis = [[0.0] * config.MAX_DIMENSIONS for _ in range(config.MAX_DIMENSIONS)]
        
        for i in range(config.MAX_DIMENSIONS):
            basis[i][i] = 1.0

        for i, (ax1, ax2) in enumerate(self.planes):
            angle_deg = self.rotations[i]
            
            if angle_deg != 0.0:
                angle_rad = math.radians(angle_deg)
                c = math.cos(angle_rad)
                s = math.sin(angle_rad)

                for j in range(config.MAX_DIMENSIONS):
                    v1 = basis[j][ax1] * c - basis[j][ax2] * s
                    v2 = basis[j][ax1] * s + basis[j][ax2] * c
                    basis[j][ax1] = v1
                    basis[j][ax2] = v2

        return basis

    def handle_input(self, width, height, basis, dt):
        # Cap dt to 0.1s so a lag spike doesn't teleport the camera
        dt = min(dt, 0.1)
        
        dt_scale = dt * 60.0 # set baseline to 60

        mouse_x, mouse_y = self.window.get_cursor_pos()
        
        mx_px = mouse_x * width
        my_px = mouse_y * height

        # Hit Detection
        if not self.is_dragging:
            self.hovered_vp_idx = 0
            for vp in reversed(self.viewports[1:]):
                if vp.dim2 < self.active_dims and vp.visible and vp.contains_mouse(mx_px, my_px):
                    self.hovered_vp_idx = vp.idx
                    break

        active_vp = self.viewports[self.hovered_vp_idx]
        right = basis[active_vp.dim1]
        up = basis[active_vp.dim2]

        base_accel = (2.0 / active_vp.zoom) * dt_scale
        
        friction = 0.85 ** dt_scale
        is_moving = False

        if self.window.is_pressed(ti.ui.SHIFT):
            base_accel *= 5.0

        if self.window.is_pressed('a'):
            self.vel_x -= base_accel
            is_moving = True
        if self.window.is_pressed('d'):
            self.vel_x += base_accel
            is_moving = True
        if self.window.is_pressed('w'):
            self.vel_y += base_accel
            is_moving = True
        if self.window.is_pressed('s'):
            self.vel_y -= base_accel
            is_moving = True

        if is_moving:
            self.momentum_mult = min(self.momentum_mult + (0.02 * dt_scale), 2.5)
        else:
            self.momentum_mult = 1.0

        self.vel_x *= friction
        self.vel_y *= friction

        total_dx = self.vel_x * self.momentum_mult * dt_scale
        total_dy = self.vel_y * self.momentum_mult * dt_scale

        if self.window.is_pressed(ti.ui.LMB):
            if self.is_dragging:
                total_dx -= (mouse_x - self.last_mouse_x) * width / active_vp.zoom
                total_dy -= (mouse_y - self.last_mouse_y) * height / active_vp.zoom
            self.is_dragging = True
        else:
            self.is_dragging = False

        self.last_mouse_x = mouse_x
        self.last_mouse_y = mouse_y

        if total_dx != 0.0 or total_dy != 0.0:
            for i in range(self.active_dims):
                self.translations[i] += total_dx * right[i] + total_dy * up[i]

        zoom_in = self.window.is_pressed('e')
        zoom_out = self.window.is_pressed('q')

        if zoom_in or zoom_out:
            if active_vp.idx > 0:
                screen_x = mx_px - (active_vp.px_x + active_vp.px_w * 0.5)
                screen_y = my_px - (active_vp.px_y + active_vp.px_h * 0.5)
            else:
                screen_x = mx_px - (width * 0.5)
                screen_y = my_px - (height * 0.5)

            math_x_before = screen_x / active_vp.zoom
            math_y_before = screen_y / active_vp.zoom

            zoom_speed = 1.05 ** dt_scale
            
            if zoom_in:
                active_vp.zoom *= zoom_speed
            if zoom_out:
                active_vp.zoom /= zoom_speed

            math_x_after = screen_x / active_vp.zoom
            math_y_after = screen_y / active_vp.zoom

            for i in range(self.active_dims):
                self.translations[i] += (math_x_before - math_x_after) * right[i]
                self.translations[i] += (math_y_before - math_y_after) * up[i]


if __name__ == "__main__":
    app = App()
    if hasattr(app, 'window'): # Make sure it initialized properly before running
        app.run()