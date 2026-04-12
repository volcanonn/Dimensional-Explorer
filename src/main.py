import math
import taichi as ti
import time
import collections

import config
import utils

ti.init(arch=ti.gpu, default_fp=ti.f32)

DIM_NAMES =[]
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

        self.px_w = 256 if idx > 0 else 0
        self.px_h = 256 if idx > 0 else 0
        self.px_x = 0
        self.px_y = 0

        if idx > 0:
            self.pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.px_w, self.px_h))
        else:
            self.pixels = None

    def contains_mouse(self, mx_px, my_px):
        in_x = self.px_x <= mx_px <= (self.px_x + self.px_w)
        in_y = self.px_y <= my_px <= (self.px_y + self.px_h)
        return in_x and in_y


class CameraState:
    def __init__(self, name):
        self.zooms =[200.0] * (config.MAX_DIMENSIONS // 2)
        
        self.translations =[0.0] * config.MAX_DIMENSIONS
        self.max_iter = 100
        self.color_freq = 0.05
        self.colormap_idx = 0

        if name == "Mandelbrot":
            self.active_dims = 6
            self.use_f64 = False
            self.translations[4] = 2.0
        elif name == "Conic Sections":
            self.active_dims = 3
            self.use_f64 = False
            self.color_freq = 0.5
        else:
            self.active_dims = 2
            self.use_f64 = False
            self.color_freq = 0.1

        self.planes =[(i, j) for i in range(self.active_dims) for j in range(i + 1, self.active_dims)]
        self.rotations =[0.0] * len(self.planes)

@ti.data_oriented
class App:
    def __init__(self):
        self.window = ti.ui.Window("Dynamic N-D Viewer", (1280, 720), vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        self.frame_times = collections.deque(maxlen=60)
        self.calc_times = collections.deque(maxlen=60)
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
        self.functions =[
            "Mandelbrot",
            "Conic Sections",
            "Voronoi",
            "Simple Wave",
            "Radial Wave",
            "Paraboloid"
        ]

        self.func_idx = 0

        self.colormaps = ["Heledron", "Psychedelic"]
        self.colormap_idx = 0

        self.states = [CameraState(name) for name in self.functions]
        
        self.viewports =[]
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
            for i, vp in enumerate(self.viewports[1:]):
                vp.px_x = width - vp.px_w - padding
                vp.px_y = height - padding - (i + 1) * vp.px_h - (i * padding)

            # Generate the Orthogonal N-D Transformation Matrix
            basis = self.get_nd_basis_matrix()
            self.handle_input(width, height, basis)


            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            self.frame_times.append(dt)
            
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

            ti.sync() 
            starttime = time.perf_counter_ns()
            
            origin_vec = config.vecMAX(self.translations)

            for vp in self.viewports:
                if vp.dim2 < self.active_dims:
                    right_vec = config.vecMAX(basis[vp.dim1])
                    up_vec = config.vecMAX(basis[vp.dim2])

                    utils.nd_slice(
                        vp.pixels,
                        origin_vec,
                        right_vec,
                        up_vec,
                        vp.zoom,
                        self.max_iter,
                        self.color_freq,
                        self.func_idx,
                        self.use_f64,
                        self.active_dims,
                        self.colormap_idx
                    )

            ti.sync()
            
            # --- 3. PICTURE-IN-PICTURE (PiP) BLIT ---
            for vp in self.viewports[1:]:
                if vp.dim2 < self.active_dims:
                    utils.blit_image(self.viewports[0].pixels, vp.pixels, vp.px_x, vp.px_y)

            self.calc_times.append(time.perf_counter_ns() - starttime)
            avg_calc_time = sum(self.calc_times) / len(self.calc_times)

            self.canvas.set_image(self.viewports[0].pixels)

            with self.gui.sub_window("Engine Stats", 0.02, 0.02, 0.3, 0.38):
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
                self.use_f64 = self.gui.checkbox("64-bit Precision", self.use_f64)

                if self.func_idx == 0:
                    self.max_iter = self.gui.slider_int("Max Iterations", self.max_iter, 10, 1000)

                self.color_freq = self.gui.slider_float("Color Scale", self.color_freq, 0.001, 1.0)

                if self.gui.button("Reset View"):
                    self.states[self.func_idx] = CameraState(self.functions[self.func_idx])
                    self.load_state()
                    for vp in self.viewports:
                        vp.zoom = 200.0

            if self.active_dims > 0:
                with self.gui.sub_window("N-D Translations", 0.02, 0.42, 0.25, 0.23):
                    for i in range(self.active_dims):
                        name = ["X", "Y", "Z", "W", "V", "U"][i] if i < 6 else f"D{i}"
                        self.translations[i] = self.smart_slider(f"Pos {name}", self.translations[i], -2.0, 2.0)

            if len(self.planes) > 0:
                with self.gui.sub_window("N-D Rotations", 0.02, 0.67, 0.25, 0.22):
                    for i, (ax1, ax2) in enumerate(self.planes):
                        name1 = DIM_NAMES[ax1]
                        name2 = DIM_NAMES[ax2]
                        
                        self.rotations[i] = self.smart_slider(
                            f"Rot {name1}{name2}", 
                            self.rotations[i], 
                            -180.0, 
                            180.0
                        )

            if self.active_dims > 2:
                with self.gui.sub_window("Viewports Zoom", 0.02, 0.91, 0.25, 0.07):
                    for vp in self.viewports:
                        if vp.dim2 < self.active_dims:
                            name1 = DIM_NAMES[vp.dim1]
                            name2 = DIM_NAMES[vp.dim2]
                            self.gui.text(f"{name1}-{name2} Plane: {vp.zoom:.1e}")

            self.window.show()
    
    def smart_slider(self, label, value, min_val, max_val):
        """Allows you to use UI sliders/text inputs without destroying 64-bit precision!"""
        new_val = self.gui.slider_float(label, value, min_val, max_val)
        if abs(new_val - value) > 1e-5:
            return new_val
        return value
    
    def save_state(self):
        s = self.states[self.func_idx]
        s.zooms =[vp.zoom for vp in self.viewports]
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

    def handle_input(self, width, height, basis):
        mouse_x, mouse_y = self.window.get_cursor_pos()
        
        mx_px = mouse_x * width
        my_px = mouse_y * height

        # Hit Detection
        if not self.is_dragging:
            self.hovered_vp_idx = 0
            for vp in reversed(self.viewports[1:]):
                if vp.dim2 < self.active_dims and vp.contains_mouse(mx_px, my_px):
                    self.hovered_vp_idx = vp.idx
                    break

        active_vp = self.viewports[self.hovered_vp_idx]
        right = basis[active_vp.dim1]
        up = basis[active_vp.dim2]

        base_accel = 2.0 / active_vp.zoom
        friction = 0.85
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
            self.momentum_mult = min(self.momentum_mult + 0.02, 2.5)
        else:
            self.momentum_mult = 1.0

        self.vel_x *= friction
        self.vel_y *= friction

        total_dx = self.vel_x * self.momentum_mult
        total_dy = self.vel_y * self.momentum_mult

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

            zoom_speed = 1.05
            
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