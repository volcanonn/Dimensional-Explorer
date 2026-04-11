import math
import taichi as ti
import time
import collections

import config
import utils

ti.init(arch=ti.vulkan, default_fp=ti.f32)

class CameraState:
    def __init__(self, name):
        self.zoom = 200.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.translations = [0.0] * config.MAX_DIMENSIONS
        self.max_iter = 100
        self.color_freq = 0.05

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
        self.rotations = {plane: 0.0 for plane in self.planes}

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
        self.pixels = None

        self.vel_x = 0.0
        self.vel_y = 0.0
        self.is_dragging = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        self.functions =["Mandelbrot", "Conic Sections", "Voronoi", "Simple Wave", "Radial Wave", "Paraboloid"]
        self.func_idx = 0
        self.states = [CameraState(name) for name in self.functions]
        self.load_state()

    def run(self):
        while self.window.running:
            self.window.get_events()
            
            actual_shape = self.window.get_window_shape()
            if actual_shape != self.current_shape:
                self.current_shape = actual_shape
                width, height = self.current_shape
                self.pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(width, height))

            width, height = self.current_shape
            right, up = self.get_nd_camera_vectors()

            self.handle_camera(width, height, right, up)


            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            self.frame_times.append(dt)
            
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

            ti.sync() 
            starttime = time.perf_counter_ns()
            
            origin_vec = config.vecMAX(self.translations)
            right_vec = config.vecMAX(right)
            up_vec = config.vecMAX(up)
            
            utils.nd_slice(
                self.pixels,
                origin_vec,
                right_vec,
                up_vec,
                self.zoom,
                self.max_iter,
                self.color_freq,
                self.func_idx,
                self.use_f64,
                self.active_dims
            )

            ti.sync()
            endtime = time.perf_counter_ns() - starttime

            self.calc_times.append(endtime)
            avg_calc_time = sum(self.calc_times) / len(self.calc_times)


            self.canvas.set_image(self.pixels)

            # --- UI RENDERING ---
            with self.gui.sub_window("Engine Stats", 0.02, 0.02, 0.3, 0.38):
                self.gui.text(f"Dimensions: {self.active_dims}D")
                self.gui.text(f"FPS: {fps:.1f}")
                self.gui.text(f"GPU Calc: {utils.format_time(avg_calc_time)}")
                self.gui.text(f"Zoom:  {self.zoom:.2e}")

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
                self.use_f64 = self.gui.checkbox("64-bit Precision", self.use_f64)

                if self.func_idx == 0:
                    self.max_iter = self.gui.slider_int("Max Iterations", self.max_iter, 10, 1000)

                self.color_freq = self.gui.slider_float("Color Scale", self.color_freq, 0.001, 1.0)

                if self.gui.button("Reset View"):
                    self.states[self.func_idx] = CameraState(self.functions[self.func_idx])
                    self.load_state()

            if self.active_dims > 0:
                with self.gui.sub_window("N-D Translations", 0.02, 0.44, 0.25, 0.20):
                    for i in range(self.active_dims):
                        name =["X", "Y", "Z", "W", "V", "U"][i] if i < 6 else f"D{i}"
                        self.translations[i] = self.smart_slider(f"Pos {name}", self.translations[i], -2.0, 2.0)

            if len(self.planes) > 0:
                with self.gui.sub_window("N-D Rotations", 0.02, 0.66, 0.25, 0.32):
                    for ax1, ax2 in self.planes:
                        name1 =["X", "Y", "Z", "W", "V", "U"][ax1] if ax1 < 6 else f"D{ax1}"
                        name2 = ["X", "Y", "Z", "W", "V", "U"][ax2] if ax2 < 6 else f"D{ax2}"
                        
                        # Use Degrees for the UI (-180 to 180)
                        self.rotations[(ax1, ax2)] = self.smart_slider(
                            f"Rot {name1}{name2}", 
                            self.rotations[(ax1, ax2)], 
                            -180.0, 
                            180.0
                        )

            self.window.show()
    
    def smart_slider(self, label, value, min_val, max_val):
        """Allows you to use UI sliders/text inputs without destroying 64-bit precision!"""
        new_val = self.gui.slider_float(label, value, min_val, max_val)
        if abs(new_val - value) > 1e-5:
            return new_val
        return value
    
    def save_state(self):
        s = self.states[self.func_idx]
        s.zoom = self.zoom
        s.pan_x = self.pan_x
        s.pan_y = self.pan_y
        s.translations = self.translations.copy()
        s.rotations = self.rotations.copy()
        s.max_iter = self.max_iter
        s.color_freq = self.color_freq
        s.use_f64 = self.use_f64

    def load_state(self):
        s = self.states[self.func_idx]
        self.zoom = s.zoom
        self.pan_x = s.pan_x
        self.pan_y = s.pan_y
        self.translations = s.translations.copy()
        self.rotations = s.rotations.copy()
        self.max_iter = s.max_iter
        self.color_freq = s.color_freq
        self.active_dims = s.active_dims
        self.planes = s.planes
        self.use_f64 = s.use_f64

    def get_nd_camera_vectors(self):
        right =[0.0] * config.MAX_DIMENSIONS
        right[0] = 1.0

        up = [0.0] * config.MAX_DIMENSIONS
        up[1] = 1.0

        for ax1, ax2 in self.planes:
            angle_deg = self.rotations[(ax1, ax2)]
            
            if angle_deg != 0.0:
                # Convert the slider's Degrees into Radians for the math engine
                angle_rad = math.radians(angle_deg)
                c = math.cos(angle_rad)
                s = math.sin(angle_rad)

                r1 = right[ax1] * c - right[ax2] * s
                r2 = right[ax1] * s + right[ax2] * c
                right[ax1] = r1
                right[ax2] = r2

                u1 = up[ax1] * c - up[ax2] * s
                u2 = up[ax1] * s + up[ax2] * c
                up[ax1] = u1
                up[ax2] = u2

        return right, up

    def handle_camera(self, width, height, right, up):
        accel = 2.0 / self.zoom
        friction = 0.85

        if self.window.is_pressed('a'):
            self.vel_x -= accel
        if self.window.is_pressed('d'):
            self.vel_x += accel
        if self.window.is_pressed('w'):
            self.vel_y += accel
        if self.window.is_pressed('s'):
            self.vel_y -= accel

        self.vel_x *= friction
        self.vel_y *= friction

        mouse_x, mouse_y = self.window.get_cursor_pos()
        total_dx = self.vel_x
        total_dy = self.vel_y

        if self.window.is_pressed(ti.ui.LMB):
            if self.is_dragging:
                total_dx -= (mouse_x - self.last_mouse_x) * width / self.zoom
                total_dy -= (mouse_y - self.last_mouse_y) * height / self.zoom
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
            screen_x = (mouse_x - 0.5) * width
            screen_y = (mouse_y - 0.5) * height

            math_x_before = screen_x / self.zoom
            math_y_before = screen_y / self.zoom

            zoom_speed = 1.05

            if zoom_in:
                self.zoom *= zoom_speed
            if zoom_out:
                self.zoom /= zoom_speed

            math_x_after = screen_x / self.zoom
            math_y_after = screen_y / self.zoom

            for i in range(self.active_dims):
                self.translations[i] += (math_x_before - math_x_after) * right[i]
                self.translations[i] += (math_y_before - math_y_after) * up[i]


if __name__ == "__main__":
    app = App()
    if hasattr(app, 'window'): # Make sure it initialized properly before running
        app.run()