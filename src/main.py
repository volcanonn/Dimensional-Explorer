import math
import taichi as ti
import taichi.math as tm
import numpy as np
import os
import time
import utils
import funcs
import collections

DIMENSIONS = 4
MIN_DIMENSIONS = 2
MAX_DIMENSIONS = 10

USE_F64 = False

MATH_TYPE = ti.f64 if USE_F64 else ti.f32

ti.init(arch=ti.vulkan, default_fp=MATH_TYPE)

vecND = ti.types.vector(DIMENSIONS, float)

@ti.data_oriented
class App:
    def __init__(self):
        if DIMENSIONS < MIN_DIMENSIONS or DIMENSIONS > MAX_DIMENSIONS:
            print("Dimensions out of range!")
            return

        self.rot_planes = math.comb(DIMENSIONS, 2)
        
        self.window = ti.ui.Window("N-Dimensional Viewer", (800, 600), vsync=True) # screen size is changed when you use hyprland
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        self.frame_times = collections.deque(maxlen=60)
        self.calc_times = collections.deque(maxlen=60)
        self.last_time = time.perf_counter() 

        self.current_shape = (0, 0)
        self.pixels = None

        self.zoom = 200.0  
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.is_dragging = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        self.translations = [0.0] * DIMENSIONS
        
        self.planes = [(i, j) for i in range(DIMENSIONS) for j in range(i + 1, DIMENSIONS)]
        self.rotations = [0.0] * len(self.planes)

        self.max_iter = 100
        self.color_freq = 0.05

        self.camera_data = ti.Vector.field(n=DIMENSIONS, dtype=float, shape=3)

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
            
            origin_vec = vecND(self.translations)
            right_vec = vecND(right)
            up_vec = vecND(up)
            
            self.camera_data[0] = self.translations
            self.camera_data[1] = right
            self.camera_data[2] = up
            
            funcs.nd_slice(
                self.pixels, 
                self.camera_data,
                self.zoom, self.pan_x, self.pan_y, 
                self.max_iter, self.color_freq
            )

            ti.sync()
            endtime = time.perf_counter_ns() - starttime

            self.calc_times.append(endtime)
            avg_calc_time = sum(self.calc_times) / len(self.calc_times)


            self.canvas.set_image(self.pixels)

            with self.gui.sub_window("Engine Stats", 0.02, 0.02, 0.3, 0.25):
                self.gui.text(f"Dimensions: {DIMENSIONS}D")
                self.gui.text(f"Rotation Planes: {self.rot_planes}")
                self.gui.text(f"FPS: {fps:.1f}")
                self.gui.text(f"Calc time: {utils.format_time(avg_calc_time)}")
                self.gui.text(f"Zoom:  {self.zoom:.2e}")
                
                self.max_iter = self.gui.slider_int("Max Iterations", self.max_iter, 10, 1000)
                self.color_freq = self.gui.slider_float("Color Freq", self.color_freq, 0.001, 0.5)

                mode_text = "64-bit (Deep Zoom)" if USE_F64 else "32-bit (Max Speed)"
                self.gui.text(f"Precision: {mode_text}")
                
                if self.gui.button("Reset View"):
                    self.zoom = 200.0
                    self.translations = [0.0] * DIMENSIONS
                    self.rotations =[0.0] * len(self.planes)

            with self.gui.sub_window("N-D Translations", 0.02, 0.28, 0.25, 0.25):
                self.gui.text("(Ctrl+Click to type exact numbers!)")
                for i in range(DIMENSIONS):
                    name =["X(c)", "Y(c)", "Z(z)", "W(z)", "V(e)", "U(e)"][i] if i < 6 else f"Dim {i}"
                    self.translations[i] = self.smart_slider(f"Pos {name}", self.translations[i], -2.0, 2.0)

            with self.gui.sub_window("N-D Rotations", 0.02, 0.55, 0.25, 0.43):
                for i, (ax1, ax2) in enumerate(self.planes):
                    name1 = ["X","Y","Z","W","V","U"][ax1] if ax1 < 6 else f"D{ax1}"
                    name2 =["X","Y","Z","W","V","U"][ax2] if ax2 < 6 else f"D{ax2}"
                    self.rotations[i] = self.smart_slider(f"Rot {name1}{name2}", self.rotations[i], -3.14, 3.14)

            self.window.show()
    
    def smart_slider(self, label, value, min_val, max_val):
        """Allows you to use UI sliders/text inputs without destroying 64-bit precision!"""
        new_val = self.gui.slider_float(label, value, min_val, max_val)
        if abs(new_val - value) > 1e-5:
            return new_val
        return value
    
    def get_nd_camera_vectors(self):
        right = [0.0] * DIMENSIONS; right[0] = 1.0
        up    = [0.0] * DIMENSIONS; up[1] = 1.0

        for angle, (ax1, ax2) in zip(self.rotations, self.planes):
            if angle != 0.0:
                c = math.cos(angle)
                s = math.sin(angle)
                
                r1 = right[ax1]*c - right[ax2]*s
                r2 = right[ax1]*s + right[ax2]*c
                right[ax1], right[ax2] = r1, r2
                
                u1 = up[ax1]*c - up[ax2]*s
                u2 = up[ax1]*s + up[ax2]*c
                up[ax1], up[ax2] = u1, u2

        return right, up

    def handle_camera(self, width, height, right, up):
        accel = 2.0 / self.zoom  
        friction = 0.85  

        if self.window.is_pressed('a'): self.vel_x -= accel
        if self.window.is_pressed('d'): self.vel_x += accel
        if self.window.is_pressed('w'): self.vel_y += accel
        if self.window.is_pressed('s'): self.vel_y -= accel

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
            for i in range(DIMENSIONS):
                self.translations[i] += total_dx * right[i] + total_dy * up[i]

        zoom_in = self.window.is_pressed('e')
        zoom_out = self.window.is_pressed('q')
        
        if zoom_in or zoom_out:
            screen_x = (mouse_x - 0.5) * width
            screen_y = (mouse_y - 0.5) * height
            
            math_x_before = screen_x / self.zoom
            math_y_before = screen_y / self.zoom
            
            zoom_speed = 1.05
            if zoom_in: self.zoom *= zoom_speed
            if zoom_out: self.zoom /= zoom_speed
                
            math_x_after = screen_x / self.zoom
            math_y_after = screen_y / self.zoom
            
            for i in range(DIMENSIONS):
                self.translations[i] += (math_x_before - math_x_after) * right[i]
                self.translations[i] += (math_y_before - math_y_after) * up[i]


if __name__ == "__main__":
    app = App()
    if hasattr(app, 'window'): # Make sure it initialized properly before running
        app.run()