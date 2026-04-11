import math
import taichi as ti
import taichi.math as tm
import numpy as np
import os
import time
import utils
import funcs
import collections

DIMENSIONS = 2
MIN_DIMENSIONS = 2
MAX_DIMENSIONS = 2

USE_F64 = False

MATH_TYPE = ti.f64 if USE_F64 else ti.f32

ti.init(arch=ti.vulkan, default_fp=MATH_TYPE)

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

        self.max_iter = 100
        self.color_freq = 0.05

        self.power = 2.0

        # --- CAMERA STATE ---
        self.zoom = 200.0  
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Momentum variables for WASD
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        # Mouse dragging state
        self.is_dragging = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

    def run(self):
        while self.window.running:
            actual_shape = self.window.get_window_shape()
            
            if actual_shape != self.current_shape:
                self.current_shape = actual_shape
                width, height = self.current_shape
                
                self.pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(width, height))

            self.handle_camera(width, height)
            
            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            self.frame_times.append(dt)
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

            
            ti.sync()
            starttime = time.perf_counter_ns()

            funcs.mandelbrot(
                self.pixels, self.zoom, self.pan_x, self.pan_y, 
                self.max_iter, self.color_freq, self.power
            )

            ti.sync()
            endtime = time.perf_counter_ns() - starttime

            self.calc_times.append(endtime)
            avg_calc_time = sum(self.calc_times) / len(self.calc_times)


            self.canvas.set_image(self.pixels)

            with self.gui.sub_window("Engine Stats", 0.05, 0.05, 0.3, 0.2):
                self.gui.text(f"Dimensions: {DIMENSIONS}")
                self.gui.text(f"Rotation Planes: {self.rot_planes}")

                self.gui.text(f"FPS: {fps:.1f}")
                self.gui.text(f"Calc time: {utils.format_time(avg_calc_time)}")

                mode_text = "64-bit (Deep Zoom)" if USE_F64 else "32-bit (Max Speed)"
                self.gui.text(f"Precision: {mode_text}")

                self.power = self.gui.slider_float("Math Power", self.power, 1.0, 6.0)

                self.gui.text(f"Zoom:  {self.zoom:.1f}")
                self.gui.text(f"Pan X: {self.pan_x:.15f}")
                self.gui.text(f"Pan Y: {self.pan_y:.15f}")

                self.max_iter = self.gui.slider_int("Max Iterations", self.max_iter, 10, 1000)
                self.color_freq = self.gui.slider_float("Color Freq", self.color_freq, 0.001, 0.5)
                
                if self.gui.button("Reset View"):
                    self.zoom = 200
                    self.pan_x = 0
                    self.pan_y = 0
                    print("View Reset!")

            self.window.show()
    
    def handle_camera(self, width, height):
        # ==========================================
        # 1. WASD MOMENTUM MOVEMENT
        # ==========================================
        # Acceleration scales perfectly with your zoom level
        accel = 2.0 / self.zoom  
        friction = 0.85  

        if self.window.is_pressed('a'): self.vel_x -= accel
        if self.window.is_pressed('d'): self.vel_x += accel
        if self.window.is_pressed('w'): self.vel_y += accel
        if self.window.is_pressed('s'): self.vel_y -= accel

        self.vel_x *= friction
        self.vel_y *= friction
        
        self.pan_x += self.vel_x
        self.pan_y += self.vel_y

        # ==========================================
        # 2. MOUSE DRAG PANNING
        # ==========================================
        mouse_x, mouse_y = self.window.get_cursor_pos()
        
        if self.window.is_pressed(ti.ui.LMB):
            if self.is_dragging:
                dx = (mouse_x - self.last_mouse_x) * width
                dy = (mouse_y - self.last_mouse_y) * height
                
                self.pan_x -= dx / self.zoom
                self.pan_y -= dy / self.zoom
            self.is_dragging = True
        else:
            self.is_dragging = False
            
        self.last_mouse_x = mouse_x
        self.last_mouse_y = mouse_y

        # ==========================================
        # 3. TARGETED ZOOM (Q to zoom out, E to zoom in)
        # ==========================================
        zoom_in = self.window.is_pressed('e')
        zoom_out = self.window.is_pressed('q')
        
        if zoom_in or zoom_out:
            # A. Find mouse position relative to the center of the screen
            screen_x = (mouse_x - 0.5) * width
            screen_y = (mouse_y - 0.5) * height
            
            # B. Find the exact math coordinate under the mouse BEFORE zooming
            math_x = (screen_x / self.zoom) + self.pan_x
            math_y = (screen_y / self.zoom) + self.pan_y
            
            # C. Apply the Zoom
            zoom_speed = 1.05
            if zoom_in:
                self.zoom *= zoom_speed
            if zoom_out:
                self.zoom /= zoom_speed
                
            # D. Adjust the pan so that the math coordinate stays EXACTLY under the mouse
            self.pan_x = math_x - (screen_x / self.zoom)
            self.pan_y = math_y - (screen_y / self.zoom)


if __name__ == "__main__":
    app = App()
    app.run()