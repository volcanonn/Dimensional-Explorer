import math
import taichi as ti
import taichi.math as tm
import numpy as np
import os
import time
import utils
import funcs

DIMENSIONS = 2
MIN_DIMENSIONS = 2
MAX_DIMENSIONS = 2

ti.init(arch=ti.vulkan)

@ti.data_oriented
class App:
    def __init__(self):
        if DIMENSIONS < MIN_DIMENSIONS or DIMENSIONS > MAX_DIMENSIONS:
            print("Dimensions out of range!")
            return

        self.rot_planes = math.comb(DIMENSIONS, 2)
        
        self.window = ti.ui.Window("N-Dimensional Viewer", (800, 600), vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        self.current_shape = (0, 0)
        self.pixels = None

        self.zoom = 200.0  
        self.pan_x = 0.0
        self.pan_y = 0.0
    
    def run(self):
        while self.window.running:
            actual_shape = self.window.get_window_shape()
            
            if actual_shape != self.current_shape:
                self.current_shape = actual_shape
                width, height = self.current_shape
                
                self.pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(width, height))

            funcs.mandelbrot(self.pixels, self.zoom, self.pan_x, self.pan_y)

            self.canvas.set_image(self.pixels)

            with self.gui.sub_window("N-D Controls", 0.05, 0.05, 0.3, 0.2):
                self.gui.text(f"Dimensions: {DIMENSIONS}")
                self.gui.text(f"Rotation Planes: {self.rot_planes}")

                self.zoom = self.gui.slider_float("Zoom (Pixels/Unit)", self.zoom, 50.0, 1000.0)
                self.pan_x = self.gui.slider_float("Pan X", self.pan_x, -2.0, 2.0)
                self.pan_y = self.gui.slider_float("Pan Y", self.pan_y, -2.0, 2.0)
                
                if self.gui.button("Reset View"):
                    print("View Reset!")

            self.window.show()

if __name__ == "__main__":
    app = App()
    app.run()