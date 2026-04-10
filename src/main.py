import math
import taichi as ti
import taichi.math as tm
import numpy as np
import os
import time
import utils

DIMENSIONS = 2
MIN_DIMENSIONS = 2
MAX_DIMENSIONS = 2

ti.init(arch=ti.vulkan)

@ti.data_oriented
class App:
    def __init__(self, width, height):
        self.initialized = False
        
        if DIMENSIONS < MIN_DIMENSIONS or DIMENSIONS > MAX_DIMENSIONS:
            print("Dimensions out of range!")
            return

        self.width = width
        self.height = height

        self.rot_planes = math.comb(DIMENSIONS, 2)
        
        self.window = ti.ui.Window("N-Dimensional Viewer", (self.width, self.height), vsync=True)
        
        self.canvas = self.window.get_canvas()
        
        self.gui = self.window.get_gui()

        self.pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.width, self.height))

        self.t = 0
        
        self.initialized = True
    
    def run(self):
        while self.window.running:
            
            self.paint(self.t)

            self.canvas.set_image(self.pixels)
            
            with self.gui.sub_window("N-D Controls", 0.05, 0.05, 0.3, 0.2):
                self.gui.text(f"Dimensions: {DIMENSIONS}")
                self.gui.text(f"Rotation Planes: {self.rot_planes}")
                self.t = self.gui.slider_float("Time (t)", self.t, 0.0, 10.0)
                
                if self.gui.button("Reset View"):
                    print("View Reset!")

            self.window.show()
    
    @ti.kernel
    def paint(self, t: float):
        # This loop automatically parallelizes over all pixels on the GPU!
        for i, j in self.pixels:  
            c = ti.Vector([-0.8, ti.cos(t) * 0.2])
            z = ti.Vector([i / self.height - 1, j / self.height - 0.5]) * 2
            
            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = utils.complex_sqr(z) + c
                iterations += 1
                
            # 3. Convert the scalar iteration value to an RGB color (grayscale)
            color = 1.0 - iterations * 0.02
            self.pixels[i, j] = ti.Vector([color, color, color])

if __name__ == "__main__":
    app = App(1280, 720)
    if getattr(app, 'initialized', False):
        app.run()