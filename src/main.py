import math
import taichi as ti
import taichi.math as tm
import numpy as np
import os
import time
import collections

DIMENSIONS = 2
MIN_DIMENSIONS = 2
MAX_DIMENSIONS = 2

ti.init(arch=ti.vulkan)

class App:
    def __init__(self, width, height):
        self.initialized = False
        
        if DIMENSIONS < MIN_DIMENSIONS or DIMENSIONS > MAX_DIMENSIONS:
            print("Dimensions out of range!")
            return

        self.rot_planes = math.comb(DIMENSIONS, 2)
        
        # 2. Create the Taichi window (vsync is supported natively)
        self.window = ti.ui.Window("N-Dimensional Viewer", (width, height), vsync=True)
        
        # 3. Get the Canvas (where your pixels/geometry are drawn)
        self.canvas = self.window.get_canvas()
        
        # 4. Get the GUI (Taichi has Dear ImGui built right in!)
        self.gui = self.window.get_gui()
        
        self.initialized = True

    def run(self):
        while self.window.running:
            
            self.canvas.set_background_color((0.1, 0.1, 0.15))
            
            with self.gui.sub_window("N-D Controls", 0.05, 0.05, 0.3, 0.2):
                self.gui.text(f"Dimensions: {DIMENSIONS}")
                self.gui.text(f"Rotation Planes: {self.rot_planes}")
                
                # Example of an ImGui button
                if self.gui.button("Reset View"):
                    print("View Reset!")

            # --- SWAP BUFFERS ---
            self.window.show()

# How to start it
if __name__ == "__main__":
    app = App(1280, 720)
    if getattr(app, 'initialized', False):
        app.run()