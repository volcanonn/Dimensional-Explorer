import taichi as ti
import time
import collections

# 1. Initialize Vulkan for best performance on Windows/CachyOS
ti.init(arch=ti.vulkan)

# Define our resolution (Original used n=320, so 640x320)
HEIGHT = 320
WIDTH = HEIGHT * 2

# 2. Modern Canvas requires an RGB vector field (n=3) instead of a scalar field
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])

@ti.kernel
def paint(t: float):
    # This loop automatically parallelizes over all pixels on the GPU!
    for i, j in pixels:  
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / HEIGHT - 1, j / HEIGHT - 0.5]) * 2
        
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
            
        # 3. Convert the scalar iteration value to an RGB color (grayscale)
        color = 1.0 - iterations * 0.02
        pixels[i, j] = ti.Vector([color, color, color])


class App:
    def __init__(self):
        # 4. Use the modern ui.Window instead of the legacy GUI
        self.window = ti.ui.Window("Taichi Julia Set", (WIDTH, HEIGHT))
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        
        # FPS Tracking setup
        self.frame_times = collections.deque(maxlen=60)
        self.last_time = time.perf_counter() 
        
        # Animation time state
        self.t = 0.0
        self.auto_play = True

    def run(self):
        while self.window.running:
            # --- FPS Tracking ---
            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            self.frame_times.append(dt)
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

            # --- Compute Physics/Math on GPU ---
            paint(self.t)
            
            if self.auto_play:
                self.t += 0.03

            # --- Render Image to Screen ---
            self.canvas.set_image(pixels)

            # --- Draw ImGui UI Over the Screen ---
            with self.gui.sub_window("Controls", 0.02, 0.02, 0.35, 0.3):
                self.gui.text(f"FPS: {fps:.1f}")
                self.gui.text(f"Frame time: {avg_dt * 1000:.2f} ms")
                
                # We can add an ImGui checkbox to pause the animation!
                self.auto_play = self.gui.checkbox("Auto Play", self.auto_play)
                
                # And a slider to manually scrub through time!
                self.t = self.gui.slider_float("Time (t)", self.t, 0.0, 10.0)

            # --- Swap Buffers ---
            self.window.show()


if __name__ == "__main__":
    app = App()
    app.run()