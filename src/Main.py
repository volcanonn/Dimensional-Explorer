import pyglet
from pyglet.window import key
import imgui
from imgui.integrations.pyglet import create_renderer
import zengl
import struct
import numpy as np
import os
import time
from Utils import *
import collections

class App(pyglet.window.Window):
    def __init__(self, width, height):

        self.initialized = False

        self.set_minimum_size(120,int(120*9/16))

        super().__init__(width, height, vsync=True, resizable=True)

        imgui.create_context()
        self.impl = create_renderer(self)

        self.ctx = zengl.context()

        self.zoom = 3.0
        self.pan_x = -0.75
        self.pan_y = 0.0
        self.max_iterations = 200
        self.color_frequency = 0.04

        self.target_fps = 60

        self.frame_times = collections.deque(maxlen=60)
        self.fps = 0.0

        self.vertex_buffer = self.ctx.buffer(np.array([
            # x,    y
            -1.0, -1.0,  # Bottom-left
             1.0, -1.0,  # Bottom-right
            -1.0,  1.0,  # Top-left
             1.0,  1.0,  # Top-right
        ], dtype='f4'))

        self.includes = {}

        for file in os.listdir("Shaders"):
            with open("Shaders\\" + file, "r") as f:
                if file == "vertex.glsl":
                    self.vertex_shader = f.read()
                elif file == "fragment.glsl":
                    self.fragment_shader = f.read()
                else:
                    self.includes.update({file:f.read()})
        
        self.uniform_buffer = self.ctx.buffer(size=32, uniform=True)

        self.on_resize(width, height)

        pyglet.clock.schedule_interval(self.update, 1.0 / self.target_fps)

        self.initialized = True

    def create_pipeline(self):
        self.pipeline = self.ctx.pipeline(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader,
            includes=self.includes,

            framebuffer=[self.image],
            layout=[
                {'name': 'Common', 'binding': 0},
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': self.uniform_buffer,
                },
            ],
            vertex_buffers=zengl.bind(self.vertex_buffer, '2f', 0),
            topology='triangle_strip',
            vertex_count=4,

        )

    def on_resize(self, width, height):
        """Pyglet's built-in event handler for window resizing."""
        # This is crucial for making the shader aware of the new window size.
        super().on_resize(width, height) # Let the parent class handle its part

        # Recreate the offscreen image with the new dimensions.
        self.image = self.ctx.image((width, height), 'rgba8unorm', samples=4)
        
        # Recreate the pipeline to link it to our new image.
        self.create_pipeline()

        # IMPORTANT: After recreating everything, we must flag the fractal as
        # dirty to force on_draw to render into our new resources.
        self.UpdateImage = True

    def on_draw(self):

        if not self.initialized:
            return
        
        self.ctx.new_frame()
        # --- Conditional Fractal Rendering ---
        if self.UpdateImage:
            uniform_data = struct.pack(
                '2f f 4x 2f i f',  # Format: vec2, float, PADDING, vec2, int, float
                self.width, self.height,
                self.zoom,
                self.pan_x, self.pan_y,
                self.max_iterations,
                self.color_frequency
            )
            self.uniform_buffer.write(uniform_data)

            # Re-render the fractal to our (already correctly-sized) offscreen image
            self.image.clear()
            self.pipeline.render()
            # Reset the flag
            self.UpdateImage = False

        # --- Unconditional Presentation ---
        
        # Blit our persistent fractal image to the screen, overwriting everything.
        self.image.blit()
        self.ctx.end_frame()

        imgui.new_frame()
        imgui.begin("Fractal Controls")
        
        # ... (slider code that sets self.fractal_dirty = True) ...
        c1, self.zoom = imgui.slider_float("Zoom", self.zoom, 0.001, 10.0)
        c2, self.pan_x = imgui.slider_float("Pan X", self.pan_x, -2.0, 1.0)
        c3, self.pan_y = imgui.slider_float("Pan Y", self.pan_y, -1.5, 1.5)
        c4, self.max_iterations = imgui.slider_int("Iterations", self.max_iterations, 50, 10000)
        c5, self.color_frequency = imgui.slider_float("Color Freq", self.color_frequency, 0.01, 0.2)
        imgui.separator()
        imgui.text(f"FPS: {self.fps:.1f}")
        if c1 or c2 or c3 or c4 or c5:
            self.UpdateImage = True

            
        imgui.end()

        # Render the UI on top.
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        

    def update(self, dt):
        """This function is called by the pyglet clock."""
        
        # Add the current frame time to our list
        self.frame_times.append(dt)

        # Calculate the average frame time
        if self.frame_times:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            
            # Avoid division by zero and calculate FPS
            if avg_dt > 0:
                self.fps = 1.0 / avg_dt

    def screen_to_complex(self, x, y):
        uv_x = x / self.width
        uv_y = y / self.height
        aspect_ratio = self.width / self.height
        corrected_x = (uv_x - 0.5) * aspect_ratio
        corrected_y = uv_y - 0.5
        complex_x = (corrected_x * self.zoom) + self.pan_x
        complex_y = (corrected_y * self.zoom) + self.pan_y
        return complex_x, complex_y

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.is_dragging = True

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.is_dragging = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.is_dragging:
            aspect_ratio = self.width / self.height
            self.pan_x -= dx * (self.zoom * aspect_ratio / self.width)
            self.pan_y -= dy * (self.zoom / self.height)
            self.UpdateImage = True

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        zoom_factor = 1.15
        p_before_zoom = self.screen_to_complex(x, y)
        if scroll_y > 0:
            self.zoom /= zoom_factor
        elif scroll_y < 0:
            self.zoom *= zoom_factor
        p_after_zoom = self.screen_to_complex(x, y)
        self.pan_x += p_before_zoom[0] - p_after_zoom[0]
        self.pan_y += p_before_zoom[1] - p_after_zoom[1]
        self.UpdateImage = True

    def on_close(self):
        self.impl.shutdown()
        super().on_close()

if __name__ == '__main__':
    app = App(1280, 720)
    pyglet.app.run()