#version 330
precision highp float;

#define PI 3.1415926538

// Define a uniform block. `std140` is a standard memory layout.
// It's a good habit to name the block (e.g., "Common") and its instance (e.g., "common").
layout (std140) uniform Common {
    vec2 u_resolution;
    float u_zoom;
    vec2 u_pan;
    int u_max_iterations;
    float u_color_frequency;
    
};


out vec4 f_color;

#include "utils.glsl"
#include "smoothing.glsl"
#include "mandelbrot.glsl"
#include "colormap.glsl"

void main() {
    // gl_FragCoord is a built-in variable containing the pixel's
    // coordinates (e.g., from 0.0 to 1280.0 on x).
    // We divide by the window resolution to "normalize" it
    // into a 0.0 to 1.0 range, which is perfect for colors.

    vec2 uv = gl_FragCoord.xy / u_resolution.xy;

    float aspect_ratio = u_resolution.x / u_resolution.y;

    vec2 corrected_uv = uv - 0.5;
    corrected_uv.x *= aspect_ratio;

    vec2 position = (corrected_uv * u_zoom) + u_pan;

    vec2 funcOutput = Mandelbrot(position, 4, u_max_iterations);
    
    float SmoothedOutput = getSmoothColor(funcOutput, u_color_frequency);
    f_color = colormap_psychedelic(SmoothedOutput);
}