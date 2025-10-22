float getSmoothColor(vec2 mandelbrotResult, float colorFrequency) {
    float iterations = mandelbrotResult.x;

    if (iterations == 0.0) {
        return 0.0; // Black for points inside the set
    }

    float final_magnitude_squared = mandelbrotResult.y;
    float smooth_val = iterations - log2(log(final_magnitude_squared) * 0.5);
    
    float color_t = fract(smooth_val * colorFrequency);
    
    return color_t;
}