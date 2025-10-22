vec4 colormap_psychedelic(float t) {
    if (t == 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0); // Black for points inside the set
    }
    float r = 0.5 * sin(t * 2.0 * PI * 1.5 + 0.0) + 0.5;
    float g = 0.5 * sin(t * 2.0 * PI * 1.5 + 1.5) + 0.5;
    float b = 0.5 * sin(t * 2.0 * PI * 1.5 + 3.0) + 0.5;
    return vec4(r, g, b, 1.0);
}