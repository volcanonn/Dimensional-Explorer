vec2 Mandelbrot(vec2 position, float bailoutRadiusSquared, int maxIterations) {
    vec2 z = vec2(0.0, 0.0);
    for (int iterations = 0; iterations < maxIterations; iterations++) {
        z = complexPow(z, vec2(2.0,0.0)) + position;
        if (dot(z, z) > bailoutRadiusSquared) {
            return vec2(float(iterations), dot(z,z));
        }
    }
    return vec2(0.0,0.0);
}