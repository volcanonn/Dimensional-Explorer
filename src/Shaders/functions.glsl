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

vec2 SimpleWave(vec2 position) {
    return vec2(sin(position.x) + cos(position.y), dot(position.x,position.y));
}

vec2 RadialWave(vec2 position) {
    return vec2(sin(distance(position.x,position.y)), dot(position.x,position.y));
}

vec2 Paraboloid(vec2 position) {
    return vec2(dot(position.x,position.y), dot(position.x,position.y));
}

vec2 Voronoi(vec2 position, vec2 uv) {
    vec2 grid_uv = fract(uv * 5.0);
    vec2 grid_id = floor(uv * 5.0);
    float min_dist = 1.0;
    for (float y = -1.0; y <= 1.0; y++) {
        for (float x = -1.0; x <= 1.0; x++) {
            vec2 neighbor_id = vec2(x, y);
            // Each cell gets a random point inside it
            vec2 point_pos = neighbor_id + random(grid_id + neighbor_id);
            float dist = length(grid_uv - point_pos);
            min_dist = min(min_dist, dist);
        }
    }
    return vec2(min_dist, dot(position.x,position.y));
}