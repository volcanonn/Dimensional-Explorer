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