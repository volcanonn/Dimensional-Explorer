vec2 complexPow(vec2 num, vec2 exponent) {
	float r = length(num);

	if (r == 0.0) return vec2(0.0, 0.0);

	float theta = atan(num.y, num.x);
	float logR = log(r);

	float newR = pow(r, exponent.x) * exp(-exponent.y * theta);
	float newTheta = exponent.x * theta + exponent.y * logR;

	return vec2(newR * cos(newTheta), newR * sin(newTheta));
}