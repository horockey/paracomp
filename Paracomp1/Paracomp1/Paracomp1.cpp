#include <iostream>
#include <immintrin.h> 
#include <chrono>


float rectangle(float a, float b, int n, float (*f)(float)) {
	float h = (b - a) / n;
	float sum = 0.0f;
	for (int i = 0; i < n; ++i) {
		sum += f(a + i * h);
	}
	return sum * h;
}

float trapezoid(float a, float b, int n, float (*f)(float)) {
	float h = (b - a) / n;
	float sum = 0.5f * (f(a) + f(b));
	for (int i = 1; i < n; ++i) {
		sum += f(a + i * h);
	}
	return sum * h;
}

float simpson(float a, float b, int n, float (*f)(float)) {
	if (n % 2 != 0) n++;
	float h = (b - a) / n;
	float sum = f(a) + f(b);
	for (int i = 1; i < n; i += 2) {
		sum += 4.0f * f(a + i * h);
	}
	for (int i = 2; i < n; i += 2) {
		sum += 2.0f * f(a + i * h);
	}
	return sum * h / 3.0f;
}

float test_function(float x) {
	return 4.0f / (1.0f + x * x);
}

__m128 test_function_simd(__m128 x) {
	__m128 ones = _mm_set1_ps(1.0f);
	__m128 x_squared = _mm_mul_ps(x, x);
	__m128 denominator = _mm_add_ps(ones, x_squared);
	__m128 result = _mm_div_ps(_mm_set1_ps(4.0f), denominator);
	return result;
}

float rectangle_simd(float a, float b, int n, float (*f)(float)) {
	float h = (b - a) / n;
	__m128 sum = _mm_setzero_ps();
	for (int i = 0; i < n; i += 4) {
		__m128 x = _mm_set_ps(a + (i + 3) * h, a + (i + 2) * h, a + (i + 1) * h, a + i * h);
		__m128 fx;
		fx = _mm_set_ps(f(a + (i + 3) * h), f(a + (i + 2) * h), f(a + (i + 1) * h), f(a + i * h));
		sum = _mm_add_ps(sum, fx);
	}
	float result[4];
	_mm_store_ps(result, sum);
	return (result[0] + result[1] + result[2] + result[3]) * h;
}

float trapezoid_simd(float a, float b, int n) {
	float h = (b - a) / n;
	__m128 sum = _mm_setzero_ps();

	for (int i = 1; i < n - 3; i += 4) {
		__m128 x = _mm_set_ps(a + (i + 3) * h, a + (i + 2) * h, a + (i + 1) * h, a + i * h);
		sum = _mm_add_ps(sum, test_function_simd(x));
	}

	float partial_sum = 0.0f;
	for (int i = ((n - 1) / 4) * 4 + 1; i < n; ++i) {
		partial_sum += test_function(a + i * h);
	}

	float result[4];
	_mm_store_ps(result, sum);
	float total = (result[0] + result[1] + result[2] + result[3]) + partial_sum;
	return (0.5f * (test_function(a) + test_function(b)) + total) * h;
}

float simpson_simd(float a, float b, int n) {
	if (n % 2 != 0) n++;
	float h = (b - a) / n;

	__m128 sum_odd = _mm_setzero_ps();  
	__m128 sum_even = _mm_setzero_ps(); 

	for (int i = 1; i < n; i += 8) {
		__m128 x_odd = _mm_set_ps(a + (i + 3) * h, a + (i + 1) * h, a + (i + 5) * h, a + (i + 7) * h);
		sum_odd = _mm_add_ps(sum_odd, test_function_simd(x_odd));

		__m128 x_even = _mm_set_ps(a + (i + 2) * h, a + i * h, a + (i + 4) * h, a + (i + 6) * h);
		sum_even = _mm_add_ps(sum_even, test_function_simd(x_even));
	}

	float partial_odd = 0.0f, partial_even = 0.0f;
	int remaining_start = ((n - 1) / 8) * 8 + 1;
	for (int i = remaining_start; i < n; ++i) {
		if (i % 2 == 1) partial_odd += test_function(a + i * h);
		else partial_even += test_function(a + i * h);
	}

	float odd[4], even[4];
	_mm_store_ps(odd, sum_odd);
	_mm_store_ps(even, sum_even);

	float total_odd = odd[0] + odd[1] + odd[2] + odd[3] + partial_odd;
	float total_even = even[0] + even[1] + even[2] + even[3] + partial_even;

	return (test_function(a) + test_function(b) + 4 * total_odd + 2 * total_even) * h / 3.0f;
}

int main() {
	const float a = 0.0f;
	const float b = 1.0f;
	const int n = 1000000; 

	auto start = std::chrono::high_resolution_clock::now();
	float pi_rect = rectangle(a, b, n, test_function);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration_rect = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	start = std::chrono::high_resolution_clock::now();
	float pi_trap = trapezoid(a, b, n, test_function);
	stop = std::chrono::high_resolution_clock::now();
	auto duration_trap = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	start = std::chrono::high_resolution_clock::now();
	float pi_simp = simpson(a, b, n, test_function);
	stop = std::chrono::high_resolution_clock::now();
	auto duration_simp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	start = std::chrono::high_resolution_clock::now();
	float pi_rect_simd = rectangle_simd(a, b, n, test_function);
	stop = std::chrono::high_resolution_clock::now();
	auto duration_rect_simd = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	start = std::chrono::high_resolution_clock::now();
	float pi_trap_simd = trapezoid_simd(a, b, n);
	stop = std::chrono::high_resolution_clock::now();
	auto duration_trap_simd = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	start = std::chrono::high_resolution_clock::now();
	float pi_simpson_simd = simpson_simd(a, b, n);
	stop = std::chrono::high_resolution_clock::now();
	auto duration_simpson_simd = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Rectangles:\n";
	std::cout << "  Result: " << pi_rect << "\n";
	std::cout << "  Time: " << duration_rect.count() << " microsec\n";
	std::cout << "  Result (SIMD): " << pi_rect_simd << "\n";
	std::cout << "  Time (SIMD): " << duration_rect_simd.count() << " microsec\n\n";

	std::cout << "Trapezoids:\n";
	std::cout << "  Result: " << pi_trap << "\n";
	std::cout << "  Time: " << duration_trap.count() << " microsec\n";
	std::cout << "  Result (SIMD): " << pi_trap_simd << "\n";
	std::cout << "  Time (SIMD): " << duration_trap_simd.count() << " microsec\n\n";

	std::cout << "Simpon:\n";
	std::cout << "  Result: " << pi_simp << "\n";
	std::cout << "  Time: " << duration_simp.count() << " microsec\n";
	std::cout << "  Result (SIMD): " << pi_simpson_simd << "\n";
	std::cout << "  Time (SIMD): " << duration_simpson_simd.count() << " microsec\n\n";

	return 0;
}