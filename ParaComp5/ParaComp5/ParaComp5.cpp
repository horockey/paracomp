#include <iostream>
#include <chrono>
#include <mpi.h>

float rectangle(float a, float b, int n, float (*f)(float)) {
	float h = (b - a) / n;
	float sum = .0f;
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

// MPI-версия метода прямоугольников
float rectangle_mpi(float a, float b, int n, float (*f)(float)) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	float h = (b - a) / n;
	float local_sum = 0.0f;

	// Распределяем итерации между процессами
	for (int i = rank; i < n; i += size) {
		local_sum += f(a + i * h);
	}

	float global_sum = 0.0f;
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	return global_sum * h;
}

// MPI-версия метода трапеций
float trapezoid_mpi(float a, float b, int n, float (*f)(float)) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	float h = (b - a) / n;
	float local_sum = 0.0f;

	// Первый и последний элемент учитываются только один раз
	if (rank == 0) {
		local_sum = 0.5f * (f(a) + f(b));
	}

	// Распределяем итерации между процессами
	for (int i = rank + 1; i < n; i += size) {
		local_sum += f(a + i * h);
	}

	float global_sum = 0.0f;
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	return global_sum * h;
}

// MPI-версия метода Симпсона
float simpson_mpi(float a, float b, int n, float (*f)(float)) {
	if (n % 2 != 0) n++;

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	float h = (b - a) / n;
	float local_sum = 0.0f;

	// Учет начального и конечного значений
	if (rank == 0) {
		local_sum = f(a) + f(b);
	}

	// Обработка нечетных точек (коэффициент 4)
	for (int i = 1 + rank; i < n; i += 2 * size) {
		local_sum += 4.0f * f(a + i * h);
	}

	// Обработка четных точек (коэффициент 2)
	for (int i = 2 + rank; i < n; i += 2 * size) {
		local_sum += 2.0f * f(a + i * h);
	}

	float global_sum = 0.0f;
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	return global_sum * h / 3.0f;
}

float test_function(float x) {
	return 4.0f / (1.0f + x * x);
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const float a = 0.0f;
	const float b = 1.0f;
	const int n = 1000000;
	const float exact_pi = 3.141592653589793f;

	if (rank == 0) {
		// Seq
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

		std::cout << "Sequential results:\n";
		std::cout << "Rectangles:\n";
		std::cout << "  Result: " << pi_rect << " (error: " << fabs(pi_rect - exact_pi) << ")\n";
		std::cout << "  Time: " << duration_rect.count() << " microsec\n";

		std::cout << "Trapezoids:\n";
		std::cout << "  Result: " << pi_trap << " (error: " << fabs(pi_trap - exact_pi) << ")\n";
		std::cout << "  Time: " << duration_trap.count() << " microsec\n";

		std::cout << "Simpson:\n";
		std::cout << "  Result: " << pi_simp << " (error: " << fabs(pi_simp - exact_pi) << ")\n";
		std::cout << "  Time: " << duration_simp.count() << " microsec\n\n";
	}

	// MPI
	MPI_Barrier(MPI_COMM_WORLD);
	auto start = std::chrono::high_resolution_clock::now();
	float mpi_rect = rectangle_mpi(a, b, n, test_function);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration_mpi_rect = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	MPI_Barrier(MPI_COMM_WORLD);
	start = std::chrono::high_resolution_clock::now();
	float mpi_trap = trapezoid_mpi(a, b, n, test_function);
	stop = std::chrono::high_resolution_clock::now();
	auto duration_mpi_trap = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	MPI_Barrier(MPI_COMM_WORLD);
	start = std::chrono::high_resolution_clock::now();
	float mpi_simp = simpson_mpi(a, b, n, test_function);
	stop = std::chrono::high_resolution_clock::now();
	auto duration_mpi_simp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	if (rank == 0) {
		std::cout << "MPI parallel results:\n";
		std::cout << "Rectangles (MPI):\n";
		std::cout << "  Result: " << mpi_rect << " (error: " << fabs(mpi_rect - exact_pi) << ")\n";
		std::cout << "  Time: " << duration_mpi_rect.count() << " microsec\n";

		std::cout << "Trapezoids (MPI):\n";
		std::cout << "  Result: " << mpi_trap << " (error: " << fabs(mpi_trap - exact_pi) << ")\n";
		std::cout << "  Time: " << duration_mpi_trap.count() << " microsec\n";

		std::cout << "Simpson (MPI):\n";
		std::cout << "  Result: " << mpi_simp << " (error: " << fabs(mpi_simp - exact_pi) << ")\n";
		std::cout << "  Time: " << duration_mpi_simp.count() << " microsec\n";
	}

	MPI_Finalize();

	// mpiexec -np 4 C:\Users\rassu\Repos\paracomp\ParaComp5\x64\Debug\ParaComp5.exe
	return 0;
}