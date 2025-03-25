#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <omp.h>

using namespace std;
using namespace std::chrono;

double pi_sequential(int n) {
	double h = 1.0 / n;
	double sum = 0.0;

	for (int i = 0; i < n; ++i) {
		double x = i * h;
		sum += 4.0 / (1.0 + x * x);
	}

	return h * sum;
}

double pi_parallel(int n) {
	double h = 1.0 / n;
	double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n; ++i) {
		double x = i * h;
		sum += 4.0 / (1.0 + x * x);
	}

	return h * sum;
}

void pi() {
	const int n = 10000000;
	const double exact_pi = 3.141592653589793;

	auto start = high_resolution_clock::now();
	double seq_pi = pi_sequential(n);
	auto end = high_resolution_clock::now();
	auto seq_time = duration_cast<milliseconds>(end - start).count();

	start = high_resolution_clock::now();
	double par_pi = pi_parallel(n);
	end = high_resolution_clock::now();
	auto par_time = duration_cast<milliseconds>(end - start).count();

	cout << "=== Pi calculation ===" << endl;
	cout << "Referencr value: " << exact_pi << endl;
	cout << "Sequential value: " << seq_pi << " (err: " << abs(seq_pi - exact_pi) << ")" << endl;
	cout << "Sequential time: " << seq_time << " ms" << endl;
	cout << "Parallel value: " << par_pi << " (err: " << abs(par_pi - exact_pi) << ")" << endl;
	cout << "Parallel value: " << par_time << " ms" << endl;
}

void selection_sort_sequential(vector<int>& arr) {
	for (size_t i = 0; i < arr.size() - 1; ++i) {
		size_t min_idx = i;
		for (size_t j = i + 1; j < arr.size(); ++j) {
			if (arr[j] < arr[min_idx]) {
				min_idx = j;
			}
		}
		swap(arr[i], arr[min_idx]);
	}
}

void selection_sort_parallel(vector<int>& arr, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
	for (auto i = 0; i < arr.size() - 1; ++i) {
		size_t min_idx = i;
		for (size_t j = i + 1; j < arr.size(); ++j) {
			if (arr[j] < arr[min_idx]) {
				min_idx = j;
			}
		}
#pragma omp critical
		swap(arr[i], arr[min_idx]);
	}
}

void sort() {
	const size_t size = 10000;
	vector<int> arr_seq(size);
	vector<int> arr_par(size);

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> dis(1, 100000);

	for (size_t i = 0; i < size; ++i) {
		arr_seq[i] = arr_par[i] = dis(gen);
	}

	auto start = high_resolution_clock::now();
	selection_sort_sequential(arr_seq);
	auto end = high_resolution_clock::now();
	auto seq_time = duration_cast<milliseconds>(end - start).count();

	start = high_resolution_clock::now();
	selection_sort_parallel(arr_par, omp_get_num_procs());
	end = high_resolution_clock::now();
	auto par_time = duration_cast<milliseconds>(end - start).count();

	cout << "=== Array sort ===" << endl;
	cout << "Size: " << size << endl;
	cout << "Sequntial time: " << seq_time << " ms" << endl;
	cout << "Parallel time: " << par_time << " ms" << endl;
}

class Matrix {
private:
	vector<vector<double>> data;
	int rows, cols;

public:
	Matrix(int r = 0, int c = 0) : rows(r), cols(c), data(r, vector<double>(c)) {}

	void random_fill() {
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<> dis(0.0, 10.0);

		for (auto& row : data) {
			for (auto& elem : row) {
				elem = dis(gen);
			}
		}
	}

	Matrix multiply_sequential(const Matrix& other) const {
		Matrix result(rows, other.cols);

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < other.cols; ++j) {
				for (int k = 0; k < cols; ++k) {
					result.data[i][j] += data[i][k] * other.data[k][j];
				}
			}
		}

		return result;
	}

	Matrix multiply_parallel(const Matrix& other) const {
		Matrix result(rows, other.cols);

#pragma omp parallel for
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < other.cols; ++j) {
				double sum = 0.0;
				for (int k = 0; k < cols; ++k) {
					sum += data[i][k] * other.data[k][j];
				}
#pragma omp critical
				result.data[i][j] = sum;
			}
		}

		return result;
	}

	bool operator==(const Matrix& other) const {
		if (rows != other.rows || cols != other.cols) return false;

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				if (fabs(data[i][j] - other.data[i][j]) > 1e-6) {
					return false;
				}
			}
		}
		return true;
	}
};

void matrix() {
	const int size = 500;
	Matrix a(size, size), b(size, size);
	a.random_fill();
	b.random_fill();

	auto start = high_resolution_clock::now();
	Matrix seq_result = a.multiply_sequential(b);
	auto end = high_resolution_clock::now();
	auto seq_time = duration_cast<milliseconds>(end - start).count();

	start = high_resolution_clock::now();
	Matrix par_result = a.multiply_parallel(b);
	end = high_resolution_clock::now();
	auto par_time = duration_cast<milliseconds>(end - start).count();

	cout << "=== Matrix mult ===" << endl;
	cout << "Size: " << size << "x" << size << endl;
	cout << "Sequential time: " << seq_time << " ms" << endl;
	cout << "Parallel time: " << par_time << " ms" << endl;
}

int main() {
	omp_set_num_threads(omp_get_num_procs());

	pi();
	sort();
	matrix();

	return 0;
}