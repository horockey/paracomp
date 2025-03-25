#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <windows.h>

using namespace std;
using namespace std::chrono;

class Matrix {
private:
	vector<vector<double>> data;
	int rows, cols;

public:
	Matrix(int r = 0, int c = 0) : rows(r), cols(c), data(r, vector<double>(c)) {}

	void random_fill() {
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<> dis(-10.0, 10.0);

		for (auto& row : data) {
			for (auto& elem : row) {
				elem = dis(gen);
			}
		}
	}

	void print() const {
		for (const auto& row : data) {
			for (auto elem : row) {
				cout << elem << "\t";
			}
			cout << endl;
		}
	}

	Matrix add_sequential(const Matrix& other) const {
		Matrix result(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				result.data[i][j] = data[i][j] + other.data[i][j];
			}
		}
		return result;
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

	Matrix add_parallel_threads(int thread_count) const {
		Matrix result(rows, cols);
		vector<thread> threads;

		auto worker = [&](int start_row, int end_row) {
			for (int i = start_row; i < end_row; ++i) {
				for (int j = 0; j < cols; ++j) {
					result.data[i][j] = data[i][j] + data[i][j];
				}
			}
			};

		int rows_per_thread = rows / thread_count;
		for (int t = 0; t < thread_count; ++t) {
			int start = t * rows_per_thread;
			int end = (t == thread_count - 1) ? rows : start + rows_per_thread;
			threads.emplace_back(worker, start, end);
		}

		for (auto& t : threads) t.join();
		return result;
	}

	Matrix multiply_parallel_threads(const Matrix& other, int thread_count) const {
		Matrix result(rows, other.cols);
		vector<thread> threads;

		auto worker = [&](int start_row, int end_row) {
			for (int i = start_row; i < end_row; ++i) {
				for (int j = 0; j < other.cols; ++j) {
					for (int k = 0; k < cols; ++k) {
						result.data[i][j] += data[i][k] * other.data[k][j];
					}
				}
			}
			};

		int rows_per_thread = rows / thread_count;
		for (int t = 0; t < thread_count; ++t) {
			int start = t * rows_per_thread;
			int end = (t == thread_count - 1) ? rows : start + rows_per_thread;
			threads.emplace_back(worker, start, end);
		}

		for (auto& t : threads) t.join();
		return result;
	}

	Matrix add_parallel_async(int thread_count) const {
		Matrix result(rows, cols);
		vector<future<void>> futures;

		auto worker = [&](int start_row, int end_row) {
			for (int i = start_row; i < end_row; ++i) {
				for (int j = 0; j < cols; ++j) {
					result.data[i][j] = data[i][j] + data[i][j];
				}
			}
			};

		int rows_per_thread = rows / thread_count;
		for (int t = 0; t < thread_count; ++t) {
			int start = t * rows_per_thread;
			int end = (t == thread_count - 1) ? rows : start + rows_per_thread;
			futures.push_back(async(launch::async, worker, start, end));
		}

		for (auto& f : futures) f.wait();
		return result;
	}

	Matrix multiply_parallel_async(const Matrix& other, int thread_count) const {
		Matrix result(rows, other.cols);
		vector<future<void>> futures;

		auto worker = [&](int start_row, int end_row) {
			for (int i = start_row; i < end_row; ++i) {
				for (int j = 0; j < other.cols; ++j) {
					for (int k = 0; k < cols; ++k) {
						result.data[i][j] += data[i][k] * other.data[k][j];
					}
				}
			}
			};

		int rows_per_thread = rows / thread_count;
		for (int t = 0; t < thread_count; ++t) {
			int start = t * rows_per_thread;
			int end = (t == thread_count - 1) ? rows : start + rows_per_thread;
			futures.push_back(async(launch::async, worker, start, end));
		}

		for (auto& f : futures) f.wait();
		return result;
	}

	static DWORD WINAPI add_worker_winapi(LPVOID param) {
		auto* args = reinterpret_cast<tuple<Matrix*, const Matrix*, int, int>*>(param);
		Matrix* result = get<0>(*args);
		const Matrix* a = get<1>(*args);
		int start_row = get<2>(*args);
		int end_row = get<3>(*args);

		for (int i = start_row; i < end_row; ++i) {
			for (int j = 0; j < a->cols; ++j) {
				result->data[i][j] = a->data[i][j] + a->data[i][j];
			}
		}
		return 0;
	}

	Matrix add_parallel_winapi(int thread_count) const {
		Matrix result(rows, cols);
		vector<HANDLE> threads;
		vector<tuple<Matrix*, const Matrix*, int, int>> args;

		int rows_per_thread = rows / thread_count;
		for (int t = 0; t < thread_count; ++t) {
			int start = t * rows_per_thread;
			int end = (t == thread_count - 1) ? rows : start + rows_per_thread;
			args.emplace_back(&result, this, start, end);
			threads.push_back(CreateThread(NULL, 0, add_worker_winapi, &args.back(), 0, NULL));
		}

		WaitForMultipleObjects(threads.size(), threads.data(), TRUE, INFINITE);
		for (auto h : threads) CloseHandle(h);
		return result;
	}

	int get_rows() const { return rows; }
	int get_cols() const { return cols; }
};

void test_operations(const Matrix& a, const Matrix& b, int thread_count) {
	auto start = high_resolution_clock::now();
	auto c_seq_add = a.add_sequential(b);
	auto end = high_resolution_clock::now();
	cout << "Sequential add: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

	start = high_resolution_clock::now();
	auto c_thread_add = a.add_parallel_threads(thread_count);
	end = high_resolution_clock::now();
	cout << "Threads add: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

	start = high_resolution_clock::now();
	auto c_async_add = a.add_parallel_async(thread_count);
	end = high_resolution_clock::now();
	cout << "Async add: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

	start = high_resolution_clock::now();
	auto c_winapi_add = a.add_parallel_winapi(thread_count);
	end = high_resolution_clock::now();
	cout << "WinAPI add: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

	start = high_resolution_clock::now();
	auto c_seq_mul = a.multiply_sequential(b);
	end = high_resolution_clock::now();
	cout << "Sequential multiply: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

	start = high_resolution_clock::now();
	auto c_thread_mul = a.multiply_parallel_threads(b, thread_count);
	end = high_resolution_clock::now();
	cout << "Threads multiply: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

	start = high_resolution_clock::now();
	auto c_async_mul = a.multiply_parallel_async(b, thread_count);
	end = high_resolution_clock::now();
	cout << "Async multiply: " << duration_cast<milliseconds>(end - start).count() << " ms\n";
}

int main() {
	const int rows = 500;
	const int cols = 500;
	const int thread_count = thread::hardware_concurrency();

	Matrix a(rows, cols);
	Matrix b(rows, cols);

	a.random_fill();
	b.random_fill();

	cout << "Testing with " << thread_count << " threads\n";
	cout << "Matrix size: " << rows << "x" << cols << "\n\n";

	test_operations(a, b, thread_count);

	return 0;
}