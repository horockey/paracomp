#include <iostream>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

using namespace std;
using namespace std::chrono;


template<typename T>
class ThreadSafeQueue {
private:
	queue<T> data_queue;
	mutable mutex mtx;
	condition_variable cv;

public:
	void push(T value) {
		lock_guard<mutex> lock(mtx);
		data_queue.push(move(value));
		cv.notify_one();
	}

	bool try_pop(T& value) {
		lock_guard<mutex> lock(mtx);
		if (data_queue.empty()) return false;
		value = move(data_queue.front());
		data_queue.pop();
		return true;
	}

	void wait_and_pop(T& value) {
		unique_lock<mutex> lock(mtx);
		cv.wait(lock, [this] { return !data_queue.empty(); });
		value = move(data_queue.front());
		data_queue.pop();
	}

	bool empty() const {
		lock_guard<mutex> lock(mtx);
		return data_queue.empty();
	}
};


void producer_consumer_cv() {
	const int max_size = 5;
	vector<int> buffer(max_size);
	mutex mtx;
	condition_variable not_full, not_empty;
	int write_pos = 0, read_pos = 0, count = 0;
	atomic<bool> done{ false };

	auto producer = [&](int id) {
		for (int i = 0; i < 10; ++i) {
			unique_lock<mutex> lock(mtx);
			not_full.wait(lock, [&] { return count < max_size || done; });
			if (done) break;

			buffer[write_pos] = i;
			write_pos = (write_pos + 1) % max_size;
			++count;
			cout << "Producer " << id << " produced " << i << endl;
			lock.unlock();
			not_empty.notify_one();
			this_thread::sleep_for(milliseconds(100));
		}
		};

	auto consumer = [&](int id) {
		while (!done || count > 0) {
			unique_lock<mutex> lock(mtx);
			not_empty.wait(lock, [&] { return count > 0 || done; });
			if (count == 0 && done) break;

			int value = buffer[read_pos];
			read_pos = (read_pos + 1) % max_size;
			--count;
			cout << "Consumer " << id << " consumed " << value << endl;
			lock.unlock();
			not_full.notify_one();
			this_thread::sleep_for(milliseconds(150));
		}
		};

	thread producer_thread(producer, 1);
	thread consumer_thread(consumer, 1);

	this_thread::sleep_for(seconds(2));
	done = true;
	not_empty.notify_all();
	not_full.notify_all();

	producer_thread.join();
	consumer_thread.join();
}

void producer_consumer_atomic() {
	const int max_size = 5;
	vector<int> buffer(max_size);
	mutex mtx;
	atomic<bool> data_ready{ false };
	atomic<bool> done{ false };
	int write_pos = 0, read_pos = 0;

	auto producer = [&](int id) {
		for (int i = 0; i < 10; ++i) {
			unique_lock<mutex> lock(mtx);
			buffer[write_pos] = i;
			write_pos = (write_pos + 1) % max_size;
			data_ready = true;
			cout << "Producer " << id << " produced " << i << endl;
			lock.unlock();

			while (data_ready && !done) {
				this_thread::yield();
			}
			this_thread::sleep_for(milliseconds(100));
		}
		};

	auto consumer = [&](int id) {
		while (!done || data_ready) {
			if (data_ready) {
				unique_lock<mutex> lock(mtx);
				int value = buffer[read_pos];
				read_pos = (read_pos + 1) % max_size;
				data_ready = false;
				cout << "Consumer " << id << " consumed " << value << endl;
				lock.unlock();
			}
			this_thread::sleep_for(milliseconds(150));
		}
		};

	thread producer_thread(producer, 1);
	thread consumer_thread(consumer, 1);

	this_thread::sleep_for(seconds(2));
	done = true;
	producer_thread.join();
	consumer_thread.join();
}


class ReaderWriter {
private:
	mutable mutex mtx;
	condition_variable no_writer;
	atomic<int> readers{ 0 };
	atomic<bool> writer_active{ false };
	atomic<bool> done{ false }; 

public:
	void read() {
		unique_lock<mutex> lock(mtx);
		no_writer.wait(lock, [this] {
			return !writer_active.load() || done.load();
			});

		if (done) return;

		++readers;
		lock.unlock();

		cout << "Reader " << this_thread::get_id() << " is reading" << endl;
		this_thread::sleep_for(milliseconds(50));

		--readers;
	}

	void write() {
		unique_lock<mutex> lock(mtx);
		no_writer.wait(lock, [this] {
			return (!writer_active.load() && readers.load() == 0) || done.load();
			});

		if (done) return;

		writer_active = true;
		lock.unlock();

		cout << "Writer " << this_thread::get_id() << " is writing" << endl;
		this_thread::sleep_for(milliseconds(100));

		writer_active = false;
		no_writer.notify_all();
	}

	void stop() {
		done = true;
		no_writer.notify_all();
	}
};

void reader_writer() {
	ReaderWriter rw;
	vector<thread> readers;
	vector<thread> writers;

	for (int i = 0; i < 3; ++i) {
		readers.emplace_back([&rw]() {
			for (int j = 0; j < 3; ++j) {
				rw.read();
			}
			});
	}

	for (int i = 0; i < 2; ++i) {
		writers.emplace_back([&rw]() {
			for (int j = 0; j < 2; ++j) {
				rw.write();
			}
			});
	}

	this_thread::sleep_for(seconds(2));

	rw.stop();

	for (auto& t : readers) t.join();
	for (auto& t : writers) t.join();
}

int main() {
	cout << "=== Thread-safe Queue Test ===" << endl;
	ThreadSafeQueue<int> tsq;
	thread t1([&]() { 
		for (int i = 0; i < 5; ++i) {
			tsq.push(i);
			cout << "Pushed: " << i << endl;
		}
	});
	thread t2([&]() {
		int val;
		for (int i = 0; i < 5; ++i) {
			tsq.wait_and_pop(val);
			cout << "Popped: " << val << endl;
		}
		});
	t1.join(); t2.join();

	cout << "\n=== Producer-Consumer (conditional vars) ===" << endl;
	producer_consumer_cv();

	cout << "\n=== Producer-Consumer (atomic) ===" << endl;
	producer_consumer_atomic();

	cout << "\n=== Reader-Writer ===" << endl;
	reader_writer();

	return 0;
}