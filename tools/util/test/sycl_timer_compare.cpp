#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

int main() {
    sycl::property_list props{sycl::property::queue::enable_profiling()};
    sycl::queue q(sycl::gpu_selector_v, props);
    std::cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

    constexpr int N = 1024 * 1024 * 32;
    constexpr int WARMUP = 20;
    constexpr int ITERS = 200;

    float *da = sycl::malloc_device<float>(N, q);
    float *db = sycl::malloc_device<float>(N, q);
    float *dc = sycl::malloc_device<float>(N, q);
    q.fill(da, 1.0f, N).wait();
    q.fill(db, 2.0f, N).wait();
    q.fill(dc, 0.0f, N).wait();

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        q.submit([&](sycl::handler &h) {
            h.parallel_for(N, [=](auto idx) { dc[idx] = da[idx] * db[idx] + dc[idx]; });
        });
        q.wait();
    }

    // Method 1: std::chrono::steady_clock
    std::vector<double> chrono_times;
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        q.submit([&](sycl::handler &h) {
            h.parallel_for(N, [=](auto idx) { dc[idx] = da[idx] * db[idx] + dc[idx]; });
        });
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        chrono_times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    // Method 2: SYCL event (GPU timestamp)
    std::vector<double> event_times;
    for (int i = 0; i < ITERS; ++i) {
        auto e = q.submit([&](sycl::handler &h) {
            h.parallel_for(N, [=](auto idx) { dc[idx] = da[idx] * db[idx] + dc[idx]; });
        });
        e.wait();
        uint64_t t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        event_times.push_back((t1 - t0) / 1e3); // ns → us
    }

    auto stats = [](const std::vector<double> &v, const char *name) {
        auto sorted = v;
        std::sort(sorted.begin(), sorted.end());
        double total = 0;
        for (double x : sorted) total += x;
        double avg = total / sorted.size();
        double median = (sorted[sorted.size()/2] + sorted[(sorted.size()-1)/2]) / 2.0;
        double var = 0;
        for (double x : sorted) { double d = x - avg; var += d * d; }
        double stddev = std::sqrt(var / sorted.size());
        int trim = sorted.size() / 20;
        double ttotal = 0;
        for (int i = trim; i < (int)sorted.size() - trim; ++i) ttotal += sorted[i];
        double tmean = ttotal / (sorted.size() - 2 * trim);
        printf("%s:\n", name);
        printf("  min=%.1fus  max=%.1fus  avg=%.1fus  median=%.1fus\n", sorted.front(), sorted.back(), avg, median);
        printf("  trimmed_mean=%.1fus  stddev=%.1fus\n\n", tmean, stddev);
    };

    stats(chrono_times, "chrono::steady_clock + queue.wait()");
    stats(event_times, "SYCL event profiling (GPU-only timestamp)");

    double c_avg = 0, e_avg = 0;
    for (size_t i = 0; i < chrono_times.size(); ++i) {
        c_avg += chrono_times[i];
        e_avg += event_times[i];
    }
    c_avg /= chrono_times.size();
    e_avg /= event_times.size();
    printf("chrono/event ratio: %.2fx\n", c_avg / e_avg);
    printf("overhead (chrono - event): %.1fus (launch + wait)\n", c_avg - e_avg);

    sycl::free(da, q); sycl::free(db, q); sycl::free(dc, q);
    return 0;
}
