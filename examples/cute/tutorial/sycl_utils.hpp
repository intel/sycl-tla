#include <iostream>
#include <sycl/sycl.hpp>

void print_device_info(sycl::queue& queue, std::ostream& output_stream) {
  output_stream << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << " ";
  output_stream << "Num CUs: " << queue.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
}
