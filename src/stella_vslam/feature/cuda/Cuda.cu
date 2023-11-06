#include <helper_cuda.h>
#include "stella_vslam/feature/cuda/Cuda.hpp"

namespace stella_vslam {
namespace feature {
namespace cuda {
  void deviceSynchronize() {
    checkCudaErrors( cudaDeviceSynchronize() );
  }
} // namespace cuda
} // namespace feature
} // namespace stella_vslam