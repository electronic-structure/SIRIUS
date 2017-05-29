//== inline __device__ uint32_t random(size_t seed)
//== {
//==     uint32_t h = 5381;
//== 
//==     return (h << (seed % 15)) + h;
//== }
//== 
//== __global__ void randomize_on_gpu_kernel
//== (
//==     double* ptr__,
//==     size_t size__
//== )
//== {
//==     int i = blockIdx.x * blockDim.x + threadIdx.x;
//==     if (i < size__) ptr__[i] = double(random(i)) / (1 << 31);
//== }
//== 
//== extern "C" void randomize_on_gpu(double* ptr, size_t size)
//== {
//==     dim3 grid_t(64);
//==     dim3 grid_b(num_blocks(size, grid_t.x));
//== 
//==     randomize_on_gpu_kernel <<<grid_b, grid_t>>>
//==     (
//==         ptr,
//==         size
//==     );
//== }
