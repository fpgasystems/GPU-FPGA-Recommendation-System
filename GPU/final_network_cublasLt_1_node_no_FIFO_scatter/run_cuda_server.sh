# Run this script before client
rm cuda_server
nvcc -std=c++11 -l cublasLt  -lpthread cuda_server.c -o cuda_server
nvprof -f --export-profile timeline.prof --concurrent-kernels on ./cuda_server
#nvprof -f --export-profile timeline.prof --concurrent-kernels off ./cuda_server
