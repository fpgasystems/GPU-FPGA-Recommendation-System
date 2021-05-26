# Run this script before client
rm cuda_server
nvcc -l cublasLt -lpthread cuda_server.c -o cuda_server
nvprof -f --export-profile timeline.prof --concurrent-kernels on ./cuda_server
#nvprof -f --export-profile timeline.prof --concurrent-kernels off ./cuda_server
