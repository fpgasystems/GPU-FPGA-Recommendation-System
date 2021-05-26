/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#include "xcl2.hpp"
#include <vector>
#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 32

#define ID_SENDER 0
#define ID_RECEIVER 1

#define IP_ADDR 0x0A01D497
#define BOARD_NUMBER 0
#define ARP 0x0A01D497

void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> " << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    cl_int err;
    cl::CommandQueue q;
    cl::Context context;

    cl::Kernel user_kernel;
    cl::Kernel network_kernel;

    auto size = DATA_SIZE;
    
    //Allocate Memory in Host Memory
    auto vector_size_bytes = sizeof(int) * size;
    std::vector<int, aligned_allocator<int>> network_ptr0(size);
    std::vector<int, aligned_allocator<int>> network_ptr1(size);
    std::vector<int, aligned_allocator<int>> user_ptr0(size);
    std::vector<int, aligned_allocator<int>> source_sw_results(size);

    // // Create the test data and Software Result
    for (int i = 0; i < size; i++) {
        network_ptr0[i] = i;
        network_ptr1[i] = 0;
        source_sw_results[i] = network_ptr0[i] + 1;
        user_ptr0[i] = 0;
    }

    //OPENCL HOST CODE AREA START
    //Create Program and Kernel
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
                  cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err,
                      network_kernel = cl::Kernel(program, "network_krnl", &err));
            OCL_CHECK(err,
                      user_kernel = cl::Kernel(program, "scatter_krnl", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
    wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");


    // Set network kernel arguments
    OCL_CHECK(err, err = network_kernel.setArg(0, IP_ADDR)); // Default IP address
    OCL_CHECK(err, err = network_kernel.setArg(1, BOARD_NUMBER)); // Board number
    OCL_CHECK(err, err = network_kernel.setArg(2, ARP)); // ARP lookup

    OCL_CHECK(err,
              cl::Buffer buffer_r1(context,
                                   CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   vector_size_bytes,
                                   network_ptr0.data(),
                                   &err));
    OCL_CHECK(err,
            cl::Buffer buffer_r2(context,
                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                vector_size_bytes,
                                network_ptr1.data(),
                                &err));

    OCL_CHECK(err, err = network_kernel.setArg(3, buffer_r1));
    OCL_CHECK(err, err = network_kernel.setArg(4, buffer_r2));


    //Copy input data to device global memory
    //    OCL_CHECK(err,
    //            err = q.enqueueMigrateMemObjects({buffer_r1}, 0));

    OCL_CHECK(err, err = q.enqueueTask(network_kernel));
    printf("enqueue network kernel\n");
    OCL_CHECK(err, err = q.finish());
    //Copy Result from Device Global Memory to Host Local Memory
    //    OCL_CHECK(err,
    //          err = q.enqueueMigrateMemObjects({buffer_r2},
    //                                           CL_MIGRATE_MEM_OBJECT_HOST));
    // Compare the results of the Device to the simulation
    /*int match = 0;
    for (int i = 0; i < size; i++) {
        if (network_ptr1[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i
                      << " Software result = " << source_sw_results[i]
                      << " Device result = " << network_ptr1[i]
                      << std::endl;
            match = 1;
            break;
        }
    }*/

    
    uint32_t numPacketWord = 10;
    uint32_t connection = 1;
    uint32_t numIpAddr = 1;
    uint32_t repetition = 1;
    uint32_t responseInKB = 1;
    uint32_t baseIpAddr = 0x0A01D481; //alveo4b
    uint32_t basePort = 5001; 
    uint32_t numPort = connection / numIpAddr;
    uint32_t delayedCycles = 0;
    uint32_t expectedRespInKBTotal = connection * responseInKB;

    double durationUs = 0.0;

    //Set user Kernel Arguments
    OCL_CHECK(err, err = user_kernel.setArg(0, connection));
    OCL_CHECK(err, err = user_kernel.setArg(1, numIpAddr));
    OCL_CHECK(err, err = user_kernel.setArg(2, numPacketWord));
    OCL_CHECK(err, err = user_kernel.setArg(3, basePort));
    OCL_CHECK(err, err = user_kernel.setArg(4, numPort));
    OCL_CHECK(err, err = user_kernel.setArg(5, responseInKB));
    OCL_CHECK(err, err = user_kernel.setArg(6, delayedCycles));
    OCL_CHECK(err, err = user_kernel.setArg(7, baseIpAddr));
    OCL_CHECK(err, err = user_kernel.setArg(8, expectedRespInKBTotal));

    OCL_CHECK(err,
            cl::Buffer buffer_w(context,
                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                vector_size_bytes,
                                user_ptr0.data(),
                                &err));

    OCL_CHECK(err, err = user_kernel.setArg(9, buffer_w));

    //Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(user_kernel));
    //printf("enqueue user kernel \n");
    OCL_CHECK(err, err = q.finish());
    auto end = std::chrono::high_resolution_clock::now();
    durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
    //OPENCL HOST CODE AREA END
    

    std::cout << "EXIT recorded" << std::endl;
}
