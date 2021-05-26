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

#include "host.hpp"

#define DATA_SIZE 62500000

// #define IP_ADDR 0x0A01D497
// #define BOARD_NUMBER 0
// #define ARP 0x0A01D497

void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
/* for U280 specifically */
const int bank[40] = {
    /* 0 ~ 31 HBM */
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31), 
    /* 32, 33 DDR */ 
    BANK_NAME(32), BANK_NAME(33), 
    /* 34 ~ 39 PLRAM */ 
    BANK_NAME(34), BANK_NAME(35), BANK_NAME(36), BANK_NAME(37), 
    BANK_NAME(38), BANK_NAME(39)};

void init_vectors(
    axi_t* DRAM_embedding,
    int table_entry_num, int axi_padded_size, int axi_start_addr) {

    // even row: 1
    // odd row: 0
    float one_arr[4] = {1, 1, 1, 1};
    float zero_arr[4] = {0, 0, 0, 0};

#define DEBUG // only init the first 100 entries to speedup the init process
#ifdef DEBUG
    for (int i = 0 ; i < 100; i++) {
#else
    for (int i = 0 ; i < table_entry_num / 2; i++) {
#endif
        for (int j = 0; j < axi_padded_size; j++) {
            memcpy(DRAM_embedding + (2 * i) * axi_padded_size + j + axi_start_addr, one_arr, 16);
        }
        for (int j = 0; j < axi_padded_size; j++) {
            memcpy(DRAM_embedding + (2 * i + 1) * axi_padded_size + j + axi_start_addr, zero_arr, 16);
        }
    }
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <batch number> <dest IP> <dest port> <local IP> <BOARD_NUMBER> <pkgWordCount> <useConn>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    cl_int err;
    cl::CommandQueue q;
    cl::Context context;

    cl::Kernel user_kernel;
    cl::Kernel network_kernel;

    int useConn = 4; 
    int pkgWordCount = 16; 
    int basePort = 5001; 
    uint32_t baseIpAddr = 0x0A01D46E; //alveo0
    uint32_t localIP = 0x0A01D498;
    uint32_t boardNum = 1;

    int batch_num = 1000000;
    size_t result_size = BATCH_SIZE * INPUT_SIZE_AXI_512; // 55 -> 880D float
    size_t idx_size = BATCH_SIZE;

    if (argc >= 3)
    {
        batch_num = strtol(argv[2], NULL, 10);
    }

    if (argc >= 4)
    {
        std::string s = argv[3];
        std::string delimiter = ".";
        int ip [4];
        size_t pos = 0;
        std::string token;
        int i = 0;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            ip [i] = stoi(token);
            s.erase(0, pos + delimiter.length());
            i++;
        }
        ip[i] = stoi(s); 
        baseIpAddr = ip[3] | (ip[2] << 8) | (ip[1] << 16) | (ip[0] << 24);
    }

    if (argc >= 5)
    {
        basePort = strtol(argv[4], NULL, 10);
    }

    if (argc >= 6)
    {
        std::string s = argv[5];
        std::string delimiter = ".";
        int ip [4];
        size_t pos = 0;
        std::string token;
        int i = 0;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            ip [i] = stoi(token);
            s.erase(0, pos + delimiter.length());
            i++;
        }
        ip[i] = stoi(s); 
        localIP = ip[3] | (ip[2] << 8) | (ip[1] << 16) | (ip[0] << 24);
    }

    if (argc >= 7)
    {
        boardNum = strtol(argv[6], NULL, 10);
    }

    if (argc >= 8)
    {
        pkgWordCount = strtol(argv[7], NULL, 10);
    }

    if (argc >= 9)
    {
        useConn = strtol(argv[8], NULL, 10);
    }

    printf("localIP:%x, boardNum:%d\n", localIP, boardNum);
    printf("batch_num:%d, dest IP:%x, dest port:%d, pkgWordCount:%d, useConn:%d \n", batch_num, baseIpAddr, basePort, pkgWordCount, useConn);



////////////////////////////////////////////////////////////////////////////////////////
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
                      user_kernel = cl::Kernel(program, "embedding_98_krnl", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
    wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");


//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
    //launch network kernel
    auto size = DATA_SIZE;
    
    //Allocate Memory in Host Memory
    auto vector_size_bytes = sizeof(int) * size;
    std::vector<int, aligned_allocator<int>> network_ptr0(size);
    std::vector<int, aligned_allocator<int>> network_ptr1(size);

    // Set network kernel arguments
    OCL_CHECK(err, err = network_kernel.setArg(0, localIP)); // Default IP address
    OCL_CHECK(err, err = network_kernel.setArg(1, boardNum)); // Board number
    OCL_CHECK(err, err = network_kernel.setArg(2, localIP)); // ARP lookup


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


    OCL_CHECK(err, err = q.enqueueTask(network_kernel));
    printf("enqueue network kernel\n");
    OCL_CHECK(err, err = q.finish());
    

//////////////////////////////   TEMPLATE START  //////////////////////////////
    size_t HBM_embedding0_size =  HBM_BANK0_SIZE;
    size_t HBM_embedding1_size =  HBM_BANK1_SIZE;
    size_t HBM_embedding2_size =  HBM_BANK2_SIZE;
    size_t HBM_embedding3_size =  HBM_BANK3_SIZE;
    size_t HBM_embedding4_size =  HBM_BANK4_SIZE;
    size_t HBM_embedding5_size =  HBM_BANK5_SIZE;
    size_t HBM_embedding6_size =  HBM_BANK6_SIZE;
    size_t HBM_embedding7_size =  HBM_BANK7_SIZE;
    size_t HBM_embedding8_size =  HBM_BANK8_SIZE;
    size_t HBM_embedding9_size =  HBM_BANK9_SIZE;
    size_t HBM_embedding10_size =  HBM_BANK10_SIZE;
    size_t HBM_embedding11_size =  HBM_BANK11_SIZE;
    size_t HBM_embedding12_size =  HBM_BANK12_SIZE;
    size_t HBM_embedding13_size =  HBM_BANK13_SIZE;
    size_t HBM_embedding14_size =  HBM_BANK14_SIZE;
    size_t HBM_embedding15_size =  HBM_BANK15_SIZE;
    size_t HBM_embedding16_size =  HBM_BANK16_SIZE;
    size_t HBM_embedding17_size =  HBM_BANK17_SIZE;
    size_t HBM_embedding18_size =  HBM_BANK18_SIZE;
    size_t HBM_embedding19_size =  HBM_BANK19_SIZE;
    size_t HBM_embedding20_size =  HBM_BANK20_SIZE;
    size_t HBM_embedding21_size =  HBM_BANK21_SIZE;
    size_t HBM_embedding22_size =  HBM_BANK22_SIZE;
    size_t HBM_embedding23_size =  HBM_BANK23_SIZE;
    size_t HBM_embedding24_size =  HBM_BANK24_SIZE;
    size_t HBM_embedding25_size =  HBM_BANK25_SIZE;
    size_t HBM_embedding26_size =  HBM_BANK26_SIZE;
    size_t HBM_embedding27_size =  HBM_BANK27_SIZE;

    size_t DDR_embedding0_size =  DDR_BANK0_SIZE;
    size_t DDR_embedding1_size =  DDR_BANK1_SIZE;

    // size_t PLRAM_embedding0_size =  PLRAM_BANK0_SIZE;
    // size_t PLRAM_embedding1_size =  PLRAM_BANK1_SIZE;
    // size_t PLRAM_embedding2_size =  PLRAM_BANK2_SIZE;
    // size_t PLRAM_embedding3_size =  PLRAM_BANK3_SIZE;
    // size_t PLRAM_embedding4_size =  PLRAM_BANK4_SIZE;
    // size_t PLRAM_embedding5_size =  PLRAM_BANK5_SIZE;
    // size_t PLRAM_embedding6_size =  PLRAM_BANK6_SIZE;
    // size_t PLRAM_embedding7_size =  PLRAM_BANK7_SIZE;
    // size_t PLRAM_embedding8_size =  PLRAM_BANK8_SIZE;
    // size_t PLRAM_embedding9_size =  PLRAM_BANK9_SIZE;
    // size_t PLRAM_embedding10_size =  PLRAM_BANK10_SIZE;
    // size_t PLRAM_embedding11_size =  PLRAM_BANK11_SIZE;
    // size_t PLRAM_embedding12_size =  PLRAM_BANK12_SIZE;
    // size_t PLRAM_embedding13_size =  PLRAM_BANK13_SIZE;
    // size_t PLRAM_embedding14_size =  PLRAM_BANK14_SIZE;
    // size_t PLRAM_embedding15_size =  PLRAM_BANK15_SIZE;
    // size_t PLRAM_embedding16_size =  PLRAM_BANK16_SIZE;

//////////////////////////////   TEMPLATE END  //////////////////////////////

    // size_t result_size = BATCH_SIZE * INPUT_SIZE_AXI_512; // 22 -> 352D float = 22 * 512 bits
    // size_t idx_size = BATCH_SIZE;
    // cl_int err;
    // unsigned fileBufSize;

    // allocate aligned 2D vectors
    std::cout << "Allocating memory..." << std::endl;
//////////////////////////////   TEMPLATE START  //////////////////////////////
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding0(HBM_embedding0_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding1(HBM_embedding1_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding2(HBM_embedding2_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding3(HBM_embedding3_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding4(HBM_embedding4_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding5(HBM_embedding5_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding6(HBM_embedding6_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding7(HBM_embedding7_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding8(HBM_embedding8_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding9(HBM_embedding9_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding10(HBM_embedding10_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding11(HBM_embedding11_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding12(HBM_embedding12_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding13(HBM_embedding13_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding14(HBM_embedding14_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding15(HBM_embedding15_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding16(HBM_embedding16_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding17(HBM_embedding17_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding18(HBM_embedding18_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding19(HBM_embedding19_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding20(HBM_embedding20_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding21(HBM_embedding21_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding22(HBM_embedding22_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding23(HBM_embedding23_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding24(HBM_embedding24_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding25(HBM_embedding25_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding26(HBM_embedding26_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> HBM_embedding27(HBM_embedding27_size, 0);

    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding0(PLRAM_embedding0_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding1(PLRAM_embedding1_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding2(PLRAM_embedding2_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding3(PLRAM_embedding3_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding4(PLRAM_embedding4_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding5(PLRAM_embedding5_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding6(PLRAM_embedding6_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding7(PLRAM_embedding7_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding8(PLRAM_embedding8_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding9(PLRAM_embedding9_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding10(PLRAM_embedding10_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding11(PLRAM_embedding11_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding12(PLRAM_embedding12_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding13(PLRAM_embedding13_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding14(PLRAM_embedding14_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding15(PLRAM_embedding15_size, 0);
    // std::vector<axi_t, aligned_allocator<axi_t>> PLRAM_embedding16(PLRAM_embedding16_size, 0);

    std::vector<axi_t, aligned_allocator<axi_t>> DDR_embedding0(DDR_embedding0_size, 0);
    std::vector<axi_t, aligned_allocator<axi_t>> DDR_embedding1(DDR_embedding1_size, 0);
//////////////////////////////   TEMPLATE END  //////////////////////////////
    std::vector<int, aligned_allocator<int>> access_idx(idx_size, 0);
    std::vector<network_t, aligned_allocator<network_t>> source_hw_results(result_size, 0);
    std::vector<network_t, aligned_allocator<network_t>> source_sw_results(result_size, 0);

//     std::cout << "test mem boundary " << DDR_embedding1[DDR_BANK1_SIZE - 1] << std::endl;
    // NOTE! the following expression will lead to seg fault!
    // since the mul expression assumes "int" input, but its result exceeds the int limit
    // std::cout << "test mem boundary " << DDR_embedding1[PADDED_SIZE_DDR_3 * TABLE_SIZE_DDR_3 - 1] << std::endl;
    // std::cout << "test mem boundary " << DDR_embedding1[ADDR_AXI_DDR_3 + PADDED_SIZE_DDR_3 * TABLE_SIZE_DDR_3 - 1] << std::endl;

    std::cout << "Allocating memory finished" << std::endl;
    
    // generate embedding table
//////////////////////////////   TEMPLATE START  //////////////////////////////

    std::cout << "Generating table contents..." << std::endl;
    std::cout << "HBM Round 0..." << std::endl;

    init_vectors(&HBM_embedding0[0], TABLE_SIZE_HBM_0, AXI_PADDED_SIZE_HBM_0, ADDR_AXI_HBM_0);
    init_vectors(&HBM_embedding1[0], TABLE_SIZE_HBM_1, AXI_PADDED_SIZE_HBM_1, ADDR_AXI_HBM_1);
    init_vectors(&HBM_embedding2[0], TABLE_SIZE_HBM_2, AXI_PADDED_SIZE_HBM_2, ADDR_AXI_HBM_2);
    init_vectors(&HBM_embedding3[0], TABLE_SIZE_HBM_3, AXI_PADDED_SIZE_HBM_3, ADDR_AXI_HBM_3);
    init_vectors(&HBM_embedding4[0], TABLE_SIZE_HBM_4, AXI_PADDED_SIZE_HBM_4, ADDR_AXI_HBM_4);
    init_vectors(&HBM_embedding5[0], TABLE_SIZE_HBM_5, AXI_PADDED_SIZE_HBM_5, ADDR_AXI_HBM_5);
    init_vectors(&HBM_embedding6[0], TABLE_SIZE_HBM_6, AXI_PADDED_SIZE_HBM_6, ADDR_AXI_HBM_6);
    init_vectors(&HBM_embedding7[0], TABLE_SIZE_HBM_7, AXI_PADDED_SIZE_HBM_7, ADDR_AXI_HBM_7);
    init_vectors(&HBM_embedding8[0], TABLE_SIZE_HBM_8, AXI_PADDED_SIZE_HBM_8, ADDR_AXI_HBM_8);
    init_vectors(&HBM_embedding9[0], TABLE_SIZE_HBM_9, AXI_PADDED_SIZE_HBM_9, ADDR_AXI_HBM_9);
    init_vectors(&HBM_embedding10[0], TABLE_SIZE_HBM_10, AXI_PADDED_SIZE_HBM_10, ADDR_AXI_HBM_10);
    init_vectors(&HBM_embedding11[0], TABLE_SIZE_HBM_11, AXI_PADDED_SIZE_HBM_11, ADDR_AXI_HBM_11);
    init_vectors(&HBM_embedding12[0], TABLE_SIZE_HBM_12, AXI_PADDED_SIZE_HBM_12, ADDR_AXI_HBM_12);
    init_vectors(&HBM_embedding13[0], TABLE_SIZE_HBM_13, AXI_PADDED_SIZE_HBM_13, ADDR_AXI_HBM_13);
    init_vectors(&HBM_embedding14[0], TABLE_SIZE_HBM_14, AXI_PADDED_SIZE_HBM_14, ADDR_AXI_HBM_14);
    init_vectors(&HBM_embedding15[0], TABLE_SIZE_HBM_15, AXI_PADDED_SIZE_HBM_15, ADDR_AXI_HBM_15);
    init_vectors(&HBM_embedding16[0], TABLE_SIZE_HBM_16, AXI_PADDED_SIZE_HBM_16, ADDR_AXI_HBM_16);
    init_vectors(&HBM_embedding17[0], TABLE_SIZE_HBM_17, AXI_PADDED_SIZE_HBM_17, ADDR_AXI_HBM_17);
    init_vectors(&HBM_embedding18[0], TABLE_SIZE_HBM_18, AXI_PADDED_SIZE_HBM_18, ADDR_AXI_HBM_18);
    init_vectors(&HBM_embedding19[0], TABLE_SIZE_HBM_19, AXI_PADDED_SIZE_HBM_19, ADDR_AXI_HBM_19);
    init_vectors(&HBM_embedding20[0], TABLE_SIZE_HBM_20, AXI_PADDED_SIZE_HBM_20, ADDR_AXI_HBM_20);
    init_vectors(&HBM_embedding21[0], TABLE_SIZE_HBM_21, AXI_PADDED_SIZE_HBM_21, ADDR_AXI_HBM_21);
    init_vectors(&HBM_embedding22[0], TABLE_SIZE_HBM_22, AXI_PADDED_SIZE_HBM_22, ADDR_AXI_HBM_22);
    init_vectors(&HBM_embedding23[0], TABLE_SIZE_HBM_23, AXI_PADDED_SIZE_HBM_23, ADDR_AXI_HBM_23);
    init_vectors(&HBM_embedding24[0], TABLE_SIZE_HBM_24, AXI_PADDED_SIZE_HBM_24, ADDR_AXI_HBM_24);
    init_vectors(&HBM_embedding25[0], TABLE_SIZE_HBM_25, AXI_PADDED_SIZE_HBM_25, ADDR_AXI_HBM_25);
    init_vectors(&HBM_embedding26[0], TABLE_SIZE_HBM_26, AXI_PADDED_SIZE_HBM_26, ADDR_AXI_HBM_26);
    init_vectors(&HBM_embedding27[0], TABLE_SIZE_HBM_27, AXI_PADDED_SIZE_HBM_27, ADDR_AXI_HBM_27);

    std::cout << "HBM Round 0..." << std::endl;
    init_vectors(&HBM_embedding0[0], TABLE_SIZE_HBM_28, AXI_PADDED_SIZE_HBM_28, ADDR_AXI_HBM_28);
    init_vectors(&HBM_embedding1[0], TABLE_SIZE_HBM_29, AXI_PADDED_SIZE_HBM_29, ADDR_AXI_HBM_29);
    init_vectors(&HBM_embedding2[0], TABLE_SIZE_HBM_30, AXI_PADDED_SIZE_HBM_30, ADDR_AXI_HBM_30);
    init_vectors(&HBM_embedding3[0], TABLE_SIZE_HBM_31, AXI_PADDED_SIZE_HBM_31, ADDR_AXI_HBM_31);
    init_vectors(&HBM_embedding4[0], TABLE_SIZE_HBM_32, AXI_PADDED_SIZE_HBM_32, ADDR_AXI_HBM_32);
    init_vectors(&HBM_embedding5[0], TABLE_SIZE_HBM_33, AXI_PADDED_SIZE_HBM_33, ADDR_AXI_HBM_33);
    init_vectors(&HBM_embedding6[0], TABLE_SIZE_HBM_34, AXI_PADDED_SIZE_HBM_34, ADDR_AXI_HBM_34);
    init_vectors(&HBM_embedding7[0], TABLE_SIZE_HBM_35, AXI_PADDED_SIZE_HBM_35, ADDR_AXI_HBM_35);
    init_vectors(&HBM_embedding8[0], TABLE_SIZE_HBM_36, AXI_PADDED_SIZE_HBM_36, ADDR_AXI_HBM_36);
    init_vectors(&HBM_embedding9[0], TABLE_SIZE_HBM_37, AXI_PADDED_SIZE_HBM_37, ADDR_AXI_HBM_37);
    init_vectors(&HBM_embedding10[0], TABLE_SIZE_HBM_38, AXI_PADDED_SIZE_HBM_38, ADDR_AXI_HBM_38);
    init_vectors(&HBM_embedding11[0], TABLE_SIZE_HBM_39, AXI_PADDED_SIZE_HBM_39, ADDR_AXI_HBM_39);
    init_vectors(&HBM_embedding12[0], TABLE_SIZE_HBM_40, AXI_PADDED_SIZE_HBM_40, ADDR_AXI_HBM_40);
    init_vectors(&HBM_embedding13[0], TABLE_SIZE_HBM_41, AXI_PADDED_SIZE_HBM_41, ADDR_AXI_HBM_41);
    init_vectors(&HBM_embedding14[0], TABLE_SIZE_HBM_42, AXI_PADDED_SIZE_HBM_42, ADDR_AXI_HBM_42);
    init_vectors(&HBM_embedding15[0], TABLE_SIZE_HBM_43, AXI_PADDED_SIZE_HBM_43, ADDR_AXI_HBM_43);
    init_vectors(&HBM_embedding16[0], TABLE_SIZE_HBM_44, AXI_PADDED_SIZE_HBM_44, ADDR_AXI_HBM_44);
    init_vectors(&HBM_embedding17[0], TABLE_SIZE_HBM_45, AXI_PADDED_SIZE_HBM_45, ADDR_AXI_HBM_45);
    init_vectors(&HBM_embedding18[0], TABLE_SIZE_HBM_46, AXI_PADDED_SIZE_HBM_46, ADDR_AXI_HBM_46);
    init_vectors(&HBM_embedding19[0], TABLE_SIZE_HBM_47, AXI_PADDED_SIZE_HBM_47, ADDR_AXI_HBM_47);
    init_vectors(&HBM_embedding20[0], TABLE_SIZE_HBM_48, AXI_PADDED_SIZE_HBM_48, ADDR_AXI_HBM_48);
    init_vectors(&HBM_embedding21[0], TABLE_SIZE_HBM_49, AXI_PADDED_SIZE_HBM_49, ADDR_AXI_HBM_49);
    init_vectors(&HBM_embedding22[0], TABLE_SIZE_HBM_50, AXI_PADDED_SIZE_HBM_50, ADDR_AXI_HBM_50);
    init_vectors(&HBM_embedding23[0], TABLE_SIZE_HBM_51, AXI_PADDED_SIZE_HBM_51, ADDR_AXI_HBM_51);
    init_vectors(&HBM_embedding24[0], TABLE_SIZE_HBM_52, AXI_PADDED_SIZE_HBM_52, ADDR_AXI_HBM_52);
    init_vectors(&HBM_embedding25[0], TABLE_SIZE_HBM_53, AXI_PADDED_SIZE_HBM_53, ADDR_AXI_HBM_53);
    init_vectors(&HBM_embedding26[0], TABLE_SIZE_HBM_54, AXI_PADDED_SIZE_HBM_54, ADDR_AXI_HBM_54);
    init_vectors(&HBM_embedding27[0], TABLE_SIZE_HBM_55, AXI_PADDED_SIZE_HBM_55, ADDR_AXI_HBM_55);


    std::cout << "DDR Round 0..." << std::endl;
    init_vectors(&DDR_embedding0[0], TABLE_SIZE_DDR_0, AXI_PADDED_SIZE_DDR_0, ADDR_AXI_DDR_0);
    init_vectors(&DDR_embedding1[0], TABLE_SIZE_DDR_1, AXI_PADDED_SIZE_DDR_1, ADDR_AXI_DDR_1);

    std::cout << "DDR Round 1..." << std::endl;
    init_vectors(&DDR_embedding0[0], TABLE_SIZE_DDR_0, AXI_PADDED_SIZE_DDR_2, ADDR_AXI_DDR_2);
    init_vectors(&DDR_embedding1[0], TABLE_SIZE_DDR_1, AXI_PADDED_SIZE_DDR_3, ADDR_AXI_DDR_3);

    std::cout << "Generating table contents finished" << std::endl;
//////////////////////////////   TEMPLATE END  //////////////////////////////
    // software result
    int idx_random[] = {3, 99, 38, 72, 29, 57, 1, 72, 36, 76, 35, 50, 37, 57, 
        13, 66, 26, 70, 41, 93, 48, 82, 44, 78, 25, 52, 3, 92, 36, 56, 46, 88};


// ================================================================
// Step 2: Setup Buffers and run Kernels
// ================================================================
//   o) Allocate Memory to store the results 
//   o) Create Buffers in Global Memory to store data
// ================================================================

// ------------------------------------------------------------------
// Step 2: Create Buffers in Global Memory to store data
//             o) buffer_in1 - stores source_in1
//             o) buffer_in2 - stores source_in2
//             o) buffer_ouput - stores Results
// ------------------------------------------------------------------   

// .......................................................
// Allocate Global Memory for source_in1
// .......................................................  
//////////////////////////////   TEMPLATE START  //////////////////////////////
    cl_mem_ext_ptr_t HBM_embedding0Ext, HBM_embedding1Ext, HBM_embedding2Ext, HBM_embedding3Ext, 
        HBM_embedding4Ext, HBM_embedding5Ext, HBM_embedding6Ext, HBM_embedding7Ext, 
        HBM_embedding8Ext, HBM_embedding9Ext, HBM_embedding10Ext, HBM_embedding11Ext, 
        HBM_embedding12Ext, HBM_embedding13Ext, HBM_embedding14Ext, HBM_embedding15Ext, 
        HBM_embedding16Ext, HBM_embedding17Ext, HBM_embedding18Ext, HBM_embedding19Ext, 
        HBM_embedding20Ext, HBM_embedding21Ext, HBM_embedding22Ext, HBM_embedding23Ext, 
        HBM_embedding24Ext, HBM_embedding25Ext, HBM_embedding26Ext, HBM_embedding27Ext, 
        // HBM_embedding28Ext, HBM_embedding29Ext, HBM_embedding30Ext, HBM_embedding31Ext, 
        DDR_embedding0Ext, DDR_embedding1Ext,
        /* PLRAM_embedding0Ext, PLRAM_embedding1Ext, PLRAM_embedding2Ext, PLRAM_embedding3Ext, 
        access_idxExt, */ source_hw_resultsExt;
//////////////////////////////   TEMPLATE END  //////////////////////////////

//////////////////////////////   TEMPLATE START  //////////////////////////////
    HBM_embedding0Ext.obj = HBM_embedding0.data();
    HBM_embedding0Ext.param = 0;
    HBM_embedding0Ext.flags = bank[0];
    HBM_embedding1Ext.obj = HBM_embedding1.data();
    HBM_embedding1Ext.param = 0;
    HBM_embedding1Ext.flags = bank[1];
    HBM_embedding2Ext.obj = HBM_embedding2.data();
    HBM_embedding2Ext.param = 0;
    HBM_embedding2Ext.flags = bank[2];
    HBM_embedding3Ext.obj = HBM_embedding3.data();
    HBM_embedding3Ext.param = 0;
    HBM_embedding3Ext.flags = bank[3];
    HBM_embedding4Ext.obj = HBM_embedding4.data();
    HBM_embedding4Ext.param = 0;
    HBM_embedding4Ext.flags = bank[4];
    HBM_embedding5Ext.obj = HBM_embedding5.data();
    HBM_embedding5Ext.param = 0;
    HBM_embedding5Ext.flags = bank[5];
    HBM_embedding6Ext.obj = HBM_embedding6.data();
    HBM_embedding6Ext.param = 0;
    HBM_embedding6Ext.flags = bank[6];
    HBM_embedding7Ext.obj = HBM_embedding7.data();
    HBM_embedding7Ext.param = 0;
    HBM_embedding7Ext.flags = bank[7];
    HBM_embedding8Ext.obj = HBM_embedding8.data();
    HBM_embedding8Ext.param = 0;
    HBM_embedding8Ext.flags = bank[8];
    HBM_embedding9Ext.obj = HBM_embedding9.data();
    HBM_embedding9Ext.param = 0;
    HBM_embedding9Ext.flags = bank[9];
    HBM_embedding10Ext.obj = HBM_embedding10.data();
    HBM_embedding10Ext.param = 0;
    HBM_embedding10Ext.flags = bank[10];
    HBM_embedding11Ext.obj = HBM_embedding11.data();
    HBM_embedding11Ext.param = 0;
    HBM_embedding11Ext.flags = bank[11];
    HBM_embedding12Ext.obj = HBM_embedding12.data();
    HBM_embedding12Ext.param = 0;
    HBM_embedding12Ext.flags = bank[12];
    HBM_embedding13Ext.obj = HBM_embedding13.data();
    HBM_embedding13Ext.param = 0;
    HBM_embedding13Ext.flags = bank[13];
    HBM_embedding14Ext.obj = HBM_embedding14.data();
    HBM_embedding14Ext.param = 0;
    HBM_embedding14Ext.flags = bank[14];
    HBM_embedding15Ext.obj = HBM_embedding15.data();
    HBM_embedding15Ext.param = 0;
    HBM_embedding15Ext.flags = bank[15];
    HBM_embedding16Ext.obj = HBM_embedding16.data();
    HBM_embedding16Ext.param = 0;
    HBM_embedding16Ext.flags = bank[16];
    HBM_embedding17Ext.obj = HBM_embedding17.data();
    HBM_embedding17Ext.param = 0;
    HBM_embedding17Ext.flags = bank[17];
    HBM_embedding18Ext.obj = HBM_embedding18.data();
    HBM_embedding18Ext.param = 0;
    HBM_embedding18Ext.flags = bank[18];
    HBM_embedding19Ext.obj = HBM_embedding19.data();
    HBM_embedding19Ext.param = 0;
    HBM_embedding19Ext.flags = bank[19];
    HBM_embedding20Ext.obj = HBM_embedding20.data();
    HBM_embedding20Ext.param = 0;
    HBM_embedding20Ext.flags = bank[20];
    HBM_embedding21Ext.obj = HBM_embedding21.data();
    HBM_embedding21Ext.param = 0;
    HBM_embedding21Ext.flags = bank[21];
    HBM_embedding22Ext.obj = HBM_embedding22.data();
    HBM_embedding22Ext.param = 0;
    HBM_embedding22Ext.flags = bank[22];
    HBM_embedding23Ext.obj = HBM_embedding23.data();
    HBM_embedding23Ext.param = 0;
    HBM_embedding23Ext.flags = bank[23];
    HBM_embedding24Ext.obj = HBM_embedding24.data();
    HBM_embedding24Ext.param = 0;
    HBM_embedding24Ext.flags = bank[24];
    HBM_embedding25Ext.obj = HBM_embedding25.data();
    HBM_embedding25Ext.param = 0;
    HBM_embedding25Ext.flags = bank[25];
    HBM_embedding26Ext.obj = HBM_embedding26.data();
    HBM_embedding26Ext.param = 0;
    HBM_embedding26Ext.flags = bank[26];
    HBM_embedding27Ext.obj = HBM_embedding27.data();
    HBM_embedding27Ext.param = 0;
    HBM_embedding27Ext.flags = bank[27];
    // HBM_embedding28Ext.obj = HBM_embedding28.data();
    // HBM_embedding28Ext.param = 0;
    // HBM_embedding28Ext.flags = bank[28];
    // HBM_embedding29Ext.obj = HBM_embedding29.data();
    // HBM_embedding29Ext.param = 0;
    // HBM_embedding29Ext.flags = bank[29];
    // HBM_embedding30Ext.obj = HBM_embedding30.data();
    // HBM_embedding30Ext.param = 0;
    // HBM_embedding30Ext.flags = bank[30];
    // HBM_embedding31Ext.obj = HBM_embedding31.data();
    // HBM_embedding31Ext.param = 0;
    // HBM_embedding31Ext.flags = bank[31];

    DDR_embedding0Ext.obj = DDR_embedding0.data();
    DDR_embedding0Ext.param = 0;
    DDR_embedding0Ext.flags = bank[0 + 32];
    DDR_embedding1Ext.obj = DDR_embedding1.data();
    DDR_embedding1Ext.param = 0;
    DDR_embedding1Ext.flags = bank[1 + 32];

//     PLRAM_embedding0Ext.obj = PLRAM_embedding0.data();
//     PLRAM_embedding0Ext.param = 0;
//     PLRAM_embedding0Ext.flags = bank[32];
//     PLRAM_embedding1Ext.obj = PLRAM_embedding1.data();
//     PLRAM_embedding1Ext.param = 0;
//     PLRAM_embedding1Ext.flags = bank[32];
//     PLRAM_embedding2Ext.obj = PLRAM_embedding2.data();
//     PLRAM_embedding2Ext.param = 0;
//     PLRAM_embedding2Ext.flags = bank[32];
//     PLRAM_embedding3Ext.obj = PLRAM_embedding3.data();
//     PLRAM_embedding3Ext.param = 0;
//     PLRAM_embedding3Ext.flags = bank[32];
//////////////////////////////   TEMPLATE END  //////////////////////////////
//     access_idxExt.obj = access_idx.data();
//     access_idxExt.param = 0;
//     access_idxExt.flags = bank[32];

    // result in PLRAM[0]
    source_hw_resultsExt.obj = source_hw_results.data();
    source_hw_resultsExt.param = 0;
    source_hw_resultsExt.flags = bank[34];

    // CL_MEM_EXT_PTR_XILINX
//////////////////////////////   TEMPLATE START  //////////////////////////////
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding0_size *sizeof(axi_t), &HBM_embedding0Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding1_size *sizeof(axi_t), &HBM_embedding1Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding2_size *sizeof(axi_t), &HBM_embedding2Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding3_size *sizeof(axi_t), &HBM_embedding3Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding4_size *sizeof(axi_t), &HBM_embedding4Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding5_size *sizeof(axi_t), &HBM_embedding5Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding6_size *sizeof(axi_t), &HBM_embedding6Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding7(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding7_size *sizeof(axi_t), &HBM_embedding7Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding8(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding8_size *sizeof(axi_t), &HBM_embedding8Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding9(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding9_size *sizeof(axi_t), &HBM_embedding9Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding10(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding10_size *sizeof(axi_t), &HBM_embedding10Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding11(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding11_size *sizeof(axi_t), &HBM_embedding11Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding12(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding12_size *sizeof(axi_t), &HBM_embedding12Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding13(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding13_size *sizeof(axi_t), &HBM_embedding13Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding14(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding14_size *sizeof(axi_t), &HBM_embedding14Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding15(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding15_size *sizeof(axi_t), &HBM_embedding15Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding16(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding16_size *sizeof(axi_t), &HBM_embedding16Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding17(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding17_size *sizeof(axi_t), &HBM_embedding17Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding18(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding18_size *sizeof(axi_t), &HBM_embedding18Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding19(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding19_size *sizeof(axi_t), &HBM_embedding19Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding20(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding20_size *sizeof(axi_t), &HBM_embedding20Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding21(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding21_size *sizeof(axi_t), &HBM_embedding21Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding22(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding22_size *sizeof(axi_t), &HBM_embedding22Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding23(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding23_size *sizeof(axi_t), &HBM_embedding23Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding24(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding24_size *sizeof(axi_t), &HBM_embedding24Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding25(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding25_size *sizeof(axi_t), &HBM_embedding25Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding26(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding26_size *sizeof(axi_t), &HBM_embedding26Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding27(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding27_size *sizeof(axi_t), &HBM_embedding27Ext, &err));
    // OCL_CHECK(err, cl::Buffer buffer_HBM_embedding28(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
    //         HBM_embedding28_size *sizeof(axi_t), &HBM_embedding28Ext, &err));
    // OCL_CHECK(err, cl::Buffer buffer_HBM_embedding29(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
    //         HBM_embedding29_size *sizeof(axi_t), &HBM_embedding29Ext, &err));
    // OCL_CHECK(err, cl::Buffer buffer_HBM_embedding30(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
    //         HBM_embedding30_size *sizeof(axi_t), &HBM_embedding30Ext, &err));
    // OCL_CHECK(err, cl::Buffer buffer_HBM_embedding31(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
    //         HBM_embedding31_size *sizeof(axi_t), &HBM_embedding31Ext, &err));

    OCL_CHECK(err, cl::Buffer buffer_DDR_embedding0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            DDR_embedding0_size *sizeof(axi_t), &DDR_embedding0Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_DDR_embedding1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            DDR_embedding1_size *sizeof(axi_t), &DDR_embedding1Ext, &err));

//     OCL_CHECK(err, cl::Buffer buffer_PLRAM_embedding0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
//             PLRAM_embedding0_size *sizeof(axi_t), &PLRAM_embedding0Ext, &err));
//     OCL_CHECK(err, cl::Buffer buffer_PLRAM_embedding1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
//             PLRAM_embedding1_size *sizeof(axi_t), &PLRAM_embedding1Ext, &err));
//     OCL_CHECK(err, cl::Buffer buffer_PLRAM_embedding2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
//             PLRAM_embedding2_size *sizeof(axi_t), &PLRAM_embedding2Ext, &err));
//     OCL_CHECK(err, cl::Buffer buffer_PLRAM_embedding3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
//             PLRAM_embedding3_size *sizeof(axi_t), &PLRAM_embedding3Ext, &err));

// .......................................................
// Allocate Global Memory for sourcce_hw_results
// .......................................................
    // OCL_CHECK(err, cl::Buffer buffer_output(
    //     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, 
    //     size_results_out *sizeof(D_TYPE), &sourcce_hw_resultsExt, &err));

// ============================================================================
// Step 2: Set Kernel Arguments and Run the Application
//         o) Set Kernel Arguments
//         o) Copy Input Data from Host to Global Memory on the device
//         o) Submit Kernels for Execution
//         o) Copy Results from Global Memory, device to Host
// ============================================================================ 
    

    


//////////////////////////////   TEMPLATE START  //////////////////////////////
    OCL_CHECK(err, err = user_kernel.setArg(0, buffer_HBM_embedding0));
    OCL_CHECK(err, err = user_kernel.setArg(1, buffer_HBM_embedding1));
    OCL_CHECK(err, err = user_kernel.setArg(2, buffer_HBM_embedding2));
    OCL_CHECK(err, err = user_kernel.setArg(3, buffer_HBM_embedding3));
    OCL_CHECK(err, err = user_kernel.setArg(4, buffer_HBM_embedding4));
    OCL_CHECK(err, err = user_kernel.setArg(5, buffer_HBM_embedding5));
    OCL_CHECK(err, err = user_kernel.setArg(6, buffer_HBM_embedding6));
    OCL_CHECK(err, err = user_kernel.setArg(7, buffer_HBM_embedding7));
    OCL_CHECK(err, err = user_kernel.setArg(8, buffer_HBM_embedding8));
    OCL_CHECK(err, err = user_kernel.setArg(9, buffer_HBM_embedding9));
    OCL_CHECK(err, err = user_kernel.setArg(10, buffer_HBM_embedding10));
    OCL_CHECK(err, err = user_kernel.setArg(11, buffer_HBM_embedding11));
    OCL_CHECK(err, err = user_kernel.setArg(12, buffer_HBM_embedding12));
    OCL_CHECK(err, err = user_kernel.setArg(13, buffer_HBM_embedding13));
    OCL_CHECK(err, err = user_kernel.setArg(14, buffer_HBM_embedding14));
    OCL_CHECK(err, err = user_kernel.setArg(15, buffer_HBM_embedding15));
    OCL_CHECK(err, err = user_kernel.setArg(16, buffer_HBM_embedding16));
    OCL_CHECK(err, err = user_kernel.setArg(17, buffer_HBM_embedding17));
    OCL_CHECK(err, err = user_kernel.setArg(18, buffer_HBM_embedding18));
    OCL_CHECK(err, err = user_kernel.setArg(19, buffer_HBM_embedding19));
    OCL_CHECK(err, err = user_kernel.setArg(20, buffer_HBM_embedding20));
    OCL_CHECK(err, err = user_kernel.setArg(21, buffer_HBM_embedding21));
    OCL_CHECK(err, err = user_kernel.setArg(22, buffer_HBM_embedding22));
    OCL_CHECK(err, err = user_kernel.setArg(23, buffer_HBM_embedding23));
    OCL_CHECK(err, err = user_kernel.setArg(24, buffer_HBM_embedding24));
    OCL_CHECK(err, err = user_kernel.setArg(25, buffer_HBM_embedding25));
    OCL_CHECK(err, err = user_kernel.setArg(26, buffer_HBM_embedding26));
    OCL_CHECK(err, err = user_kernel.setArg(27, buffer_HBM_embedding27));
    // OCL_CHECK(err, err = user_kernel.setArg(28, buffer_HBM_embedding28));
    // OCL_CHECK(err, err = user_kernel.setArg(29, buffer_HBM_embedding29));
    // OCL_CHECK(err, err = user_kernel.setArg(30, buffer_HBM_embedding30));
    // OCL_CHECK(err, err = user_kernel.setArg(31, buffer_HBM_embedding31));

    OCL_CHECK(err, err = user_kernel.setArg(28, buffer_DDR_embedding0));
    OCL_CHECK(err, err = user_kernel.setArg(29, buffer_DDR_embedding1));

    OCL_CHECK(err, err = user_kernel.setArg(46, useConn));
    OCL_CHECK(err, err = user_kernel.setArg(47, pkgWordCount));
    OCL_CHECK(err, err = user_kernel.setArg(48, basePort));
    OCL_CHECK(err, err = user_kernel.setArg(49, baseIpAddr));
    OCL_CHECK(err, err = user_kernel.setArg(50, batch_num));
    // OCL_CHECK(err, err = user_kernel.setArg(34, buffer_output));
//////////////////////////////   TEMPLATE END  //////////////////////////////
// ------------------------------------------------------
// Step 2: Copy Input data from Host to Global Memory on the device
// ------------------------------------------------------
//////////////////////////////   TEMPLATE START  //////////////////////////////
    std::cout << "Starting copy from Host to device..." << std::endl;
    OCL_CHECK(
        err, err = q.enqueueMigrateMemObjects({
        buffer_HBM_embedding0, buffer_HBM_embedding1, buffer_HBM_embedding2, buffer_HBM_embedding3, 
        buffer_HBM_embedding4, buffer_HBM_embedding5, buffer_HBM_embedding6, buffer_HBM_embedding7, 
        buffer_HBM_embedding8, buffer_HBM_embedding9, buffer_HBM_embedding10, buffer_HBM_embedding11, 
        buffer_HBM_embedding12, buffer_HBM_embedding13, buffer_HBM_embedding14, buffer_HBM_embedding15, 
        buffer_HBM_embedding16, buffer_HBM_embedding17, buffer_HBM_embedding18, buffer_HBM_embedding19, 
        buffer_HBM_embedding20, buffer_HBM_embedding21, buffer_HBM_embedding22, buffer_HBM_embedding23, 
        buffer_HBM_embedding24, buffer_HBM_embedding25, buffer_HBM_embedding26, buffer_HBM_embedding27, 
        // buffer_HBM_embedding28, buffer_HBM_embedding29, 
        buffer_DDR_embedding0, buffer_DDR_embedding1}, 0/* 0 means from host*/));   
    std::cout << "Host to device finished..." << std::endl;
//////////////////////////////   TEMPLATE END  //////////////////////////////
// ----------------------------------------
// Step 2: Submit Kernels for Execution
// ----------------------------------------
    OCL_CHECK(err, err = q.enqueueTask(user_kernel));
// --------------------------------------------------
// Step 2: Copy Results from Device Global Memory to Host
// --------------------------------------------------
    // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));

    q.finish();
// OPENCL HOST CODE AREA END


// ============================================================================
// Step 3: Release Allocated Resources
// ============================================================================
    // delete[] fileBuf;

    // std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
    // return (match ? EXIT_SUCCESS : EXIT_FAILURE);
    

    std::cout << "EXIT recorded" << std::endl;
}
