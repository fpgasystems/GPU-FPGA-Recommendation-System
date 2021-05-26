### Vitis with Network Stack

Adding the network stack to the Vitis shell.

## Setup
Git Clone 

	git clone	
	git submodule update --init --recursive

Setup the HLS IPs:

    mkdir build
    cd build
    cmake .. -DFDEV_NAME=u280 -DTCP_STACK_EN=1 -DTCP_STACK_RX_DDR_BYPASS_EN=1 
    make installip

Create the Vitis kernel:

    make all DEVICE=/opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm USER_KRNL=scatter_krnl NETH=4
