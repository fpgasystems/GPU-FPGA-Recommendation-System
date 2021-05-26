# $(TEMP_DIR)/${KRNL_NAME_2}.xo: kernel/user_krnl/src/C/*.cpp
# 	mkdir -p $(TEMP_DIR)
# 	$(VPP) $(CLFLAGS) -c -k ${KRNL_NAME_2} -o $(TEMP_DIR)/${KRNL_NAME_2}.xo --input_files kernel/user_krnl/src/C/*.cpp

# VIVADO := $(XILINX_VIVADO)/bin/vivado
# $(TEMP_DIR)/${KRNL_NAME_1}.xo: kernel/network_krnl.xml scripts/package_network_krnl.tcl scripts/gen_xo.tcl kernel/network_krnl/src/hdl/*.sv
# 	mkdir -p $(TEMP_DIR)
# 	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(TEMP_DIR)/${KRNL_NAME_1}.xo ${KRNL_NAME_1} $(TARGET) $(DEVICE) $(XSA)

# # $(TEMP_DIR)/${KRNL_NAME_2}.xo: kernel/user_krnl.xml scripts/package_user_krnl.tcl scripts/gen_xo.tcl kernel/user_krnl/src/hdl/*.sv
# # 	mkdir -p $(TEMP_DIR)
# # 	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(TEMP_DIR)/${KRNL_NAME_2}.xo ${KRNL_NAME_2} $(TARGET) $(DEVICE) $(XSA)


# $(TEMP_DIR)/${KRNL_NAME_3}.xo: kernel/cmac_krnl.xml scripts/package_cmac_krnl.tcl scripts/gen_xo.tcl kernel/cmac_krnl/src/hdl/*.sv
# 	mkdir -p $(TEMP_DIR)
# 	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(TEMP_DIR)/${KRNL_NAME_3}.xo ${KRNL_NAME_3} $(TARGET) $(DEVICE) $(XSA)


VIVADO := $(XILINX_VIVADO)/bin/vivado
$(TEMP_DIR)/${KRNL_1}.xo: kernel/network_krnl/network_krnl.xml scripts/package_network_krnl.tcl scripts/gen_xo.tcl kernel/network_krnl/src/hdl/*.sv
	mkdir -p $(TEMP_DIR)
	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(TEMP_DIR)/${KRNL_1}.xo ${KRNL_1} $(TARGET) $(DEVICE) $(XSA) kernel/network_krnl/network_krnl.xml ./scripts/package_network_krnl.tcl

$(TEMP_DIR)/${KRNL_2}.xo: kernel/user_krnl/${KRNL_2}/src/hls/*.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(CLFLAGS) -c -k ${KRNL_2} -o $(TEMP_DIR)/${KRNL_2}.xo --input_files kernel/user_krnl/${KRNL_2}/src/hls/*.cpp


$(TEMP_DIR)/${KRNL_3}.xo: kernel/cmac_krnl/cmac_krnl.xml scripts/package_cmac_krnl.tcl scripts/gen_xo.tcl kernel/cmac_krnl/src/hdl/*.sv
	mkdir -p $(TEMP_DIR)
	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(TEMP_DIR)/${KRNL_3}.xo ${KRNL_3} $(TARGET) $(DEVICE) $(XSA) kernel/cmac_krnl/cmac_krnl.xml ./scripts/package_cmac_krnl.tcl
