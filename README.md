# GPU-FPGA-Recommendation-System
The source code of our paper published in KDD 2021--- [FleetRec: Large-Scale Recommendation Inference on Hybrid GPU-FPGA Clusters](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/485153/1/FleetRec_camera_ready.pdf).



There are two folders for the FPGA and GPU implementations. To build the FPGA, refer to the README.md in the folder. The supported device is Alveo U280 and we used an [open-source TCP/IP stack for Vitis](https://github.com/fpgasystems/Vitis_with_100Gbps_TCP-IP). The GPU implementation requires CUDA version of at least 11.0 and should support a wide range of GPU models.



There are three experiments on different recommendation models. The FPGA kernels can be found [here](./FPGA/kernel/user_krnl), and the GPU kernels can be found [here](./GPU). There are respective READMEs in those FPGA and GPU folders.



## Reference

The paper corresponds to this repository:

Jiang, W., He, Z., Zhang, S., Zeng, K., Feng, L., Zhang, J., ... & Alonso, G. (2021, August). FleetRec: Large-Scale Recommendation Inference on Hybrid GPU-FPGA Clusters. In *27th SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2021)*.

The FPGA implementation is based on a previous paper:

Jiang, W., He, Z., Zhang, S., Preußer, T. B., Zeng, K., Feng, L., ... & Alonso, G. (2021). MicroRec: efficient recommendation inference by hardware and data structure solutions. *Proceedings of Machine Learning and Systems*, *3*.

The FPGA network stack we used:

Zhenhao He, Dario Korolija, and Gustavo Alonso. 2021. EasyNet: 100 Gbps Network for HLS. In 2021 31th International Conference on Field Programmable Logic and Applications (FPL)