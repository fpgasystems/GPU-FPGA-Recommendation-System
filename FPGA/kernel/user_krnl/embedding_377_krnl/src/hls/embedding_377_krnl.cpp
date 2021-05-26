#include "embedding_377_krnl.hpp"

#include "ap_axi_sdata.h"
#include <ap_fixed.h>
#include "ap_int.h" 
#include "hls_stream.h"

#include "axi_utils.hpp"
#include "toe.hpp"



extern "C" {
void openConnections(int useConn, int baseIpAddress, int basePort, hls::stream<pkt64>& m_axis_tcp_open_connection, hls::stream<pkt32>& s_axis_tcp_open_status, ap_uint<16>* sessionID)
{
#pragma HLS dataflow

     int numOpenedCon = 0;
     pkt64 openConnection_pkt;
     for (int i = 0; i < useConn; ++i)
     {
     #pragma HLS PIPELINE II=1
          openConnection_pkt.data(31,0) = baseIpAddress;
          openConnection_pkt.data(47,32) = basePort+i;
          m_axis_tcp_open_connection.write(openConnection_pkt);
     }

     openStatus status;
     for (int i = 0; i < useConn; ++i)
     {
     #pragma HLS PIPELINE II=1
          pkt32 open_status_pkt = s_axis_tcp_open_status.read();
          status.sessionID = open_status_pkt.data(15,0);
          status.success = open_status_pkt.data(16,16);

          if (status.success)
          {
               sessionID[numOpenedCon] = status.sessionID;
               numOpenedCon++;
               std::cout << "Connection successfully opened." << std::endl;
          }
     }
}

void sendData(hls::stream<pkt32>& m_axis_tcp_tx_meta, 
               hls::stream<pkt512>& m_axis_tcp_tx_data, 
               hls::stream<pkt64>& s_axis_tcp_tx_status,
               hls::stream<ap_uint<512> >& s_data_in,
               ap_uint<16>* sessionID,
               int useConn,
               ap_uint<64> expectedTxByteCnt, 
               int pkgWordCount
                )
{
     bool first_round = true;
     ap_uint<64> sentByteCnt = 0;
     ap_uint<64> currentPkgWordCnt = 0;
     int currentPkgSessionIndex = 0;

     do{
          pkt32 tx_meta_pkt;
          appTxRsp resp;

          if (first_round)
          {
               
               for (int i = 0; i < useConn; ++i)
               {
               #pragma HLS PIPELINE II=1
                    tx_meta_pkt.data(15,0) = sessionID[i];
                    tx_meta_pkt.data(31,16) = pkgWordCount*(512/8);
                    m_axis_tcp_tx_meta.write(tx_meta_pkt);
               }
               first_round = false;
          }
          else
          {
               if (!s_axis_tcp_tx_status.empty())
               {
                    pkt64 txStatus_pkt = s_axis_tcp_tx_status.read();
                    resp.sessionID = txStatus_pkt.data(15,0);
                    resp.length = txStatus_pkt.data(31,16);
                    resp.remaining_space = txStatus_pkt.data(61,32);
                    resp.error = txStatus_pkt.data(63,62);

                    if (resp.error == 0)
                    {
                         sentByteCnt = sentByteCnt + resp.length;

                         currentPkgSessionIndex++;
                         if (currentPkgSessionIndex == useConn)
                         {
                              currentPkgSessionIndex = 0;
                         }

                         if (sentByteCnt < expectedTxByteCnt)
                         {
                              tx_meta_pkt.data(15,0) = resp.sessionID;
                              if (sentByteCnt + pkgWordCount*64 < expectedTxByteCnt )
                              {
                                  tx_meta_pkt.data(31,16) = pkgWordCount*(512/8);
                                  currentPkgWordCnt = pkgWordCount;
                              }
                              else
                              {
                                  tx_meta_pkt.data(31,16) = expectedTxByteCnt - sentByteCnt;
                                  currentPkgWordCnt = (expectedTxByteCnt - sentByteCnt)>>6;
                              }
                              
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                         }
                         
                         for (int j = 0; j < currentPkgWordCnt; ++j)
                         {
                         #pragma HLS PIPELINE II=1
                              ap_uint<512> s_data = s_data_in.read();
                              pkt512 currWord;
                              for (int i = 0; i < (512/64); i++) 
                              {
                                   #pragma HLS UNROLL
                                   currWord.data(i*64+63, i*64) = s_data(i*64+63, i*64);
                                   currWord.keep(i*8+7, i*8) = 0xff;
                              }
                              currWord.last = (j == currentPkgWordCnt-1);
                              m_axis_tcp_tx_data.write(currWord);
                         }
                    }
                    else
                    {
                         //Check if connection  was torn down
                         if (resp.error == 1)
                         {
                              std::cout << "Connection was torn down. " << resp.sessionID << std::endl;
                         }
                         else
                         {
                              tx_meta_pkt.data(15,0) = resp.sessionID;
                              tx_meta_pkt.data(31,16) = resp.length;
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                         }
                    }
               }
          }
          
     }
     while(sentByteCnt<expectedTxByteCnt);
}




void embedding_377_krnl(  
    const axi_t* table_HBM0, const axi_t* table_HBM1, 
    const axi_t* table_HBM2, const axi_t* table_HBM3, 
    const axi_t* table_HBM4, const axi_t* table_HBM5, 
    const axi_t* table_HBM6, const axi_t* table_HBM7, 
    const axi_t* table_HBM8, const axi_t* table_HBM9, 
    const axi_t* table_HBM10, const axi_t* table_HBM11, 
    const axi_t* table_HBM12, const axi_t* table_HBM13, 
    const axi_t* table_HBM14, const axi_t* table_HBM15, 
    const axi_t* table_HBM16, const axi_t* table_HBM17, 
    const axi_t* table_HBM18, const axi_t* table_HBM19, 
    const axi_t* table_HBM20, const axi_t* table_HBM21, 
    const axi_t* table_HBM22, const axi_t* table_HBM23, 
    const axi_t* table_HBM24, const axi_t* table_HBM25, 
    const axi_t* table_HBM26, const axi_t* table_HBM27, 
    const axi_t* table_DDR0, const axi_t* table_DDR1,

    // Internal Stream
    hls::stream<pkt512>& s_axis_udp_rx, 
    hls::stream<pkt512>& m_axis_udp_tx, 
    hls::stream<pkt256>& s_axis_udp_rx_meta, 
    hls::stream<pkt256>& m_axis_udp_tx_meta, 

    hls::stream<pkt16>& m_axis_tcp_listen_port, 
    hls::stream<pkt8>& s_axis_tcp_port_status, 
    hls::stream<pkt64>& m_axis_tcp_open_connection, 
    hls::stream<pkt32>& s_axis_tcp_open_status, 
    hls::stream<pkt16>& m_axis_tcp_close_connection, 
    hls::stream<pkt128>& s_axis_tcp_notification, 
    hls::stream<pkt32>& m_axis_tcp_read_pkg, 
    hls::stream<pkt16>& s_axis_tcp_rx_meta, 
    hls::stream<pkt512>& s_axis_tcp_rx_data, 
    hls::stream<pkt32>& m_axis_tcp_tx_meta, 
    hls::stream<pkt512>& m_axis_tcp_tx_data, 
    hls::stream<pkt64>& s_axis_tcp_tx_status,

    int useConn, 
    int pkgWordCount, 
    int basePort, 
    int baseIpAddress,
    int batch_num   )
{
    
#pragma HLS DATAFLOW

#pragma HLS INTERFACE m_axi port=table_HBM0  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table_HBM1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=table_HBM2  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=table_HBM3  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=table_HBM4  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=table_HBM5  offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=table_HBM6  offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=table_HBM7  offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=table_HBM8  offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=table_HBM9  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=table_HBM10  offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=table_HBM11  offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=table_HBM12  offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=table_HBM13  offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=table_HBM14  offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=table_HBM15  offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=table_HBM16  offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=table_HBM17  offset=slave bundle=gmem17
#pragma HLS INTERFACE m_axi port=table_HBM18  offset=slave bundle=gmem18
#pragma HLS INTERFACE m_axi port=table_HBM19  offset=slave bundle=gmem19
#pragma HLS INTERFACE m_axi port=table_HBM20  offset=slave bundle=gmem20
#pragma HLS INTERFACE m_axi port=table_HBM21  offset=slave bundle=gmem21
#pragma HLS INTERFACE m_axi port=table_HBM22  offset=slave bundle=gmem22
#pragma HLS INTERFACE m_axi port=table_HBM23  offset=slave bundle=gmem23
#pragma HLS INTERFACE m_axi port=table_HBM24  offset=slave bundle=gmem24
#pragma HLS INTERFACE m_axi port=table_HBM25  offset=slave bundle=gmem25
#pragma HLS INTERFACE m_axi port=table_HBM26  offset=slave bundle=gmem26
#pragma HLS INTERFACE m_axi port=table_HBM27  offset=slave bundle=gmem27

#pragma HLS INTERFACE m_axi port=table_DDR0  offset=slave bundle=gmem32
#pragma HLS INTERFACE m_axi port=table_DDR1  offset=slave bundle=gmem33

#pragma HLS INTERFACE s_axilite port=table_HBM0  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM1  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM2  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM3  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM4  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM5  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM6  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM7  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM8  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM9  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM10  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM11  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM12  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM13  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM14  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM15  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM16  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM17  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM18  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM19  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM20  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM21  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM22  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM23  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM24  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM25  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM26  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM27  bundle=control

#pragma HLS INTERFACE s_axilite port=table_DDR0  bundle=control
#pragma HLS INTERFACE s_axilite port=table_DDR1  bundle=control

#pragma HLS INTERFACE axis port = s_axis_udp_rx
#pragma HLS INTERFACE axis port = m_axis_udp_tx
#pragma HLS INTERFACE axis port = s_axis_udp_rx_meta
#pragma HLS INTERFACE axis port = m_axis_udp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_listen_port
#pragma HLS INTERFACE axis port = s_axis_tcp_port_status
#pragma HLS INTERFACE axis port = m_axis_tcp_open_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_open_status
#pragma HLS INTERFACE axis port = m_axis_tcp_close_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_notification
#pragma HLS INTERFACE axis port = m_axis_tcp_read_pkg
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_meta
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_data
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_data
#pragma HLS INTERFACE axis port = s_axis_tcp_tx_status

#pragma HLS INTERFACE s_axilite port=useConn bundle = control
#pragma HLS INTERFACE s_axilite port=pkgWordCount bundle = control
#pragma HLS INTERFACE s_axilite port=basePort bundle = control
#pragma HLS INTERFACE s_axilite port=baseIpAddress bundle=control
#pragma HLS INTERFACE s_axilite port=batch_num bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
    
hls::stream<ap_uint<16> > listenPort;
hls::stream<bool> listenPortStatus;
hls::stream<appNotification> notifications;
hls::stream<appReadRequest> readRequest;
hls::stream<ap_uint<16> > rxMetaData;
hls::stream<net_axis<DATA_WIDTH> > rxData;
hls::stream<ap_uint<16> > closeConnection;
hls::stream<ipTuple> openConnection;
hls::stream<openStatus> openConStatus;
hls::stream<appTxMeta> txMetaData;
hls::stream<net_axis<DATA_WIDTH> > txData;
hls::stream<appTxRsp> txStatus;

#pragma HLS stream variable=openConStatus depth=128
#pragma HLS stream variable=txStatus depth=128
#pragma HLS stream variable=openConnection depth=128
#pragma HLS stream variable=txMetaData depth=128
#pragma HLS stream variable=txData depth=128



ap_uint<16> sessionID [16];
          
    openConnections( useConn,  baseIpAddress,  basePort, m_axis_tcp_open_connection, s_axis_tcp_open_status, sessionID);



    hls::stream<hbm_t> s_embedding_buffer_HBM0;
    hls::stream<hbm_t> s_embedding_buffer_HBM1;
    hls::stream<hbm_t> s_embedding_buffer_HBM2;
    hls::stream<hbm_t> s_embedding_buffer_HBM3;
    hls::stream<hbm_t> s_embedding_buffer_HBM4;
    hls::stream<hbm_t> s_embedding_buffer_HBM5;
    hls::stream<hbm_t> s_embedding_buffer_HBM6;
    hls::stream<hbm_t> s_embedding_buffer_HBM7;
    hls::stream<hbm_t> s_embedding_buffer_HBM8;
    hls::stream<hbm_t> s_embedding_buffer_HBM9;
    hls::stream<hbm_t> s_embedding_buffer_HBM10;
    hls::stream<hbm_t> s_embedding_buffer_HBM11;
    hls::stream<hbm_t> s_embedding_buffer_HBM12;
    hls::stream<hbm_t> s_embedding_buffer_HBM13;
    hls::stream<hbm_t> s_embedding_buffer_HBM14;
    hls::stream<hbm_t> s_embedding_buffer_HBM15;
    hls::stream<hbm_t> s_embedding_buffer_HBM16;
    hls::stream<hbm_t> s_embedding_buffer_HBM17;
    hls::stream<hbm_t> s_embedding_buffer_HBM18;
    hls::stream<hbm_t> s_embedding_buffer_HBM19;
    hls::stream<hbm_t> s_embedding_buffer_HBM20;
    hls::stream<hbm_t> s_embedding_buffer_HBM21;
    hls::stream<hbm_t> s_embedding_buffer_HBM22;
    hls::stream<hbm_t> s_embedding_buffer_HBM23;
    hls::stream<hbm_t> s_embedding_buffer_HBM24;
    hls::stream<hbm_t> s_embedding_buffer_HBM25;
    hls::stream<hbm_t> s_embedding_buffer_HBM26;
    hls::stream<hbm_t> s_embedding_buffer_HBM27;
#pragma HLS stream variable=s_embedding_buffer_HBM0 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM1 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM2 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM3 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM4 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM5 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM6 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM7 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM8 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM9 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM10 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM11 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM12 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM13 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM14 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM15 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM16 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM17 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM18 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM19 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM20 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM21 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM22 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM23 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM24 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM25 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM26 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM27 depth=8

#pragma HLS resource variable=s_embedding_buffer_HBM0 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM1 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM2 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM3 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM4 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM5 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM6 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM7 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM8 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM9 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM10 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM11 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM12 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM13 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM14 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM15 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM16 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM17 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM18 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM19 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM20 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM21 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM22 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM23 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM24 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM25 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM26 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM27 core=FIFO_SRL


    hls::stream<plram_t> s_embedding_buffer_PLRAM0;
    hls::stream<plram_t> s_embedding_buffer_PLRAM1;
    hls::stream<plram_t> s_embedding_buffer_PLRAM2;
    hls::stream<plram_t> s_embedding_buffer_PLRAM3;
    hls::stream<plram_t> s_embedding_buffer_PLRAM4;
    hls::stream<plram_t> s_embedding_buffer_PLRAM5;
    hls::stream<plram_t> s_embedding_buffer_PLRAM6;
    hls::stream<plram_t> s_embedding_buffer_PLRAM7;
    hls::stream<plram_t> s_embedding_buffer_PLRAM8;
    hls::stream<plram_t> s_embedding_buffer_PLRAM9;
    hls::stream<plram_t> s_embedding_buffer_PLRAM10;
#pragma HLS stream variable=s_embedding_buffer_PLRAM0 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM1 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM2 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM3 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM4 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM5 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM6 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM7 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM8 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM9 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM10 depth=8

#pragma HLS resource variable=s_embedding_buffer_PLRAM0 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM1 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM2 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM3 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM4 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM5 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM6 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM7 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM8 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM9 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM10 core=FIFO_SRL

    hls::stream<ddr_t> s_embedding_buffer_DDR0;
    hls::stream<ddr_t> s_embedding_buffer_DDR1;
#pragma HLS stream variable=s_embedding_buffer_DDR0 depth=8
#pragma HLS stream variable=s_embedding_buffer_DDR1 depth=8

#pragma HLS resource variable=s_embedding_buffer_DDR0 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_DDR1 core=FIFO_SRL

    hls::stream<int> s_idx_buffer_HBM0;
    hls::stream<int> s_idx_buffer_HBM1;
    hls::stream<int> s_idx_buffer_HBM2;
    hls::stream<int> s_idx_buffer_HBM3;
    hls::stream<int> s_idx_buffer_HBM4;
    hls::stream<int> s_idx_buffer_HBM5;
    hls::stream<int> s_idx_buffer_HBM6;
    hls::stream<int> s_idx_buffer_HBM7;
    hls::stream<int> s_idx_buffer_HBM8;
    hls::stream<int> s_idx_buffer_HBM9;
    hls::stream<int> s_idx_buffer_HBM10;
    hls::stream<int> s_idx_buffer_HBM11;
    hls::stream<int> s_idx_buffer_HBM12;
    hls::stream<int> s_idx_buffer_HBM13;
    hls::stream<int> s_idx_buffer_HBM14;
    hls::stream<int> s_idx_buffer_HBM15;
    hls::stream<int> s_idx_buffer_HBM16;
    hls::stream<int> s_idx_buffer_HBM17;
    hls::stream<int> s_idx_buffer_HBM18;
    hls::stream<int> s_idx_buffer_HBM19;
    hls::stream<int> s_idx_buffer_HBM20;
    hls::stream<int> s_idx_buffer_HBM21;
    hls::stream<int> s_idx_buffer_HBM22;
    hls::stream<int> s_idx_buffer_HBM23;
    hls::stream<int> s_idx_buffer_HBM24;
    hls::stream<int> s_idx_buffer_HBM25;
    hls::stream<int> s_idx_buffer_HBM26;
    hls::stream<int> s_idx_buffer_HBM27;
#pragma HLS stream variable=s_idx_buffer_HBM0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM1 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM2 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM3 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM4 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM5 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM6 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM7 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM8 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM9 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM10 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM11 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM12 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM13 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM14 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM15 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM16 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM17 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM18 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM19 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM20 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM21 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM22 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM23 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM24 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM25 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM26 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM27 depth=fifo_batch_size

    hls::stream<int> s_idx_buffer_PLRAM0;
    hls::stream<int> s_idx_buffer_PLRAM1;
    hls::stream<int> s_idx_buffer_PLRAM2;
    hls::stream<int> s_idx_buffer_PLRAM3;
    hls::stream<int> s_idx_buffer_PLRAM4;
    hls::stream<int> s_idx_buffer_PLRAM5;
    hls::stream<int> s_idx_buffer_PLRAM6;
    hls::stream<int> s_idx_buffer_PLRAM7;
    hls::stream<int> s_idx_buffer_PLRAM8;
    hls::stream<int> s_idx_buffer_PLRAM9;
    hls::stream<int> s_idx_buffer_PLRAM10;
#pragma HLS stream variable=s_idx_buffer_PLRAM0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM1 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM2 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM3 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM4 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM5 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM6 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM7 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM8 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM9 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM10 depth=fifo_batch_size

    hls::stream<int> s_idx_buffer_DDR0;
    hls::stream<int> s_idx_buffer_DDR1;
#pragma HLS stream variable=s_idx_buffer_DDR0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_DDR1 depth=fifo_batch_size

    hls::stream<network_t> s_network;
#pragma HLS stream variable=s_network depth=128
    
    plram_t table_PLRAM0[PLRAM_BANK0_SIZE]; 
    plram_t table_PLRAM1[PLRAM_BANK1_SIZE]; 
    plram_t table_PLRAM2[PLRAM_BANK2_SIZE]; 
    plram_t table_PLRAM3[PLRAM_BANK3_SIZE]; 
    plram_t table_PLRAM4[PLRAM_BANK4_SIZE]; 
    plram_t table_PLRAM5[PLRAM_BANK5_SIZE]; 
    plram_t table_PLRAM6[PLRAM_BANK6_SIZE]; 
    plram_t table_PLRAM7[PLRAM_BANK7_SIZE]; 
    plram_t table_PLRAM8[PLRAM_BANK8_SIZE]; 
    plram_t table_PLRAM9[PLRAM_BANK9_SIZE]; 
    plram_t table_PLRAM10[PLRAM_BANK10_SIZE]; 
#pragma HLS resource variable=table_PLRAM0 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM1 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM2 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM3 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM4 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM5 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM6 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM7 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM8 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM9 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM10 core=RAM_2P_URAM

    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_0, AXI_PADDED_SIZE_PLRAM_0, TABLE_SIZE_PLRAM_0,
         ADDR_AXI_PLRAM_11, AXI_PADDED_SIZE_PLRAM_11, TABLE_SIZE_PLRAM_11,
         ADDR_AXI_PLRAM_22, AXI_PADDED_SIZE_PLRAM_22, TABLE_SIZE_PLRAM_22,
         ADDR_AXI_PLRAM_33, AXI_PADDED_SIZE_PLRAM_33, TABLE_SIZE_PLRAM_33>(table_PLRAM0);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_1, AXI_PADDED_SIZE_PLRAM_1, TABLE_SIZE_PLRAM_1,
         ADDR_AXI_PLRAM_12, AXI_PADDED_SIZE_PLRAM_12, TABLE_SIZE_PLRAM_12,
         ADDR_AXI_PLRAM_23, AXI_PADDED_SIZE_PLRAM_23, TABLE_SIZE_PLRAM_23,
         ADDR_AXI_PLRAM_34, AXI_PADDED_SIZE_PLRAM_34, TABLE_SIZE_PLRAM_34>(table_PLRAM1);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_2, AXI_PADDED_SIZE_PLRAM_2, TABLE_SIZE_PLRAM_2,
         ADDR_AXI_PLRAM_13, AXI_PADDED_SIZE_PLRAM_13, TABLE_SIZE_PLRAM_13,
         ADDR_AXI_PLRAM_24, AXI_PADDED_SIZE_PLRAM_24, TABLE_SIZE_PLRAM_24,
         ADDR_AXI_PLRAM_35, AXI_PADDED_SIZE_PLRAM_35, TABLE_SIZE_PLRAM_35>(table_PLRAM2);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_3, AXI_PADDED_SIZE_PLRAM_3, TABLE_SIZE_PLRAM_3,
         ADDR_AXI_PLRAM_14, AXI_PADDED_SIZE_PLRAM_14, TABLE_SIZE_PLRAM_14,
         ADDR_AXI_PLRAM_25, AXI_PADDED_SIZE_PLRAM_25, TABLE_SIZE_PLRAM_25,
         ADDR_AXI_PLRAM_36, AXI_PADDED_SIZE_PLRAM_36, TABLE_SIZE_PLRAM_36>(table_PLRAM3);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_4, AXI_PADDED_SIZE_PLRAM_4, TABLE_SIZE_PLRAM_4,
         ADDR_AXI_PLRAM_15, AXI_PADDED_SIZE_PLRAM_15, TABLE_SIZE_PLRAM_15,
         ADDR_AXI_PLRAM_26, AXI_PADDED_SIZE_PLRAM_26, TABLE_SIZE_PLRAM_26,
         ADDR_AXI_PLRAM_37, AXI_PADDED_SIZE_PLRAM_37, TABLE_SIZE_PLRAM_37>(table_PLRAM4);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_5, AXI_PADDED_SIZE_PLRAM_5, TABLE_SIZE_PLRAM_5,
         ADDR_AXI_PLRAM_16, AXI_PADDED_SIZE_PLRAM_16, TABLE_SIZE_PLRAM_16,
         ADDR_AXI_PLRAM_27, AXI_PADDED_SIZE_PLRAM_27, TABLE_SIZE_PLRAM_27,
         ADDR_AXI_PLRAM_38, AXI_PADDED_SIZE_PLRAM_38, TABLE_SIZE_PLRAM_38>(table_PLRAM5);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_6, AXI_PADDED_SIZE_PLRAM_6, TABLE_SIZE_PLRAM_6,
         ADDR_AXI_PLRAM_17, AXI_PADDED_SIZE_PLRAM_17, TABLE_SIZE_PLRAM_17,
         ADDR_AXI_PLRAM_28, AXI_PADDED_SIZE_PLRAM_28, TABLE_SIZE_PLRAM_28,
         ADDR_AXI_PLRAM_39, AXI_PADDED_SIZE_PLRAM_39, TABLE_SIZE_PLRAM_39>(table_PLRAM6);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_7, AXI_PADDED_SIZE_PLRAM_7, TABLE_SIZE_PLRAM_7,
         ADDR_AXI_PLRAM_18, AXI_PADDED_SIZE_PLRAM_18, TABLE_SIZE_PLRAM_18,
         ADDR_AXI_PLRAM_29, AXI_PADDED_SIZE_PLRAM_29, TABLE_SIZE_PLRAM_29,
         ADDR_AXI_PLRAM_40, AXI_PADDED_SIZE_PLRAM_40, TABLE_SIZE_PLRAM_40>(table_PLRAM7);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_8, AXI_PADDED_SIZE_PLRAM_8, TABLE_SIZE_PLRAM_8,
         ADDR_AXI_PLRAM_19, AXI_PADDED_SIZE_PLRAM_19, TABLE_SIZE_PLRAM_19,
         ADDR_AXI_PLRAM_30, AXI_PADDED_SIZE_PLRAM_30, TABLE_SIZE_PLRAM_30,
         ADDR_AXI_PLRAM_41, AXI_PADDED_SIZE_PLRAM_41, TABLE_SIZE_PLRAM_41>(table_PLRAM8);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_9, AXI_PADDED_SIZE_PLRAM_9, TABLE_SIZE_PLRAM_9,
         ADDR_AXI_PLRAM_20, AXI_PADDED_SIZE_PLRAM_20, TABLE_SIZE_PLRAM_20,
         ADDR_AXI_PLRAM_31, AXI_PADDED_SIZE_PLRAM_31, TABLE_SIZE_PLRAM_31,
         ADDR_AXI_PLRAM_42, AXI_PADDED_SIZE_PLRAM_42, TABLE_SIZE_PLRAM_42>(table_PLRAM9);
    init_plram_t_4_tables
        <ADDR_AXI_PLRAM_10, AXI_PADDED_SIZE_PLRAM_10, TABLE_SIZE_PLRAM_10,
         ADDR_AXI_PLRAM_21, AXI_PADDED_SIZE_PLRAM_21, TABLE_SIZE_PLRAM_21,
         ADDR_AXI_PLRAM_32, AXI_PADDED_SIZE_PLRAM_32, TABLE_SIZE_PLRAM_32,
         ADDR_AXI_PLRAM_43, AXI_PADDED_SIZE_PLRAM_43, TABLE_SIZE_PLRAM_43>(table_PLRAM10);

    load_access_idx(s_idx_buffer_HBM0, batch_num);
    load_access_idx(s_idx_buffer_HBM1, batch_num);
    load_access_idx(s_idx_buffer_HBM2, batch_num);
    load_access_idx(s_idx_buffer_HBM3, batch_num);
    load_access_idx(s_idx_buffer_HBM4, batch_num);
    load_access_idx(s_idx_buffer_HBM5, batch_num);
    load_access_idx(s_idx_buffer_HBM6, batch_num);
    load_access_idx(s_idx_buffer_HBM7, batch_num);
    load_access_idx(s_idx_buffer_HBM8, batch_num);
    load_access_idx(s_idx_buffer_HBM9, batch_num);
    load_access_idx(s_idx_buffer_HBM10, batch_num);
    load_access_idx(s_idx_buffer_HBM11, batch_num);
    load_access_idx(s_idx_buffer_HBM12, batch_num);
    load_access_idx(s_idx_buffer_HBM13, batch_num);
    load_access_idx(s_idx_buffer_HBM14, batch_num);
    load_access_idx(s_idx_buffer_HBM15, batch_num);
    load_access_idx(s_idx_buffer_HBM16, batch_num);
    load_access_idx(s_idx_buffer_HBM17, batch_num);
    load_access_idx(s_idx_buffer_HBM18, batch_num);
    load_access_idx(s_idx_buffer_HBM19, batch_num);
    load_access_idx(s_idx_buffer_HBM20, batch_num);
    load_access_idx(s_idx_buffer_HBM21, batch_num);
    load_access_idx(s_idx_buffer_HBM22, batch_num);
    load_access_idx(s_idx_buffer_HBM23, batch_num);
    load_access_idx(s_idx_buffer_HBM24, batch_num);
    load_access_idx(s_idx_buffer_HBM25, batch_num);
    load_access_idx(s_idx_buffer_HBM26, batch_num);
    load_access_idx(s_idx_buffer_HBM27, batch_num);

    load_access_idx(s_idx_buffer_DDR0, batch_num);
    load_access_idx(s_idx_buffer_DDR1, batch_num);

    load_access_idx(s_idx_buffer_PLRAM0, batch_num);
    load_access_idx(s_idx_buffer_PLRAM1, batch_num);
    load_access_idx(s_idx_buffer_PLRAM2, batch_num);
    load_access_idx(s_idx_buffer_PLRAM3, batch_num);
    load_access_idx(s_idx_buffer_PLRAM4, batch_num);
    load_access_idx(s_idx_buffer_PLRAM5, batch_num);
    load_access_idx(s_idx_buffer_PLRAM6, batch_num);
    load_access_idx(s_idx_buffer_PLRAM7, batch_num);
    load_access_idx(s_idx_buffer_PLRAM8, batch_num);
    load_access_idx(s_idx_buffer_PLRAM9, batch_num);
    load_access_idx(s_idx_buffer_PLRAM10, batch_num);

//////////////////////////////     Load Vectors & Concatenate     ////////////////////////////// 

    load_single_embedding_5_tables<
        ADDR_AXI_HBM_0, AXI_PADDED_SIZE_HBM_0, 
        ADDR_AXI_HBM_28, AXI_PADDED_SIZE_HBM_28,
        ADDR_AXI_HBM_56, AXI_PADDED_SIZE_HBM_56,
        ADDR_AXI_HBM_84, AXI_PADDED_SIZE_HBM_84,
        ADDR_AXI_HBM_112, AXI_PADDED_SIZE_HBM_112>(
        s_idx_buffer_HBM0, table_HBM0, s_embedding_buffer_HBM0, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_1, AXI_PADDED_SIZE_HBM_1, 
        ADDR_AXI_HBM_29, AXI_PADDED_SIZE_HBM_29,
        ADDR_AXI_HBM_57, AXI_PADDED_SIZE_HBM_57,
        ADDR_AXI_HBM_85, AXI_PADDED_SIZE_HBM_85,
        ADDR_AXI_HBM_113, AXI_PADDED_SIZE_HBM_113>(
        s_idx_buffer_HBM1, table_HBM1, s_embedding_buffer_HBM1, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_2, AXI_PADDED_SIZE_HBM_2, 
        ADDR_AXI_HBM_30, AXI_PADDED_SIZE_HBM_30,
        ADDR_AXI_HBM_58, AXI_PADDED_SIZE_HBM_58,
        ADDR_AXI_HBM_86, AXI_PADDED_SIZE_HBM_86,
        ADDR_AXI_HBM_114, AXI_PADDED_SIZE_HBM_114>(
        s_idx_buffer_HBM2, table_HBM2, s_embedding_buffer_HBM2, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_3, AXI_PADDED_SIZE_HBM_3, 
        ADDR_AXI_HBM_31, AXI_PADDED_SIZE_HBM_31,
        ADDR_AXI_HBM_59, AXI_PADDED_SIZE_HBM_59,
        ADDR_AXI_HBM_87, AXI_PADDED_SIZE_HBM_87,
        ADDR_AXI_HBM_115, AXI_PADDED_SIZE_HBM_115>(
        s_idx_buffer_HBM3, table_HBM3, s_embedding_buffer_HBM3, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_4, AXI_PADDED_SIZE_HBM_4, 
        ADDR_AXI_HBM_32, AXI_PADDED_SIZE_HBM_32,
        ADDR_AXI_HBM_60, AXI_PADDED_SIZE_HBM_60,
        ADDR_AXI_HBM_88, AXI_PADDED_SIZE_HBM_88,
        ADDR_AXI_HBM_116, AXI_PADDED_SIZE_HBM_116>(
        s_idx_buffer_HBM4, table_HBM4, s_embedding_buffer_HBM4, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_5, AXI_PADDED_SIZE_HBM_5, 
        ADDR_AXI_HBM_33, AXI_PADDED_SIZE_HBM_33,
        ADDR_AXI_HBM_61, AXI_PADDED_SIZE_HBM_61,
        ADDR_AXI_HBM_89, AXI_PADDED_SIZE_HBM_89,
        ADDR_AXI_HBM_117, AXI_PADDED_SIZE_HBM_117>(
        s_idx_buffer_HBM5, table_HBM5, s_embedding_buffer_HBM5, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_6, AXI_PADDED_SIZE_HBM_6, 
        ADDR_AXI_HBM_34, AXI_PADDED_SIZE_HBM_34,
        ADDR_AXI_HBM_62, AXI_PADDED_SIZE_HBM_62,
        ADDR_AXI_HBM_90, AXI_PADDED_SIZE_HBM_90,
        ADDR_AXI_HBM_118, AXI_PADDED_SIZE_HBM_118>(
        s_idx_buffer_HBM6, table_HBM6, s_embedding_buffer_HBM6, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_7, AXI_PADDED_SIZE_HBM_7, 
        ADDR_AXI_HBM_35, AXI_PADDED_SIZE_HBM_35,
        ADDR_AXI_HBM_63, AXI_PADDED_SIZE_HBM_63,
        ADDR_AXI_HBM_91, AXI_PADDED_SIZE_HBM_91,
        ADDR_AXI_HBM_119, AXI_PADDED_SIZE_HBM_119>(
        s_idx_buffer_HBM7, table_HBM7, s_embedding_buffer_HBM7, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_8, AXI_PADDED_SIZE_HBM_8, 
        ADDR_AXI_HBM_36, AXI_PADDED_SIZE_HBM_36,
        ADDR_AXI_HBM_64, AXI_PADDED_SIZE_HBM_64,
        ADDR_AXI_HBM_92, AXI_PADDED_SIZE_HBM_92,
        ADDR_AXI_HBM_120, AXI_PADDED_SIZE_HBM_120>(
        s_idx_buffer_HBM8, table_HBM8, s_embedding_buffer_HBM8, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_9, AXI_PADDED_SIZE_HBM_9, 
        ADDR_AXI_HBM_37, AXI_PADDED_SIZE_HBM_37,
        ADDR_AXI_HBM_65, AXI_PADDED_SIZE_HBM_65,
        ADDR_AXI_HBM_93, AXI_PADDED_SIZE_HBM_93,
        ADDR_AXI_HBM_121, AXI_PADDED_SIZE_HBM_121>(
        s_idx_buffer_HBM9, table_HBM9, s_embedding_buffer_HBM9, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_10, AXI_PADDED_SIZE_HBM_10, 
        ADDR_AXI_HBM_38, AXI_PADDED_SIZE_HBM_38,
        ADDR_AXI_HBM_66, AXI_PADDED_SIZE_HBM_66,
        ADDR_AXI_HBM_94, AXI_PADDED_SIZE_HBM_94,
        ADDR_AXI_HBM_122, AXI_PADDED_SIZE_HBM_122>(
        s_idx_buffer_HBM10, table_HBM10, s_embedding_buffer_HBM10, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_11, AXI_PADDED_SIZE_HBM_11, 
        ADDR_AXI_HBM_39, AXI_PADDED_SIZE_HBM_39,
        ADDR_AXI_HBM_67, AXI_PADDED_SIZE_HBM_67,
        ADDR_AXI_HBM_95, AXI_PADDED_SIZE_HBM_95,
        ADDR_AXI_HBM_123, AXI_PADDED_SIZE_HBM_123>(
        s_idx_buffer_HBM11, table_HBM11, s_embedding_buffer_HBM11, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_12, AXI_PADDED_SIZE_HBM_12, 
        ADDR_AXI_HBM_40, AXI_PADDED_SIZE_HBM_40,
        ADDR_AXI_HBM_68, AXI_PADDED_SIZE_HBM_68,
        ADDR_AXI_HBM_96, AXI_PADDED_SIZE_HBM_96,
        ADDR_AXI_HBM_124, AXI_PADDED_SIZE_HBM_124>(
        s_idx_buffer_HBM12, table_HBM12, s_embedding_buffer_HBM12, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_13, AXI_PADDED_SIZE_HBM_13, 
        ADDR_AXI_HBM_41, AXI_PADDED_SIZE_HBM_41,
        ADDR_AXI_HBM_69, AXI_PADDED_SIZE_HBM_69,
        ADDR_AXI_HBM_97, AXI_PADDED_SIZE_HBM_97,
        ADDR_AXI_HBM_125, AXI_PADDED_SIZE_HBM_125>(
        s_idx_buffer_HBM13, table_HBM13, s_embedding_buffer_HBM13, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_14, AXI_PADDED_SIZE_HBM_14, 
        ADDR_AXI_HBM_42, AXI_PADDED_SIZE_HBM_42,
        ADDR_AXI_HBM_70, AXI_PADDED_SIZE_HBM_70,
        ADDR_AXI_HBM_98, AXI_PADDED_SIZE_HBM_98,
        ADDR_AXI_HBM_126, AXI_PADDED_SIZE_HBM_126>(
        s_idx_buffer_HBM14, table_HBM14, s_embedding_buffer_HBM14, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_15, AXI_PADDED_SIZE_HBM_15, 
        ADDR_AXI_HBM_43, AXI_PADDED_SIZE_HBM_43,
        ADDR_AXI_HBM_71, AXI_PADDED_SIZE_HBM_71,
        ADDR_AXI_HBM_99, AXI_PADDED_SIZE_HBM_99,
        ADDR_AXI_HBM_127, AXI_PADDED_SIZE_HBM_127>(
        s_idx_buffer_HBM15, table_HBM15, s_embedding_buffer_HBM15, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_16, AXI_PADDED_SIZE_HBM_16, 
        ADDR_AXI_HBM_44, AXI_PADDED_SIZE_HBM_44,
        ADDR_AXI_HBM_72, AXI_PADDED_SIZE_HBM_72,
        ADDR_AXI_HBM_100, AXI_PADDED_SIZE_HBM_100,
        ADDR_AXI_HBM_128, AXI_PADDED_SIZE_HBM_128>(
        s_idx_buffer_HBM16, table_HBM16, s_embedding_buffer_HBM16, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_17, AXI_PADDED_SIZE_HBM_17, 
        ADDR_AXI_HBM_45, AXI_PADDED_SIZE_HBM_45,
        ADDR_AXI_HBM_73, AXI_PADDED_SIZE_HBM_73,
        ADDR_AXI_HBM_101, AXI_PADDED_SIZE_HBM_101,
        ADDR_AXI_HBM_129, AXI_PADDED_SIZE_HBM_129>(
        s_idx_buffer_HBM17, table_HBM17, s_embedding_buffer_HBM17, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_18, AXI_PADDED_SIZE_HBM_18, 
        ADDR_AXI_HBM_46, AXI_PADDED_SIZE_HBM_46,
        ADDR_AXI_HBM_74, AXI_PADDED_SIZE_HBM_74,
        ADDR_AXI_HBM_102, AXI_PADDED_SIZE_HBM_102,
        ADDR_AXI_HBM_130, AXI_PADDED_SIZE_HBM_130>(
        s_idx_buffer_HBM18, table_HBM18, s_embedding_buffer_HBM18, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_19, AXI_PADDED_SIZE_HBM_19, 
        ADDR_AXI_HBM_47, AXI_PADDED_SIZE_HBM_47,
        ADDR_AXI_HBM_75, AXI_PADDED_SIZE_HBM_75,
        ADDR_AXI_HBM_103, AXI_PADDED_SIZE_HBM_103,
        ADDR_AXI_HBM_131, AXI_PADDED_SIZE_HBM_131>(
        s_idx_buffer_HBM19, table_HBM19, s_embedding_buffer_HBM19, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_20, AXI_PADDED_SIZE_HBM_20, 
        ADDR_AXI_HBM_48, AXI_PADDED_SIZE_HBM_48,
        ADDR_AXI_HBM_76, AXI_PADDED_SIZE_HBM_76,
        ADDR_AXI_HBM_104, AXI_PADDED_SIZE_HBM_104,
        ADDR_AXI_HBM_132, AXI_PADDED_SIZE_HBM_132>(
        s_idx_buffer_HBM20, table_HBM20, s_embedding_buffer_HBM20, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_21, AXI_PADDED_SIZE_HBM_21, 
        ADDR_AXI_HBM_49, AXI_PADDED_SIZE_HBM_49,
        ADDR_AXI_HBM_77, AXI_PADDED_SIZE_HBM_77,
        ADDR_AXI_HBM_105, AXI_PADDED_SIZE_HBM_105,
        ADDR_AXI_HBM_133, AXI_PADDED_SIZE_HBM_133>(
        s_idx_buffer_HBM21, table_HBM21, s_embedding_buffer_HBM21, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_22, AXI_PADDED_SIZE_HBM_22, 
        ADDR_AXI_HBM_50, AXI_PADDED_SIZE_HBM_50,
        ADDR_AXI_HBM_78, AXI_PADDED_SIZE_HBM_78,
        ADDR_AXI_HBM_106, AXI_PADDED_SIZE_HBM_106,
        ADDR_AXI_HBM_134, AXI_PADDED_SIZE_HBM_134>(
        s_idx_buffer_HBM22, table_HBM22, s_embedding_buffer_HBM22, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_23, AXI_PADDED_SIZE_HBM_23, 
        ADDR_AXI_HBM_51, AXI_PADDED_SIZE_HBM_51,
        ADDR_AXI_HBM_79, AXI_PADDED_SIZE_HBM_79,
        ADDR_AXI_HBM_107, AXI_PADDED_SIZE_HBM_107,
        ADDR_AXI_HBM_135, AXI_PADDED_SIZE_HBM_135>(
        s_idx_buffer_HBM23, table_HBM23, s_embedding_buffer_HBM23, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_24, AXI_PADDED_SIZE_HBM_24, 
        ADDR_AXI_HBM_52, AXI_PADDED_SIZE_HBM_52,
        ADDR_AXI_HBM_80, AXI_PADDED_SIZE_HBM_80,
        ADDR_AXI_HBM_108, AXI_PADDED_SIZE_HBM_108,
        ADDR_AXI_HBM_136, AXI_PADDED_SIZE_HBM_136>(
        s_idx_buffer_HBM24, table_HBM24, s_embedding_buffer_HBM24, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_25, AXI_PADDED_SIZE_HBM_25, 
        ADDR_AXI_HBM_53, AXI_PADDED_SIZE_HBM_53,
        ADDR_AXI_HBM_81, AXI_PADDED_SIZE_HBM_81,
        ADDR_AXI_HBM_109, AXI_PADDED_SIZE_HBM_109,
        ADDR_AXI_HBM_137, AXI_PADDED_SIZE_HBM_137>(
        s_idx_buffer_HBM25, table_HBM25, s_embedding_buffer_HBM25, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_26, AXI_PADDED_SIZE_HBM_26, 
        ADDR_AXI_HBM_54, AXI_PADDED_SIZE_HBM_54,
        ADDR_AXI_HBM_82, AXI_PADDED_SIZE_HBM_82,
        ADDR_AXI_HBM_110, AXI_PADDED_SIZE_HBM_110,
        ADDR_AXI_HBM_138, AXI_PADDED_SIZE_HBM_138>(
        s_idx_buffer_HBM26, table_HBM26, s_embedding_buffer_HBM26, batch_num);
    load_single_embedding_5_tables<
        ADDR_AXI_HBM_27, AXI_PADDED_SIZE_HBM_27, 
        ADDR_AXI_HBM_55, AXI_PADDED_SIZE_HBM_55,
        ADDR_AXI_HBM_83, AXI_PADDED_SIZE_HBM_83,
        ADDR_AXI_HBM_111, AXI_PADDED_SIZE_HBM_111,
        ADDR_AXI_HBM_139, AXI_PADDED_SIZE_HBM_139>(
        s_idx_buffer_HBM27, table_HBM27, s_embedding_buffer_HBM27, batch_num);

    load_single_embedding_2_tables<ADDR_AXI_DDR_0, AXI_PADDED_SIZE_DDR_0, ADDR_AXI_DDR_2, AXI_PADDED_SIZE_DDR_2>(
        s_idx_buffer_DDR0, table_DDR0, s_embedding_buffer_DDR0, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_DDR_1, AXI_PADDED_SIZE_DDR_1, ADDR_AXI_DDR_3, AXI_PADDED_SIZE_DDR_3>(
        s_idx_buffer_DDR1, table_DDR1, s_embedding_buffer_DDR1, batch_num);

    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_0, AXI_PADDED_SIZE_PLRAM_0, ADDR_AXI_PLRAM_11, AXI_PADDED_SIZE_PLRAM_11,
        ADDR_AXI_PLRAM_22, AXI_PADDED_SIZE_PLRAM_22, ADDR_AXI_PLRAM_33, AXI_PADDED_SIZE_PLRAM_33>(
        s_idx_buffer_PLRAM0, table_PLRAM0, s_embedding_buffer_PLRAM0, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_1, AXI_PADDED_SIZE_PLRAM_1, ADDR_AXI_PLRAM_12, AXI_PADDED_SIZE_PLRAM_12,
        ADDR_AXI_PLRAM_23, AXI_PADDED_SIZE_PLRAM_23, ADDR_AXI_PLRAM_34, AXI_PADDED_SIZE_PLRAM_34>(
        s_idx_buffer_PLRAM1, table_PLRAM1, s_embedding_buffer_PLRAM1, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_2, AXI_PADDED_SIZE_PLRAM_2, ADDR_AXI_PLRAM_13, AXI_PADDED_SIZE_PLRAM_13,
        ADDR_AXI_PLRAM_24, AXI_PADDED_SIZE_PLRAM_24, ADDR_AXI_PLRAM_35, AXI_PADDED_SIZE_PLRAM_35>(
        s_idx_buffer_PLRAM2, table_PLRAM2, s_embedding_buffer_PLRAM2, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_3, AXI_PADDED_SIZE_PLRAM_3, ADDR_AXI_PLRAM_14, AXI_PADDED_SIZE_PLRAM_14,
        ADDR_AXI_PLRAM_25, AXI_PADDED_SIZE_PLRAM_25, ADDR_AXI_PLRAM_36, AXI_PADDED_SIZE_PLRAM_36>(
        s_idx_buffer_PLRAM3, table_PLRAM3, s_embedding_buffer_PLRAM3, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_4, AXI_PADDED_SIZE_PLRAM_4, ADDR_AXI_PLRAM_15, AXI_PADDED_SIZE_PLRAM_15,
        ADDR_AXI_PLRAM_26, AXI_PADDED_SIZE_PLRAM_26, ADDR_AXI_PLRAM_37, AXI_PADDED_SIZE_PLRAM_37>(
        s_idx_buffer_PLRAM4, table_PLRAM4, s_embedding_buffer_PLRAM4, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_5, AXI_PADDED_SIZE_PLRAM_5, ADDR_AXI_PLRAM_16, AXI_PADDED_SIZE_PLRAM_16,
        ADDR_AXI_PLRAM_27, AXI_PADDED_SIZE_PLRAM_27, ADDR_AXI_PLRAM_38, AXI_PADDED_SIZE_PLRAM_38>(
        s_idx_buffer_PLRAM5, table_PLRAM5, s_embedding_buffer_PLRAM5, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_6, AXI_PADDED_SIZE_PLRAM_6, ADDR_AXI_PLRAM_17, AXI_PADDED_SIZE_PLRAM_17,
        ADDR_AXI_PLRAM_28, AXI_PADDED_SIZE_PLRAM_28, ADDR_AXI_PLRAM_39, AXI_PADDED_SIZE_PLRAM_39>(
        s_idx_buffer_PLRAM6, table_PLRAM6, s_embedding_buffer_PLRAM6, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_7, AXI_PADDED_SIZE_PLRAM_7, ADDR_AXI_PLRAM_18, AXI_PADDED_SIZE_PLRAM_18,
        ADDR_AXI_PLRAM_29, AXI_PADDED_SIZE_PLRAM_29, ADDR_AXI_PLRAM_40, AXI_PADDED_SIZE_PLRAM_40>(
        s_idx_buffer_PLRAM7, table_PLRAM7, s_embedding_buffer_PLRAM7, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_8, AXI_PADDED_SIZE_PLRAM_8, ADDR_AXI_PLRAM_19, AXI_PADDED_SIZE_PLRAM_19,
        ADDR_AXI_PLRAM_30, AXI_PADDED_SIZE_PLRAM_30, ADDR_AXI_PLRAM_41, AXI_PADDED_SIZE_PLRAM_41>(
        s_idx_buffer_PLRAM8, table_PLRAM8, s_embedding_buffer_PLRAM8, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_9, AXI_PADDED_SIZE_PLRAM_9, ADDR_AXI_PLRAM_20, AXI_PADDED_SIZE_PLRAM_20,
        ADDR_AXI_PLRAM_31, AXI_PADDED_SIZE_PLRAM_31, ADDR_AXI_PLRAM_42, AXI_PADDED_SIZE_PLRAM_42>(
        s_idx_buffer_PLRAM9, table_PLRAM9, s_embedding_buffer_PLRAM9, batch_num);
    load_single_embedding_4_tables<
        ADDR_AXI_PLRAM_10, AXI_PADDED_SIZE_PLRAM_10, ADDR_AXI_PLRAM_21, AXI_PADDED_SIZE_PLRAM_21,
        ADDR_AXI_PLRAM_32, AXI_PADDED_SIZE_PLRAM_32, ADDR_AXI_PLRAM_43, AXI_PADDED_SIZE_PLRAM_43>(
        s_idx_buffer_PLRAM10, table_PLRAM10, s_embedding_buffer_PLRAM10, batch_num);


    gather_embeddings(
        s_embedding_buffer_HBM0, s_embedding_buffer_HBM1, 
        s_embedding_buffer_HBM2, s_embedding_buffer_HBM3, 
        s_embedding_buffer_HBM4, s_embedding_buffer_HBM5, 
        s_embedding_buffer_HBM6, s_embedding_buffer_HBM7, 
        s_embedding_buffer_HBM8, s_embedding_buffer_HBM9, 
        s_embedding_buffer_HBM10, s_embedding_buffer_HBM11, 
        s_embedding_buffer_HBM12, s_embedding_buffer_HBM13, 
        s_embedding_buffer_HBM14, s_embedding_buffer_HBM15, 
        s_embedding_buffer_HBM16, s_embedding_buffer_HBM17, 
        s_embedding_buffer_HBM18, s_embedding_buffer_HBM19, 
        s_embedding_buffer_HBM20, s_embedding_buffer_HBM21, 
        s_embedding_buffer_HBM22, s_embedding_buffer_HBM23, 
        s_embedding_buffer_HBM24, s_embedding_buffer_HBM25, 
        s_embedding_buffer_HBM26, s_embedding_buffer_HBM27, 
        
        s_embedding_buffer_DDR0, s_embedding_buffer_DDR1,

        s_embedding_buffer_PLRAM0, s_embedding_buffer_PLRAM1, 
        s_embedding_buffer_PLRAM2, s_embedding_buffer_PLRAM3,
        s_embedding_buffer_PLRAM4, s_embedding_buffer_PLRAM5,
        s_embedding_buffer_PLRAM6, s_embedding_buffer_PLRAM7,
        s_embedding_buffer_PLRAM8, s_embedding_buffer_PLRAM9,
        s_embedding_buffer_PLRAM10, 

        s_network, 
        
        batch_num);

    ap_uint<64> expectedTxByteCnt = batch_num * BATCH_SIZE * INPUT_SIZE_AXI_512 * 64;
    
    sendData(m_axis_tcp_tx_meta, 
             m_axis_tcp_tx_data, 
             s_axis_tcp_tx_status,
             s_network,
             sessionID,
             useConn,
             expectedTxByteCnt, 
             pkgWordCount
             );
    

    if (!s_axis_udp_rx.empty())
    {
    pkt512 udp_rx = s_axis_udp_rx.read();
    m_axis_udp_tx.write(udp_rx);
    }

    if (!s_axis_udp_rx_meta.empty())
    {
    pkt256 udp_rx_meta = s_axis_udp_rx_meta.read();
    m_axis_udp_tx_meta.write(udp_rx_meta);
    }

    pkt16 listenPort_pkt;
    if (!listenPort.empty())
    {
    ap_uint<16> listenPort_data = listenPort.read();
    listenPort_pkt.data = listenPort_data;
    m_axis_tcp_listen_port.write(listenPort_pkt);
    }

    if (!s_axis_tcp_port_status.empty())
    {
    pkt8 port_status_pkt = s_axis_tcp_port_status.read();
    bool port_status_data = (port_status_pkt.data == 1) ? true: false;
    listenPortStatus.write(port_status_data);
    }

    pkt16 close_connection_pkt;
    if (!closeConnection.empty())
    {
    close_connection_pkt.data = closeConnection.read();
    m_axis_tcp_close_connection.write(close_connection_pkt);
    }

    appNotification tcp_notification_data;
    if (!s_axis_tcp_notification.empty())
    {
    pkt128 tcp_notification_pkt = s_axis_tcp_notification.read();
    tcp_notification_data.sessionID = tcp_notification_pkt.data(15,0);
    tcp_notification_data.length = tcp_notification_pkt.data(31,16);
    tcp_notification_data.ipAddress = tcp_notification_pkt.data(63,32);
    tcp_notification_data.dstPort = tcp_notification_pkt.data(79,64);
    tcp_notification_data.closed = tcp_notification_pkt.data(80,80);
    notifications.write(tcp_notification_data);

    }

    pkt32 readRequest_pkt;
    if (!readRequest.empty())
    {
    appReadRequest readRequest_data = readRequest.read();
    readRequest_pkt.data(15,0) = readRequest_data.sessionID;
    readRequest_pkt.data(31,16) = readRequest_data.length;
    m_axis_tcp_read_pkg.write(readRequest_pkt);
    }

    if (!s_axis_tcp_rx_meta.empty())
    {
    pkt16 tcp_rx_meta_pkt = s_axis_tcp_rx_meta.read();
    ap_uint<16> tcp_rx_meta_data = tcp_rx_meta_pkt.data;
    rxMetaData.write(tcp_rx_meta_data);
    }

    net_axis<512> tcp_rx_data;
    if (!s_axis_tcp_rx_data.empty())
    {
    pkt512 tcp_rx_pkt = s_axis_tcp_rx_data.read();
    tcp_rx_data.data = tcp_rx_pkt.data;
    tcp_rx_data.keep = tcp_rx_pkt.keep;
    tcp_rx_data.last = tcp_rx_pkt.last;
    rxData.write(tcp_rx_data);
    }


    // consume_W(s_feature_in);



//////////////////////////////     Write results     ////////////////////////////// 
}

}

template<
    const int start_addr_0, const int axi_padded_size_0, const int table_entry_num_0,
    const int start_addr_1, const int axi_padded_size_1, const int table_entry_num_1>
void init_plram_t_2_tables(
    plram_t* table_PLRAM) {

    // even row: 1 
    // odd row: -1
    // NOTE! must use ap_uint<128> here
    // float one_arr[4] = {1, 1, 1, 1};
    // float zero_arr[4] = {0, 0, 0, 0};
    int one = 1065353216; // float 1 -> same bit ->  int 1065353216
    int zero = 0; // float 0 -> int 0
    for (int i = 0 ; i < table_entry_num_0 / 2; i++) {
        for (int j = 0; j < axi_padded_size_0; j++) {
            // memcpy(&table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j], one_arr, 16);
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_0; j++) {
            // memcpy(&table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j], zero_arr, 16);
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(127, 96) = zero;
        }
    }
    for (int i = 0 ; i < table_entry_num_1 / 2; i++) {
        for (int j = 0; j < axi_padded_size_1; j++) {
            // memcpy(&table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j], one_arr, 16);
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_1; j++) {
            // memcpy(&table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j], zero_arr, 16);
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(127, 96) = zero;
        }
    }
}


template<
    const int start_addr_0, const int axi_padded_size_0, const int table_entry_num_0,
    const int start_addr_1, const int axi_padded_size_1, const int table_entry_num_1,
    const int start_addr_2, const int axi_padded_size_2, const int table_entry_num_2,
    const int start_addr_3, const int axi_padded_size_3, const int table_entry_num_3>
void init_plram_t_4_tables(
    plram_t* table_PLRAM) {

    // even row: 1 
    // odd row: -1
    // NOTE! must use ap_uint<128> here
    // float one_arr[4] = {1, 1, 1, 1};
    // float zero_arr[4] = {0, 0, 0, 0};
    int one = 1065353216; // float 1 -> same bit ->  int 1065353216
    int zero = 0; // float 0 -> int 0

    for (int i = 0 ; i < table_entry_num_0 / 2; i++) {
        for (int j = 0; j < axi_padded_size_0; j++) {
            // memcpy(&table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j], one_arr, 16);
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_0; j++) {
            // memcpy(&table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j], zero_arr, 16);
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(127, 96) = zero;
        }
    }

    for (int i = 0 ; i < table_entry_num_1 / 2; i++) {
        for (int j = 0; j < axi_padded_size_1; j++) {
            // memcpy(&table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j], one_arr, 16);
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_1; j++) {
            // memcpy(&table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j], zero_arr, 16);
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(127, 96) = zero;
        }
    }

    for (int i = 0 ; i < table_entry_num_2 / 2; i++) {
        for (int j = 0; j < axi_padded_size_2; j++) {
            // memcpy(&table_PLRAM[start_addr_2 + (2 * i) * axi_padded_size_2 + j], one_arr, 16);
            table_PLRAM[start_addr_2 + (2 * i) * axi_padded_size_2 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_2 + (2 * i) * axi_padded_size_2 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_2 + (2 * i) * axi_padded_size_2 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_2 + (2 * i) * axi_padded_size_2 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_2; j++) {
            // memcpy(&table_PLRAM[start_addr_2 + (2 * i + 1) * axi_padded_size_2 + j], zero_arr, 16);
            table_PLRAM[start_addr_2 + (2 * i + 1) * axi_padded_size_2 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_2 + (2 * i + 1) * axi_padded_size_2 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_2 + (2 * i + 1) * axi_padded_size_2 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_2 + (2 * i + 1) * axi_padded_size_2 + j].range(127, 96) = zero;
        }
    }

    for (int i = 0 ; i < table_entry_num_3 / 2; i++) {
        for (int j = 0; j < axi_padded_size_3; j++) {
            // memcpy(&table_PLRAM[start_addr_3 + (2 * i) * axi_padded_size_3 + j], one_arr, 16);
            table_PLRAM[start_addr_3 + (2 * i) * axi_padded_size_3 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_3 + (2 * i) * axi_padded_size_3 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_3 + (2 * i) * axi_padded_size_3 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_3 + (2 * i) * axi_padded_size_3 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_3; j++) {
            // memcpy(&table_PLRAM[start_addr_3 + (2 * i + 1) * axi_padded_size_3 + j], zero_arr, 16);
            table_PLRAM[start_addr_3 + (2 * i + 1) * axi_padded_size_3 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_3 + (2 * i + 1) * axi_padded_size_3 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_3 + (2 * i + 1) * axi_padded_size_3 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_3 + (2 * i + 1) * axi_padded_size_3 + j].range(127, 96) = zero;
        }
    }
}

void load_access_idx(
    hls::stream<int>& s_idx_buffer_single_channel,
    int batch_num) { 

    int idx_random[] = {3, 99, 38, 72, 29, 57, 1, 72, 36, 76, 35, 50, 37, 57, 
        13, 66, 26, 70, 41, 93, 48, 82, 44, 78, 25, 52, 3, 92, 36, 56, 46, 88};

    for (int i = 0; i < batch_num; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
            #pragma HLS pipeline II=1

            int idx = idx_random[j]; 
            s_idx_buffer_single_channel.write(idx);
        }
    }
}

template<
    const long start_addr_0, const long AXI_padded_size_0, 
    const long start_addr_1, const long AXI_padded_size_1>
void load_single_embedding_2_tables(
    hls::stream<int>& s_idx_buffer, const hbm_t* table_RAM, 
    hls::stream<hbm_t>& s_embedding_buffer,
    int batch_num) {
#pragma HLS INLINE off

    // 8 < data size <= 16, load 2 times
    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {

        long idx = s_idx_buffer.read();

        long base_addr_0 = start_addr_0 + idx * AXI_padded_size_0;
        for (int j = 0; j < AXI_padded_size_0; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_0 + j]);
        }
        long base_addr_1 = start_addr_1 + idx * AXI_padded_size_1;
        for (int j = 0; j < AXI_padded_size_1; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_1 + j]);
        }
    }
}


template<
    const long start_addr_0, const long AXI_padded_size_0, 
    const long start_addr_1, const long AXI_padded_size_1,
    const long start_addr_2, const long AXI_padded_size_2,
    const long start_addr_3, const long AXI_padded_size_3>
void load_single_embedding_4_tables(
    hls::stream<int>& s_idx_buffer, const hbm_t* table_RAM, 
    hls::stream<hbm_t>& s_embedding_buffer,
    int batch_num) {
#pragma HLS INLINE off

    // 8 < data size <= 16, load 2 times
    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {

        long idx = s_idx_buffer.read();

        long base_addr_0 = start_addr_0 + idx * AXI_padded_size_0;
        for (int j = 0; j < AXI_padded_size_0; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_0 + j]);
        }
        long base_addr_1 = start_addr_1 + idx * AXI_padded_size_1;
        for (int j = 0; j < AXI_padded_size_1; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_1 + j]);
        }
        long base_addr_2 = start_addr_2 + idx * AXI_padded_size_2;
        for (int j = 0; j < AXI_padded_size_2; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_2 + j]);
        }
        long base_addr_3 = start_addr_3 + idx * AXI_padded_size_3;
        for (int j = 0; j < AXI_padded_size_3; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_3 + j]);
        }
    }
}


template<
    const long start_addr_0, const long AXI_padded_size_0, 
    const long start_addr_1, const long AXI_padded_size_1,
    const long start_addr_2, const long AXI_padded_size_2,
    const long start_addr_3, const long AXI_padded_size_3,
    const long start_addr_4, const long AXI_padded_size_4>
void load_single_embedding_5_tables(
    hls::stream<int>& s_idx_buffer, const hbm_t* table_RAM, 
    hls::stream<hbm_t>& s_embedding_buffer,
    int batch_num) {
#pragma HLS INLINE off

    // 8 < data size <= 16, load 2 times
    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {

        long idx = s_idx_buffer.read();

        long base_addr_0 = start_addr_0 + idx * AXI_padded_size_0;
        for (int j = 0; j < AXI_padded_size_0; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_0 + j]);
        }
        long base_addr_1 = start_addr_1 + idx * AXI_padded_size_1;
        for (int j = 0; j < AXI_padded_size_1; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_1 + j]);
        }
        long base_addr_2 = start_addr_2 + idx * AXI_padded_size_2;
        for (int j = 0; j < AXI_padded_size_2; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_2 + j]);
        }
        long base_addr_3 = start_addr_3 + idx * AXI_padded_size_3;
        for (int j = 0; j < AXI_padded_size_3; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_3 + j]);
        }
        long base_addr_4 = start_addr_4 + idx * AXI_padded_size_4;
        for (int j = 0; j < AXI_padded_size_4; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_4 + j]);
        }
    }
}


void group_4_embeddings_len7_out_7(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<axi_t>& s_embedding_2, hls::stream<axi_t>& s_embedding_3,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num
) {
    // Concatenate 4 embedding streams
    //   each stream: 7 x axi 128

    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {
        #pragma HLS pipeline II=7

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        axi_t tmp4 = s_embedding_0.read();
        axi_t tmp5 = s_embedding_0.read();
        axi_t tmp6 = s_embedding_0.read();
        
        axi_t tmp7 = s_embedding_1.read();
        axi_t tmp8 = s_embedding_1.read();
        axi_t tmp9 = s_embedding_1.read();
        axi_t tmp10 = s_embedding_1.read();
        axi_t tmp11 = s_embedding_1.read();
        axi_t tmp12 = s_embedding_1.read();
        axi_t tmp13 = s_embedding_1.read();

        axi_t tmp14 = s_embedding_2.read();
        axi_t tmp15 = s_embedding_2.read();
        axi_t tmp16 = s_embedding_2.read();
        axi_t tmp17 = s_embedding_2.read();
        axi_t tmp18 = s_embedding_2.read();
        axi_t tmp19 = s_embedding_2.read();
        axi_t tmp20 = s_embedding_2.read();
        
        axi_t tmp21 = s_embedding_3.read();
        axi_t tmp22 = s_embedding_3.read();
        axi_t tmp23 = s_embedding_3.read();
        axi_t tmp24 = s_embedding_3.read();
        axi_t tmp25 = s_embedding_3.read();
        axi_t tmp26 = s_embedding_3.read();
        axi_t tmp27 = s_embedding_3.read();
        
        network_t out0, out1, out2, out3, out4, out5, out6;
        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        out3.range(127, 0) = tmp12;
        out3.range(255, 128) = tmp13;
        out3.range(383, 256) = tmp14;
        out3.range(511, 384) = tmp15;

        out4.range(127, 0) = tmp16;
        out4.range(255, 128) = tmp17;
        out4.range(383, 256) = tmp18;
        out4.range(511, 384) = tmp19;

        out5.range(127, 0) = tmp20;
        out5.range(255, 128) = tmp21;
        out5.range(383, 256) = tmp22;
        out5.range(511, 384) = tmp23;

        out6.range(127, 0) = tmp24;
        out6.range(255, 128) = tmp25;
        out6.range(383, 256) = tmp26;
        out6.range(511, 384) = tmp27;

        s_gathered_embedding.write(out0);
        s_gathered_embedding.write(out1);
        s_gathered_embedding.write(out2);
        s_gathered_embedding.write(out3);
        s_gathered_embedding.write(out4);
        s_gathered_embedding.write(out5);
        s_gathered_embedding.write(out6);
    }
}

void group_2_embeddings_len14_out_7(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num
) {
    // Concatenate 2 embedding streams
    //   each stream: 14 x axi 128

    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {
        #pragma HLS pipeline II=14

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        axi_t tmp4 = s_embedding_0.read();
        axi_t tmp5 = s_embedding_0.read();
        axi_t tmp6 = s_embedding_0.read();
        axi_t tmp7 = s_embedding_0.read();
        axi_t tmp8 = s_embedding_0.read();
        axi_t tmp9 = s_embedding_0.read();
        axi_t tmp10 = s_embedding_0.read();
        axi_t tmp11 = s_embedding_0.read();
        axi_t tmp12 = s_embedding_0.read();
        axi_t tmp13 = s_embedding_0.read();

        axi_t tmp14 = s_embedding_1.read();
        axi_t tmp15 = s_embedding_1.read();
        axi_t tmp16 = s_embedding_1.read();
        axi_t tmp17 = s_embedding_1.read();
        axi_t tmp18 = s_embedding_1.read();
        axi_t tmp19 = s_embedding_1.read();
        axi_t tmp20 = s_embedding_1.read();
        axi_t tmp21 = s_embedding_1.read();
        axi_t tmp22 = s_embedding_1.read();
        axi_t tmp23 = s_embedding_1.read();
        axi_t tmp24 = s_embedding_1.read();
        axi_t tmp25 = s_embedding_1.read();
        axi_t tmp26 = s_embedding_1.read();
        axi_t tmp27 = s_embedding_1.read();
        
        network_t out0, out1, out2, out3, out4, out5, out6;

        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        out3.range(127, 0) = tmp12;
        out3.range(255, 128) = tmp13;
        out3.range(383, 256) = tmp14;
        out3.range(511, 384) = tmp15;

        out4.range(127, 0) = tmp16;
        out4.range(255, 128) = tmp17;
        out4.range(383, 256) = tmp18;
        out4.range(511, 384) = tmp19;

        out5.range(127, 0) = tmp20;
        out5.range(255, 128) = tmp21;
        out5.range(383, 256) = tmp22;
        out5.range(511, 384) = tmp23;

        out6.range(127, 0) = tmp24;
        out6.range(255, 128) = tmp25;
        out6.range(383, 256) = tmp26;
        out6.range(511, 384) = tmp27;

        s_gathered_embedding.write(out0);
        s_gathered_embedding.write(out1);
        s_gathered_embedding.write(out2);
        s_gathered_embedding.write(out3);
        s_gathered_embedding.write(out4);
        s_gathered_embedding.write(out5);
        s_gathered_embedding.write(out6);
    }
}

void group_1_embeddings_len8_out_2(
    hls::stream<axi_t>& s_embedding_0,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num
) {
    // Concatenate 1 embedding sub-vectors
    //     each stream: 8 x axi 128

    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {
        #pragma HLS pipeline II=8

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        axi_t tmp4 = s_embedding_0.read();
        axi_t tmp5 = s_embedding_0.read();
        axi_t tmp6 = s_embedding_0.read();
        axi_t tmp7 = s_embedding_0.read();
        
        network_t out0, out1;
        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;
        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        s_gathered_embedding.write(out0);
        s_gathered_embedding.write(out1);
    }
}

void group_1_embeddings_len12_out_3(
    hls::stream<axi_t>& s_embedding_0,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num
) {
    // Concatenate 1 embedding sub-vectors
    //     each stream: 12 x axi 128

    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {
        #pragma HLS pipeline II=12

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        
        network_t out0;
        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        s_gathered_embedding.write(out0);

        axi_t tmp4 = s_embedding_0.read();
        axi_t tmp5 = s_embedding_0.read();
        axi_t tmp6 = s_embedding_0.read();
        axi_t tmp7 = s_embedding_0.read();
        
        network_t out1;
        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        s_gathered_embedding.write(out1);

        axi_t tmp8 = s_embedding_0.read();
        axi_t tmp9 = s_embedding_0.read();
        axi_t tmp10 = s_embedding_0.read();
        axi_t tmp11 = s_embedding_0.read();
        
        network_t out2;
        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        s_gathered_embedding.write(out2);
    }
}


void group_1_embeddings_len16_out_4(
    hls::stream<axi_t>& s_embedding_0,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num
) {
    // Concatenate 1 embedding sub-vectors
    //     each stream: 16 x axi 128

    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {
        #pragma HLS pipeline II=16

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        
        network_t out0;
        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        s_gathered_embedding.write(out0);

        axi_t tmp4 = s_embedding_0.read();
        axi_t tmp5 = s_embedding_0.read();
        axi_t tmp6 = s_embedding_0.read();
        axi_t tmp7 = s_embedding_0.read();
        
        network_t out1;
        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        s_gathered_embedding.write(out1);

        axi_t tmp8 = s_embedding_0.read();
        axi_t tmp9 = s_embedding_0.read();
        axi_t tmp10 = s_embedding_0.read();
        axi_t tmp11 = s_embedding_0.read();
        
        network_t out2;
        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        s_gathered_embedding.write(out2);

        axi_t tmp12 = s_embedding_0.read();
        axi_t tmp13 = s_embedding_0.read();
        axi_t tmp14 = s_embedding_0.read();
        axi_t tmp15 = s_embedding_0.read();

        network_t out3;
        out3.range(127, 0) = tmp12;
        out3.range(255, 128) = tmp13;
        out3.range(383, 256) = tmp14;
        out3.range(511, 384) = tmp15;

        s_gathered_embedding.write(out3);
    }
}

void gather_122_embedding_streams(
    hls::stream<network_t> (&s_embedding_gather_level_A_out_2)[3],
    hls::stream<network_t> (&s_embedding_gather_level_A_out_3)[8],
    hls::stream<network_t> (&s_embedding_gather_level_A_out_4)[2],
    hls::stream<network_t> (&s_embedding_gather_level_A_out_7)[12],
    hls::stream<network_t>& s_network, 
    int batch_num) {

    for (int i = 0; i < BATCH_SIZE * batch_num; i++) {

        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 2; k++) {
                #pragma HLS pipeline II=1
                s_network.write(s_embedding_gather_level_A_out_2[j].read());
            }
        }

        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 3; k++) {
                #pragma HLS pipeline II=1
                s_network.write(s_embedding_gather_level_A_out_3[j].read());
            }
        }

        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                #pragma HLS pipeline II=1
                s_network.write(s_embedding_gather_level_A_out_4[j].read());
            }
        }

        for (int j = 0; j < 12; j++) {
            for (int k = 0; k < 7; k++) {
                #pragma HLS pipeline II=1
                s_network.write(s_embedding_gather_level_A_out_7[j].read());
            }
        }
    }
}

void gather_embeddings(
    hls::stream<hbm_t>& s_embedding_buffer_HBM0, hls::stream<hbm_t>& s_embedding_buffer_HBM1, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM2, hls::stream<hbm_t>& s_embedding_buffer_HBM3, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM4, hls::stream<hbm_t>& s_embedding_buffer_HBM5, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM6, hls::stream<hbm_t>& s_embedding_buffer_HBM7, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM8, hls::stream<hbm_t>& s_embedding_buffer_HBM9, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM10, hls::stream<hbm_t>& s_embedding_buffer_HBM11, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM12, hls::stream<hbm_t>& s_embedding_buffer_HBM13, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM14, hls::stream<hbm_t>& s_embedding_buffer_HBM15, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM16, hls::stream<hbm_t>& s_embedding_buffer_HBM17, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM18, hls::stream<hbm_t>& s_embedding_buffer_HBM19, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM20, hls::stream<hbm_t>& s_embedding_buffer_HBM21, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM22, hls::stream<hbm_t>& s_embedding_buffer_HBM23, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM24, hls::stream<hbm_t>& s_embedding_buffer_HBM25, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM26, hls::stream<hbm_t>& s_embedding_buffer_HBM27, 
    
    hls::stream<ddr_t>& s_embedding_buffer_DDR0, hls::stream<ddr_t>& s_embedding_buffer_DDR1,

    hls::stream<plram_t>& s_embedding_buffer_PLRAM0, hls::stream<plram_t>& s_embedding_buffer_PLRAM1, 
    hls::stream<plram_t>& s_embedding_buffer_PLRAM2, hls::stream<plram_t>& s_embedding_buffer_PLRAM3,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM4, hls::stream<plram_t>& s_embedding_buffer_PLRAM5,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM6, hls::stream<plram_t>& s_embedding_buffer_PLRAM7,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM8, hls::stream<plram_t>& s_embedding_buffer_PLRAM9,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM10, 
    
    hls::stream<network_t>& s_network,

    int batch_num
) {
#pragma HLS inline

    /// NOTE THAT THIS FUNCTION IS ONLY APPLICABLE TO THE CURRENT TABLE SIZES ///
    /// THIS IS NOT A GENERALIZED GATHER FUNCTION, IT IS DATA DEPENDENT ///

// #define VECTOR_SIZE_PLRAM_BANK_0 28
// #define VECTOR_SIZE_PLRAM_BANK_1 28
// #define VECTOR_SIZE_PLRAM_BANK_2 28
// #define VECTOR_SIZE_PLRAM_BANK_3 28
// #define VECTOR_SIZE_PLRAM_BANK_4 28
// #define VECTOR_SIZE_PLRAM_BANK_5 28
// #define VECTOR_SIZE_PLRAM_BANK_6 28
// #define VECTOR_SIZE_PLRAM_BANK_7 28 -> (4 * 28) * 32 / 512
// #define VECTOR_SIZE_PLRAM_BANK_8 32
// #define VECTOR_SIZE_PLRAM_BANK_9 32
// #define VECTOR_SIZE_PLRAM_BANK_10 32 -> 32 * 32 / 512 = 2 x 512 bits

// PLRAM 0~7: gather_4_embeddings_len_7
// PLRAM 8~10: gather_1_embeddings_len_8

// #define VECTOR_SIZE_HBM_BANK_0 48
// #define VECTOR_SIZE_HBM_BANK_1 48
// #define VECTOR_SIZE_HBM_BANK_2 48
// #define VECTOR_SIZE_HBM_BANK_3 48
// #define VECTOR_SIZE_HBM_BANK_4 48
// #define VECTOR_SIZE_HBM_BANK_5 48
// #define VECTOR_SIZE_HBM_BANK_6 48
// #define VECTOR_SIZE_HBM_BANK_7 48 -> 3 x 512 bits

// #define VECTOR_SIZE_HBM_BANK_8 56
// #define VECTOR_SIZE_HBM_BANK_9 56
// #define VECTOR_SIZE_HBM_BANK_10 56
// #define VECTOR_SIZE_HBM_BANK_11 56
// #define VECTOR_SIZE_HBM_BANK_12 56
// #define VECTOR_SIZE_HBM_BANK_13 56
// #define VECTOR_SIZE_HBM_BANK_14 56
// #define VECTOR_SIZE_HBM_BANK_15 56
// #define VECTOR_SIZE_HBM_BANK_16 56
// #define VECTOR_SIZE_HBM_BANK_17 56
// #define VECTOR_SIZE_HBM_BANK_18 56
// #define VECTOR_SIZE_HBM_BANK_19 56
// #define VECTOR_SIZE_HBM_BANK_20 56
// #define VECTOR_SIZE_HBM_BANK_21 56
// #define VECTOR_SIZE_HBM_BANK_22 56
// #define VECTOR_SIZE_HBM_BANK_23 56
// #define VECTOR_SIZE_HBM_BANK_24 56
// #define VECTOR_SIZE_HBM_BANK_25 56
// #define VECTOR_SIZE_HBM_BANK_26 56
// #define VECTOR_SIZE_HBM_BANK_27 56 -> (2 * 56) * 32 / 512 = 7 x 512 bits

// HBM 0~7: gather_1_embeddings_len_12
// HBM 8~27: gather_2_embeddings_len_14

// #define VECTOR_SIZE_DDR_BANK_0 64
// #define VECTOR_SIZE_DDR_BANK_1 64 -> 64 * 32 / 512 = 4 x 512 bits

// DDR 0~1: gather_1_embeddings_len_16

    // 1952 * 32 / 512 = 122
//     hls::stream<network_t> s_embedding_gather_level_A[INPUT_SIZE_AXI_512];
// #pragma HLS stream variable=s_embedding_gather_level_A depth=2
// #pragma HLS resource variable=s_embedding_gather_level_A core=FIFO_SRL
// #pragma HLS array_partition variable=s_embedding_gather_level_A dim=1 complete


    hls::stream<network_t> s_embedding_gather_level_A_out_2[3];
#pragma HLS stream variable=s_embedding_gather_level_A_out_2 depth=64
#pragma HLS array_partition variable=s_embedding_gather_level_A_out_2 dim=1 complete

    hls::stream<network_t> s_embedding_gather_level_A_out_3[8];
#pragma HLS stream variable=s_embedding_gather_level_A_out_3 depth=64
#pragma HLS array_partition variable=s_embedding_gather_level_A_out_3 dim=1 complete

    hls::stream<network_t> s_embedding_gather_level_A_out_4[2];
#pragma HLS stream variable=s_embedding_gather_level_A_out_4 depth=64
#pragma HLS array_partition variable=s_embedding_gather_level_A_out_4 dim=1 complete

    hls::stream<network_t> s_embedding_gather_level_A_out_7[12];
#pragma HLS stream variable=s_embedding_gather_level_A_out_7 depth=64
#pragma HLS array_partition variable=s_embedding_gather_level_A_out_7 dim=1 complete

// 3 * 8 + 3 * 2 + 12 * 7 + 2 * 4 = 24 + 6 + 84 + 8 = 122

    // PLRAM 0~7: gather_4_embeddings_len_7
    // PLRAM 8~10: gather_1_embeddings_len_8
    // 2 * 7 + 3 * 2 = 20 -> s_embedding_gather_level_A[0:19]
    group_4_embeddings_len7_out_7(
        s_embedding_buffer_PLRAM0, s_embedding_buffer_PLRAM1,
        s_embedding_buffer_PLRAM2, s_embedding_buffer_PLRAM3,
        s_embedding_gather_level_A_out_7[0], batch_num);
    group_4_embeddings_len7_out_7(
        s_embedding_buffer_PLRAM4, s_embedding_buffer_PLRAM5,
        s_embedding_buffer_PLRAM6, s_embedding_buffer_PLRAM7,
        s_embedding_gather_level_A_out_7[1], batch_num);
    group_1_embeddings_len8_out_2(
        s_embedding_buffer_PLRAM8, 
        s_embedding_gather_level_A_out_2[0], batch_num);
    group_1_embeddings_len8_out_2(
        s_embedding_buffer_PLRAM9, 
        s_embedding_gather_level_A_out_2[1], batch_num);
    group_1_embeddings_len8_out_2(
        s_embedding_buffer_PLRAM10, 
        s_embedding_gather_level_A_out_2[2], batch_num);

    // HBM 0~7: gather_1_embeddings_len_12
    // HBM 8~27: gather_2_embeddings_len_14
    // 8 * 3 + 10 * 7 = 94 -> s_embedding_gather_level_A[20:113]
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM0, 
        s_embedding_gather_level_A_out_3[0], batch_num);
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM1, 
        s_embedding_gather_level_A_out_3[1], batch_num);
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM2, 
        s_embedding_gather_level_A_out_3[2], batch_num);
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM3, 
        s_embedding_gather_level_A_out_3[3], batch_num);
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM4, 
        s_embedding_gather_level_A_out_3[4], batch_num);
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM5, 
        s_embedding_gather_level_A_out_3[5], batch_num);
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM6, 
        s_embedding_gather_level_A_out_3[6], batch_num);
    group_1_embeddings_len12_out_3(
        s_embedding_buffer_HBM7, 
        s_embedding_gather_level_A_out_3[7], batch_num);
    
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM8, s_embedding_buffer_HBM9,
        s_embedding_gather_level_A_out_7[2], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM10, s_embedding_buffer_HBM11,
        s_embedding_gather_level_A_out_7[3], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM12, s_embedding_buffer_HBM13,
        s_embedding_gather_level_A_out_7[4], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM14, s_embedding_buffer_HBM15,
        s_embedding_gather_level_A_out_7[5], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM16, s_embedding_buffer_HBM17,
        s_embedding_gather_level_A_out_7[6], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM18, s_embedding_buffer_HBM19,
        s_embedding_gather_level_A_out_7[7], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM20, s_embedding_buffer_HBM21,
        s_embedding_gather_level_A_out_7[8], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM22, s_embedding_buffer_HBM23,
        s_embedding_gather_level_A_out_7[9], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM24, s_embedding_buffer_HBM25,
        s_embedding_gather_level_A_out_7[10], batch_num);
    group_2_embeddings_len14_out_7(
        s_embedding_buffer_HBM26, s_embedding_buffer_HBM27,
        s_embedding_gather_level_A_out_7[11], batch_num);

    // DDR 0~1: gather_1_embeddings_len_16
    // 2 x 4 = 8 -> s_embedding_gather_level_A[114:121]
    group_1_embeddings_len16_out_4(
        s_embedding_buffer_DDR0, 
        s_embedding_gather_level_A_out_4[0], batch_num);
    group_1_embeddings_len16_out_4(
        s_embedding_buffer_DDR1, 
        s_embedding_gather_level_A_out_4[1], batch_num);

    // gather all 22 streams into 1
    gather_122_embedding_streams(
        s_embedding_gather_level_A_out_2,
        s_embedding_gather_level_A_out_3,
        s_embedding_gather_level_A_out_4,
        s_embedding_gather_level_A_out_7, 
        s_network, batch_num);
}

void consume_and_write(
    // 22 = 352 (input feature len) * 4 * 8 / 512 bit
    hls::stream<network_t>& s_network, network_t* out_RAM, int batch_num
) {
    // consume all gathered embedding vectors, only write the last batch
    network_t out_local[BATCH_SIZE * INPUT_SIZE_AXI_512];

    for (int i = 0 ; i < batch_num; i++){
        for (int j = 0; j < BATCH_SIZE; j++) {
            for (int k = 0; k < INPUT_SIZE_AXI_512; k++) {
                #pragma HLS pipeline II=1
                out_local[j * INPUT_SIZE_AXI_512 + k] = s_network.read();
            }
        }
    }
    for (int k = 0; k < BATCH_SIZE * INPUT_SIZE_AXI_512; k++) {
        #pragma HLS pipeline II=1
        out_RAM[k] = out_local[k];
    }
}
