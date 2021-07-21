Universal Memory Allocation in DLRM

Running the UMA/RMA 

1) create the criteoTB data using the bench/DATAGEN_TB.sh 
  - this takes a lot of time to run. 
2) Download the repo https://github.com/apd10/universal_memory_allocation
  - and run python3 setup install in that repo
3) run the bench/train_tb_rma_final.sh to run 1000x compression on official MLPerf DLRM Model
  - You can change the --rma-size to use required memory  



Deep Learning Recommendation Model for Personalization and Recommendation Systems:
Refer to the Original dlrm repo for dlrm description

---------------------------
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
