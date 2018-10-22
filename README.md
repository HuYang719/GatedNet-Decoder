# GatedNet: Neural Network Decoding for LDPC over Impulsive Noise Channels

A novel neural network decoder (NND) for [low-density parity-check](https://en.wikipedia.org/wiki/Low-density_parity-check_code) (LDPC) decoding over impulsive noise channels. 

## System Model
Use [Symmetric alpha-stable](https://en.wikipedia.org/wiki/Stable_distribution) to model the impulsive noise. The structure of the system model is depicted as follow:
<p align="center">
  <img src=/images/Fig1.png/ width="90%" height="90%" alt="The structure of the system model.">
</p>

* The neurons are optimized with control gates, which shows much greater performance and robustness.

<p align="center">
  <img src=/images/Fig3.png/ width="75%" height="75%" alt="Shematics of a gated neuron and a normal neuron.">
</p>

* A partially connected layer (PCL) with specific connection is proposed to decrease the computation complexity and it brings an extra performance benefit.

<p align="center">
  <img src=/images/Fig2.png/ width="75%" height="75%" alt="Example structure of a GatedNet with PCL">
</p>

## Performance Result
* Simulation results show the proposed GatedNet decoder can improve the traditional decoding method - belief propagation (BP) at least 1 dB in different degrees of impulsive noise. The closer that parameter alpha is to 1, the stronger weaker impulsive noise is. Results show in [Result_Report.ipynb](reproduce_result.ipynb).

<p align="center">
  <img src=/images/Fig4.png/ alt="Comparison between GatedNet decoder with BP decoder ">
</p>

* GatedNet decoder has better performance than DNN decoder under the same computation complexity.
<p align="center">
  <img src=/images/Fig5.png/  alt="Comparison between GatedNet decoder with DNN decoder ">
</p>

* PCL can bring an extra performance benefit. PCL-Partially Connected Layer; FCL-Fully Connected Layer; RCL-Random Connected Layer(Use the same number weights of PCL)
<p align="center">
  <img src=/images/Fig6_1.png/ width="75%" height="75%" alt="Comparison between fully connected, partially connected,  and random connected Layer. ">
</p>

## Usage

#### 1. Install dependencies
```
conda env create -f environment.yml
source activate deepcom2
```
#### 2. Add PCL Layers in keras.layers.core

* [GatedPCL.py](layers/GatedPCL.py) and [GatedRanPCL.py](layers/GatedRanPCL.py) are implementation of Partially Connected Layer and Random Partially Connected Layer. Add them in keras.layers.core.

#### 3. (Recommend) IPython Notebook for training/benchmarking GatedNet decoder.

* [reproduce_result.ipynb](reproduce_result.ipynb): A Jypyter notebook demonstrates how to train a Neural Decoder and compare  the performance with  other Decoders. You can see the details about these models.

#### 4. (Optional) Modify the model for your own projects.

* The main models are implementation in [Model](Model), and some important communication functions are in [CommFunc.py](Model/CommFunc.py). The BP algorithm is implementation in matlab. 




