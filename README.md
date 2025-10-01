# CLVTNet  
Efficient CNN-based Spatial Channel Enhanced Vision Transformer for Auditory Attention Detection Using EEG

![CLVTNet Architecture](assets/model.png)

---

## Overview  
This repository contains the official implementation of **CLVTNet**, an efficient **CNN-based Spatial Channel Enhanced Vision Transformer** for **Auditory Attention Detection (AAD)** using EEG signals.  

Our model integrates **CNN feature extraction** with a **Vision Transformer backbone** enhanced by a **spatial channel attention module**, enabling robust and efficient auditory attention decoding.

---

---

## Installation  

1. Clone this repository:
```bash
git clone https://github.com/tasleemkausar845/CLVTNet.git
cd CLVTNet
pip install -r requirements.txt
python src/training/train.py --config configs/train_config.yaml

