# CLVTNet: CNN-based Spatial Channel Enhanced Vision Transformer for EEG Classification

![CLVTNet Architecture](figure%202.png)

---

## 📌 Overview
CLVTNet is a hybrid deep learning framework that combines convolutional neural networks (CNN) with spatial-channel enhanced vision transformers to classify EEG signals effectively.  
This repository contains code for data preprocessing, training, and evaluation.

---

## 📂 Project Structure
Clone the repository and install all required dependencies:
```bash
git clone https://github.com/tasleemkausar845/CLVTNet.git
cd CLVTNet
pip install -r requirements.txt

python3 main.py \
    --config configs/default.yaml \
    --mode train \
    --data-dir /path/to/processed_eeg_data \
    --output-dir outputs \
    --log-level INFO
python3 main.py \
    --config configs/default.yaml \
    --mode evaluate \
    --model-path outputs/checkpoints/best_model.pth \
    --data-dir /path/to/processed_eeg_data \
    --output-dir outputs/evaluation \
    --log-level INFO
