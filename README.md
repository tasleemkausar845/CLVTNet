This repository contains code for running **CLVTNet: Efficient CNN-based Spatial Channel
Enhanced Vision Transformer for Auditory
Attention Detection Using EEG**.  
It supports:
- Data loading (KUL / DTU datasets)
- Preprocessing
- SSF feature map generation
- Training & evaluation
- Ablation studies
- Window-length sweeps
- Noise robustness (EEG-domain & image-domain)

---

## Project Structure
configs/ # Configuration files src/ config.py # Config definitions dataset_loaders.py # Dataset loaders for KUL & DTU data_pipeline.py # End-to-end data processing ssf_extraction.py # SSF map generation evaluation.py # Evaluation utilities training.py # Training utilities utils/ # Helper functions experiments/ run_noise_robustness.py # EEG & image noise tests run_window_length_sweep.py# Window length experiments run_ablation_studies.py # Ablation experiments main.py # Main entry point requirements.txt # Dependencies setup.py # Package setup


---

## Installation
git clone [https://github.com/yourusername/EEG-SSF-Classification.git](https://github.com/tasleemkausar845/CLVTNet) cd CLVTNet python3 -m venv venv source venv/bin/activate pip install -r requirements.txt


---
python src/training/train.py --dataset KUL --dataset_path data/KUL --output_dir outputs/train_kul --device cuda
python src/training/train.py --dataset DTU --dataset_path data/DTU --output_dir outputs/train_dtu --device cuda
---
python src/evaluate.py --dataset KUL --dataset_path data/KUL --checkpoint outputs/train_kul/best_model.pth --output_dir outputs/eval_kul
python src/experiments/run_noise_robustness.py --dataset KUL --dataset_path data/KUL --checkpoint outputs/train_kul/best_model.pth --output_dir outputs/noise_results_img --mode image_only 
python src/experiments/run_window_length_sweep.py --dataset KUL --dataset_path data/KUL --output_dir outputs/window_sweep --window_lengths 1.0 2.0 5.0 10.0


python src/experiments/run_ablation_studies.py --dataset KUL --dataset_path data/KUL --output_dir outputs/ablation
---



## Notes
- Update dataset paths in `configs` or pass via CLI.
- Supported datasets: `KUL`, `DTU`
- Make sure `locs_orig.mat` (electrode positions) is available in your dataset folder.

`

