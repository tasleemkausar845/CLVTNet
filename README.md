Installation
git clone https://github.com/tasleemkausar845/CLVTNet.git
cd CLVTNet

python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
Usage
Training
python src/training/train.py --dataset KUL --dataset_path data/KUL --output_dir outputs/train_kul --device cuda
python src/training/train.py --dataset DTU --dataset_path data/DTU --output_dir outputs/train_dtu --device cuda
Evaluation
python src/evaluate.py --dataset KUL --dataset_path data/KUL --checkpoint outputs/train_kul/best_model.pth --output_dir outputs/eval_kul
Noise Robustness
python src/experiments/run_noise_robustness.py --dataset KUL --dataset_path data/KUL --checkpoint outputs/train_kul/best_model.pth --output_dir outputs/noise_results_img --mode image_only
Window-Length Sweep
python src/experiments/run_window_length_sweep.py --dataset KUL --dataset_path data/KUL --output_dir outputs/window_sweep --window_lengths 1.0 2.0 5.0 10.0
Ablation Studies
python src/experiments/run_ablation_studies.py --dataset KUL --dataset_path data/KUL --output_dir outputs/ablation
Notes
Update dataset paths in configs/ or provide via CLI.
Supported datasets: KUL, DTU.
Ensure locs_orig.mat (electrode positions) is present in your dataset folder.
