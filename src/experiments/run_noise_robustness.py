import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scipy import signal as scipy_signal

from src.config import Config
from ablation_configs import AblationConfig
from dataset_loaders import KULDatasetLoader, DTUDatasetLoader
from ssf_extraction import SSFExtractor
from data_pipeline_updated import EEGDataPipeline

# -------------------
# Logging setup
# -------------------
def setup_logging(output_dir: str):
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'noise_robustness_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# -------------------
# EEG-domain noise injections
# -------------------
def add_line_noise(eeg_data, fs, noise_freq=50.0, noise_amplitude=0.1):
    n_samples = eeg_data.shape[0]
    t = np.arange(n_samples) / fs
    line_noise = noise_amplitude * np.sin(2 * np.pi * noise_freq * t[:, None])
    noisy_data = eeg_data + line_noise * np.std(eeg_data, axis=0, keepdims=True)
    return noisy_data

def add_eog_noise(eeg_data, fs, noise_amplitude=0.5):
    n_samples, n_channels = eeg_data.shape
    blink_frequency = 0.2  # Hz
    n_blinks = int(n_samples / fs * blink_frequency)
    eog_signal = np.zeros((n_samples, n_channels))
    for _ in range(n_blinks):
        blink_start = np.random.randint(0, n_samples - int(0.3 * fs))
        blink_duration = int(0.3 * fs)
        blink_template = np.exp(-np.linspace(0, 10, blink_duration))
        blink_template /= np.max(blink_template)
        frontal_channels = slice(0, min(8, n_channels))
        eog_signal[blink_start:blink_start+blink_duration, frontal_channels] += blink_template[:, None] * noise_amplitude
    noisy_data = eeg_data + eog_signal * np.std(eeg_data, axis=0, keepdims=True)
    return noisy_data

def add_emg_noise(eeg_data, fs, noise_amplitude=0.3):
    n_samples, n_channels = eeg_data.shape
    b, a = scipy_signal.butter(4, [20, 100], btype='band', fs=fs)
    emg_noise = np.random.randn(n_samples, n_channels)
    emg_noise = scipy_signal.filtfilt(b, a, emg_noise, axis=0)
    temporal_channels = slice(max(0, n_channels//2), n_channels)
    noisy_data = eeg_data.copy()
    noisy_data[:, temporal_channels] += emg_noise[:, temporal_channels] * noise_amplitude * \
        np.std(eeg_data[:, temporal_channels], axis=0, keepdims=True)
    return noisy_data

# -------------------
# Image-space Gaussian noise
# -------------------
def add_gaussian_image_noise(img_batch, n=0.5, m=1.0):
    """
    img_batch: (B, C, H, W) tensor — SSF maps
    """
    noisy_batch = img_batch.clone()
    noise = torch.randn_like(noisy_batch)
    noisy_batch = n * noisy_batch + m * noise
    return noisy_batch

# -------------------
# Noise injection controller
# -------------------
def inject_noise(eeg_data, noise_type, fs, noise_level):
    if noise_type == 'none':
        return eeg_data
    elif noise_type == 'line':
        return add_line_noise(eeg_data, fs, noise_amplitude=noise_level)
    elif noise_type == 'eog':
        return add_eog_noise(eeg_data, fs, noise_amplitude=noise_level)
    elif noise_type == 'emg':
        return add_emg_noise(eeg_data, fs, noise_amplitude=noise_level)
    else:
        raise ValueError(f"Unknown EEG noise type: {noise_type}")

# -------------------
# Evaluation
# -------------------
def evaluate_noise_robustness(model, test_loader, device, noise_type, noise_level, fs, domain='eeg'):
    """
    domain='eeg' → apply noise to EEG before SSF extraction in tensor  
    domain='image' → apply Gaussian noise directly to SSF maps
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if domain == 'eeg' and noise_type != 'none':
                # Inputs here are SSF maps — in real EEG noise test, you'd inject before SSF creation
                # Since we already have SSF maps, we can't re-create from EEG here — pipeline would need pre-noisy EEG
                # In practice you'd run noise injection before pipeline preprocessing
                inputs_np = inputs.numpy()
                noisy_inputs = []
                for sample in inputs_np:
                    if sample.ndim == 3:
                        sample = sample[0]
                    noisy_sample = inject_noise(sample, noise_type, fs, noise_level)
                    noisy_inputs.append(noisy_sample[np.newaxis, ...])
                inputs = torch.FloatTensor(np.array(noisy_inputs))

            elif domain == 'image' and noise_type == 'gaussian_image':
                inputs = add_gaussian_image_noise(inputs, n=noise_level, m=1.0)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description='EEG/Image noise robustness evaluation')
    parser.add_argument('--dataset', type=str, default='KUL', choices=['KUL', 'DTU'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/noise_robustness')
    parser.add_argument('--mode', type=str, choices=['all_eeg', 'image_only'], default='all_eeg')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load config & pipeline
    config = Config()
    config.data.dataset_name = args.dataset
    config.data.dataset_path = args.dataset_path
    config.experiment.seed = args.seed

    if args.dataset == 'KUL':
        dataset_loader = KULDatasetLoader(config.data.dataset_path, config.data)
    else:
        dataset_loader = DTUDatasetLoader(config.data.dataset_path, config.data)

    ssf_extractor = SSFExtractor(config.data)
    pipeline = EEGDataPipeline(config, dataset_loader=dataset_loader, ssf_extractor=ssf_extractor)

    processed_data_dir = Path(args.output_dir) / 'processed_data'
    if not processed_data_dir.exists():
        pipeline.process_raw_data(output_dir=str(processed_data_dir))

    _, _, test_loader = pipeline.create_data_loaders(str(processed_data_dir))

    from clvtnet import CLVTNet
    ablation_config = AblationConfig()
    model = CLVTNet(config, ablation_config=ablation_config)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    results = {}
    if args.mode == 'all_eeg':
        noise_types = ['none', 'line', 'eog', 'emg']
        for nt in noise_types:
            results[nt] = {}
            for nl in args.noise_levels:
                if nt == 'none' and nl > 0:
                    continue
                acc = evaluate_noise_robustness(model, test_loader, args.device, nt, nl, config.data.sampling_rate, domain='eeg')
                logger.info(f"[EEG noise] {nt} @ {nl}: {acc:.2f}%")
                results[nt][str(nl)] = acc

    elif args.mode == 'image_only':
        nt = 'gaussian_image'
        results[nt] = {}
        for nl in args.noise_levels:
            acc = evaluate_noise_robustness(model, test_loader, args.device, nt, nl, config.data.sampling_rate, domain='image')
            logger.info(f"[Image noise] Gaussian @ {nl}: {acc:.2f}%")
            results[nt][str(nl)] = acc

    results_file = Path(args.output_dir) / 'noise_robustness_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()