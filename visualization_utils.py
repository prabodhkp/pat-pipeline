# visualization_utils.py

import numpy as np
import matplotlib.pyplot as plt

def visualize_prediction(input_data, ground_truth, prediction, correlation, pred_num, start_frame, end_frame):
    fig = plt.figure(figsize=(18, 8))

    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(input_data, cmap='seismic', aspect='auto',
                     vmin=-np.max(np.abs(input_data)), 
                     vmax=np.max(np.abs(input_data)))
    ax1.set_title(f'Input (frames {start_frame}-{end_frame} avg)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Detectors')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(ground_truth, cmap='seismic', aspect='auto',
                     vmin=-np.max(np.abs(ground_truth)), 
                     vmax=np.max(np.abs(ground_truth)))
    ax2.set_title('Ground Truth (RFsum)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Detectors')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(prediction, cmap='seismic', aspect='auto',
                     vmin=-np.max(np.abs(ground_truth)), 
                     vmax=np.max(np.abs(ground_truth)))
    ax3.set_title(f'Prediction (Corr: {correlation:.4f})', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Detectors')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    detector_idx = 121
    ax4 = plt.subplot(2, 1, 2)
    ax4.plot(input_data[detector_idx, :], 'g-', label='Input', alpha=0.7)
    ax4.plot(ground_truth[detector_idx, :], 'b-', label='Ground Truth', alpha=0.7)
    ax4.plot(prediction[detector_idx, :], 'r--', label='Prediction', alpha=0.7)
    ax4.set_title(f'Signals at Detector {detector_idx}', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Prediction {pred_num}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_all_correlations(correlations):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, len(correlations) + 1)
    ax.plot(x, correlations, 'o-', color='steelblue', linewidth=2)
    ax.axhline(0.9, linestyle='--', color='green', label='0.9 threshold')
    ax.axhline(0.8, linestyle='--', color='orange', label='0.8 threshold')
    ax.set_title('Correlation: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediction Number')
    ax.set_ylabel('Correlation')
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
    stats = f"Mean: {np.mean(correlations):.4f}\nStd: {np.std(correlations):.4f}\nMin: {np.min(correlations):.4f}\nMax: {np.max(correlations):.4f}"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, va='top',
            bbox=dict(facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.show()
