# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:31:18 2025

@author: TRUE Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:03:30 2025

@author: TRUE Lab
"""

# run_utils.py

from preprocessing_utils import (
    load_test_data, apply_delay_padding, zero_dead_channels,
    bandpass_filter, downsample_data, crop_or_pad
)
from model_utils import load_model_and_normalization, predict_in_chunks
from visualization_utils import visualize_prediction, plot_all_correlations
import numpy as np


def test_new_data2(
    file_path,
    model_path,
    norm_params_path,
    fc1,
    fc2,
    Fs_model,
    target_time_samples,
    n_avg
):
    print("=" * 70)
    print("RUNNING TEST ON NEW DATA")
    print("=" * 70)

    # --- Load and preprocess
    RF, Fs, delay = load_test_data(file_path)
    
    #RF = apply_delay_padding(RF, delay, Fs)
    RF = zero_dead_channels(RF)
    #RFsum = RF.mean(axis=2)

    RF = bandpass_filter(RF, fc1, fc2, Fs)
    #RFsum = bandpass_filter(RFsum, fc1, fc2, Fs)

    RF = downsample_data(RF, Fs, Fs_model)
    #RFsum = downsample_data(RFsum, Fs, Fs_model)
    print(f"DELAY IS: {delay}")

    #RF = apply_delay_padding(RF, delay, Fs_model)
    

    RF = crop_or_pad(RF, target_time_samples)
    RFsum = RF.mean(axis=2)
    #RFsum = crop_or_pad(RFsum, target_time_samples)

    # --- Load model
    model, norm_params = load_model_and_normalization(model_path, norm_params_path)

    # --- Predict
    predictions, correlations = predict_in_chunks(
        RF, RFsum, model, norm_params, target_time_samples, n_avg,
        visualize_callback=visualize_prediction
    )
    
    predictions = apply_delay_padding(predictions, delay, Fs_model)
    predictions = crop_or_pad(predictions, target_time_samples)


    # --- Visualize overall performance
    plot_all_correlations(correlations)

    # --- Print final results
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Min correlation: {np.min(correlations):.4f}")
    print(f"Max correlation: {np.max(correlations):.4f}")

    return {
        'predictions': predictions,
        'correlations': correlations,
        'RFsum': RFsum,
        'preprocessed_RF': RF
    }
