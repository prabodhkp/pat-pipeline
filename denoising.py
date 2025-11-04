# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 19:41:22 2025

@author: TRUE Lab
"""

"""
Denoising module for PAT signal processing
"""
import os
import numpy as np
import scipy.io as sio
from run_denoise_utils_Delay_Modified import test_new_data2
from my_visualization import make_sinogram_video


class Denoiser:
    """Handle denoising of PAT signals"""
    
    def __init__(self, config):
        self.config = config
        self.results_denoise = None
        self.results = None
        self.RF = None
        self.RFsum = None
    
    def denoise(self):
        """Run denoising on the input data"""
        print("🔍 Starting denoising...")
        
        self.results_denoise = test_new_data2(
            file_path=self.config.get_file_path(),
            model_path=self.config.model_path,
            norm_params_path=self.config.norm_params_path,
            fc1=self.config.fc1,
            fc2=self.config.fc2,
            Fs_model=self.config.Fs_model,
            target_time_samples=self.config.target_samples,
            n_avg=self.config.n_avg
        )
        
        self.results = self.results_denoise['predictions']
        self.RF = self.results_denoise['preprocessed_RF']
        self.RFsum = self.results_denoise['RFsum']
        
        print(f"✅ Denoising completed. Shape: {self.results.shape}")
        return self.results
    
    def save_results(self):
        """Save denoised results to MAT file"""
        base_name = self.config.get_base_name()
        save_name = f"Denoised_{base_name}_nAvg{self.config.n_avg}.mat"
        save_path = os.path.join(self.config.folder_name, save_name)
        
        sio.savemat(
            save_path,
            {'results': self.results, 'Fs_model': self.config.Fs_model},
            do_compression=True
        )
        
        print(f"✅ Saved denoised data: {save_path}")
        return save_path
    
    def create_visualization(self):
        """Create video visualization of denoising results"""
        base_name = self.config.get_base_name()
        video_name = f"Denoising_{base_name}_nAvg{self.config.n_avg}.mp4"
        
        make_sinogram_video(
            self.results,
            self.RF,
            self.RFsum,
            self.config.n_avg,
            video_name,
            self.config.folder_name,
            fps=self.config.fps
        )
        
        print(f"✅ Created visualization: {video_name}")
        return video_name