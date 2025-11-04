# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 19:40:56 2025

@author: TRUE Lab
"""

"""
Configuration module for PAT reconstruction pipeline
"""
import os
import sys

class Config:
    """Configuration class for PAT reconstruction"""
    
    def __init__(self):
        # Paths
        self.base_dir = r"Y:\Denoising_PINN_GUI_Nov2025"
        self.folder_name = r'Y:\Denoising_PINN_GUI_Nov2025\Bot_left_Trial_Data'
        self.file_name = '18_40_52timeBottomLeft.mat'
        self.detector_file = "MatrixAray_det_cords.mat"
        
        # Model paths
        self.n_avg = 10
        self.model_path = f'best_sinogram_denoiser_nAvg{self.n_avg}frames_07Oct2025.h5'
        self.norm_params_path = f'sinogram_denoiser_final_nAvg{self.n_avg}frames_07Oct2025_norm_params.npy'
        self.pinn_weights = 'epoch_194_val_loss_0.00083.h5'
        
        # Signal processing
        self.fc1 = 0.01e6
        self.fc2 = 1e6
        self.Fs_model = 13.33e6
        self.target_samples = 1024
        self.Fs_exp = 40e6 / 3
        
        # Domain parameters
        self.Lx, self.Ly, self.Lz = 0.10, 0.10, 0.18
        self.dx, self.dy, self.dz = 1e-3, 1e-3, 1e-3
        self.Nx_predict, self.Ny_predict, self.Nz_predict = 64, 64, 96
        
        # Medium properties
        self.c0 = 1500.0  # sound speed [m/s]
        self.density = 1000.0  # [kg/m³]
        
        # Simulation options
        self.pml_size = [10, 10, 10]
        self.use_gpu = True
        self.detector_z = 0.07
        
        # Video parameters
        self.fps = 4
        
    def setup_paths(self):
        """Add base directory and subdirectories to Python path"""
        for root, dirs, files in os.walk(self.base_dir):
            sys.path.append(root)
    
    def get_grid_size(self):
        """Calculate grid dimensions"""
        import numpy as np
        Nx = int(np.round(self.Lx / self.dx))
        Ny = int(np.round(self.Ly / self.dy))
        Nz = int(np.round(self.Lz / self.dz))
        return Nx, Ny, Nz
    
    def get_file_path(self):
        """Get full path to data file"""
        return os.path.join(self.folder_name, self.file_name)
    
    def get_base_name(self):
        """Get base filename without extension"""
        return os.path.splitext(self.file_name)[0]