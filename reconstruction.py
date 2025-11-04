"""
Time reversal reconstruction module for PAT
"""
import os
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import cv2 as cv  # make cv point to cv2
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from my_visualization import plot_mip, save_volume_video


class TimeReversalReconstructor:
    """Handle time reversal reconstruction"""
    
    def __init__(self, config):
        self.config = config
        self.Nx, self.Ny, self.Nz = config.get_grid_size()
        self.kgrid = None
        self.medium = None
        self.sensor_mask = None
        self.det_cords = None
        
        print(f"Grid size: ({self.Nx}, {self.Ny}, {self.Nz})")
        print(f"Domain size: {config.Lx}m x {config.Ly}m x {config.Lz}m")
    
    def setup_grid(self):
        """Initialize k-Wave grid"""
        self.kgrid = kWaveGrid(
            [self.Nx, self.Ny, self.Nz],
            [self.config.dx, self.config.dy, self.config.dz]
        )
        
        dt = 0.3 * min(self.config.dx, self.config.dy, self.config.dz) / self.config.c0
        self.kgrid.makeTime(self.config.c0, dt)
        
        print("✅ Grid initialized")
    
    def setup_medium(self):
        """Initialize medium properties"""
        self.medium = kWaveMedium(
            sound_speed=self.config.c0,
            density=self.config.density
        )
        print("✅ Medium initialized")
    
    def load_detector_positions(self):
        """Load and setup detector positions"""
        mat = sio.loadmat(self.config.detector_file)
        self.det_cords = mat["det_cords"]
        self.det_cords[:, 2] = self.config.detector_z
        
        print(f"Number of detectors: {self.det_cords.shape[0]}")
        print(f"Detector Z position: {self.config.detector_z} m")
        
        # Convert to grid indices
        det_x_ind = np.round((self.det_cords[:, 0] + self.config.Lx/2) / self.config.dx).astype(int)
        det_y_ind = np.round((self.det_cords[:, 1] + self.config.Ly/2) / self.config.dy).astype(int)
        det_z_ind = np.round((self.det_cords[:, 2] + self.config.Lz/2) / self.config.dz).astype(int)
        
        # Clip to bounds
        det_x_ind = np.clip(det_x_ind, 0, self.Nx-1)
        det_y_ind = np.clip(det_y_ind, 0, self.Ny-1)
        det_z_ind = np.clip(det_z_ind, 0, self.Nz-1)
        
        # Create sensor mask
        self.sensor_mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        for i in range(len(det_x_ind)):
            self.sensor_mask[det_x_ind[i], det_y_ind[i], det_z_ind[i]] = True
        
        print(f"✅ Active sensor points: {np.sum(self.sensor_mask)}")
        return det_z_ind[0]
    
    def reconstruct_frame(self, sensor_data, frame_idx):
        """Reconstruct a single frame using time reversal"""
        N_det, Nt = sensor_data.shape
        
        print(f"Sensor data shape: {sensor_data.shape}")
        print(f"Max signal amplitude: {np.max(np.abs(sensor_data)):.2e}")
        
        # Update grid time parameters
        dt_data = 1.0 / self.config.Fs_exp
        t_end = (Nt - 1) * dt_data
        self.kgrid.setTime(Nt, dt_data)
        
        print(f"Time parameters:")
        print(f"  Sampling frequency: {self.config.Fs_exp/1e6:.2f} MHz")
        print(f"  Time step: {dt_data*1e6:.2f} μs")
        print(f"  Total time: {t_end*1e3:.2f} ms")
        print(f"  Number of time steps: {self.kgrid.Nt}")
        
        # Setup sensor
        sensor = kSensor()
        sensor.mask = self.sensor_mask
        sensor.time_reversal_boundary_data = sensor_data.astype(np.float32)
        
        # Empty source
        source = kSource()
        source.p_mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        
        # Simulation options
        sim_opts = SimulationOptions()
        sim_opts.pml_inside = False
        sim_opts.pml_size = self.config.pml_size
        sim_opts.data_cast = 'single'
        sim_opts.save_to_disk = True
        sim_opts.record_movie = False
        
        exec_opts = SimulationExecutionOptions()
        exec_opts.is_gpu_simulation = self.config.use_gpu
        exec_opts.verbose_level = 1
        
        print("\nSimulation options:")
        print(f"  GPU enabled: {exec_opts.is_gpu_simulation}")
        print(f"  PML size: {sim_opts.pml_size}")
        print(f"  Data cast: {sim_opts.data_cast}")
        
        # Run reconstruction
        print("\nStarting time reversal simulation...")
        recon_result = kspaceFirstOrder3D(
            self.kgrid, source, sensor, self.medium,
            sim_opts, exec_opts
        )
        
        print("Reconstruction complete!")
        
        # Extract and process pressure field
        pressure_field = recon_result['p_final']
        pressure_field = np.transpose(pressure_field, (2, 1, 0))
        pressure_field[pressure_field < 0] = 0
        pressure_field[:, :, 154:] = 0
        
        # Display the reconstruction
        plot_mip(pressure_field, h=1e-3)
        
        print(f"Pressure field shape: {pressure_field.shape}")
        print(f"Pressure field dtype: {pressure_field.dtype}")
        print(f"Max reconstructed pressure: {np.max(np.abs(pressure_field)):.2e} Pa")
        print(f"Min reconstructed pressure: {np.min(pressure_field):.2e} Pa")
        
        return pressure_field
    
    def extract_subvolume(self, pressure_field, detector_slc):
        """Extract and prepare subvolume for PINN"""
        # Find max location
        linear_idx = np.argmax(pressure_field)
        max_val = pressure_field.flat[linear_idx]
        BP_row, BP_col, BP_slc = np.unravel_index(linear_idx, pressure_field.shape)
        
        # Adjust boundaries
        if BP_slc + 48 > pressure_field.shape[2] - 1:
            BP_slc = pressure_field.shape[2] - 48 - 1
        
        BP_row = np.clip(BP_row, 32, 66)
        BP_col = np.clip(BP_col, 32, 66)
        
        # Extract subvolume
        subvol = pressure_field[
            BP_row-31:BP_row+33,
            BP_col-31:BP_col+33,
            BP_slc-47:BP_slc+49
        ]
        
        # Flip and normalize
        subvol = subvol[:, :, ::-1]
        max_subvol = np.max(subvol)
        subvol = subvol / max_subvol if max_subvol > 0 else subvol
        
        return subvol, (BP_row, BP_col, BP_slc), max_val
    
    def reconstruct_all_frames(self, sensor_data_all):
        """Reconstruct all frames"""
        self.setup_grid()
        self.setup_medium()
        detector_slc = self.load_detector_positions()
        
        Nframes = sensor_data_all.shape[2]
        PB_rec_exp = np.zeros(
            (Nframes, self.config.Nx_predict, self.config.Ny_predict, self.config.Nz_predict),
            dtype=float
        )
        BP_loc_shotwise = np.zeros((Nframes, 3), dtype=int)
        TR_max = np.zeros((Nframes, 1), dtype=float)
        TR_Visual_with_Det = np.zeros((Nframes, self.Nx, self.Ny, self.Nz), dtype=float)
        
        for frame_idx in range(Nframes):
            sensor_data = sensor_data_all[:, :, frame_idx]
            
            # Reconstruct
            pressure_field = self.reconstruct_frame(sensor_data, frame_idx)
            
            # Extract subvolume
            subvol, bp_loc, max_val = self.extract_subvolume(pressure_field, detector_slc)
            
            # Store results
            PB_rec_exp[frame_idx] = subvol
            BP_loc_shotwise[frame_idx] = bp_loc
            TR_max[frame_idx] = max_val
            TR_Visual_with_Det[frame_idx] = pressure_field + self.sensor_mask * max_val
            
            print(f"=======================================================")
            print(f"PB_rec_exp for frame number #{frame_idx + 1} Completed")
            print(f"=======================================================\n")
        
        return {
            'PB_rec_exp': PB_rec_exp,
            'BP_loc_shotwise': BP_loc_shotwise,
            'TR_max': TR_max,
            'TR_Visual_with_Det': TR_Visual_with_Det
        }
    
    def save_results(self, results):
        """Save reconstruction results"""
        base_name = self.config.get_base_name()
        save_name = f"TR_{base_name}_nAvg{self.config.n_avg}.mat"
        save_path = os.path.join(self.config.folder_name, save_name)
        
        sio.savemat(save_path, results, do_compression=True)
        print(f"✅ Saved TR results: {save_path}")
        
        # Create videos
        save_volume_video(
            results['PB_rec_exp'],
            f"TR_for_Pred{base_name}_nAvg{self.config.n_avg}.mp4",
            self.config.folder_name,
            h=1e-3,
            fps=self.config.fps
        )
        
        save_volume_video(
            results['TR_Visual_with_Det'],
            f"TR_Aligned_Det{base_name}_nAvg{self.config.n_avg}.mp4",
            self.config.folder_name,
            h=1e-3,
            fps=self.config.fps
        )
        
        return save_path
