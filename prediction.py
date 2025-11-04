"""
PINN prediction module for PAT reconstruction
"""
import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from Mymodels import unet_3d
from my_visualization import plot_mip, save_volume_video


class PINNPredictor:
    """Handle PINN-based predictions"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.setup_gpu()
        self.load_model()
    
    def setup_gpu(self):
        """Configure GPU settings"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        physical_devices = tf.config.list_physical_devices('GPU')
        print(f"Number of physical GPUs: {len(physical_devices)}")
        
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    def load_model(self):
        """Load the trained PINN model"""
        inp_size = (
            self.config.Nx_predict,
            self.config.Ny_predict,
            self.config.Nz_predict
        )
        
        with tf.device('/GPU:0'):
            self.model = unet_3d(inp_size)
        
        self.model.load_weights(self.config.pinn_weights)
        print(f"✅ Model loaded: {self.config.pinn_weights}")
    
    def predict(self, PB_rec_exp, TR_max):
        """Run predictions on all frames"""
        print(f"🔮 Starting PINN predictions...")
        print(f"Input shape: {PB_rec_exp.shape}")
        
        num_samples = PB_rec_exp.shape[0]
        PINN_Pred = np.zeros_like(PB_rec_exp, dtype=np.float32)
        
        for i in range(num_samples):
            print(f"  Predicting frame {i+1}/{num_samples}...")
            
            sample = PB_rec_exp[i:i+1]
            pred = self.model.predict(sample, verbose=0)
            pred = np.squeeze(pred)  # Remove all singleton dimensions
            pred *= TR_max[i]
            
            PINN_Pred[i] = pred
            plot_mip(pred, h=1e-3)
        
        print(f"✅ PINN predictions complete. Shape: {PINN_Pred.shape}")
        return PINN_Pred
    
    def align_predictions(self, PINN_Pred, BP_loc_shotwise, TR_max, sensor_mask):
        """Align predictions back to full domain"""
        Nx, Ny, Nz = self.config.get_grid_size()
        Nframes = PINN_Pred.shape[0]
        
        PINN_aligned = np.zeros((Nframes, Nx, Ny, Nz), dtype=float)
        PINN_Visual_with_Det = np.zeros((Nframes, Nx, Ny, Nz), dtype=float)
        
        PINN_Pred_flipped = PINN_Pred[:, :, :, ::-1]
        
        for frame_idx in range(Nframes):
            BP_row, BP_col, BP_slc = BP_loc_shotwise[frame_idx, :]
            
            # Define slices
            row_slice = slice(BP_row - 31, BP_row + 33)
            col_slice = slice(BP_col - 31, BP_col + 33)
            slc_slice = slice(BP_slc - 47, BP_slc + 49)
            
            # Assign prediction
            PINN_aligned[frame_idx, row_slice, col_slice, slc_slice] = \
                PINN_Pred_flipped[frame_idx, :, :, :]
            
            # Add detector visualization
            PINN_Visual_with_Det[frame_idx] = \
                PINN_aligned[frame_idx] + sensor_mask * TR_max[frame_idx]
        
        print("✅ Predictions aligned to full domain")
        return PINN_aligned, PINN_Visual_with_Det
    
    def save_results(self, PINN_Pred, PINN_aligned, PINN_Visual_with_Det):
        """Save prediction results"""
        base_name = self.config.get_base_name()
        
        # Save raw predictions
        save_name = f"PINN_{base_name}_nAvg{self.config.n_avg}.mat"
        save_path = os.path.join(self.config.folder_name, save_name)
        sio.savemat(save_path, {'PINN_Pred': PINN_Pred}, do_compression=True)
        print(f"✅ Saved PINN predictions: {save_path}")
        
        # Save aligned predictions
        save_name = f"PINN_Aligned_{base_name}_nAvg{self.config.n_avg}.mat"
        save_path = os.path.join(self.config.folder_name, save_name)
        sio.savemat(save_path, {'PINN_aligned': PINN_aligned}, do_compression=True)
        print(f"✅ Saved aligned predictions: {save_path}")
        
        # Create videos
        save_volume_video(
            PINN_Pred,
            f"PINN_{base_name}_nAvg{self.config.n_avg}.mp4",
            self.config.folder_name,
            h=1e-3,
            fps=self.config.fps
        )
        
        save_volume_video(
            PINN_Visual_with_Det,
            f"PINN_Aligned_{base_name}_nAvg{self.config.n_avg}.mp4",
            self.config.folder_name,
            h=1e-3,
            fps=self.config.fps
        )
        
        return save_path