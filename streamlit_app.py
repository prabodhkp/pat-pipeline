"""
PAT Reconstruction Pipeline - Streamlit Web App (Fixed)
Integrates: Denoising → Time Reversal → PINN Prediction
"""
import cv2 as cv  # allow old k-Wave imports using 'cv' to work
import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import your modules
from config import Config
from denoising import Denoiser
from reconstruction import TimeReversalReconstructor
from prediction import PINNPredictor

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PAT Reconstruction Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = ""
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'working_folder' not in st.session_state:
    st.session_state.working_folder = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_uploaded_file(uploaded_file, save_dir):
    """Save uploaded file to temporary directory"""
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def create_denoise_plot(results, RF, RFsum, frame_idx):
    """Create denoising comparison plot"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    pred = results[:, :, frame_idx]
    rf_input = RF[:, :, frame_idx]
    
    # Plot 1: Input
    im1 = axes[0, 0].imshow(rf_input, cmap='seismic', aspect='auto', origin='upper')
    axes[0, 0].set_title(f'Input (frames {frame_idx*10}-{frame_idx*10+9} avg)')
    axes[0, 0].set_xlabel('Samples')
    axes[0, 0].set_ylabel('Detectors')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Ground Truth
    im2 = axes[0, 1].imshow(RFsum, cmap='seismic', aspect='auto', origin='upper')
    axes[0, 1].set_title('Ground Truth (RFsum)')
    axes[0, 1].set_xlabel('Samples')
    axes[0, 1].set_ylabel('Detectors')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Prediction
    im3 = axes[0, 2].imshow(pred, cmap='seismic', aspect='auto', origin='upper')
    corr = np.corrcoef(RFsum.flatten(), pred.flatten())[0, 1]
    axes[0, 2].set_title(f'Prediction (Corr: {corr:.4f})')
    axes[0, 2].set_xlabel('Samples')
    axes[0, 2].set_ylabel('Detectors')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Signal comparison
    detector_idx = 121
    axes[1, 0].plot(rf_input[detector_idx, :], label='Input', alpha=0.7)
    axes[1, 0].plot(RFsum[detector_idx, :], label='Ground Truth', alpha=0.7)
    axes[1, 0].plot(pred[detector_idx, :], label='Prediction', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title(f'Signals at Detector {detector_idx}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hide unused subplots
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    
    fig.suptitle(f'Prediction {frame_idx + 1}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig

def create_mip_plot(volume, title, cmap='hot'):
    """Create 3-panel MIP plot"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    h = 1e-3
    Nx, Ny, Nz = volume.shape
    x = (np.arange(Nx) - Nx // 2) * h
    y = (np.arange(Ny) - Ny // 2) * h
    z = (np.arange(Nz) - Nz // 2) * h
    
    # MIP along Z
    mip_xy = np.max(volume, axis=2)
    im1 = axes[0].imshow(mip_xy.T, extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3],
                         cmap=cmap, aspect='equal', origin='lower')
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].set_title('MIP along Z')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # MIP along Y
    mip_xz = np.max(volume, axis=1)
    im2 = axes[1].imshow(mip_xz.T, extent=[x[0]*1e3, x[-1]*1e3, z[0]*1e3, z[-1]*1e3],
                         cmap=cmap, aspect='equal', origin='lower')
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('z (mm)')
    axes[1].set_title('MIP along Y')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # MIP along X
    mip_yz = np.max(volume, axis=0)
    im3 = axes[2].imshow(mip_yz.T, extent=[y[0]*1e3, y[-1]*1e3, z[0]*1e3, z[-1]*1e3],
                         cmap=cmap, aspect='equal', origin='lower')
    axes[2].set_xlabel('y (mm)')
    axes[2].set_ylabel('z (mm)')
    axes[2].set_title('MIP along X')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🔬 PAT Reconstruction Pipeline")
    st.markdown("### Denoising → Time Reversal → PINN Prediction")
    
    # ========================================================================
    # SIDEBAR: Input Configuration
    # ========================================================================
    with st.sidebar:
        st.header("📁 Input Configuration")
        
        # Working Folder Input
        st.subheader("1. Working Folder")
        working_folder_input = st.text_input(
            "Enter Working Folder Path",
            value=st.session_state.working_folder if st.session_state.working_folder else "",
            placeholder="e.g., Y:\\Denoising_PINN_GUI_Nov2025",
            help="Full path to folder where results will be saved"
        )
        
        if working_folder_input:
            # Create folder if it doesn't exist
            if not os.path.exists(working_folder_input):
                try:
                    os.makedirs(working_folder_input)
                    st.success(f"✓ Created folder: {working_folder_input}")
                except Exception as e:
                    st.error(f"❌ Could not create folder: {str(e)}")
            else:
                st.success(f"✓ Using folder: {working_folder_input}")
            
            st.session_state.working_folder = working_folder_input
        else:
            st.warning("⚠️ Please specify a working folder")
        
        st.markdown("---")
        
        # RF Data File Upload
        st.subheader("2. RF Data File")
        rf_file = st.file_uploader(
            "Upload RF Data File (.mat)",
            type=['mat'],
            help="MATLAB file containing RF, RFsum, Fs, delay"
        )
        
        # n_avg input
        n_avg = st.number_input(
            "Number of Frames (n_avg)",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Number of frames to average"
        )
        
        st.markdown("---")
        st.header("🤖 Model Configuration")
        
        # Denoising Model
        denoise_model = st.file_uploader(
            "Denoising Model (.h5) - Optional",
            type=['h5'],
            help="Auto-detects if not provided"
        )
        
        # Denoising Norm Params
        denoise_npy = st.file_uploader(
            "Denoising Norm (.npy) - Optional",
            type=['npy'],
            help="Auto-detects if not provided"
        )
        
        # PINN Model
        pinn_model = st.file_uploader(
            "PINN Model (.h5) - Optional",
            type=['h5'],
            help="Uses default if not provided"
        )
        
        st.markdown("---")
        
        # Validation check
        can_start = (
            st.session_state.working_folder is not None and 
            os.path.exists(st.session_state.working_folder) and 
            rf_file is not None
        )
        
        # Start button
        start_btn = st.button(
            "🚀 Start Pipeline",
            type="primary",
            disabled=st.session_state.pipeline_running or not can_start,
            use_container_width=True
        )
        
        if not can_start and not st.session_state.pipeline_running:
            if not st.session_state.working_folder:
                st.error("❌ Working folder required")
            if not rf_file:
                st.error("❌ RF data file required")
    
    # ========================================================================
    # MAIN AREA: Status and Visualization
    # ========================================================================
    
    if rf_file is None or not st.session_state.working_folder:
        st.info("👈 Please configure the pipeline in the sidebar to begin")
        st.markdown("""
        ### Pipeline Stages:
        1. **Denoising**: Clean RF signals using trained neural network
        2. **Time Reversal**: Reconstruct 3D pressure field using k-Wave
        3. **PINN Prediction**: Enhance reconstruction using physics-informed neural network
        
        ### Required Configuration:
        1. **Working Folder**: Specify where results will be saved
        2. **RF Data (.mat)**: Contains RF signals, RFsum, sampling frequency
        3. Model files are auto-detected based on n_avg value
        
        ### File Size Limit:
        - Maximum upload size: **1 GB** (configured in .streamlit/config.toml)
        """)
        return
    
    # ========================================================================
    # RUN PIPELINE
    # ========================================================================
    
    if start_btn:
        st.session_state.pipeline_running = True
        
        # Use the user-specified working folder
        working_dir = st.session_state.working_folder
        
        try:
            # Save uploaded RF file to working folder
            rf_path = save_uploaded_file(rf_file, working_dir)
            st.success(f"✓ Saved RF data to: {rf_path}")
            
            # Setup configuration
            config = Config()
            config.folder_name = working_dir
            config.file_name = rf_file.name
            config.n_avg = n_avg
            
            # Handle optional model uploads
            if denoise_model:
                config.model_path = save_uploaded_file(denoise_model, working_dir)
            else:
                config.model_path = f'best_sinogram_denoiser_nAvg{n_avg}frames_07Oct2025.h5'
            
            if denoise_npy:
                config.norm_params_path = save_uploaded_file(denoise_npy, working_dir)
            else:
                config.norm_params_path = f'sinogram_denoiser_final_nAvg{n_avg}frames_07Oct2025_norm_params.npy'
            
            if pinn_model:
                config.pinn_weights = save_uploaded_file(pinn_model, working_dir)
            
            config.setup_paths()
            
            # Configure GPU
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                st.success(f"✓ GPU detected: {len(physical_devices)} device(s)")
            
            # ================================================================
            # STAGE 1: Denoising
            # ================================================================
            st.session_state.current_stage = "Denoising"
            st.header("🔊 Stage 1: Denoising")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            viz_placeholder = st.empty()
            
            status_text.text("Running signal denoising...")
            
            denoiser = Denoiser(config)
            denoised_results = denoiser.denoise()
            
            # Visualize denoising results
            num_predictions = denoised_results.shape[2]
            for pred_idx in range(num_predictions):
                progress_bar.progress((pred_idx + 1) / num_predictions)
                status_text.text(f"Displaying prediction {pred_idx + 1}/{num_predictions}")
                
                fig = create_denoise_plot(denoised_results, denoiser.RF, denoiser.RFsum, pred_idx)
                viz_placeholder.pyplot(fig)
                plt.close(fig)
            
            denoiser.save_results()
            st.success("✅ Denoising complete!")
            
            # ================================================================
            # STAGE 2: Time Reversal
            # ================================================================
            st.session_state.current_stage = "Time Reversal"
            st.header("🔄 Stage 2: Time Reversal Reconstruction")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            viz_placeholder = st.empty()
            
            reconstructor = TimeReversalReconstructor(config)
            reconstructor.setup_grid()
            reconstructor.setup_medium()
            detector_slc = reconstructor.load_detector_positions()
            
            Nframes = denoised_results.shape[2]
            PB_rec_exp = np.zeros(
                (Nframes, config.Nx_predict, config.Ny_predict, config.Nz_predict),
                dtype=float
            )
            BP_loc_shotwise = np.zeros((Nframes, 3), dtype=int)
            TR_max = np.zeros((Nframes, 1), dtype=float)
            TR_Visual_with_Det = np.zeros((Nframes, reconstructor.Nx, reconstructor.Ny, reconstructor.Nz), dtype=float)
            
            for frame_idx in range(Nframes):
                progress_bar.progress((frame_idx + 1) / Nframes)
                status_text.text(f"Reconstructing frame {frame_idx + 1}/{Nframes}")
                
                sensor_data = denoised_results[:, :, frame_idx]
                pressure_field = reconstructor.reconstruct_frame(sensor_data, frame_idx)
                
                # Visualize
                fig = create_mip_plot(pressure_field, f'Time Reversal - Frame {frame_idx + 1}', cmap='hot')
                viz_placeholder.pyplot(fig)
                plt.close(fig)
                
                subvol, bp_loc, max_val = reconstructor.extract_subvolume(pressure_field, detector_slc)
                
                PB_rec_exp[frame_idx] = subvol
                BP_loc_shotwise[frame_idx] = bp_loc
                TR_max[frame_idx] = max_val
                TR_Visual_with_Det[frame_idx] = pressure_field + reconstructor.sensor_mask * max_val
            
            tr_results = {
                'PB_rec_exp': PB_rec_exp,
                'BP_loc_shotwise': BP_loc_shotwise,
                'TR_max': TR_max,
                'TR_Visual_with_Det': TR_Visual_with_Det
            }
            
            reconstructor.save_results(tr_results)
            st.success("✅ Time reversal complete!")
            
            # ================================================================
            # STAGE 3: PINN Prediction
            # ================================================================
            st.session_state.current_stage = "PINN Prediction"
            st.header("🧠 Stage 3: PINN Prediction")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            viz_placeholder = st.empty()
            
            predictor = PINNPredictor(config)
            num_samples = PB_rec_exp.shape[0]
            PINN_Pred = np.zeros_like(PB_rec_exp, dtype=np.float32)
            
            for i in range(num_samples):
                progress_bar.progress((i + 1) / num_samples)
                status_text.text(f"Predicting frame {i + 1}/{num_samples}")
                
                sample = PB_rec_exp[i:i+1]
                pred = predictor.model.predict(sample, verbose=0)
                pred = np.squeeze(pred)
                pred *= TR_max[i]
                
                PINN_Pred[i] = pred
                
                # Visualize
                fig = create_mip_plot(pred, f'PINN Prediction - Frame {i + 1}', cmap='viridis')
                viz_placeholder.pyplot(fig)
                plt.close(fig)
            
            # Align predictions
            pinn_aligned, pinn_visual = predictor.align_predictions(
                PINN_Pred,
                BP_loc_shotwise,
                TR_max,
                reconstructor.sensor_mask
            )
            
            predictor.save_results(PINN_Pred, pinn_aligned, pinn_visual)
            st.success("✅ PINN prediction complete!")
            
            # ================================================================
            # COMPLETION
            # ================================================================
            st.balloons()
            st.success("🎉 Pipeline completed successfully!")
            
            # Display results summary
            st.header("📊 Results Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Denoised Data", f"{denoised_results.shape}")
            with col2:
                st.metric("TR Reconstructions", f"{PB_rec_exp.shape}")
            with col3:
                st.metric("PINN Predictions", f"{PINN_Pred.shape}")
            
            # Show output folder
            st.info(f"📁 All results saved to: **{working_dir}**")
            
            # List saved files
            st.header("💾 Saved Files")
            saved_files = [f for f in os.listdir(working_dir) if f.endswith(('.mat', '.mp4'))]
            if saved_files:
                for file in saved_files:
                    file_path = os.path.join(working_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    st.write(f"✓ {file} ({file_size:.2f} MB)")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            st.session_state.pipeline_running = False


if __name__ == "__main__":
    main()
