# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:12:05 2025

@author: TRUE Lab
"""

# my_visualization.py

#-------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import imageio

def make_sinogram_video(results, RF, RFsum, n_avg, video_name="sinogram_results.mp4", folder_name=None, fps=5):
    """
    Create a video of noisy, ground truth, and predicted sinograms.

    Parameters
    ----------
    results : np.ndarray
        Denoised results array of shape (N_det, Nt, Nframes).
    RF : np.ndarray
        Raw/noisy sinogram array of shape (N_det, Nt, Nraw_frames).
    RFsum : np.ndarray
        Ground truth sinogram (2D).
    n_avg : int
        Number of raw frames averaged per denoised frame.
    video_name : str, optional
        Output video filename (default: "sinogram_results.mp4").
    folder_name : str, optional
        Folder in which to save the video. Defaults to base_dir.
    fps : int, optional
        Frames per second for output video (default: 5).
    """
    import os
    import imageio
    import numpy as np
    import matplotlib.pyplot as plt

    # Default base folder
    base_dir = os.getcwd()  # Or replace with your preferred default
    if folder_name is None:
        folder_name = base_dir

    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Full output path
    out_path = os.path.join(folder_name, video_name)

    writer = imageio.get_writer(out_path, fps=fps)
    Tot_frames = results.shape[2]

    for frame_ind in range(Tot_frames):
        reshaped_frame = results[:, :, frame_ind]
        noisy_frame = np.mean(RF[:, :, frame_ind * n_avg:(frame_ind + 1) * n_avg], axis=2)

        r_denoised_rf = np.corrcoef(RFsum.flatten(), reshaped_frame.flatten())[0, 1]
        print(f"Correlation coefficient (denoised vs RFsum) for frame {frame_ind}: {r_denoised_rf:.4f}")

        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        im0 = axes[0].imshow(noisy_frame, aspect='auto', cmap='gray')
        axes[0].set_title(f"Noisy Sinogram (frame {frame_ind})")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(RFsum, aspect='auto', cmap='gray')
        axes[1].set_title("Ground Truth Sinogram (RFsum)")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(reshaped_frame, aspect='auto', cmap='gray')
        axes[2].set_title(f"Predicted Sinogram (r = {r_denoised_rf:.4f})")
        fig.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()

        # convert fig → image (RGBA → RGB)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        image = image[:, :, :3]
        writer.append_data(image)

        plt.close(fig)

    writer.close()
    print(f"✅ Video saved successfully!")
    print(f"📂 Location: {folder_name}")
    print(f"🎥 File: {os.path.basename(out_path)}")

















#------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def plot_mip(volume, h=None, cmap="hot"):
    """
    Plot maximum intensity projections (MIPs) of a 3D volume along x, y, z axes.
    """

    if h is not None:
        if np.isscalar(h):
            dx = dy = dz = float(h)
        else:
            dx, dy, dz = h
    else:
        dx = dy = dz = 1.0

    Nx, Ny, Nz = volume.shape
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    z = (np.arange(Nz) - Nz // 2) * dz

    proj_x = np.max(volume, axis=0)   # (Ny, Nz)
    proj_y = np.max(volume, axis=1)   # (Nx, Nz)
    proj_z = np.max(volume, axis=2)   # (Nx, Ny)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axs[0].imshow(proj_x.T, extent=[y.min(), y.max(), z.min(), z.max()],
                        origin="lower", cmap=cmap, aspect="equal")
    axs[0].set_title("Max over X-axis")
    axs[0].set_xlabel("y [m]" if h else "y [voxels]")
    axs[0].set_ylabel("z [m]" if h else "z [voxels]")
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(proj_y.T, extent=[x.min(), x.max(), z.min(), z.max()],
                        origin="lower", cmap=cmap, aspect="equal")
    axs[1].set_title("Max over Y-axis")
    axs[1].set_xlabel("x [m]" if h else "x [voxels]")
    axs[1].set_ylabel("z [m]" if h else "z [voxels]")
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    im3 = axs[2].imshow(proj_z.T, extent=[x.min(), x.max(), y.min(), y.max()],
                        origin="lower", cmap=cmap, aspect="equal")
    axs[2].set_title("Max over Z-axis")
    axs[2].set_xlabel("x [m]" if h else "x [voxels]")
    axs[2].set_ylabel("y [m]" if h else "y [voxels]")
    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return fig, axs



import pyvista as pv
import numpy as np
import os

def save_volume_video(frames, movie_name="volume_video.mp4", folder_name=None, h=None, fps=10, nticks=5):
    """
    Save a video of 4D volumetric frames with Maximum Intensity Projection (MIP) rendering.

    Parameters
    ----------
    frames : np.ndarray
        4D array with shape (Nframes, Nx, Ny, Nz).
    movie_name : str
        Output filename for the movie.
    folder_name : str, optional
        Folder in which to save the video. Defaults to base_dir.
    h : float, optional
        Grid resolution in meters. If given, axes are shown in cm with origin at grid center.
    fps : int
        Frames per second for the movie.
    nticks : int
        Number of ticks per axis.
    """
    import os
    import numpy as np
    import pyvista as pv

    # Default base folder
    base_dir = os.getcwd()  # Or replace with your preferred default
    if folder_name is None:
        folder_name = base_dir

    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Full output path
    out_path = os.path.join(folder_name, movie_name)

    Nframes, Nx, Ny, Nz = frames.shape

    print(f"Saving video '{os.path.basename(out_path)}' with {Nframes} frames...")
    print(f"➡️  Video will be stored in folder: {folder_name}")

    # Set up plotter (off_screen for movie writing)
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(out_path, framerate=fps)

    # spacing and origin for axes (in cm if h provided)
    if h is not None:
        spacing = (h * 100, h * 100, h * 100)  # cm
        origin = (-Nx/2 * spacing[0], -Ny/2 * spacing[1], -Nz/2 * spacing[2])
        label_units = " (cm)"
    else:
        spacing = (1, 1, 1)
        origin = (0, 0, 0)
        label_units = ""

    # Axis extents
    x_range = (origin[0], origin[0] + Nx * spacing[0])
    y_range = (origin[1], origin[1] + Ny * spacing[1])
    z_range = (origin[2], origin[2] + Nz * spacing[2])

    # Tick positions
    x_ticks = np.linspace(x_range[0], x_range[1], nticks)
    y_ticks = np.linspace(y_range[0], y_range[1], nticks)
    z_ticks = np.linspace(z_range[0], z_range[1], nticks)

    for frame_ind in range(Nframes):
        print(f"Rendering frame {frame_ind+1}/{Nframes}...")
        volume = frames[frame_ind].astype(np.float32)

        # Wrap into a grid
        grid = pv.wrap(volume)
        grid.spacing = spacing
        grid.origin = origin

        plotter.clear()
        actor = plotter.add_volume(grid, cmap="seismic", opacity="linear")
        actor.GetMapper().SetBlendModeToMaximumIntensity()
        plotter.add_mesh(grid.outline(), color="green", line_width=2)
        plotter.add_text(f"Frame #{frame_ind+1}", position="upper_left", font_size=14, color="green", shadow=True)

        # Manual axes
        plotter.add_lines(np.array([[x_range[0], origin[1], origin[2]], [x_range[1], origin[1], origin[2]]]), color="white", width=2)
        plotter.add_lines(np.array([[origin[0], y_range[0], origin[2]], [origin[0], y_range[1], origin[2]]]), color="white", width=2)
        plotter.add_lines(np.array([[origin[0], origin[1], z_range[0]], [origin[0], origin[1], z_range[1]]]), color="white", width=2)

        tick_points, tick_labels = [], []
        for xt in x_ticks:
            tick_points.append([xt, origin[1], origin[2]])
            tick_labels.append(f"{xt:.1f}")
        for yt in y_ticks:
            tick_points.append([origin[0], yt, origin[2]])
            tick_labels.append(f"{yt:.1f}")
        for zt in z_ticks:
            tick_points.append([origin[0], origin[1], zt])
            tick_labels.append(f"{zt:.1f}")

        plotter.add_point_labels(tick_points, tick_labels, font_size=10, text_color="white", point_size=0, render_points_as_spheres=False)
        plotter.add_text(f"X{label_units}", position="lower_right", font_size=12, color="white")
        plotter.add_text(f"Y{label_units}", position="lower_left", font_size=12, color="white")
        plotter.add_text(f"Z{label_units}", position="upper_right", font_size=12, color="white")

        plotter.write_frame()

    plotter.close()
    print(f"✅ Video saved successfully!")
    print(f"📂 Location: {folder_name}")
    print(f"🎥 File: {os.path.basename(out_path)}")



# def save_volume_video(frames, movie_name="volume_video.mp4", h=None, fps=10, nticks=5):
#     """
#     Save a video of 4D volumetric frames with Maximum Intensity Projection (MIP) rendering.

#     Parameters
#     ----------
#     frames : np.ndarray
#         4D array with shape (Nframes, Nx, Ny, Nz).
#     movie_name : str
#         Output filename for the movie (absolute or relative path).
#     h : float, optional
#         Grid resolution in meters. If given, axes are shown in cm with origin at grid center.
#     fps : int
#         Frames per second for the movie.
#     nticks : int
#         Number of ticks per axis.
#     """
#     Nframes, Nx, Ny, Nz = frames.shape

#     # Resolve absolute output path
#     out_path = os.path.abspath(movie_name)
#     out_dir = os.path.dirname(out_path)

#     print(f"Saving video '{os.path.basename(out_path)}' with {Nframes} frames...")
#     print(f"➡️  Video will be stored in folder: {out_dir}")

#     # Set up plotter (off_screen for movie writing)
#     plotter = pv.Plotter(off_screen=True)
#     plotter.open_movie(out_path, framerate=fps)

#     # spacing and origin for axes (in cm if h provided)
#     if h is not None:
#         spacing = (h * 100, h * 100, h * 100)  # cm
#         origin = (-Nx/2 * spacing[0], -Ny/2 * spacing[1], -Nz/2 * spacing[2])
#         label_units = " (cm)"
#     else:
#         spacing = (1, 1, 1)
#         origin = (0, 0, 0)
#         label_units = ""

#     # Axis extents
#     x_range = (origin[0], origin[0] + Nx * spacing[0])
#     y_range = (origin[1], origin[1] + Ny * spacing[1])
#     z_range = (origin[2], origin[2] + Nz * spacing[2])

#     # Tick positions
#     x_ticks = np.linspace(x_range[0], x_range[1], nticks)
#     y_ticks = np.linspace(y_range[0], y_range[1], nticks)
#     z_ticks = np.linspace(z_range[0], z_range[1], nticks)

#     for frame_ind in range(Nframes):
#         print(f"Rendering frame {frame_ind+1}/{Nframes}...")

#         volume = frames[frame_ind].astype(np.float32)

#         # Wrap into a grid
#         grid = pv.wrap(volume)
#         grid.spacing = spacing
#         grid.origin = origin

#         plotter.clear()

#         # Add volume rendering (MIP mode)
#         actor = plotter.add_volume(grid, cmap="seismic", opacity="linear")
#         actor.GetMapper().SetBlendModeToMaximumIntensity()

#         # Add cuboid outline
#         plotter.add_mesh(grid.outline(), color="green", line_width=2)

#         # Add frame title
#         plotter.add_text(
#             f"Frame #{frame_ind+1}",
#             position="upper_left",
#             font_size=14,
#             color="green",
#             shadow=True
#         )

#         # --- Manual axes ---
#         # X axis line
#         plotter.add_lines(np.array([[x_range[0], origin[1], origin[2]],
#                                     [x_range[1], origin[1], origin[2]]]),
#                           color="white", width=2)
#         # Y axis line
#         plotter.add_lines(np.array([[origin[0], y_range[0], origin[2]],
#                                     [origin[0], y_range[1], origin[2]]]),
#                           color="white", width=2)
#         # Z axis line
#         plotter.add_lines(np.array([[origin[0], origin[1], z_range[0]],
#                                     [origin[0], origin[1], z_range[1]]]),
#                           color="white", width=2)

#         # Tick marks + labels
#         tick_points = []
#         tick_labels = []

#         for xt in x_ticks:
#             tick_points.append([xt, origin[1], origin[2]])
#             tick_labels.append(f"{xt:.1f}")
#         for yt in y_ticks:
#             tick_points.append([origin[0], yt, origin[2]])
#             tick_labels.append(f"{yt:.1f}")
#         for zt in z_ticks:
#             tick_points.append([origin[0], origin[1], zt])
#             tick_labels.append(f"{zt:.1f}")

#         plotter.add_point_labels(
#             tick_points, tick_labels,
#             font_size=10, text_color="white", point_size=0, render_points_as_spheres=False
#         )

#         # Axis titles
#         plotter.add_text(f"X{label_units}", position="lower_right", font_size=12, color="white")
#         plotter.add_text(f"Y{label_units}", position="lower_left", font_size=12, color="white")
#         plotter.add_text(f"Z{label_units}", position="upper_right", font_size=12, color="white")

#         # Write this frame
#         plotter.write_frame()

#     plotter.close()

#     print(f"✅ Video saved successfully!")
#     print(f"📂 Location: {out_dir}")
#     print(f"🎥 File: {os.path.basename(out_path)}")
