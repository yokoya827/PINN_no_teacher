import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import pyvista as pv
import matplotlib.pyplot as plt


def box_liner(b, TH = 50):
    #(64, 64, 64, 3)
    stride = 1
    mask = np.abs(b[:, :, 0, 2]) > TH
    seeds = np.stack([np.where(mask == True)[0], 
                    np.where(mask == True)[1], 
                    np.zeros_like(np.where(mask == True)[0])], axis=1)
    seeds = seeds[::stride]
    seeds.shape

    b_resampled = b
    nx, ny, nz, _ = b_resampled.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    mesh = pv.StructuredGrid(xv, yv, zv)

    bx, by, bz = b_resampled[..., 0], b_resampled[..., 1], b_resampled[..., 2]
    vectors = np.stack([bx, by, bz], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)

    mesh["vector"] = vectors
    mesh.active_vectors_name = "vector"

    xx, yy = np.meshgrid(x, y, indexing='ij')
    seeds_xs = np.stack([xx[seeds[:, 0], seeds[:, 1]], 
                        yy[seeds[:, 0], seeds[:, 1]], 
                        np.zeros_like(seeds[:, 0])], axis=1)
    seeds_xs.shape
    pv.global_theme.notebook = True
    pv.global_theme.jupyter_backend = 'static'
    # pv.global_theme.jupyter_backend = 'trame'
    p = pv.Plotter(window_size=(1000, 600))

    p.show_bounds(
        grid='front',
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
    )
    p.add_mesh(mesh.outline(), color='k')

    seed = pv.PolyData(seeds_xs)
    strl = mesh.streamlines_from_source(seed, vectors='vector', integration_direction='both',
                                        max_time=10000, initial_step_length=0.1)
    p.add_mesh(strl.tube(radius=0.4), color='blue')

    bottom_subset = mesh.extract_subset((0, nx-1, 0, ny-1, 0, 0)).extract_surface()
    p.add_mesh(bottom_subset, cmap='gray', scalars='vector', component=2, clim=(-2000, 2000), 
            lighting=False, show_scalar_bar=False)

    p.camera_position = "xy"
    p.camera.roll = -30
    p.camera.elevation = -70
    p.camera.zoom(1.3)
    p.show()

def xyzmap_relative_error(train_b, ref_b, index, xyz_axis):
    #(64, 64, 64, 3)
    diff = np.linalg.norm(train_b - ref_b, axis=-1)  
    norm = np.linalg.norm(ref_b, axis = -1) +  1e-8 
    relative_error = diff/norm

    # --- z固定の2Dスライスを取り出し ---
    if xyz_axis == 0:
        relative_error_slice = relative_error[:, index, :]  # (64,64)
    else:
        relative_error_slice = relative_error[:, :, index]
    

    print(relative_error_slice.shape)
    # --- 可視化 ---
    plt.figure(figsize=(6,5))
    plt.imshow(relative_error_slice.T, origin="lower", cmap="inferno")
    plt.colorbar(label="Relative Error")
    plt.title(f"Relative Error (z={index})")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.show()
    return relative_error_slice

import os

def remove_empty_dirs(root_dir):
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for d in dirs:
            path = os.path.join(root, d)
            try:
                if not os.listdir(path):  # 中身が空
                    os.rmdir(path)
                    print(f"Removed empty dir: {path}")
            except OSError:
                pass
