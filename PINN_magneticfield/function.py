import numpy as np
import matplotlib.pyplot as plt
import ast
import re
import pyvista as pv
import matplotlib.pyplot as plt

class Function:
    def box_liner(b):#b(64, 64, 64)の形式で入力する磁場の描画
        #(64, 64, 64, 3)
        stride = 1
        mask = np.abs(b[:, :, 0, 2]) > 500
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


    def formatting_b(bx_pred, by_pred, bz_pred):#(x, 3)の形式の磁場を(x, x, x, 3)に変更
        x = bx_pred.reshape(-1)  # (262144,)
        y = by_pred.reshape(-1)
        z = bz_pred.reshape(-1)

        x3 = x.reshape(64, 64, 64)
        y3 = y.reshape(64, 64, 64)
        z3 = z.reshape(64, 64, 64)

        b = np.stack([x3, y3, z3], axis=-1)

        max_abs_bz = np.max(np.abs(b[:, :, :, 2]))
        print(max_abs_bz)
        return b


    def load_model_config(txt_path):#機械学習用のテキストファイルをニューラルネットワークに入力としてわたす用に変更
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Layers
        layers = ast.literal_eval(re.search(r"Layers:\s*(\[[^\]]+\])", text).group(1))
        # weight1
        weight1 = ast.literal_eval(re.search(r"weight1:\s*(\[[^\]]+\])", text).group(1))
        # weight2
        weight2 = ast.literal_eval(re.search(r"weight2\s*(\[[^\]]+\])", text).group(1))
        # Threshold
        TH = int(re.search(r"Threshold\s*(\d+)", text).group(1))
        # learning rate
        lr = float(re.search(r"learning:\s*([0-9.eE+-]+)", text).group(1))

        return layers, weight1, weight2, TH, lr

    def ploter(self, slice):#指定したsliceの描画関数
        #plt.figure(figsize=(6,5))
        plt.imshow(slice.T, origin="lower", cmap="inferno", vmin = 0, vmax = 1.5)
        #plt.colorbar(label="Relative Error")
        #plt.title(f"Relative Error (z={index})")
        #plt.xlabel("x index")
        #plt.ylabel("y index")
        plt.axis('off') #軸を消す
        plt.show()

    def xyzmap_relative_error(self, train_b, ref_b, index, xyz_axis):#相対誤差を出し描画
        #(64, 64, 64, 3)
        diff = np.linalg.norm(train_b - ref_b, axis=-1)  
        norm = np.linalg.norm(ref_b, axis = -1) +  1e-8 
        relative_error = diff/norm

        # --- z固定の2Dスライスを取り出し ---
        if xyz_axis == 0:
            relative_error_slice = relative_error[:, index, :]  # (64,64)
        else:
            relative_error_slice = relative_error[:, :, index]

        # --- 可視化 ---
        self.ploter(relative_error_slice)
        return relative_error


    def physics(b):#物理式の関数でdivbとjxbを計算
        bx = b[:, :, :, 0]
        by = b[:, :, :, 1]
        bz = b[:, :, :, 2]
    
        bx_x = np.gradient(bx, axis=0)
        by_y = np.gradient(by, axis=1)
        bz_z = np.gradient(bz, axis=2)
        
        bz_y = np.gradient(bz, axis=1)
        by_z = np.gradient(by, axis=2)
        bx_z = np.gradient(bx, axis=2)
        bz_x = np.gradient(bz, axis=0)
        by_x = np.gradient(by, axis=0)
        bx_y = np.gradient(bx, axis=1)

        jx = bz_y - by_z
        jy = bx_z - bz_x
        jz = by_x - bx_y

        jxb_x = jy*bz - jz*by
        jxb_y = jz*bx - jx*bz
        jxb_z = jx*by - jy*bx
        #print(min(jxb_x**2 + jxb_y**2 + jxb_z**2))
        return np.sqrt((bx_x + by_y + bz_z)**2), np.sqrt(jxb_x**2 + jxb_y**2 + jxb_z**2)


    def divb(self, b, index):#divbに対する描画関数
        div, _ = self.physics(b)
        D = div[:, :, index]

        self.ploter(D.T)
        return div

    def jxb(self, b, index):#jxbに対する描画関数
        _, ab_f = self.physics(b)
        F = ab_f[:, :, index]
        
        self.ploter(F.T)
        return ab_f

if __name__ == "__main__":
    print("OK")