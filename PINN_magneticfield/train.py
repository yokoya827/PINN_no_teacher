import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from scipy.interpolate import griddata
from pyDOE import lhs
from scipy.stats import qmc
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import torch
import torch.nn as nn
import pyvista as pv
import matplotlib.pyplot as plt
from streamtracer import StreamTracer, VectorGrid
import pathlib

class PhysicsInformedNN(nn.Module):

    def __init__(self, x0, y0, bx0, by0, bz0, xyz_f, layers, min_xyz, max_xyz, TH, weight1, weight2, lr):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        z0 = np.zeros_like(x0)
        i = 0


        self.lb = torch.tensor(min_xyz, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(max_xyz, dtype=torch.float32).to(self.device)

        # training data
        self.x0 = torch.tensor(x0, dtype=torch.float32).to(self.device)
        self.y0 = torch.tensor(y0, dtype=torch.float32).to(self.device)
        self.z0 = torch.tensor(z0, dtype=torch.float32).to(self.device)

        self.bx0 = torch.tensor(bx0, dtype=torch.float32).to(self.device)
        self.by0 = torch.tensor(by0, dtype=torch.float32).to(self.device)
        self.bz0 = torch.tensor(bz0, dtype=torch.float32).to(self.device)

        self.x_f = torch.tensor(xyz_f[:,0:1], dtype=torch.float32).to(self.device)
        self.y_f = torch.tensor(xyz_f[:,1:2], dtype=torch.float32).to(self.device)
        self.z_f = torch.tensor(xyz_f[:,2:3], dtype=torch.float32).to(self.device)

        self.weight = weight1
        self.weight1 = weight1
        self.weight2 = weight2
        self.TH = TH
        self.lr = lr
        self.i = i


        # NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.to(self.device)

    # ---------------- NN ----------------
    def initialize_NN(self, layers):
        weights = nn.ParameterList()
        biases = nn.ParameterList()

        for l in range(len(layers)-1):
            W = nn.Parameter(torch.empty(layers[l], layers[l+1]))
            nn.init.xavier_normal_(W)
            b = nn.Parameter(torch.zeros(1, layers[l+1]))
            weights.append(W)
            biases.append(b)

        return weights, biases

    def neural_net(self, XYZ):
        H = 2.0*(XYZ - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(len(self.weights)-1):
            H = torch.tanh(H @ self.weights[l] + self.biases[l])
        return H @ self.weights[-1] + self.biases[-1]

    def net_b_bc(self, x, y, z):
        if x.dim() == 1:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            z = z.unsqueeze(1)

        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        bxyz = self.neural_net(torch.cat([x, y, z], dim=1))
        #bxyz = 1e3*bxyz
        bx, by, bz = bxyz[:,0:1], bxyz[:,1:2], bxyz[:,2:3]
        #print(max(by))
        return bx, by, bz

    def net_b_div(self, x, y, z):
        bx, by, bz = self.net_b_bc(x, y, z)

        bx_x = self.safe_grad(bx, x)
        by_y = self.safe_grad(by, y)
        bz_z = self.safe_grad(bz, z)
        #print(max(bx_x + by_y + bz_z))

        return bx_x + by_y + bz_z


    def net_b_jxb(self, x, y, z):
        bx, by, bz = self.net_b_bc(x, y, z)

        bz_y = self.safe_grad(bz, y)
        by_z = self.safe_grad(by, z)
        bx_z = self.safe_grad(bx, z)
        bz_x = self.safe_grad(bz, x)
        by_x = self.safe_grad(by, x)
        bx_y = self.safe_grad(bx, y)

        jx = bz_y - by_z
        jy = bx_z - bz_x
        jz = by_x - bx_y

        jxb_x = jy*bz - jz*by
        jxb_y = jz*bx - jx*bz
        jxb_z = jx*by - jy*bx
        #print(min(jxb_x**2 + jxb_y**2 + jxb_z**2))

        return jxb_x**2 + jxb_y**2 + jxb_z**2

    
    def safe_grad(self, f, x):
        g = torch.autograd.grad(
            f, x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            allow_unused=True
        )[0]
        if g is None:
            return torch.zeros_like(f)
        return g

        


    # ---------------- Loss ----------------
    def loss_fn(self):
        bx0_pred, by0_pred, bz0_pred = self.net_b_bc(self.x0, self.y0, self.z0)
        div_pred = self.net_b_div(self.x_f, self.y_f, self.z_f)
        jxb_pred = self.net_b_jxb(self.x_f, self.y_f, self.z_f)


        loss = self.weight[0]*torch.mean((bx0_pred - self.bx0)**2) +\
            self.weight[0]*torch.mean((by0_pred - self.by0)**2) +\
            self.weight[0]*torch.mean((bz0_pred - self.bz0)**2) +\
            self.weight[1]*torch.mean(div_pred**2) +\
            self.weight[2]*torch.mean(jxb_pred)      
        #assert torch.isfinite(by0_pred).all(), f"contains NaN or Inf"assertを負の数に関しても作成
        print(f"bx0 = {torch.mean((bx0_pred - self.bx0)**2)}, by0 = {torch.mean((by0_pred - self.by0)**2)}, bz0 = {torch.mean((bz0_pred - self.bz0)**2)}, div = {torch.mean((div_pred)**2)}, jxb = {torch.mean(jxb_pred)}")
        
        return loss
    def train_lbfgs(self):
        self.train()

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            max_iter=50000,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn()

            if torch.isnan(loss):
                raise RuntimeError("NaN detected in LBFGS loss")

            loss.backward()
            return loss

        optimizer.step(closure)

    def predict(self, xyz_star):
        with torch.no_grad():  #計算グラフを作成しない
            xyz_star = torch.tensor(
                xyz_star, dtype=torch.float32, device=self.device
            )

            x = xyz_star[:, 0:1]
            y = xyz_star[:, 1:2]
            z = xyz_star[:, 2:3]

            bx, by, bz = self.net_b_bc(x, y, z)

        return (
            bx.cpu().numpy(),
            by.cpu().numpy(),
            bz.cpu().numpy()
        )

    def save_checkpoint(self, path, optimizer, step):
        torch.save({
            "step": step,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)


    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)#CPUとGPUの違いを読み込みload

        # case 1: full checkpoint dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            step = checkpoint.get("step", 0)

        # case 2: state_dict only
        else:
            self.load_state_dict(checkpoint)
            step = 0

        return step

    
    def train_model(self, n_iter, ckpt_dir, checkpoint_path=None,  save_every=50000):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)

        start_iter = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            start_iter = self.load_checkpoint(checkpoint_path, optimizer)
            ckpt_dir = os.path.dirname(checkpoint_path)
            print(f"Resumed from step {start_iter}")

        for i in range(start_iter, n_iter):
            optimizer.zero_grad()
            self.weight = self.weight1#[bc, div, jxb]
            if i > self.TH:
                self.weight = self.weight2
          
            loss = self.loss_fn()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            if i % 100 == 0:
                print(f"{i}, loss={loss.item():.3e}")

            if (i + 1) % save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{i+1}.pt")
                self.save_checkpoint(ckpt_path, optimizer, i + 1)
                print(f"Checkpoint saved at step {i+1}")


if __name__ == "__main__":        

    # 空間の大きさと分解能
    x = np.linspace(0, 63, 64)
    y = np.linspace(0, 63, 64)
    z = np.linspace(0, 63, 64)
    max_xyz = np.array([63, 63, 63])
    min_xyz = np.array([0.0, 0.0, 0.0])
    #print(min_xyz)

    N_b = 64#境界条件の点
    N_f = 64*64*8#コロケーション 
    layers = [3, 256, 256, 256, 256,  256, 256,  3]#層の接続
    lowlou_f = "/workspaces/template_pytorch/rtmag/lowlou/test/b_0.210_0.124.npz"

    data = np.load(lowlou_f)
    Exa_b = data["b"]#(64, 64, 64, 3)
    bottom = Exa_b[:, :, 0, :]

    Exa_bx = bottom[:, :, 0:1]
    Exa_by = bottom[:, :, 1:2]
    Exa_bz = bottom[:, :, 2:3]

    X, Y, Z = np.meshgrid(x, y, z)
    #print(X)

    xyz_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None], Z.flatten()[:,None]))

    x_star = Exa_bx.T.flatten()[:,None] 
    y_star = Exa_by.T.flatten()[:,None]
    z_star = Exa_bz.T.flatten()[:,None]

    idx_x = np.random.choice(x.shape[0], N_b, replace=False)
    idx_y = np.random.choice(y.shape[0], N_b, replace=False)

    x0 = x[idx_x]
    y0 = y[idx_y]

    bx0 = Exa_bx[idx_x,idx_y, 0:1]#後ろの軸指定の有無
    by0 = Exa_by[idx_x,idx_y, 0:1]
    bz0 = Exa_bz[idx_x,idx_y, 0:1]


    sampler = qmc.Sobol(d=3, scramble=True, seed=1234) 
    xyz_f = sampler.random(N_f)
    xyz_bc = sampler.random(100)

    xyz_f = np.array([0,0,0]) + np.array([63,63,63]) * xyz_f 
    xyz_bc_points = np.array([63, 63, 0]) * xyz_bc 
    xyz_f = np.vstack([xyz_f, xyz_bc_points])


    path = "/workspaces/template_pytorch/PINNs-master/model"
    num  = sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))
    os.makedirs(f"/workspaces/template_pytorch/PINNs-master/model/model{num}", exist_ok=True)
    model_dir = f"/workspaces/template_pytorch/PINNs-master/model/model{num}"

    weight1,weight2, weight3 = [100, 1, 1], [1, 10, 10], [1, 1, 1] 
    TH = 100000
    lr = 1e-3

    print(f"Layers:{layers}")
    print(f"sampling_points:{N_f}, bc_sampling_points:{N_b}")
    print(f"weight1:{weight1}, weight2{weight2}, Threshold{TH}, learning:{lr}")


    model = PhysicsInformedNN(x0, y0, bx0, by0, bz0, xyz_f, layers, min_xyz, max_xyz, TH, weight1, weight2, lr)#lossをプロット
    #学習率を途中で変更してみる


    start_time = time.time()
    model.train_model(500000, model_dir, save_every=50000)
    #model.train_model(100)
    #model.train_lbfgs()
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.4f}")

    model.eval()
    bx_pred, by_pred, bz_pred= model.predict(xyz_star)