import numpy as np
import copy
import os
import torch
import torch.nn as nn


class PhysicsInformedNN(nn.Module):

    def __init__(self, x0, y0, bx0, by0, bz0, xyz_f, layers, min_xyz, max_xyz, TH, weight1, weight2, lr):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        x0, y0 = np.meshgrid(x0, y0)
        z0 = np.zeros_like(x0)
        xyz0 = np.stack([x0, y0, z0], axis=-1).reshape(-1,3)
        z_inv = 1/(xyz_f[:, 2:3]+1)
        z_weight = 1

        self.lb = torch.tensor(min_xyz, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(max_xyz, dtype=torch.float32).to(self.device)

        # training data
        self.x0 = torch.tensor(xyz0[:,0:1], dtype=torch.float32).to(self.device)
        self.y0 = torch.tensor(xyz0[:,1:2], dtype=torch.float32).to(self.device)
        self.z0 = torch.tensor(xyz0[:,2:3], dtype=torch.float32).to(self.device)

        self.bx0 = torch.tensor(bx0, dtype=torch.float32).to(self.device)
        self.by0 = torch.tensor(by0, dtype=torch.float32).to(self.device)
        self.bz0 = torch.tensor(bz0, dtype=torch.float32).to(self.device)

        self.x_f = torch.tensor(xyz_f[:,0:1], dtype=torch.float32).to(self.device)
        self.y_f = torch.tensor(xyz_f[:,1:2], dtype=torch.float32).to(self.device)
        self.z_f = torch.tensor(xyz_f[:,2:3], dtype=torch.float32).to(self.device)

        self.z_inv =  torch.tensor(z_inv, dtype=torch.float32).to(self.device)
        print(self.z_inv.shape)
        self.z_weight = torch.tensor(z_weight, dtype=torch.float32).to(self.device)

        self.weight = weight1
        self.weight1 = weight1
        self.weight2 = weight2
        self.TH = TH
        self.lr = lr

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
        xyz = torch.cat([x, y, z], dim=1)

        bxyz = self.neural_net(xyz)
        bx, by, bz = bxyz[:,0:1], bxyz[:,1:2], bxyz[:,2:3]
        
        return bx, by, bz

    def physics_strict(self, x, y, z):
        bx, by, bz = self.net_b_bc(x, y, z)

        abs_b = self.abs_b(bx, by, bz) + 1e-5

        bx_x = self.safe_grad(bx, x)
        by_y = self.safe_grad(by, y)
        bz_z = self.safe_grad(bz, z)
        
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

        return (bx_x + by_y + bz_z), (jxb_x**2 + jxb_y**2 + jxb_z**2)/abs_b#次元をそろえる
        
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

    def abs_b(self, bx, by, bz):
        num = bx**2 + by**2 + bz**2
        return torch.sqrt(num)
        
    # ---------------- Loss ----------------
    def loss_fn(self):
        bx0_pred, by0_pred, bz0_pred = self.net_b_bc(self.x0, self.y0, self.z0)
        #div_pred = self.net_b_div(self.x_f, self.y_f, self.z_f)
        #jxb_pred = self.net_b_jxb(self.x_f, self.y_f, self.z_f)

        div_pred, jxb_pred = self.physics_strict(self.x_f, self.y_f, self.z_f)

        #x_loss = torch.mean(((bx0_pred - self.bx0)/self.bx0)**2)
        x_loss = torch.mean((bx0_pred - self.bx0)**2)
        #y_loss = torch.mean(((by0_pred - self.by0)/self.by0)**2)
        y_loss = torch.mean((by0_pred - self.by0)**2)
        #z_loss = torch.mean((((bz0_pred - self.bz0)/self.bz0)**2))
        z_loss = torch.mean((bz0_pred - self.bz0)**2)

        div_loss = torch.mean(self.z_weight*div_pred**2)
        jxb_loss = torch.mean(self.z_weight*jxb_pred)#磁場で正規化

        #abs_loss = torch.mean((self.abs_b(bx0_pred, by0_pred, bz0_pred)-self.b0_abs)**2)

        loss = self.weight[0]*x_loss + self.weight[0]*y_loss + self.weight[0]*z_loss + self.weight[1]*div_loss + self.weight[2]*jxb_loss 
        loss_txt = f"bx0 = {x_loss}, by0 = {y_loss}, bz0 = {z_loss}, div = {div_loss}, jxb = {jxb_loss}"   
        
        print(loss_txt)
        
        return loss, loss_txt, jxb_loss
    
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
            loss, _ , _= self.loss_fn()

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
        loss_path = os.path.join(ckpt_dir, "log.txt")

        start_iter = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            start_iter = self.load_checkpoint(checkpoint_path, optimizer)
            ckpt_dir = os.path.dirname(checkpoint_path)
            print(f"Resumed from step {start_iter}")

        best_loss = float("inf")
        best_model_state = None
        best_opt_state = None
        best_step = 0
        monitor_after = 100000
        

        for i in range(start_iter, n_iter):
            optimizer.zero_grad()
            self.weight = self.weight1#[bc, div, jxb]
            if i > self.TH:
                self.weight = self.weight2
          
            loss, loss_txt, jxb_loss = self.loss_fn()

            with open(loss_path, "a", encoding="utf-8") as f:
                f.write(loss_txt)
                f.write("\n")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            if i >= monitor_after:
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                    best_opt_state = copy.deepcopy(optimizer.state_dict())
                    best_step = i + 1

            if i % 100 == 0:
                print(f"{i}, loss={loss.item():.3e}")

            if (i + 1) % save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{i+1}.pt")
                self.save_checkpoint(ckpt_path, optimizer, i + 1)
                print(f"Checkpoint saved at step {i+1}")
                    
        if best_model_state is not None:
            best_path = os.path.join(ckpt_dir, "best.pt")

            torch.save({
                "step": best_step,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": best_opt_state,
                "loss": best_loss
            }, best_path)

            print(f" Best model saved at step {best_step}, loss={best_loss:.3e}")


