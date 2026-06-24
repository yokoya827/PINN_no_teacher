import numpy as np
import os
import torch
import time
import random
from PINN_magneticfield.train  import PhysicsInformedNN

#ssh -l yokoyakd g00.cfca.nao.ac.jp
#sbatch train.sh

seed = 1234 
random.seed(seed)          # Python標準乱数
np.random.seed(seed)       # NumPy
torch.manual_seed(seed)    # PyTorch (CPU)

x = np.linspace(0, 63, 64)#規格化(0, 63, 64)でなく(0, 2, 64)など
y = np.linspace(0, 63, 64)
z = np.linspace(0, 63, 64)

x_index = np.arange(64)
y_index = np.arange(64)

max_xyz = np.array([63, 63, 63])
min_xyz = np.array([0.0, 0.0, 0.0])

N_b = 64#境界条件の点
N_f = 64*64*64#コロケーション 

layers = [3, 256, 256, 256, 256,  256,  256, 256,  3]#層の接続
lowlou_f = "sample/b_0.210_0.124.npz"

data = np.load(lowlou_f)
Exa_b = data["b"]#(64, 64, 64, 3)
bottom = Exa_b[:, :, 0, :]

Exa_bx = bottom[:, :, 0:1]
Exa_by = bottom[:, :, 1:2]
Exa_bz = bottom[:, :, 2:3]

X, Y, Z = np.meshgrid(x, y, z)

xyz_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None], Z.flatten()[:,None]))

bx0 = Exa_bx.flatten()[:, None]#後ろの軸指定の有無
by0 = Exa_by.flatten()[:, None]
bz0 = Exa_bz.flatten()[:, None]

x0 = x
y0 = y

np.random.seed(seed)
xyz_f = np.random.rand(N_f, 3) * 63

#new model
path = "output"
num  = sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))
os.makedirs(f"{path}/model{num}", exist_ok=True)
model_dir = f"{path}/model{num}"
loss_path = os.path.join(model_dir, "log.txt")
print(model_dir)


weight1,weight2, weight3 = [1, 3, 3, 3], [1, 25, 100, 200], [1, 1, 1] 
TH = 200000
lr =1.4*1e-4

tmp_txt = f"Layers:{layers}\nsampling_points(N_f):{N_f}, ,bc_sampling_points(N_b):{N_b}\nweight1:{weight1}, weight2{weight2}, Threshold{TH}, learning:{lr}\nData_dir:{lowlou_f}"
print(tmp_txt)

with open(loss_path, "a", encoding="utf-8") as f:
    f.write(tmp_txt)
    f.write("\n")

model = PhysicsInformedNN(x0, y0, bx0, by0, bz0, xyz_f, layers, min_xyz, max_xyz, TH, weight1, weight2, lr)#model作成

#学習率を途中で変更してみる
start_time = time.time()
model.train_model(800000, model_dir)#training
elapsed = time.time() - start_time
print(f"Training time: {elapsed:.4f}")

model.eval()

bx_pred, by_pred, bz_pred= model.predict(xyz_star)