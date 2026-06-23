教師データなしでPINNの物理式のみを用いて磁場を推定するプログラム
## サンプリングポイント

ランダムなサンプリングポイントを生成しそこで物理式が満たされるように学習を行う

![NF2 field visualization in ParaView](image/sampling.png)

## 物理式
$$
\nabla \cdot \mathbf{B}
=
\frac{\partial B_x}{\partial x}
+
\frac{\partial B_y}{\partial y}
+
\frac{\partial B_z}{\partial z}
= 0
$$