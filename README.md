# hinspheres

球对称多层 HINSA 辐射转移正向建模与 CMA-ES 拟合包。

独立于 `hinsapack`，可单独使用。`hinsapack` 通过 `prepare_hinspheres_input` 准备数据后，直接调用本包进行拟合。

---

## 安装

```bash
pip install numpy scipy astropy matplotlib joblib cma
```

---

## 一、正向建模

### 核心思路

给定球对称云核的径向剖面（密度、温度、HI 丰度、内落速度、旋转速度），建立 n 层同心球壳结构，沿视线方向从远到近逐体素进行辐射转移计算。

### Python API

```python
from hinspheres import generate_sim_hinsa

# 方式1：均匀常数背景
result = generate_sim_hinsa(
    output_path='output.fits',
    background=None,              # None = 均匀背景 (T=40 K)
    vlsr_kms=9.3,
    R_out_pc=0.91,
    distance_pc=1290.0,
    rho0=500, r0=0.05, alpha=1.0,
    T0=10.0, T1=40.0, rT=0.15,
    f_ff=0.15, turb_kms=0.15,
    v_rot_kms=0.3, rot_pa_deg=45.0,
    n_shells=9, n_jobs=4,
)

# 方式2：真实背景 FITS（速度/空间分辨率自动匹配输入）
result = generate_sim_hinsa(
    output_path='output.fits',
    background='G206_HI_background.fits',
    center_pixel=(10, 10),
    vlsr_kms=9.3, R_out_pc=0.91,
    rho0=800, T0=5.0, T1=12.0,  # T_warm < T_bg 才有吸收
    f_ff=0.15, v_rot_kms=0.2,
)

# 方式3：直接指定每层 HI 丰度
result = generate_sim_hinsa(
    output_path='output.fits',
    background='G206_HI_background.fits',
    abundance=[0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.05, 0.04, 0.03],
    # 其余参数...
)

cube = result['out_cube']     # (n_v, ny, nx) ndarray
velo = result['velo_kms']     # velocity axis (km/s)
```

也可通过 `forward_model_cube` 直接操作已有 cube：

```python
from hinspheres import forward_model_cube, Config
import numpy as np

cfg = Config(n_shells=9, R_out_pc=0.91, vlsr_kms=9.3,
             v_min_kms=-10.0, v_max_kms=25.0, n_v_channels=301)

params = {
    'rho0': 500, 'r0': 0.05, 'alpha': 1.0,
    'T0': 10.0, 'T1': 40.0, 'rT': 0.15,
    'f_HI': [0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.05, 0.04, 0.03],
    'f_ff': 0.15, 'turb_kms': 0.15, 'v_offset': 0.0,
    'v_rot_kms': 0.3, 'rot_pa_deg': 45.0,
}

velo_kms = np.linspace(-10.0, 25.0, 301)
out_cube = forward_model_cube(cfg, params, bg_cube, velo_kms,
                              center_yx=(10, 10), pixel_scale_pc=0.56)
```

### 物理参数

| 参数 | 典型值 | 说明 |
|---|---|---|
| `rho0` | 100-10000 cm⁻³ | 中心 H 核密度，决定光深 |
| `r0` | 0.01-0.15 pc | Plummer 核半径 |
| `alpha` | 1.0-3.0 | 密度幂律指数，越小越平坦 |
| `T0` | 3-15 K | 中心自旋温度（冷核） |
| `T1` | 10-60 K | 环境自旋温度（暖包层） |
| `rT` | 0.01-0.15 pc | 温度转变半径 |
| `f_ff` | 0-0.5 | 自由下落速度分数 |
| `turb_kms` | 0.05-0.5 km/s | 湍流速度 |
| `v_rot_kms` | 0-5 km/s | 旋转速度（云半径处） |
| `rot_pa_deg` | 0-180° | 旋转轴位置角（北→东） |
| `v_offset` | -3~+3 km/s | 相对 Vlsr 速度偏移 |

### HI 丰度指定方式

**方式 A：peak_shell + multipliers**（默认）

```python
# peak_shell=5 表示第5层丰度最高，multipliers 控制向外递减
params = {'peak_shell': 5, 'multipliers': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]}
```

**方式 B：直接指定每层 f_HI**

```python
# 9层云核，每层的 HI 占总氢的比例
params = {'f_HI': [0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.05, 0.04, 0.03]}
```

方式 B 优先级高于方式 A（当 `params` 中存在 `f_HI` 键时直接使用）。

### 旋转模型

刚体旋转，旋转轴在天空平面上，位置角 PA 从北向东度量。

视线速度公式：

$$v_{los}^{rot} = \Omega \cdot (-dx \cdot \cos PA - z \cdot \sin PA)$$

- `Ω = v_rot_kms / R_out`：角速度
- `dx`：天空平面上的偏移（pc），正值 = 西（FITS 像素坐标约定）
- `z`：沿视线方向的深度（pc），正值 = 朝向观察者
- 不同壳层、不同深度 `z` 处的旋转视线分量不同（3D 效应）

### Double-dip 条件

Goldsmith (2007) 自反转吸收轮廓：
1. **T_cold < T_warm < T_bg**
2. 暖包层在线心 τ >> 1，在翼部 τ ~ 1
3. 密度轮廓较平坦（alpha ≈ 1）

---

## 二、CMA-ES 拟合

拟合本质上是正向建模的迭代包装——每次评估候选参数时调用相同的 RT 内核。

### Python API

```python
from hinspheres import Config, fit_hinspheres

cfg = Config(
    n_shells=9,
    R_out_pc=0.91,
    vlsr_kms=9.3,
    v_min_kms=-10.0, v_max_kms=25.0,
    n_v_channels=301,
)

best_params, history = fit_hinspheres(
    cfg=cfg,
    obs_hinsa_map=hinsa_map,          # 2D 观测吸收图
    T_HI_true_map=T_HI_true_map,      # 2D 背景温度图
    max_gen=200,
    popsize=24,
    n_jobs=4,
)
```

### 拟合参数与边界

| 参数 | 下界 | 上界 | 说明 |
|---|---|---|---|
| `rho0` | 100 cm⁻³ | 100000 cm⁻³ | log10 尺度优化 |
| `r0` | 0.01 pc | 0.15 pc | Plummer 核半径 |
| `alpha` | 1.0 | 5.0 | 密度幂律指数 |
| `T0` | 5.0 K | 30.0 K | 中心温度 |
| `T1` | 10.0 K | 80.0 K | 环境温度 |
| `rT` | 0.01 pc | 0.15 pc | 温度转变半径 |
| `peak_shell` | 1 | 9 | 丰度峰值壳层 |
| `multipliers` | 0.1 | 1.0 | 丰度递减因子 |
| `f_ff` | 0.01 | 0.5 | 自由下落分数 |
| `turb_kms` | 0.05 km/s | 0.5 km/s | 湍流速度 |
| `v_offset` | -3 km/s | +3 km/s | 速度偏移 |
| `v_rot_kms` | 0 km/s | 5 km/s | 旋转速度 |
| `rot_pa_deg` | 0° | 180° | 旋转轴位置角 |

默认 n_shells=9 时，CMA-ES 优化维度 = 11 + 7 + 2 = **20**。

---

## 三、与 hinsapack 的协作

```python
import hinsapack as hp
from hinspheres import fit_hinspheres, Config

# 1. hinsapack 准备数据
result = hp.prepare_hinspheres_input(
    target_id='G206.10-15.77',
    output_dir='HIfig/hinspheres_input/',
)

# 2. hinspheres 拟合
cfg = Config(n_shells=9, R_out_pc=result['R_out_pc'],
             vlsr_kms=result['vlsr_kms'])
best_params, _ = fit_hinspheres(cfg, result['hinsa_map'], result['T_HI_true'])
```

---

## 四、包结构

```
hinspheres/
├── __init__.py      # 公有 API
├── config.py        # 物理常数与配置 (Config, bounds_pc)
├── profiles.py      # Plummer 密度/温度/丰度/内落轮廓
├── rt.py            # 光线追踪与辐射转移 (los_velocity 含旋转)
├── models.py        # 正向建模 (forward_model_cube, generate_sim_hinsa)
├── fitters.py       # CMA-ES 拟合器 (fit_hinspheres)
└── utils.py         # 诊断图
```

---

## 参考文献

- Goldsmith, P. F. (2007), ApJ, 668, 1043
- Li, D. & Goldsmith, P. F. (2003), ApJ, 585, 823
- Zuo, P. et al. (2018), ApJ, 858, 89
