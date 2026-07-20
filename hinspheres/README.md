# hinspheres

球对称多层 HINSA 辐射转移正向建模与 CMA-ES 拟合包。

独立于 `hinsapack`，可单独使用。`hinsapack` 通过 `prepare_hinspheres_input` 准备数据后，直接调用本包进行拟合。

---

## 安装

```bash
pip install numpy scipy astropy matplotlib joblib cma
```

---

## 一、辐射转移全过程

### 1.1 球对称壳层模型

云核被离散为 `n_shells` 层同心球壳，每层具有均匀物理属性。壳层均匀分布从 0 到 `R_out`：

$$r_{\text{inner},k} = (k-1) \cdot \Delta r, \quad r_{\text{outer},k} = k \cdot \Delta r, \quad \Delta r = \frac{R_{\text{out}}}{n}$$

$$r_{\text{mid},k} = \frac{r_{\text{inner},k} + r_{\text{outer},k}}{2}$$

### 1.2 径向物理剖面

**密度**（Plummer 剖面）：

$$n_H(r) = \frac{\rho_0}{1 + (r / r_0)^\alpha}$$

- $\rho_0$：中心 H 核密度（cm⁻³）
- $r_0$：Plummer 核半径（pc）
- $\alpha$：幂律指数，越小越平坦

**温度**（Plummer 型）：

$$T_{\text{spin}}(r) = \max\left(T_1 + \frac{T_0 - T_1}{1 + (r / r_T)^2},\ T_{\text{CMB}}\right)$$

- $T_0$：中心自旋温度（冷核）
- $T_1$：环境自旋温度（暖包层）
- $r_T$：温度转变半径

**HI 丰度**：

- 方式 A（peak_shell + multipliers）：$f_{\text{HI}}$ 在 `peak_shell` 层达到峰值 1.0，向外通过 `multipliers` 逐层递减
- 方式 B（直接指定）：$f_{\text{HI}} = [f_1, f_2, \ldots, f_n]$，每层 HI 占总氢的比例

**HI 数密度**：

$$n_{\text{HI}}(r) = n_H(r) \cdot f_{\text{HI}}(r)$$

### 1.3 运动学

**内落速度**（自由下落的分数）：

$$v_{\text{infall}}(r) = f_{ff} \cdot \sqrt{\frac{2G \cdot M(<r)}{r}}$$

- $f_{ff}$：自由下落速度分数（0-1）
- $M(<r)$：半径 r 以内的包裹质量（由密度剖面积分得到）
- 方向：径向向内

**旋转速度**（刚体旋转）：

$$v_{\text{los}}^{\text{rot}} = \Omega \cdot (-dx \cdot \cos PA - z \cdot \sin PA)$$

- $\Omega = v_{\text{rot}} / R_{\text{out}}$：角速度
- $PA$：旋转轴位置角（从北向东度量，度）
- $dx$：天空平面上的偏移（pc），正值 = 西（FITS 像素约定）
- $z$：沿视线方向的深度（pc），正值 = 朝向观察者
- 不同壳层、不同深度 $z$ 处的旋转视线分量不同（3D 效应）

**总视线速度**：

$$v_{\text{los}}(r, z) = v_{\text{los}}^{\text{infall}}(r, z) + v_{\text{los}}^{\text{rot}}(dx, z)$$

其中内落分量：

$$v_{\text{los}}^{\text{infall}} = -v_{\text{infall}} \cdot \text{sign}(z) \cdot \cos\theta, \quad \cos\theta = \frac{z}{\sqrt{b^2 + z^2}}$$

### 1.4 光学深度计算

对每个像素（impact parameter $b$），沿视线追踪穿过所有壳层。每层的峰值光学深度：

$$\tau_{0,k} = \frac{3 c^2 A_{10}}{8\pi \nu_{21}^2} \cdot \frac{n_{\text{HI},k} \cdot \Delta l_k}{T_{\text{spin},k} \cdot \sigma_v \sqrt{2\pi}}$$

- $c$：光速
- $A_{10} = 2.884 \times 10^{-15}$ s⁻¹：21cm 跃迁 Einstein A 系数
- $\nu_{21} = 1.420$ GHz：21cm 频率
- $\Delta l_k$：视线穿过第 k 层的路径长度（pc）
- $\sigma_v$：速度展宽（热运动 + 湍流）

每层的速度展宽：

$$\sigma_v = \sqrt{\sigma_{\text{thermal}}^2 + \sigma_{\text{turb}}^2}$$

$$\sigma_{\text{thermal}} = \sqrt{\frac{k_B T}{\mu m_H}}, \quad \sigma_{\text{turb}} \text{ (用户指定, km/s)}$$

### 1.5 辐射转移方程

采用 Li & Goldsmith (2003) 三组分模型：

$$\text{背景 HI} \rightarrow \text{冷云核（N 层）} \rightarrow \text{前景 HI} \rightarrow \text{观察者}$$

每个速度通道 $v$ 的辐射转移：

$$T_B(v) = T_{\text{bg}} \cdot e^{-\tau_{\text{bg}}(v)} \cdot \prod_{k=1}^{N} e^{-\tau_k(v)} + \sum_{k=1}^{N} T_{s,k} \cdot \left(1 - e^{-\tau_k(v)}\right) \cdot \prod_{j=k+1}^{N} e^{-\tau_j(v)}$$

其中每层的光深随速度呈 Gaussian 分布：

$$\tau_k(v) = \tau_{0,k} \cdot \exp\left(-\frac{(v - v_{\text{center},k})^2}{2\sigma_k^2}\right)$$

- $v_{\text{center},k} = v_{\text{los}}(r_k, z_k) + V_{\text{lsr}} + v_{\text{offset}}$：第 k 层的中心速度（含内落 + 旋转 + 系统速度）
- $T_{s,k}$：第 k 层的自旋温度
- $T_{\text{bg}}$：背景 HI 亮温度
- $\tau_{\text{bg}}, \tau_{\text{fg}}$：背景/前景 HI 光深（通常很小）

**HINSA 吸收**：

$$\Delta T_B(v) = T_{\text{bg}}(v) - T_B(v)$$

正值 = 吸收，负值 = 发射。

### 1.6 逐像素计算流程

```
对每个像素 (i, j):
    1. 计算 impact parameter: b = sqrt(dx² + dy²)
    2. 若 b > R_out: 输出背景谱，跳过
    3. 计算视线穿过的壳层: layers = los_path_lengths(b, r_outer, n_shells)
    4. 对每层 k:
        a. 路径长度 dl_k
        b. 3D 半径 r = sqrt(b² + z²) → 找到所在壳层索引
        c. 计算 τ₀_k (由 n_HI, T_spin, dl, σ_v 决定)
        d. 计算 v_center_k = v_infall + v_rot + Vlsr + v_offset
        e. 读取 T_spin_k
    5. 对所有速度通道求解辐射转移方程
    6. 输出 T_B(v) 或吸收谱 ΔT_B(v)
```

### 1.7 Double-dip 机制

Goldsmith (2007) 自反转吸收轮廓的产生条件：

1. **温度层级**：$T_{\text{cold}} < T_{\text{warm}} < T_{\text{bg}}$
2. **光深分布**：暖包层在线心 $\tau \gg 1$（不透明），在翼部 $\tau \sim 1$（透明）
3. **密度轮廓**：较平坦（$\alpha \approx 1$），使暖包层有足够柱密度

**物理图像**：
- **线心**：暖包层 $\tau \gg 1$，观察者只看到暖包层表面（$T_{\text{warm}} < T_{\text{bg}}$）→ 吸收
- **翼部**：暖包层 $\tau \sim 1$，辐射穿透暖层，进入冷核 → 冷核温度更低 → 吸收更深
- **结果**：翼部吸收深于线心 → 两个吸收极小值夹一个中心局部极大值 → "double-dip"

---

## 二、Python API

### 正向建模

```python
from hinspheres import generate_sim_hinsa

# 均匀常数背景
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

# 真实背景 FITS（速度/空间分辨率自动匹配输入）
result = generate_sim_hinsa(
    output_path='output.fits',
    background='G206_HI_background.fits',
    center_pixel=(10, 10),
    vlsr_kms=9.3, R_out_pc=0.91,
    rho0=800, T0=5.0, T1=12.0,
    f_ff=0.15, v_rot_kms=0.2,
)

# 直接指定每层 HI 丰度
result = generate_sim_hinsa(
    output_path='output.fits',
    background='G206_HI_background.fits',
    abundance=[0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.05, 0.04, 0.03],
)

cube = result['out_cube']     # (n_v, ny, nx) ndarray
velo = result['velo_kms']     # velocity axis (km/s)
```

也可通过 `build_synthetic_hinsa` 直接操作已有 cube：

```python
from hinspheres import build_synthetic_hinsa, Config
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

out_cube = build_synthetic_hinsa(cfg, params, bg_cube,
                                  center_yx=(10, 10), pixel_scale_pc=0.56)
```

### 输出文件

`generate_sim_hinsa` 自动生成三个同名文件：

| 文件 | 内容 |
|---|---|
| `output.fits` | 合成数据 cube |
| `output.json` | 所有配置参数（含输入背景文件名） |
| `output.png` | 6 面板诊断图（密度/温度/丰度剖面 + Moment 0 + 吸收图 + 谱线） |

`fit_hinsa_model` 自动保存三个文件：

| 文件 | 内容 |
|---|---|
| `{name}_bestfit.fits` | 最佳拟合模型 cube |
| `{name}_bestfit.png` | 6 面板诊断图（剖面 + Moment 0 + 峰值吸收 + 中心谱线） |
| `{name}_grid_spectra.png` | 空间网格谱线图（Moment 0 背景 + 网格位置叠加观测/模型谱） |
| `{name}_fit_result.npz` | 综合结果（params, param_stds, model_cube, obs_cube, 等） |

### 空间与速度平滑

```python
result = generate_sim_hinsa(
    ...,
    spatial_res_arcmin=4.0,   # 平滑到 FAST 4' beam
    vel_res_kms=0.3,       # 平滑速度分辨率到 0.3 km/s
)
```

两个参数默认 `None`，不指定则不做相应方向的卷积。

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
params = {'peak_shell': 5, 'multipliers': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]}
```

**方式 B：直接指定每层 f_HI**

```python
params = {'f_HI': [0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.05, 0.04, 0.03]}
```

方式 B 优先级高于方式 A。

---

## 三、CMA-ES 拟合

拟合本质上是正向建模的迭代包装——每次评估候选参数时调用相同的 RT 内核。

```python
from hinspheres import Config, fit_hinspheres

cfg = Config(
    n_shells=9,
    R_out_pc=0.91,
    vlsr_kms=9.3,
    v_min_kms=-10.0, v_max_kms=25.0,
    n_v_channels=301,
)

# 模式 1：正向建模（需要背景）
best_params, history, param_stds = fit_hinspheres(
    cfg=cfg,
    obs_hinsa_map=hinsa_map,          # 3D 观测 cube
    T_HI_true_map=T_HI_true_map,      # 3D 背景 cube
    max_gen=200,
    popsize=24,
    n_jobs=4,
)

# 模式 2：二阶导数法（不需要背景）
best_params, history, param_stds = fit_hinspheres(
    cfg=cfg,
    obs_hinsa_map=hinsa_cube,          # 3D 观测 cube
    # T_HI_true_map 不需要
    mode='second_derivative',
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

### 残差权重

CMA-ES 拟合最小化的残差为：

```
R = Σ[ w(j,i) · (obs(j,i) − model(j,i))² / |obs(j,i)| ] / Σ w(j,i)
```

其中 **径向权重** `w(j,i)` 定义为：

```
w(j,i) = 1 / r(j,i)
```

`r(j,i)` 是像素 `(j,i)` 到云核中心的距离（单位 pc）。中心像素（`r=0`）的权重设为最近邻一圈的权重（`1 / pixel_scale_pc`），避免除以零。

**为什么要加 1/r 权重？**

在球对称模型中，距中心 `r` 处的环形区域包含 `∝ 2πr` 个像素。如果不加权重，外圈像素数量多，会主导残差，导致拟合过度关注边缘而忽略核心区域。`1/r` 权重正好补偿了这个几何效应，使每个等距环对残差的贡献相同。

**归一化：** 残差除以权重总和 `Σw`，使其：
- 完美拟合 = 0
- 与图的大小（像素数）无关
- 单位为 K（加权平均每个像素的相对误差）

### 参数置信区间

`fit_hinsa_model` 返回 `param_stds` dict，包含每个参数的1σ不确定性。

**原理：** CMA-ES 在优化过程中维护一个关于参数空间的多元高斯搜索分布 `N(μ, Σ)`，其中 `μ` 是当前最优解，`Σ` 是协方差矩阵。收敛时，`Σ` 的对角元素 `σ_i²` 反映了参数 `θ_i` 的不确定度：在残差 landscape 上，使得残差接近最优值的参数空间区域近似为高斯分布，其宽度就是 `σ_i`。

```python
result = fit_hinsa_model(...)
for k, v in result['best_params'].items():
    std = result['param_stds'][k]
    print(f'{k} = {v:.4f} ± {std:.4f}')
```

**1/r 权重的作用：** 权重使残差均匀覆盖云核各区域（而非被外圈像素主导），因此 `Σ` 反映的是全局参数不确定度，而非局部偏差。

**适用范围与 caveat：**

| 条件 | 说明 |
|---|---|
| 残差 landscape 近似高斯 | CMA-ES 协方差矩阵给出可靠的 σ 估计 |
| CMA-ES 已收敛 | `history.stop` 显示收敛原因，未收敛的 σ 不可靠 |
| 参数间相关性弱 | 若存在强简并（如 `rho0` 和 `r0`），个别参数的 σ 可能被低估 |

对于发表级结果，建议对关键参数用 MCMC（如 `emcee` 包）验证后验分布，确认 CMA-ES 的高斯近似是否足够好。

---

## 三-B、拟合模式：forward vs second_derivative

`fit_hinsa_model` 支持两种拟合模式，通过 `mode` 参数选择：

### mode='forward'（默认）

标准正向建模：需要 `obs_background_fits`（未吸收的背景 HI 亮温度 cube），残差为：

```
R = Σ[ w(j,i) · (obs(j,i) − model(j,i))² / |obs(j,i)| ] / Σ w(j,i)
```

其中 `model` 是正向 RT 模型输出。背景 cube 由 `prepare_hinspheres_input` 生成。

### mode='second_derivative'

Liu Method 2（Liu+2021），**无需 `obs_background_fits`**：

1. 用物理模型计算各壳层光深 τ₀
2. 从观测 cube 出发，用逆 RT 逐层剥离吸收，重建背景温度谱 `T_bg(v)`
3. 对重建的背景谱求二阶导数 `d²T_bg/dv²`
4. 最小化 R-value（含 1/r 径向权重，归一化）：

```
R = ∫[ Σ_pixel w(j,i) · (d²T_bg/dv²)² ] dv / Σ w(j,i)
```

其中 `w(j,i) = 1/r(j,i)`，与 forward 模式相同的径向权重。

**物理原理：** 干净的背景 HI 谱在速度方向上是光滑的（R→0），而 HINSA 吸收导致二阶导数出现尖峰。模型越接近真实物理参数，重建的背景谱越光滑，R-value 越小。

**优点：** 不需要单独提取背景 cube，避免背景提取误差引入拟合。

**⚠ 已知局限性：τ→0 退化。** 当 τ→0 时，逆 RT 几乎不改变观测谱，重建的 T_bg ≈ T_obs，R-value→0。CMA-ES 总是收敛到这个退化解（低密度+低丰度=微小 τ）。实际测试中（G206.10-15.77），拟合收敛到 τ≈0 的非物理解。**建议实际拟合使用 `mode='forward'`。**

```python
from hinspheres import fit_hinsa_model

# 模式 1：正向建模（需要背景 FITS）
result_forward = fit_hinsa_model(
    obs_hinsa_fits='G206_HI_cube.fits',
    obs_background_fits='G206_HI_background.fits',  # 必需
    mode='forward',
)

# 模式 2：二阶导数法（不需要背景 FITS）
result_sd = fit_hinsa_model(
    obs_hinsa_fits='G206_HI_cube.fits',
    # obs_background_fits 不需要
    mode='second_derivative',
)
```

---

## 四、与 hinsapack 的协作

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

## 五、包结构

```
hinspheres/
├── __init__.py      # 公有 API
├── config.py        # 物理常数与配置 (Config, bounds_pc)
├── profiles.py      # Plummer 密度/温度/丰度/内落轮廓
├── rt.py            # 光线追踪与辐射转移 (含 inverse RT)
├── models.py        # 正向建模 (build_synthetic_hinsa, inverse_build_hinsa_cube)
├── fitters.py       # CMA-ES 拟合器 (fit_hinsa_model, reload_fit_result)
└── utils.py         # 诊断图
```

---

## 参考文献

- Goldsmith, P. F. (2007), ApJ, 668, 1043
- Li, D. & Goldsmith, P. F. (2003), ApJ, 585, 823
- Zuo, P. et al. (2018), ApJ, 858, 89
