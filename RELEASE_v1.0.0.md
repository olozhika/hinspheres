# hinspheres v1.0.0

**No-foreground version** — 适用于无前景HI的云核建模与拟合。

## 工具
- `generate_sim_hinsa`: 从冷云核模型生成合成HINSA观测
- `fit_hinsa_model`: 使用CMA-ES优化拟合观测HINSA数据

## 模型
Li & Goldsmith (2003) 三组分辐射转移模型：`bg HI → cold cloud (N shells) → observer`（无前景HI）

## 功能
- Forward模式和二阶导模式
- [0,1]归一化CMA-ES参数空间
- 径向权重 (1/r^weight_index)
- 空间+速度分辨率卷积（在优化循环中）
- fit_velocity_radius_kms谱窗口限制
- 1σ置信区间
- 诊断PNG含N(HI)柱密度图（对数色标）

## 依赖
- numpy, scipy, astropy, matplotlib, py-cma
