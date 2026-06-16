# Fashion-MNIST 分布外（OOD）泛化实验

> 机器学习课程作业 — 题目 2：图像分布外泛化分类

## 任务设定

- **训练集**：原始 Fashion-MNIST 经二值化处理 → 黑背景 + 白前景 (28×28)
- **测试集**：
  - *In-distribution (ID)*：同训练集分布，黑白
  - *Out-of-distribution (OOD)*：随机彩色背景 + 随机彩色前景，且前后景颜色差异 (L1 距离) ≥ 80/765

## 方法

| 类别 | 模型 | 说明 |
|---|---|---|
| 传统机器学习 | 随机森林 | 输入 784 维 灰度像素特征 (RGB→灰度) |
| 神经网络 | SmallCNN | 2 × Conv-BN-ReLU + GAP + FC，输入 3×28×28 |
| 改进策略 | CNN + 颜色随机化数据增强 | 训练时在线随机重染色，模拟测试分布 |

## 文件结构

```
experiment/
├── data.py              # 数据加载与 OOD 构造
├── augment.py           # 训练时颜色随机化
├── models.py            # SmallCNN
├── train_rf.py          # 随机森林
├── train_cnn.py         # CNN 训练循环
├── visualize.py         # 样本图/混淆矩阵/t-SNE/曲线
├── main.py              # 完整实验入口
├── requirements.txt     # Python 依赖
├── README.md
├── figures/             # 输出图（PNG，18 张）
└── results/             # 数值指标 (summary.json + 逐类报告)
```

数据集请确保 `../data/fashion/` 下放有 4 个 Fashion-MNIST 原始 gz 文件
（`train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`、`t10k-images-…`、`t10k-labels-…`）。

## 环境搭建

实验在 Python 3.10+（推荐 3.13）+ PyTorch CPU 上即可运行，约需 15-20 分钟。

### PyCharm

1. File → Settings → Project → Python Interpreter → Add Local Interpreter → New Virtualenv，
   位置选 `fashion-mnist-master/venv`
2. 在 PyCharm 自带终端执行 `pip install -r experiment/requirements.txt`
3. 右键 `experiment/main.py` → Run；运行配置里把 Working directory 设为 `experiment/`

### 命令行

```bash
cd fashion-mnist-master
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r experiment/requirements.txt
cd experiment
python main.py
```

## 运行参数

```bash
# 完整实验 (8 epochs, RF 200 trees, 约 15-20 分钟)
python main.py

# 快速冒烟测试 (1 epoch, RF 30 trees, 约 2-3 分钟)
python main.py --smoke

# 自定义
python main.py --epochs 15 --rf-estimators 300 --seed 1
```

## 评价指标

- 准确率 (Accuracy) 在三个集合 (Train / Test-ID / Test-OOD) 上的对比
- 混淆矩阵 (10 × 10)
- 训练曲线 (loss / accuracy 随 epoch)
- t-SNE 特征空间可视化（CNN 倒数第二层 128 维特征）
- 错误样本可视化

## 主要结论 (完整 8-epoch 训练)

| 方法 | Train | Test-ID | Test-OOD |
|---|---|---|---|
| 随机森林 (200 trees) | 0.9999 | 0.8363 | 0.2759 |
| CNN baseline | 0.9202 | 0.8688 | **0.1610** |
| **CNN + ColorAug** | 0.8369 | 0.8219 | **0.8409** |

关键发现：
1. CNN baseline 的 OOD 准确率 (16.1%) 反而比随机森林 (27.6%) 还差 — BatchNorm 的三通道分布假设在彩色测试时崩坏。
2. **颜色随机化数据增强让 OOD 准确率从 16.1% → 84.1%**，+68 个百分点，5.2×。
3. 增强后 OOD (84.1%) 反超 ID (82.2%)，说明模型学到了颜色无关的形状表征。
