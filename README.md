# Project 1：MNIST 手写数字识别

> v3 指南 第三章 3.3 / 第十二章 Level 2 的第一个项目。
> 目标：跑通 PyTorch 训练循环，测试准确率 ≥ 98%。

## 环境

```bash
conda activate ai          # Miniconda 环境，Python 3.11
python train_mnist.py
```

测试环境：macOS (Apple Silicon, arm64) · Python 3.11 · PyTorch（MPS 后端）
依赖版本见 `requirements.txt`。

## 做了什么

- **数据**：MNIST，60000 训练 / 10000 测试，标准化用 mean=0.1307, std=0.3081
- **模型**：MLP 784 → 256 → 128 → 10，ReLU 激活，Dropout(0.2)
- **优化器**：Adam, lr=1e-3；损失：交叉熵
- **训练**：batch_size=128，10 epochs
- **设备**：自动选择 MPS（苹果 GPU）→ CUDA → CPU

## 结果

| 指标 | 数值 |
|---|---|
| 最终测试准确率 | **97.95%**（第 10 epoch，峰值 98.0% @ epoch 5） |
| 最终训练 loss | ≈ 0.04（见 `training_curve.png`） |

> MLP 在 MNIST 上的天花板大致就在 98% 附近；loss 持续下降而测试准确率横盘，是轻微过拟合的信号。下一步换 CNN 冲 99%+。

产出：`mnist_mlp.pt`（模型权重）、`training_curve.png`（loss/acc 曲线，"科研第一张图"）

## 训练循环五步（要能默写）

```python
optimizer.zero_grad()           # 1. 梯度清零
out = model(x)                  # 2. 前向传播
loss = F.cross_entropy(out, y)  # 3. 计算损失
loss.backward()                 # 4. 反向传播
optimizer.step()                # 5. 参数更新
```

## 踩过的坑 / 笔记

### 坑 1：MNIST 自动下载失败（2026-07-14）

**报错**：
```
RuntimeError: Error downloading train-images-idx3-ubyte.gz:
Tried https://ossci-datasets.s3.amazonaws.com/mnist/, got:
<urlopen error [Errno 54] Connection reset by peer>
Tried http://yann.lecun.com/exdb/mnist/, got: HTTP Error 404: Not Found
```

**怎么读这个报错**（v3 第九章：从下往上）：
- 最后一行说的是"下载失败"，不是代码错 → 第一行 `使用设备: mps` 已经打印出来了，说明 PyTorch/GPU 都正常
- **关键判断：这是网络问题，不是我的 bug**。新手最容易在这里怀疑代码，其实一行都不用改

**排查过程**：分别 curl 测试三个域名的连通性
- GitHub (raw.githubusercontent.com) → 完全不通
- AWS 镜像 (ossci-datasets.s3.amazonaws.com) → **返回 200，其实是通的**
- HuggingFace → 通

**根因**：AWS 镜像可达，torchvision 那次是**偶发的连接重置**（国内网络抖动），不是源挂了。
Yann LeCun 原站 404 是因为那个老网站已经下线——这是所有教程里的历史遗留。

**解决**：用 curl + 重试参数手动下载到 `data/MNIST/raw/`，torchvision 检测到文件存在就不再下载。

```bash
mkdir -p data/MNIST/raw && cd data/MNIST/raw
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sfL --retry 5 --retry-all-errors -O "https://ossci-datasets.s3.amazonaws.com/mnist/$f"
done
gunzip -kf *.gz
```

**教训**：
1. 遇到下载失败先 `curl -I` 测源的连通性，再决定是换源还是重试
2. `--retry 5 --retry-all-errors` 能救掉大部分国内网络抖动
3. 报错要分清"我的代码错了"和"环境/网络错了"——看报错发生在哪一层

### 坑 2：CUDA 在 Mac 上永远不可用

教程里的 `torch.cuda.is_available()` 在 Apple Silicon 上恒为 False，这不是坏了。
苹果的 GPU 加速叫 **MPS**，本项目已用自动选择：
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
以后看任何教程遇到 `.cuda()` / `device='cuda'`，都要做这个转换。

## 下一步

- [ ] 改成 CNN，看准确率能到多少（应该 >99%）
- [ ] 去掉 Dropout 对比效果
- [ ] 换 Fashion-MNIST 数据集再跑一遍
- [ ] Project 3：CIFAR-10 + ResNet18
