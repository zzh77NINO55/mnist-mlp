"""
Project 1: MNIST 手写数字识别（v3 指南 第三章 3.3）
目标：跑通 PyTorch 训练循环，准确率 >= 98%

用法：
    conda activate ai
    python train_mnist.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── 0. 设备选择（Apple Silicon 用 mps，没有就用 cpu）──
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}")

# ── 1. 数据准备 ──
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的标准均值/标准差
])
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
print(f"训练集 {len(train_set)} 张，测试集 {len(test_set)} 张")


# ── 2. 定义模型 ──
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)            # 展平：(B,1,28,28) -> (B,784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ── 3. 训练循环（这五步要背下来）──
def train_one_epoch():
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()           # 1. 梯度清零
        out = model(x)                  # 2. 前向传播
        loss = F.cross_entropy(out, y)  # 3. 计算损失
        loss.backward()                 # 4. 反向传播
        optimizer.step()                # 5. 参数更新
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():               # 推理时关掉梯度记录
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(test_set)


# ── 4. 主循环 ──
if __name__ == "__main__":
    history = []
    for epoch in range(10):
        loss = train_one_epoch()
        acc = evaluate()
        history.append((epoch + 1, loss, acc))
        print(f"Epoch {epoch+1:2d}: loss={loss:.4f}, acc={acc:.4f}")

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\n最终准确率: {history[-1][2]:.4f}  (目标 >= 0.98)")
    print("模型已保存到 mnist_mlp.pt")

    # 画 loss 曲线——你的"科研第一张图"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h[0] for h in history]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(epochs, [h[1] for h in history], "b-o", label="train loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="b")
        ax2 = ax1.twinx()
        ax2.plot(epochs, [h[2] for h in history], "r-s", label="test acc")
        ax2.set_ylabel("Accuracy", color="r")
        plt.title("MNIST MLP Training")
        fig.tight_layout()
        plt.savefig("training_curve.png", dpi=120)
        print("训练曲线已保存到 training_curve.png")
    except Exception as e:
        print(f"(画图跳过: {e})")
