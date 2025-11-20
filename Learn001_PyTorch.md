[Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
<details>
  <summary>点击展开代码块</summary>

import torch
import numpy as np

#init
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

#np array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#like
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones like Tensor: \n {x_ones} \n")
#like and override
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random like Tensor: \n {x_rand} \n")

#shape
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print("shape:")
print(f"Random Tensor: \n {rand_tensor} ")
print(f"Ones Tensor: \n {ones_tensor} ")
print(f"Zeros Tensor: \n {zeros_tensor}")

</details>

# 1120 
MLP + 隐藏层 + 激活函数 + 梯度下降 + Adam 优化器
[copilot:learn pytorch](https://copilot.microsoft.com/shares/gs2Dy6e3kvTPnzUwkvcjk)
## 基本概念
- 前向传播 (Forward pass)：输入》张量运算》输出
- 损失函数：模型输出和真实标签之间的差异，比如均方误差 (MSE)、交叉熵 (Cross-Entropy)。
  > 损失函数 𝐿 是参数的函数：
  > 𝐿=𝑓(𝑊,𝑏,𝑥)
- 梯度：损失函数对参数的偏导数。即：如果参数𝑊改变一点点，损失𝐿会怎么变化
  > 梯度反方向：往哪个方向调整参数能让损失减小。
  > 梯度大小：调整多少合适。
- 反向传播 (Backward pass)：损失函数对每个参数求导，计算梯度。
- 梯度下降法 (Gradient Descent)：新参数=旧参数−𝜂⋅梯度
  > 𝜂 是 学习率 (learning rate)，控制每次更新的步长
代码示例：
```py
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2  # y = x^2
y.backward()  # 计算 dy/dx
print(x.grad)  # 输出: tensor([4.])
```
## 手动更新参数
```py
import torch

# 初始化参数
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([2.0])
t = torch.tensor([10.0])
lr = 0.01

# 前向传播
y = (w * x + b)**2
loss = (y - t)**2

# 反向传播
loss.backward()

# 更新参数（手动）
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

# 清除梯度
w.grad.zero_()
b.grad.zero_()

```
## 优化器 (Optimizer)更新参数
```py
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的线性模型 y = wx + b
model = nn.Linear(1, 1)  

# 损失函数：均方误差
criterion = nn.MSELoss()

# 优化器：随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练数据：只有一组 (x=2, y=10)
x = torch.tensor([[2.0]])
y_true = torch.tensor([[10.0]])

# 迭代训练 10 次
for epoch in range(10):
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    # 反向传播
    optimizer.zero_grad()   # 清空梯度
    loss.backward()         # 计算梯度
    optimizer.step()        # 更新参数

    # 打印权重和偏置
    w = model.weight.data.item()
    b = model.bias.data.item()
    print(f"Epoch {epoch}: loss={loss.item():.4f}, weight={w:.4f}, bias={b:.4f}")
```
## 多层神经网络 (MLP) 拟合二次函数
```py
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 生成训练数据：y = 2x^2 + 3x + 1
x = torch.linspace(-5, 5, steps=200).unsqueeze(1)   # 输入维度 [200,1]
y_true = 2 * x**2 + 3 * x + 1                       # 输出维度 [200,1]

# 2. 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, 16)   # 输入层 -> 隐藏层1
        self.hidden2 = nn.Linear(16, 16)  # 隐藏层1 -> 隐藏层2
        self.output = nn.Linear(16, 1)    # 隐藏层2 -> 输出层
        self.relu = nn.ReLU()             # 激活函数

    def forward(self, x):
        x = self.relu(self.hidden1(x))    # 第一层 + ReLU
        x = self.relu(self.hidden2(x))    # 第二层 + ReLU
        x = self.output(x)                # 输出层（不加激活）
        return x

# 3. 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.MSELoss()                          # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练过程
for epoch in range(200):
    y_pred = model(x)                             # 前向传播
    loss = criterion(y_pred, y_true)              # 计算损失

    optimizer.zero_grad()                         # 清空梯度
    loss.backward()                               # 反向传播
    optimizer.step()                              # 更新参数

    if epoch % 20 == 0:                           # 每20轮打印一次
        print(f"Epoch {epoch:03d}: loss={loss.item():.4f}")
```
#### 总结
1. **`nn.Linear` 的矩阵和偏置大小**  
   - 权重矩阵形状：\((out\_features, in\_features)\)  
   - 偏置向量形状：\((out\_features)\)，在批维度上广播成 \([batch\_size, out\_features]\)。  

2. **MLP 拟合二次函数的完整流程**  
   - 数据准备：生成输入 \(x\) 和真实输出 \(y\)。  
   - 模型结构：两层隐藏层 + ReLU 激活，输出层不加激活。  
   - 损失函数：MSE。  
   - 优化器：Adam。  
   - 训练过程：前向传播 → 计算损失 → 反向传播 → 更新参数。  

3. **ReLU 激活函数的作用**  
   - 定义：\(\text{ReLU}(x) = \max(0, x)\)。  
   - 在隐藏层逐个作用于神经元输出，引入非线性。  
   - 输出层通常不加 ReLU，避免限制输出只能为正数。  

4. **梯度下降与优化器**  
   - 梯度是损失函数上升最快的方向。  
   - 更新参数时减去梯度，保证往损失下降的方向走。  
   - Adam 优化器结合了动量（平滑方向）和自适应学习率（不同参数不同步长）。  

5. **矩阵维度的完整追踪**  
   - 输入 \([200,1]\) → 第一层 \([200,16]\) → 第二层 \([200,16]\) → 输出层 \([200,1]\)。  
   - 偏置在广播时从 \([16]\) 扩展成 \([200,16]\)。  
