在 **Normal-Inverse Gamma (NIG)** 分布中，**联合概率密度函数 (PDF)** 和**负对数似然损失 (NLL Loss)** 用于联合估计正态分布的**均值**和**方差**的不确定性。下面详细讲解 NIG 的概率密度函数和如何基于此构建 NLL 损失。

### 1. **NIG 分布的联合概率密度函数 (PDF)**

NIG 分布的联合概率密度函数描述了均值 \( \mu \) 和方差 \( \sigma^2 \) 的联合分布，形式为：
$P(\mu, \sigma^2 | \alpha, \beta, \gamma, \nu) = \frac{\beta^\alpha \sqrt{\nu}}{\Gamma(\alpha) \sqrt{2\pi \sigma^2}} \left( \frac{1}{\sigma^2} \right)^{1+\alpha} \exp\left(- \frac{2\beta + \nu(\gamma - \mu)^2}{2\sigma^2}\right)$

其中：
- \( \alpha \) 是形状参数（控制方差的先验分布）。
- \( \beta \) 是尺度参数（控制方差的扩展性）。
- \( \gamma \) 是均值的期望（控制均值的中心位置）。
- \( \nu \) 是精度参数（控制均值的精确度）。

#### 解释：
- 该分布是**正态分布**和**逆伽马分布**的联合形式，专门用于建模同时具有**均值不确定性**和**方差不确定性**的问题。
- 正态分布用于描述在给定方差 \( \sigma^2 \) 条件下的**均值** \( \mu \) 的分布。
- 逆伽马分布用于描述 \( \sigma^2 \) 的不确定性。

### 2. **NIG 的负对数似然损失 (NLL Loss)**

为了通过神经网络或其他优化方法拟合 NIG 分布中的参数 \( \alpha \)、\( \beta \)、\( \gamma \)、\( \nu \)，我们需要基于概率密度函数构建**负对数似然损失 (Negative Log-Likelihood Loss, NLL Loss)**。

负对数似然损失的表达式通过对联合概率密度函数取对数并取负号得到。对于 NIG 分布，NLL Loss 的表达式为：

$
\text{NLLLoss} = -\log P(\mu, \sigma^2 | \alpha, \beta, \gamma, \nu)
$

将上面的 PDF 代入并取对数，可以得到：

$
\text{NLLLoss} = -\log\left( \frac{\beta^\alpha \sqrt{\nu}}{\Gamma(\alpha) \sqrt{2\pi \sigma^2}} \left( \frac{1}{\sigma^2} \right)^{1+\alpha} \exp\left(- \frac{2\beta + \nu(\gamma - \mu)^2}{2\sigma^2}\right) \right)
$

这可以进一步简化为：

$
\text{NLLLoss} = \alpha \log(\beta) - \log(\Gamma(\alpha)) + \frac{1}{2} \log(2\pi \sigma^2) + (1 + \alpha) \log(\sigma^2) + \frac{2\beta + \nu (\gamma - \mu)^2}{2\sigma^2}
$

其中：
- \( \log(\Gamma(\alpha)) \) 是形状参数 \( \alpha \) 的伽马函数的对数。
- 其他项分别描述了 \( \sigma^2 \) 和 \( \mu \) 的贡献。

#### NLL Loss 的各部分解释：
- **\( \alpha \log(\beta) - \log(\Gamma(\alpha)) \)**：这是由逆伽马分布引入的项，表示方差 \( \sigma^2 \) 的不确定性。
- **\( \frac{1}{2} \log(2\pi \sigma^2) \)**：这来自正态分布，表示均值 \( \mu \) 的不确定性。
- **\( (1 + \alpha) \log(\sigma^2) \)**：来自方差的逆伽马部分。
- **\( \frac{2\beta + \nu (\gamma - \mu)^2}{2\sigma^2} \)**：这是正态分布与逆伽马分布共同作用的一部分，反映了均值和方差的联合不确定性。

### 3. **在 PyTorch 中实现 NIG 的 NLL Loss**

基于 NIG 的联合概率密度函数，我们可以在 PyTorch 中构建损失函数来优化模型。以下是实现 NLL Loss 的代码示例：

```python
import torch
import torch.nn as nn

class NIGLoss(nn.Module):
    def __init__(self):
        super(NIGLoss, self).__init__()

    def forward(self, mu, sigma2, alpha, beta, gamma, nu, target_mu):
        """
        mu: 模型预测的均值
        sigma2: 模型预测的方差
        alpha: NIG 分布中的形状参数
        beta: NIG 分布中的尺度参数
        gamma: NIG 分布中的均值
        nu: NIG 分布中的精度参数
        target_mu: 真实的目标均值
        """
        eps = 1e-6  # 防止log(0)和除以0

        # NLL公式的各项
        term1 = alpha * torch.log(beta + eps) - torch.lgamma(alpha + eps)
        term2 = 0.5 * torch.log(2 * torch.pi * sigma2 + eps)
        term3 = (1 + alpha) * torch.log(sigma2 + eps)
        term4 = (2 * beta + nu * (gamma - target_mu) ** 2) / (2 * sigma2 + eps)

        # 负对数似然损失
        nll_loss = term1 + term2 + term3 + term4
        return torch.mean(nll_loss)

# 测试 NLL Loss 的实现
mu_pred = torch.tensor([2.5], requires_grad=True)  # 模型预测的均值
sigma2_pred = torch.tensor([0.5], requires_grad=True)  # 模型预测的方差
alpha_pred = torch.tensor([2.0], requires_grad=True)
beta_pred = torch.tensor([1.0], requires_grad=True)
gamma_pred = torch.tensor([2.0], requires_grad=True)
nu_pred = torch.tensor([3.0], requires_grad=True)

# 真实的均值
target_mu = torch.tensor([3.0])

# 实例化 NIG Loss
criterion = NIGLoss()

# 计算 NLL Loss
loss = criterion(mu_pred, sigma2_pred, alpha_pred, beta_pred, gamma_pred, nu_pred, target_mu)
print(f'NLL Loss: {loss.item()}')

# 反向传播
loss.backward()
```

### 4. **总结**

- **联合概率密度函数 (PDF)**：NIG 分布联合描述了正态分布的均值和方差的不确定性，是由正态分布和逆伽马分布联合构成的。
- **负对数似然损失 (NLL Loss)**：通过 NIG 分布的联合概率密度函数构造 NLL Loss，我们可以在神经网络中优化 NIG 分布的四个参数 \( \alpha, \beta, \gamma, \nu \)，以更好地估计数据中的均值和方差的不确定性。
- **在机器学习中的应用**：这种方法在回归任务、不确定性估计、时间序列预测等领域非常有效，能够让模型同时估计均值和方差的不确定性。

通过这种方式，你可以在机器学习任务中构建一个强大的模型，不仅能进行点预测，还能生成完整的**不确定性估计**。