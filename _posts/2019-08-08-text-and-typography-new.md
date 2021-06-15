---
title: 13 面向非结构化数据的神经网络
author: Cotes Chung
date: 2019-08-08 11:33:00 +0800
categories: [ML, DNN]
tags: [typography]
math: true
mermaid: true
comments: true
---

本章我们将开始讨论深度学习，深度学习可以用于提取非结构化数据的判别特征。

## 13.1 引言

在部分 $\mathrm{II}$， 我们讨论了在回归和分类任务中的线性模型。其中，在第$\textrm{11}$章，我们讨论了线性回归模型，即 $p(y|\mathbf{x}, \mathbf{w})=\mathcal{N}\left(y|\mathbf{w}^{\top}\mathbf{x}, \sigma^{2}\right)$  。在第$\textrm{10}$章，我们讨论了逻辑回归，在二分类任务中，模型定义为 $p(y|\mathbf{x}, \mathbf{w})={\rm{Ber}}(y|\sigma(\mathbf{w}^{\top}\mathbf{x}))$，在多分类任务中，模型定义为 $ p(y|\mathbf{x},\mathbf{w})=\operatorname{Cat}(y|\mathcal{S}(\mathbf{W} \mathbf{x}))$。在第$\textrm{12}$章，我们讨论了广义线性模型，定义为:
$$
p(\mathbf{y}|\mathbf{x};\pmb{\theta})=p(\mathbf{y}|g^{-1}(f(\mathbf{x};\pmb{\theta}))) \tag{13.1}
$$
其中 $p(\mathbf{y}|\pmb{\mu})$ 表示均值为 $\pmb{\mu}$ 的指数族分布，$g^{-1}()$ 表示该指数族分布对应的逆连接函数。上式中:
$$
f(\mathbf{x};\pmb{\theta})=\mathbf{Wx} + \mathbf{b} \tag{13.2}
$$
表示关于输入的一个线性（仿射）变换函数， 其中 $\mathbf{W}$ 被称为 **权重** ($\textrm{weights}$)，$\mathbf{b}$ 被称为 **偏置** ($\textrm{biases}$)。

```markdown
线性回归：linear regression
逻辑回归: logistic regression
指数族分布：exponential family distribution;
逆连接函数：inverse link function;
线性(仿射)变换：linear(affine) transformation.
```

------

线性模型中的线性关系假设具有很强的局限性。为了增加此类线性模型的灵活性，一种简单方法是使用特征变换，即利用 $\phi(\mathbf{x})$ 替代 $\mathbf{x}$。举例来说，我们可以使用多项式变换，在 $\textrm{1}$ 维数据中，该变换函数定义为 $\phi(x)=[1,x,x^2,x^3,...]$，我们在 $\textrm{1.2.2.2}$ 节中对该方法进行了讨论。这种方法有时被称为 **基函数拓展** ($\textrm{basis function expansion}$)。基于该变换，上述模型定义为:
$$
f(\mathbf{x}; \pmb{\theta})=\mathbf{W}\mathbf{\phi}(\mathbf{x}) + \mathbf{b} \tag{13.3}
$$
上式关于参数 $\pmb{\theta}=(\mathbf{W}, \mathbf{b})$ 依然是线性关系，从而降低了模型的拟合难度。然而，手动设计的特征变换函数依然具有很强的局限性。

一个很自然的拓展是为特征提取器赋予自己的参数 $\pmb{\theta}^\prime$， 即:
$$
f(\mathbf{x};\pmb{\theta},\pmb{\theta}^\prime)=\mathbf{W}\phi(\mathbf{x};\pmb{\theta}^\prime)+\mathbf{b} \tag{13.4}
$$
我们可以递归地重复上述过程，从而构造一个越来越复杂的函数。如果我们组合 $L$ 个函数，即
$$
f(\mathbf{x};\pmb{\theta})=f_L(f_{L-1}(...(f_1(\mathbf{x}))...)) \label{eq:13.5} \tag{13.5}
$$
其中 $f_l(\mathbf{x})=f(\mathbf{x};\pmb{\theta}_l)$ 为第 $l$ 层的函数。这便是 **深度神经网络** ($\textrm{deep neural networks, DNNs}$) 背后的关键思想。

```markdown
特征变换： feature transformation
特征提取器： feature extractor
```

------

术语 “$\textrm{ DNN}$ ” 实际上包含了一系列模型，其中我们将可微函数组合到任何类型的 $\textrm{DAG}$（有向无环图）中，实现输入到输出的映射函数建模。 式 (\ref{eq:13.5}) 是链式 $\textrm{DAG}$ 的最简单示例。 这被称为 **前馈神经网络**（$\textrm{feedforward neural network, FFNN}$）或 **多层感知器**（$\textrm{multilayer perceptron, MLP}$）。

$\textrm{MLP}$ 假定输入是一个维度固定的矢量，即 $\mathbf{x} \in \mathbb{R}^D$。 我们称此类数据为“**非结构化数据**” ($\textrm{unstructured data}$)，因为我们没有对输入的形式进行任何假设。 但是，这使得该模型难以应用于具有可变大小或形状的输入。 在第 $\textrm{14}$ 章中，我们讨论了**卷积神经网络**（$\textrm{convolutional neural networks, CNN}$），用于处理可变大小的图像。 在第 $\textrm{15}$ 章中，我们讨论了**递归神经网络**（$\textrm{recurrent neural networks, RNN}$），用于处理可变大小的序列。 在第 $\textrm{23}$ 章中，我们讨论了**图神经网络**（$\textrm{graph neural networks, GNN}$），用于处理可变大小的图。 有关 $\textrm{DNN}$ 的更多信息，请参见其他书籍，例如 [HG20][^HG20], [Zha19a][^Zha19a], [Ger19][^Ger19]。

[^HG20]: J. Howard and S. Gugger. Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD. en. 1st ed. O’Reilly Media, Aug. 2020.
[^Zha19a]: A. Zhang, Z. Lipton, M. Li, and A. Smola. Dive into deep learning. 2019.
[^Ger19]: A. Géron. Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques for Building Intelligent Systems (2nd edition). en. O’Reilly Media, Incorporated, 2019.

## 13.2 多层感知机

在第 $\textrm{10.2.5}$ 节，我们表明 **感知机** ($\textrm{perceptron}$) 就是逻辑回归模型的一个确定性版本。具体来说，它是一个具备如下形式映射函数:
$$
f(\mathbf{x};\mathbf{\theta})=\mathbb{I}(\mathbf{w}^{\rm{T}}\mathbf{x}+b\ge0)=H(\mathbf{w}^{\rm{T}}\mathbf{x}+b) \tag{13.6}
$$
其中 $H(a)$ 表示 **单位阶跃函数**（$\textrm{heaviside step function}$）， 又被称为 **线性阈值函数** ($\textrm{linear threshold function}$）。由于感知机的决策边界依然是线性的，所以其表达能力十分有限。$\textrm{1969}$ 年，$\textrm{Marvin Minsky}$ 和 $\textrm{Seymour Papert}$ 出版了一本名为 $\textrm{《 Perceptrons》}$ [MP69][^MP69] 的著名著作，其中他们给出了许多感知机无法解决的模式识别的问题。 在讨论如何解决问题之前，我们首先举一个具体的例子。

[^MP69]: M. Minsky and S. Papert. Perceptrons. MIT Press, 1969.



| $x_1$ | $x_2$ | $y$  |
| ----- | ----- | :--: |
| 0     | 0     |  0   |
| 0     | 1     |  1   |
| 1     | 0     |  1   |
| 1     | 1     |  0   |

表$\textrm{ 13.1}$： 抑或问题的真值表，$y=x_{1} \underline{\vee} x_{2}$。

![xor-heaviside](/assets/img/figures/xor-heaviside.png)

图 $\textrm{13.1}$：$\textrm{(a)}$ 抑或函数无法实现线性可分，但基于单位阶跃函数构建的两层模型可以将数据分开。程序由 $xor-heaviside.py$ 生成。 $\textrm{(b)}$ 包含一个隐藏层的神经网络，其中的权重由人工设计，该网络实现了抑或函数。 $h_1$ 表示 $AND$ 函数，$h_2$ 表示 $OR$ 函数。 偏置项表示为常数节点（值为 $\textrm{1}$）的连接权重。

```python
import numpy as np
# Show that 2 layer MLP (with manually chosen weights) can solve the XOR problem
# xor-heaviside.py
import numpy as np
import matplotlib.pyplot as plt
def heaviside(z):
    return (z >= 0).astype(z.dtype)
def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) - 0.5)
x1s = np.linspace(-0.2, 1.2, 100)
x2s = np.linspace(-0.2, 1.2, 100)
x1, x2 = np.meshgrid(x1s, x2s)

z1 = mlp_xor(x1, x2, activation=heaviside)
z2 = mlp_xor(x1, x2, activation=sigmoid)

plt.figure()
plt.contourf(x1, x2, z1)
plt.plot([0, 1], [0, 1], "gs", markersize=20)
plt.plot([0, 1], [1, 0], "r^", markersize=20)
plt.title("Activation function: heaviside", fontsize=14)
plt.grid(True)
plt.show()
```



### 13.2.1 抑或问题

$\textrm{《 Perceptrons》}$书中最著名的例子之一就是 $\textrm{XOR}$ 问题。 这里的目标是学习一个计算两个二进制输入的异或函数。 表 $\textrm{13.1}$ 给出了该函数的真值表。 我们在图 $\textrm{13.1a}$ 中对该函数进行了可视化。 显然，数据不是线性可分离的，因此感知机模型无法表示该映射函数。

但是，我们可以通过叠加多个感知机来克服这个问题。 这称为**多层感知机**（$\textrm{multilayer perceptron, MLP}$）。 例如，要解决 $\textrm{XOR}$ 问题，我们可以使用图$\textrm{13.1b}$ 所示的 $\textrm{MLP}$。 它由 $\textrm{3}$ 个感知机组成，分别表示为 $h_1$，$h_2$ 和 $y$。 节点 $x$ 表示输入，节点 $1$ 表示常数项。 节点 $h_1$ 和 $h_2$ 被称为**隐藏单元** （$\textrm{hidden units}$），因为在训练数据中未观察到它们的真值。

第一个隐藏单元通过使用设置的合理权重来计算 $h_{1}=x_{1} \wedge x_{2}$。（此处 $\wedge$ 表示 $\rm{AND}$ 操作。）特别地，它的输入为 $x_1$和 $x_2$，且权重均为为 $\textrm{1.0}$，同时具有 $\textrm{-1.5}$ 的偏置项（通过虚拟一个常量节点 $\textrm{1}$ 来实现偏置）。 因此，如果 $x_1$ 和 $x_2$ 都等于 $\textrm{1}$，则 $h_1$ 将被激活，因为
$$
\mathbf{w}_1^{\rm{T}}\mathbf{x}-b_1=[1.0, 1.0]^{\rm{T}}[1, 1] - 1.5 =0.5 > 0 \tag{13.7}
$$
类似的，第二个隐藏单元计算 $h_{2}=x_{1} \vee x_{2}$，其中 $\vee$ 为 $\rm{OR}$ 操作，第三个节点计算输出 $y=\overline{h_1} \wedge h_2$，其中 $\bar{h}=\neg h$ 为 $\rm{NOT}$ 操作 （ 逻辑非）。 所以节点 $y$ 计算
$$
y=f\left(x_{1}, x_{2}\right)=\overline{\left(x_{1} \wedge x_{2}\right)} \wedge\left(x_{1} \vee x_{2}\right) \tag{13.8}
$$
上式等价于 $\rm{XOR}$ 函数。

通过扩展上述示例，我们可以证明 $\textrm{MLP}$ 可以表示任何逻辑函数。 但是，我们显然希望避免手动指定权重和偏差。 在本章的其余部分，我们将讨论从数据中学习这些参数的方法。

### 13.2.2 可微多层感知机

我们在第 $\textrm{13.2.1}$ 节中讨论的 $\textrm{MLP}$ 被定义为多个感知机的叠加，每个感知机都包含不可微的 $\textrm{Heaviside}$ 函数。 这使得这种模型很难训练，这就是为什么它们从未被广泛使用的原因。然而，如果我们将阶跃函数 $H:\mathbb{R}\rightarrow \{0,1\}$ 替换为一个可微的 **激活函数** （$\textrm{activation function}$）$\varphi:\mathbb{R} \rightarrow \mathbb{R}$ 。更精确地讲，我们将每一层 $l$ 的隐藏单元 $\mathbf{z}_l$ 定义为通过激活函数逐元素传递的上一层隐藏单元的线性变换：

```markdown
More precisely, we define the hidden units $\mathbf{z}_l$ at each layer $l$ to be a linear transformation of the hidden units at the previous layer passed elementwise through this activation function.
```


$$
\mathbf{z}_l=f_l(\mathbf{z}_{l-1})=\varphi(\mathbf{b}_l+\mathbf{W}_l\mathbf{z}_{l-1}) \tag{13.9}
$$


或者，以标量的形式：


$$
z_{kl}=\varphi_l \left( b_{kl}+\sum_{j=1}^{K_{l-1}}w_{jkl}z_{jl-1} \right) \tag{13.10}
$$

如式 (\ref{eq:13.5}) 中所示，如果我们现在将 $L$ 个诸如此类的激活函数叠加在一起。然后我们可以使用链式规则，计算输出关于每一层中的参数的梯度，也称为**反向传播** （$\textrm{backpropagation}$），如我们在第 $\textrm{13.3}$ 节中所解释的。 （这对于任何一种可微的激活函数都是正确的，尽管某些类型的函数要比其他类型的函数更适用，正如我们在第 $\textrm{13.2.3}$ 节中讨论的那样。）然后，我们可以将梯度传递给优化器，从而最小化某些训练目标，正如我们在 $\textrm{13.4}$ 节讨论的那样。 因此，术语“$\textrm{ MLP}$”几乎总是指可微的模型，而不是指具有不可微分线性阈值单位的历史版本。

```markdown
不可微: non-differentiable
```

| $\textrm{Name}$                    | $\textrm{Definition}$                                        | $\textrm{Range}$     | $\textrm{Reference}$            |
| ---------------------------------- | ------------------------------------------------------------ | -------------------- | ------------------------------- |
| $\textrm{Sigmoid}$                 | $\sigma_{a}=\frac{1}{1+e^{-a}}$                              | $[0, 1]$             |                                 |
| $\textrm{Hyperbolic tangent}$      | $\tanh(a)=2\sigma(2a)-1$                                     | $[-1, 1]$            |                                 |
| $\textrm{Softplus}$                | $\sigma_{+}(a)=\log(1+e^a)$                                  | $[0, \infty]$        | [GBBB11][^GBB11]                |
| $\textrm{Rectified linear unit}$   | $\operatorname{ReLU}(a)=\max (a, 0)$                         | $[0, \infty]$        | [GBB11][^GBB11];[KSH12][^KSH12] |
| $\textrm{Leaky ReLU}$              | $\max (a, 0)+\alpha \min (a, 0)$                             | $[-\infty, +\infty]$ | [MHN13][^MHN13]                 |
| $\textrm{Exponential linear unit}$ | $\max (a, 0)+\min \left(\alpha\left(e^{a}-1\right), 0\right)$ | $[-\infty, +\infty]$ | [CUH16][^CUH16]                 |
| $\textrm{Swish}$                   | $a \sigma(a)$                                                | $[-\infty, +\infty]$ | [RZL17][^RZL17]                 |

表 $\textrm{13.2}$：神经网络中常用的一些激活函数

[^GBB11]: text
[^KSH12]: text
[^MHN13]: text
[^CUH16]: text
[^RZL17]: text

![activations](/assets/img/figures/activations.png)

图 $\textrm{13.2}$：$\textrm{(a)}$ 对于 $sigmoid$ 函数而言，当输入在 $0$ 附近时，输出与输入呈线性关系，但对于较大的正值或负值输入，则输出存在饱和区。图形由程序 生成。 $\textrm{(b)}$ 一些常用的非饱和激活函数的可视化。图形由程序 生成。

### 13.2.3 激活函数

我们可以在每一层使用任何一种可微的激活函数。然而，如果我们使用 *线性* （$\textrm{linear}$）激活函数  $\varphi_l(a)=c_la$，那整个模型将退化为一个常规的线性模型。以式  (\ref{eq:13.5})为例，该模型将退化为：


$$
f(\mathbf{x};\mathbf{\theta})=\mathbf{W}_Lc_L(\mathbf{W}_{L-1}c_{L-1}(...(\mathbf{W}_1\mathbf{x})...)) \propto \mathbf{W}_L\mathbf{W}_{L-1}...\mathbf{W}_1\mathbf{x}=\mathbf{W}^\prime\mathbf{x} \tag{13.11}
$$


在上式中，为了符号上的简洁性，我们丢弃了偏置项。基于上述原因，使用非线性激活函数就显得十分重要。

在神经网络的早期发展阶段，一个常见的选择是使用 $\textrm{S}$ 型 ($\textrm{logistic}$) 激活函数，该函数可以看做是单位阶跃函数的平滑近似版本。然而，如图 $\textrm{13.2a}$ 所示，对于较大的正值输入，$\textrm{S}$ 形函数存在饱和值 $\textrm{1}$；对于较大的负值输入，$\textrm{S}$ 形函数存在饱和值 $\textrm{0}$。 $\textrm{tanh}$ 激活函数具有相似的形状，但其饱和值分别为 $\textrm{-1}$ 和 $\textrm{+1}$。 在这些饱和区域，输出关于输入的斜率将接近于零。因此，如我们在第 $\textrm{13.4.2}$  节中所讨论的，来自深层网络的任何梯度信号都将“消失”。

要想成功训练一个非常深的神经网络模型，一个关键因素是使用 **非饱和激活函数** （$\textrm{non-saturating activation functions}$）。几种不同的激活函数如表 $\textrm{13.2}$ 所示。其中最常用的是 **整流线性单元** （$\textrm{rectifled linear unit, ReLU}$）。定义为


$$
{\rm{ReLU}}(a)=\max(a,0)=a\mathbb{I}(a>0) \tag{13.12}
$$


该 $\textrm{ReLU}$ 函数简单地将负值输入置零，并保持正值输入保持不变。如 $\textrm{13.3.3.2}$ 节所介绍的，这种形式至少保证对于正值输入，梯度的值为 $\textrm{1}$，从而避免了梯度的消失。

不幸的是，对于负值输入，$\textrm{ReLU}$的梯度依然为 $\textrm{0}$，因此，该单元将永远无法获得任何反馈信号来帮助其摆脱当前的参数设置 （**译者注：**即无法逃离负值区域）； 这被称为 “**垂死ReLU**” ($\textrm{dying ReLU}$) 问题。

一种简单的解决方法是使用[MHN13][^MHN13]中提出的 **泄漏ReLU** （$\textrm{leaky ReLU}$）。 定义为


$$
{\rm{LReLU}}(a;\alpha)=\max(\alpha a,a) \tag{13.13}
$$

其中 $0 \lt \alpha \lt 1$。该函数对于正值输入的斜率为 $\textrm{1}$，对于负值输入的斜率为 $\alpha$， 所以可以确保当输入为负值时，依然可以有信号可以从更深的网络层中反传回来。如果我们允许参数 $\alpha$ 可以学习，而非固定， $\textrm{leaky ReLU}$ 将被称为 **参数化ReLU** （$\textrm{parametric ReLU}$）。

另一个广泛的选择是 [CUH16][^CUH16] 中提出的 $\textrm{ELU}$，定义为


$$
{\rm{ELU}}(a;\alpha) = \begin{cases}
\alpha(e^a-1) & \text{if } a\le0\\
a & \text{if } a \gt 0
\end{cases} \tag{13.14}
$$

[^CUH16]: 

与 $\textrm{leaky ReLU}$ 相比，它具有平滑函数的优点。

在[Kla+17][^Kla17]中提出了一种 $\textrm{ELU}$ 的轻微变体，称为 $\textrm{SELU}$（自规范化$\textrm{ELU}$）。 形式为


$$
{\rm{SELU}}(a;\alpha,\lambda)=\lambda{\rm{ELU}}(a;\alpha) \tag{13.15}
$$

[^Kla17]: text

出乎意料的是，他们证明了通过为 $\alpha$ 和 $\lambda$ 设置精心选择的值，即使不使用 $\textrm{batchnorm}$ 技术（见$\textrm{13.4.5}$节），也可以确保通过激活函数来确保每个网络层的输出是被标准化的（假设输入也已标准化）。 这可以促进模型的拟合。

作为手动发现良好的激活函数的替代方法，我们可以使用黑盒优化方法来对激活函数空间进行搜索。 [RZL17][^RZL17]使用这种方法发现了称为 $\mathrm{swish}$ 的函数，该函数在某些图像分类数据集上似乎表现很好。该函数定义为


$$
{\rm{swish}}(a;\beta)=a\sigma(\beta a)\tag{13.16}
$$

有关这些函数的可视化对比，请参见图$\textrm{13.2b}$。 我们看到，它们主要是在处理负输入的方式上存在差异。

[^RZL17]: 

### 13.2.4 案例模型

$\textrm{MLPs}$ 可以对很多类型的数据进行分类和回归，接下来我们将给出一些案例。

#### 13.2.4.1 MLP用于表格数据分类

图 $\textrm{13.3}$给出了一个包含两个隐藏层的 $\textrm{MLP}$ 示意图，将该$\textrm{MLP}$应用于$\textrm{1.2.1.1}$节中的表格鸢尾花数据集，该数据集具有 $\textrm{4}$ 个特征和 $\textrm{3}$ 个类别。 该模型具有如下形式


$$
\begin{align}
p(y|\mathbf{x};\mathbf{\theta})=& {\rm{Cat}}(y|f_3(\mathbf{x};\mathbf{\theta})) \tag{13.17}\\
f_3(\mathbf{x};\mathbf{\theta})=&\mathcal{S}(\mathbf{W}_3f_2(\mathbf{x};\mathbf{\theta})+\mathbf{b}_3) \tag{13.18} \\
f_2(\mathbf{x};\mathbf{\theta})=&\varphi_2(\mathbf{W}_2f_1(\mathbf{x};\mathbf{\theta})+\mathbf{b}_2) \tag{13.19} \\
f_1(\mathbf{x};\mathbf{\theta})=&\varphi_1(\mathbf{W}_1f_0(\mathbf{x};\mathbf{\theta})+\mathbf{b}_1) \tag{13.19} \\
f_0(\mathbf{x};\mathbf{\theta})=&\mathbf{x} \tag{13.21}
\end{align}
$$


其中 $\mathbf{\theta}=(\mathbf{W}_3,\mathbf{b}_3,\mathbf{W}_2,\mathbf{b}_2,\mathbf{W}_1,\mathbf{b}_1)$ 为模型中的参数，对应于 $\textrm{3}$ 组可调节权重的连接边。我们看到最终（输出）层的激活函数为 $\textrm{softmax}$ 函数，$\textrm{softmax}$ 函数是分类分布的反向连接函数。对于隐藏层，我们可以自由选择所需的不同形式的激活函数，正如我们在第 $\textrm{13.2.3}$ 节中所讨论的。

#### 13.2.4.2 MLP用于图像分类

要将 $\textrm{MLP}$ 应用于图像分类，我们需要将 $\textrm{2d}$ 输入“**展开**”（$\textrm{flatten}$）为 $\textrm{1d}$ 向量。 然后，我们可以使用类似于第$\textrm{13.2.4.1}$ 节中所述的前馈网络。 例如，考虑构建一个 $\textrm{MLP}$ 以对 $\textrm{MNIST}$ 数字进行分类（第$\textrm{3.7.2}$ 节）。 这些数字表示为 $28\times28 = 784$ 维向量。 如果我们使用 $\textrm{2}$ 个具有$\textrm{128}$个单位的隐藏层，和 $\textrm{1}$ 个包含 $\textrm{10}$ 个输出单元的$\textrm{softmax}$ 层，将得到如图$\textrm{13.4}$ 所示的模型。

我们在图 $\textrm{13.5}$ 中展示了一些该模型的预测结果。 我们对训练集仅训练两个“**周期**”（$\textrm{epochs}$, 遍历数据集的次数），但是该模型已经具备较好的性能，测试集的准确率为 $\textrm{97.1％}$。 此外，错误的预测案例似乎也是可以理解的，例如将 $\textrm{9}$ 误认为是 $\textrm{3}$。训练更多的时间可以进一步提高测试的准确性。

在第 $\textrm{14}$ 章中，我们讨论了另一种称为卷积神经网络的模型，该模型更适用于图像数据的处理。 通过利用与图像数据相关的空间结构的先验知识，它可以获得更好的性能，并使用更少的参数。相比之下，$\textrm{MLP}$ 对输入的排列具有不变性。 换句话说，我们可以随机地对像素进行排列，并且可以获得相同的结果（前提是我们对所有的输入使用相同的随机排列算法）。

#### 13.2.4.3 MLP用于电影评论的情感分析

[Maa+11][^Maa11]的 $\textrm{IMDB}$ 电影评论数据集（$\textrm{IMDB}$ 代表“互联网电影数据库”。）被称为 “文本分类的 $\textrm{MNIST}$”。该数据集包含 $\textrm{25k}$ 带有标签的样本用于训练，而 $\textrm{25k}$的样本用于测试。 每个样本都有一个二进制标签，代表积极或消极的评分。 此任务称为（二进制）**情感分析** （$\textrm{sentiment analysis}$）。 例如，以下是训练集中的两个样本：

```markdown
1. this film was just brilliant casting location scenery story direction everyone’s really suited the part they played robert \<UNK\> is an amazing actor ...
2. big hair big boobs bad music and a giant safety pin these are the words to best describe this terrible movie i love cheesy horror movies and i’ve seen hundreds...
```

> 1. this film was just brilliant casting location scenery story direction everyone’s really suited the part they played robert \<UNK\> is an amazing actor ...
> 2. big hair big boobs bad music and a giant safety pin these are the words to best describe this terrible movie i love cheesy horror movies and i’ve seen hundreds...

毫不奇怪，第一个样本被标记为正例，第二个标记为负例。

[^Maa11]: 

我们可以设计一个 $\textrm{MLP}$ 来进行情感分析，如下所示。假设输入是一个包含 $T$ 个符号的序列 $\mathbf{x}_{1:T}$，其中 $\mathbf{x}_t$ 是一个长度为 $V$ 的 $\textrm{one-hot}$ 向量，其中 $V$ 为词语料库的大小。我们将此视为无序的单词袋（第$\textrm{10.4.3.1}$ 节）。模型的第一层为 $E\times V$  的嵌入矩阵 $\mathbf{W}_1$，该层将每一个稀疏的 $V$ 维向量映射到一个稠密的 $E$ 维向量 ${\rm{e}}_t=\mathbf{W}_1\mathbf{x}_t$（见$\textrm{19.5}$ 节学习更多关于词嵌入的细节）。接着我们使用 **全局平均池化**（$\textrm{global average pooling}$）将 $T\times D$ 的序列嵌入向量转化为一个固定长度的向量 ${\rm{\bar{\mathbf{e}}}}=\frac{1}{T}\sum_{t=1}^T{\rm{\mathbf{e}}}_t$。接着我们将该向量传入一个非线性隐藏层，计算一个 $K$ 维向量 $\mathbf{h}$，并将其传入最后的线性 $\textrm{logistic}$ 层。 综上所述，模型定义如下：


$$
\begin{align}
p(y|\mathbf{x};\mathbf{\theta}) = & {\rm{Ber}}(y|\sigma(\mathbf{w}_3^{\rm{T}}\mathbf{h}+b_3)) \tag{13.22} \\
\mathbf{h}= & \varphi(\mathbf{W}_2{\rm{\bar{\mathbf{e}}}}+\mathbf{b}_2) \tag{13.23} \\
{\bar{\mathbf{e}}}=& \frac{1}{T}\sum_{t=1}^T\mathbf{e}_t \tag{13.24} \\
\mathbf{e}_t=& \mathbf{W}_1\mathbf{x}_t \tag{13.25}
\end{align}
$$


如果我们使用的语料库大小为 $V = 1000$，嵌入向量维度为 $E = 16$，隐藏层的维度为$\textrm{16}$，则得到的模型如图 $\textrm{13.6}$ 所示。 模型在验证集的准确度为 $\textrm{86％}$。

我们看到模型中大多数参数都分布在嵌入矩阵中，这可能会导致过拟合问题。 幸运的是，正如我们在第 $\textrm{19.5}$ 节中讨论的那样，我们可以执行词嵌入模型的无监督预训练，然后我们只需要微调此特定标记任务的输出层参数即可。

#### 13.2.4.4 MLP用于异方差回归

我们还可以使用 $\textrm{MLP}$ 进行回归。 图 $\textrm{13.7}$ 显示了如何为异方差 （$\textrm{heteroskedastic}$）非线性回归任务建立模型。 （术语“异方差”仅表示预测的输出方差与输入有关，如第$\textrm{3.3.3}$节中所述。）该函数具有两个输出，分别表示 $f_\mu(\mathbf{x})=\mathbb{E}[y|\mathbf{x},\mathbf{\theta}]$ 和 $f_\sigma(\mathbf{x})=\sqrt{\mathbb{V}[y|\mathbf{x},\mathbf{\theta}]}$。如图$\textrm{13.7}$所示，通过使用一个共享的**主干网络**（$\textrm{backbone}$）和两个输出**头**（$\textrm{heads}$）， 我们可以在这两个函数之间共享大多数层（因此也可以共享参数）。对于 $\mu$ 头，我们使用一个线性激活函数 $\varphi(a)=a$。对于 $\sigma$ 头，我们使用 $\textrm{softplus}$ 激活函数 $\varphi(a)=\sigma_{+}(a)$。如果我们使用线性头和一个非线性主干网络，整个模型定义为
$$
p(y|\mathbf{x},\mathbf{\theta})=\mathcal{N}(y|\mathbf{w}_\mathbf{\mu}^{\rm{T}}f(\mathbf{x};\mathbf{w}_{\rm{shared}}), \sigma_{+}(\mathbf{w}_{\mathbf{\sigma}}^{\rm{T}}f(\mathbf{x};\mathbf{w}_{\rm{shared}})))\tag{13.26}
$$
图$\textrm{13.8}$ 显示了这种模型在某些数据集上的优势，在该数据集中，预测的期望值随时间线性增长，并且随季节波动，与此同时，数据的方差呈二次方增加趋势。（这是 **随机波动率模型** （$\textrm{stochastic volatility model}$）的一个简单示例；它可以用于对财务数据以及地球的全球温度进行建模，其中地球温度的（由于气候变化）均值和方差不断增加。）我们发现 将输出方差 $\sigma^2$ 视为固定（与输入无关）参数的回归模型置信度有时会比较低，因为模型需要适应整体的噪声水平，并且无法适应输入空间中每个点的噪声水平。

### 13.2.5 深度的重要性

研究表明包含一个隐藏层的 $\textrm{MLP}$ 是一个**通用函数逼近器**（$\textrm{universal function approximator}$），这意味着只要给定足够的隐藏单元，$\textrm{MLP}$ 就可以逼近任何平滑函数，并达到任何所需的精度水平[HSW89][^HSW89]; [cyb89][^cyb89]; [Hor91][^Hor91]。直观地讲，这样做的原因是每个隐藏的单元都可以指定一个半平面，并且这些单元的足够大的组合可以“划分”空间的任何区域，我们可以将其与任何响应相关联（这在分段使用时最容易看到 线性激活函数，如图13.9所示。

但是，实验和理论上的各种论证（例如[Has87][^Has87]; [Mon+14][^Mon14]; [Rag+17][^Gag17]; [Pog+17][^Pog17]）都表明，深层网络比浅层网络更有效。 原因是更高的层次可以利用先前的层次所学习的功能。 也就是说，该功能是以组合或分层的方式定义的。 例如，假设我们要对 $\textrm{DNA}$ 字符串进行分类，并且正类与正则表达式$\textrm{* AA ?? CGCG ?? AA *}$相关联。 尽管我们可以将其与单个隐藏层模型配合使用，但是从直观上来说，如果模型首先学会使用第 $\textrm{1}$ 层中的隐藏单元来检测 $\textrm{AA}$ 和 $\textrm{CG}$ “基元”，然后使用这些功能来定义一个简单的模型，则将更容易学习 第 $\textrm{2}$ 层中的线性分类器，类似于我们如何解决第 $\textrm{13.2.1}$ 节中的 $\textrm{XOR}$ 问题。

[^HSW89]: 
[^cyb89]: 
[^Hor91]: 
[^Has87]: 
[^Mon14]: 
[^Rag17]: 
[^Pog17]: 

#### 13.2.5.1 深度学习革命

尽管 $\textrm{DNN}$ 背后的思想可以追溯到几十年前，但直到 $\textrm{2010}$ 年代，它们才开始被广泛使用。 突破性的时刻发生在$\textrm{2012}$ 年，当时[KSH12][^KSH12]表明深层的 $\textrm{CNN}$ 可以在具有挑战性的 $\textrm{ImageNet}$ 图像分类基准上显着提高性能，将错误率从一年的$\textrm{26％}$降低到$\textrm{16％}$（见图$\textrm{14.14b}$）； 与之前每年约减少$\textrm{2％}$的进度相比，这是一个巨大的飞跃。 大约在同一时间，[DHK13][^DHK13]表明，在各种语音识别任务上，深度神经网络可以大大优于现有技术。

$\textrm{DNN}$ 的使用中的“爆炸”有几个促成因素。 一是便宜的$\textrm{GPU}$（图形处理单元）的可用性。 它们最初是为了加快视频游戏的图像渲染速度而开发的，但是它们也可以大大减少适合大型 $\textrm{CNN}$ 的时间，而大型 $\textrm{CNN}$ 涉及类似的矩阵矢量计算。 另一个是大型标记数据集的增长，这使我们能够在不过度拟合的情况下将具有许多参数的复杂函数逼近器拟合。 （例如，$\textrm{ImageNet}$ 具有$\textrm{130}$ 万个带标签的图像，并用于拟合具有数百万个参数的模型。）的确，如果将深度学习系统视为“火箭”，那么大型数据集就被称为燃料。

由于$\textrm{DNN}$取得了巨大的经验成功，许多公司开始对该技术产生兴趣。 这导致开发了高质量的开源软件库，例如$\textrm{Tensorflow}$（由 $\textrm{Google}$ 开发），$\textrm{PyTorch}$（由 $\textrm{Facebook}$ 开发）和 $\textrm{MXNet}$（由亚马逊开发）。 这些库支持复杂的微分函数的自动微分（请参阅第 $\textrm{13.3}$ 节）和基于可伸缩的基于梯度的优化（请参见第 $\textrm{5.4}$ 节）。 在本书的各个地方，我们将使用其中的一些库来实现各种模型，而不仅仅是 $\textrm{DNN}$。

有关“深度学习革命”历史的更多详细信息，请参见[Sej18][^Sej18]。

[^KSH12]: 
[^DHK13]: 
[^Sej18]: 

### 13.2.6 与生物学的联系

在本节中，我们讨论了上文讨论过的各种神经网络（称为人工神经网络或 $\textrm{ANN}$）与实际神经网络之间的联系。 真正的生物大脑如何工作的细节非常复杂（例如，参见[Kan+12][^Kan12]），但是我们可以给出一个简单的“卡通”。

我们首先考虑单个神经元的模型。 对于第一近似，我们可以说神经元 $k$ 是否发射，用 $h_{k} \in\{0,1\}$ 表示，取决于其输入的活动（用$\mathbf{x} \in \mathbb{R}^{D}$表示）以及传入连接的强度（我们用$\mathbf{w}_{k} \in \mathbb{R}^{D}$表示）。 我们可以使用$a_{k}=\mathbf{w}_{k}^{\top} \mathbf{x}$来计算输入的加权和。 这些权重可以看作是将输入$x_d$ 连接到神经元$h_k$的“电线”。 这些类似于真实神经元中的树突（见图$\textrm{13.10}$）。然后将该加权总和与阈值$b_k$进行比较，如果激活超过阈值，则神经元触发； 这类似于神经元发出电输出或动作电位。 因此，我们可以使用 $h_{k}(\mathbf{x})=H\left(\mathbf{w}_{k}^{\top} \mathbf{x}-b_{k}\right)$ 来建模神经元的行为，其中 $H(a)=\mathbb{I}(a>0)$ 是$\textrm{Heaviside}$函数。 这称为神经元的$\textrm{McCulloch-Pitts}$模型，并于$\textrm{1943}$年提出[MP43][^MP43]。

[^Kan12]: 
[^MP43]: 



我们可以将多个这样的神经元组合在一起以构成一个人工神经网络。 结果有时被视为大脑的模型。 但是，人工神经网络在许多方面与生物大脑不同，包括以下方面：

- 大多数 $\textrm{ANN}$ 使用反向传播来修改其连接强度（请参阅第$\textrm{13.3}$节）。 但是，真正的大脑不会使用反向传播，因为无法沿着轴突向后发送信息[Ben+15b][^Ben15b]; [BS16][^BS16]；[KH19][^KH19]。 相反，他们使用本地更新规则来调整突触强度。
- 大多数$\textrm{ANN}$都是严格的前馈，但是真实的大脑有很多反馈连接。 可以相信，这种反馈的作用类似于先验，可以与来自感官系统的自下而上的可能性结合起来，计算出世界上隐藏状态的后验，然后可以将其用于最佳决策（例如，参见[Doy+07][^Doy07]）。
- 大多数人工神经网络使用简化的神经元，该神经元由经过非线性处理的加权总和组成，但实际的生物神经元具有复杂的树状树结构（见图$\textrm{13.10}$），具有复杂的时空动态。
- 大多数人工神经网络的大小和连接数均小于生物大脑（见图$\textrm{13.11}$）。 当然，在各种新型硬件加速器（例如$\textrm{GPU}$和$\textrm{TPU}$（张量处理单元）等）的推动下，人工神经网络每周都会变得越来越大。但是，即使人工神经网络在单元数量上与生物大脑相匹配，这种比较也具有误导性，因为 生物神经元的处理能力远高于人工神经元（见上文）。
- 大多数ANN被设计为对单个函数建模，例如将图像映射到标签，或将单词序列映射到另一个单词序列。 相比之下，生物大脑是非常复杂的系统，由多个专门的交互模块组成，这些模块实现不同种类的功能或行为，例如感知，控制，记忆，语言等（请参见[Sha88][^Sha88]; [Kan+12][^Kan12]）。

[^Ben15b]: 
[^BS16]: 
[^KH19]: 
[^Doy07]: 
[^Sha88]: 
[^Kan12]: 

当然，我们正在努力建立逼真的生物大脑模型（例如，蓝脑计划[Mar06][^Mar06]; [Yon19][^Yon19]）。但是，一个有趣的问题是，以这种详细程度研究大脑是否对“解决$\textrm{AI}$”有用？通常认为，生物大脑的低级细节并不重要，我们的目标是否是建造“智能机器”，就像飞机不会拍打自己的飞机一样。 翅膀。但是，大概“ $\textrm{AI}$”将遵循与智能生物代理类似的“智能定律”，就像飞机和鸟类遵循相同的空气动力学定律一样。 不幸的是，我们尚不知道什么是“智力法则”，或者甚至是否存在这样的法则。在本书中，我们假设任何智能主体都应遵循信息处理和贝叶斯决策理论的基本原理，这是在不确定性下做出决策的最佳方法（请参见第 $\textrm{8.4.2}$ 节）。 当然，生物因子受到许多约束（例如，计算，生态），这通常需要算法“捷径”才能获得最佳解决方案。这可以解释人们在日常推理中使用的许多启发式方法。[KST82][^KST82]; [GTA00][^GTA00]; [Gri20][^Gri20]。随着我们希望机器解决的任务变得越来越困难，我们也许能够从神经科学和认知科学的其他领域获得见识（例如，参见[MWK16][^MWK16]; [Has+17][^Has17]; [Lak+17][^Lak17]）。

[^Mar06]: 
[^Yon19]: 
[^KST82]: 
[^GTA00]: 
[^Gri20]: 
[^MWK16]: 
[^Has17]: 
[^Lak17]: 

