---
title: 34 不确定性下的决策
author: fengliang qi
date: 2025-11-22 11:33:00 +0800
categories: [BOOK-2, PART-VI]
tags: [Action,Decision making]
math: true
mermaid: true
toc: true
comments: true
---

> 本章将介绍统计决策理论，其背后的动机在于，在已知外部世界的潜在状态后，我们如何选择最优的行动，从而实现外部奖励的最大化。
>
> 本章实质上构成了强化学习的基础，具有承上启下的关键作用，乘上是指，它继承于AI算法对外部世界的感知推断，启下是指，它需要根据推断的结论进行决策（即选择具体的行动），以实现外部世界奖励的最大化。
>
> 我们将理清一些重要的基本概念，譬如智能体，策略行动等。
>
> 限于本人能力，目前只翻译了本章大约9成的核心内容。同时感谢 deepseek 的鼎力相助。
>
> 之后的强化学习一章，将在不久后完成，尽请期待....

* TOC
{:toc}
## 34.1 统计决策理论

贝叶斯推断提供了根据观测值 $\boldsymbol{X}=\boldsymbol{x}$ 更新关于隐变量 $H$ 信念状态的最佳方式——计算后验分布 $p(H\mid\boldsymbol{x})$。然而，无论如何，我们都需要将信念状态转换成影响外部世界的**行动**（actions）。那么，我们又该如何决定哪一种行动是最优的呢？这便是**决策理论**（decision theroy）需要回答的问题。本章，我们将简单介绍相关的内容。

### 34.1.1 基础知识

在**统计决策理论**（statistical deciosn theory）中，我们有一个**智能体**（agent）或者决策者，它需要在一系列可选决策 $a\in\mathcal{A}$ 中选择一个**行动**（action），作为应对某个观测或数据 $\boldsymbol{x}$ 的回应。假设数据来自智能体外部的某个环境；我们将该环境的状态表示为某个隐变量或者未知变量 $h\in\mathcal{H}$ ——即**本质状态**（state of nature）。最后，假设存在某个**损失函数**（loss function）$\ell(h, a)$，它反应了本质状态为 $h$ 时执行行动 $a$ 所导致的损失。我们的目标是定义一种**策略**（policy）$a=\delta(\boldsymbol{x})$——又被称为**估计量**（estimator）或**决策程序**（decision procedure），它决定了每种观测下需要采取的行动，从而最小化期望损失——又被称为**风险**（risk）：
$$
\delta^*(\cdot)=\underset{\delta}{\operatorname{argmin}} R(\delta) \tag{34.1}
$$
其中风险定义为：
$$
R(\delta)=\mathbb{E}[\ell(h, \delta(\mathbf{X}))] \tag{34.2}
$$
上式的关键在于如何定义期望。接下来我们将介绍频率学派和贝叶斯学派两种定义视角。

### 34.1.2 频率学派决策理论

**频率学派决策理论**（frequentist decision theory）假设本质状态 $h$ 是固定但未知的量，并将数据 $\mathbf{X}$ 视为随机变量。所以，我们需要关于数据计算期望，即考虑**频率风险**（frequentist risk）：
$$
r(\delta \mid h)=\mathbb{E}_{p(\boldsymbol{x} \mid h)}[\ell(h, \delta(\boldsymbol{x}))]=\int p(\boldsymbol{x} \mid h) \ell(h, \delta(\boldsymbol{x})) d \boldsymbol{x} \tag{34.3}
$$
其背后思想是一个好的策略在不同的数据集上都应该表现为低风险。

遗憾的是，本质状态 $h$ 是未知的，所以上式无法计算。存在一些解决方案，一种思路是假设 $h$ 服从某种先验分布，然后计算**贝叶斯风险**（Bayes risk），又被称为 **综合风险**（integrated risk）:
$$
R_B(\delta) \triangleq \mathbb{E}_{p(h)}[r(\delta \mid h)]=\int p(h) p(\boldsymbol{x} \mid h) \ell(h, \delta(\boldsymbol{x})) d h d \boldsymbol{x} \tag{34.4}
$$
最小化贝叶斯风险的决策规则被称为 **贝叶斯估计量**（Bayes estimator）。

当然，引入先验分布的做法似乎与频率学派是不相容的。因此，我们可以使用**最大风险**（maximum risk）替代。定义为：
$$
R_{\max }(\delta)=\max _h r(\delta \mid h) \tag{34.5}
$$
使最大风险最小的决策规则称为**minimax 估计量**：
$$
\delta^*=\min _\delta \max _h r_h(\delta) \tag{34.6}
$$
Minimax估计量具有一定的吸引力。然而，计算该估计量可能非常困难。此外，该思想的基础是悲观的（因为它只考虑极端情况）。事实上，可以证明，所有minimax估计量都等价于**最不利先验**（least favorable prior）下得到的贝叶斯估计量。在多数情境下（博弈论除外），将自然界作为对手的假设并不合理。有关该观点的进一步讨论，请参考 [BS94, p449]。

### 34.1.3 贝叶斯学派决策理论

**贝叶斯学派决策理论**（Bayesian decision theory）将数据作为可观测常量 $\boldsymbol{x}$，而本质状态作为一个未知的随机变量。行动 $a$的**后验期望损失**（posterior expected loss）定义为：
$$
\rho(a \mid \boldsymbol{x}) \triangleq \mathbb{E}_{p(h \mid \boldsymbol{x})}[\ell(h, a)]=\int \ell(h, a) p(h \mid \boldsymbol{x}) d h \tag{34.7}
$$
对于某个估计量，可以定义后验期望损失或者**贝叶斯式风险**（Bayesian risk）
$$
\rho(\delta \mid \boldsymbol{x})=\rho(\delta(\boldsymbol{x}) \mid \boldsymbol{x}) \tag{34.8}
$$
**最优策略**（optimal policy）决定如何采取行动以最小化期望损失，即
$$
\delta^*(\boldsymbol{x})=\underset{a \in \mathcal{A}}{\operatorname{argmin}} \mathbb{E}_{p(h \mid \boldsymbol{x})}[\ell(h, a)] \tag{34.9}
$$
另一种等效表述如下：定义**效用函数**（utility function）$U(h, a)$，用以衡量在每个可能状态下采取每种可能行动的满意度。如果令 $U(h, a)=-\ell(h, a)$，最优策略可以定义为：
$$
\delta^*(\boldsymbol{x})=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \mathbb{E}_h[U(h, a)] \tag{34.10}
$$
这被称为**最大期望效用原则**（maximum expected utility principle）。

![image-20260102103414285](/assets/img/figures/book2/34.1.png)

### 34.1.4 贝叶斯方法的频率学派最优

我们发现式（34.9）中定义的贝叶斯方法——对每个观测值 $\boldsymbol{x}$ 选择的最优行动，同时使式 （34.4）定义的贝叶斯式风险达到最优——对所有可能的观测选择的最优行动。根据**富比尼定理**（Fubini’s theorem），该定理允许在二重积分中交换积分顺序（等价于**迭代期望法则**（law of iterated expectation））:
$$
\begin{align}
R_B(\delta) & =\mathbb{E}_{p(\boldsymbol{x})}[\rho(\delta \mid \boldsymbol{x})]=\mathbb{E}_{p(h \mid \boldsymbol{x}) p(\boldsymbol{x})}[\ell(h, \delta(\boldsymbol{x}))] \tag{34.11}\\
& =\mathbb{E}_{p(h)}[r(\delta \mid h)]=\mathbb{E}_{p(h) p(\boldsymbol{x} \mid h)}[\ell(h, \delta(\boldsymbol{x}))] \tag{34.12}
\end{align}
$$
如图34.1所示。上式说明贝叶斯学派下的最优策略在频率学派下一样是最优的。

更一般地，可以证明，任何一个**可接受的策略**（admissable policy）¹ 都等价于基于某个（可能为非正常）先验分布的贝叶斯策略，这一结果被称为**瓦尔德定理**（Wald’s theorem）[Wal47]。（该定理更通用的版本可参见[DR21]。）**贝叶斯框架在理论上已经足够广泛，足以涵盖所有“好”的决策策略。**因此，可以说，我们“局限于”贝叶斯方法并不会损失任何可能性（尽管我们仍需检验模型的假设是否充分，这是我们在第3.9节讨论的主题）。关于这一点的进一步讨论，请参见[BS94, p448]。

贝叶斯方法的另一个优点在于其**构造性**，也就是说，它明确了如何根据给定的特定数据集来构建最优策略（估计量）。相比之下，频率主义方法允许你使用任何你喜欢的估计量；它只是推导出该估计量在多个数据集上的统计性质，但并未告诉你如何创建这个估计量。

[^1]: 若一个估计量不存在严格优于它的其他估计量，则称该估计量是可接受的。我们称$\delta^1$**支配**（dominates）$\delta^2$，如果对于所有$\boldsymbol{\theta}$，满足$R\left(\boldsymbol{\theta}, \delta^1\right) \leq R\left(\boldsymbol{\theta}, \delta^2\right)$。若该不等式对某个$\boldsymbol{\theta}^*$严格成立，则称这种支配是严格的。

### 34.1.5 one-shot 决策问题案例

本节将介绍一些常见的机器学习应用中的 **one-shot** 决策问题（即，作单一决策而非连续决策）。

#### 34.1.5.1 分类

假设本质状态对应于类别标签，此时 $\mathcal{H}=\mathcal{Y}=\{1, \ldots, C\}$。进一步讲，假设行动也对应于类别标签，此时 $\mathcal{A}=\mathcal{Y}$。在该设定下，最常用的损失函数为 **zero-one loss** $\ell_{01}\left(y^*, \hat{y}\right)$，定义为：

$$
\begin{array}{c|cc} 
& \hat{y}=0 & \hat{y}=1 \\
\hline y^*=0 & 0 & 1 \\
y^*=1 & 1 & 0
\end{array} \tag{34.13}
$$

更精确的写法为：

$$
\ell_{01}\left(y^*, \hat{y}\right)=\mathbb{I}\left(y^* \neq \hat{y}\right) \tag{34.14}
$$

在这种情况下，后验期望损失为

$$
\rho(\hat{y} \mid \boldsymbol{x})=p\left(\hat{y} \neq y^* \mid \boldsymbol{x}\right)=1-p\left(y^*=\hat{y} \mid \boldsymbol{x}\right) \tag{34.15}
$$

所以最小化期望损失的行动即选择最有可能的标签：

$$
\delta(\boldsymbol{x})=\underset{y \in \mathcal{Y}}{\operatorname{argmax}} p(y \mid \boldsymbol{x}) \tag{34.16}
$$

上式对应后验分布的**众数**（mode），又被称为 **maximum a posteriori** 或者 **MAP estimate**。

我们可以对损失函数进行泛化，为假阳性（误报）和假阴性（漏报）分配不同的代价。此外，还可以引入“**拒绝决策**”（reject action）机制，即当决策者信心不足时选择暂不进行分类。这种方法被称为**选择性预测**（selective prediction），详见第19.3.3节。

#### 34.1.5.2 回归

现在假设本质状态是标量 $h\in \mathbb{R}$，对应的行动也是标量 $y\in\mathbb{R}$。对于连续状态和行动的最常见损失是 $\ell_2$ 损失，又被称为 **平方误差**（squared error）或者 **二次损失**（quadratic loss），定义为：

$$
\ell_2(h, y)=(h-y)^2 \tag{34.17}
$$

在这种情况下，风险定义为

$$
\rho(y \mid \boldsymbol{x})=\mathbb{E}\left[(h-y)^2 \mid \boldsymbol{x}\right]=\mathbb{E}\left[h^2 \mid \boldsymbol{x}\right]-2 y \mathbb{E}[h \mid \boldsymbol{x}]+y^2 \tag{34.18}
$$

最优行动必须满足风险的导数（在最优点）为0（参考第6章）。所以最优行动是选择后验期望：

$$
\frac{\partial}{\partial y} \rho(y \mid \boldsymbol{x})=-2 \mathbb{E}[h \mid \boldsymbol{x}]+2 y=0 \Rightarrow \delta(\boldsymbol{x})=\mathbb{E}[h \mid \boldsymbol{x}]=\int h p(h \mid \boldsymbol{x}) d h \tag{34.19}
$$

上式被称为**最小均方差**（minimum mean squared error，MMSE）估计。

#### 34.1.5.3 参数估计

假设自然状态对应于未知参数，即 $\mathcal{H} = \Theta = \mathbb{R}^D$。进一步，假设行动也对应于参数，即 $\mathcal{A} = \Theta$。最后，我们假设观测数据（作为策略/估计量的输入）是一个数据集，例如 $\mathcal{D} = \{(\boldsymbol{x}_n, \boldsymbol{y}_n): n = 1: N\}$。若采用二次损失，则最优行动是选择后验均值；若采用0-1损失，则最优行动是选择后验众数，即最大后验概率估计：

$$
\delta(\mathcal{D}) = \hat{\boldsymbol{\theta}} = \operatorname*{argmax}_{\boldsymbol{\theta} \in \Theta} p(\boldsymbol{\theta}|\mathcal{D}) \tag{34.20}
$$

<img src="/assets/img/figures/book2/34.2.png" alt="image-20260102115202189" style="zoom:70%;" />

#### 34.1.5.4 估计离散参数

当损失函数为 0-1 损失 $\ell(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}})=\mathbb{I}(\boldsymbol{\theta} \neq \hat{\boldsymbol{\theta}})$ 时，最大后验估计是最优估计，我们在第 34.1.5.1 节说明了这一点。然而，该损失函数不会对正确估计出 $\boldsymbol{\theta}$ 的部分分量给予任何“部分奖励”。一种替代方案是使用**汉明损失**（Hamming loss）：

$$
\ell(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}})=\sum_{d=1}^D \mathbb{I}\left(\theta_d \neq \hat{\theta}_d\right) \tag{34.21}
$$

在这种情况下，可以证明最优估计量是**最大边缘概率**（max marginals）构成的向量：

$$
\hat{\boldsymbol{\theta}}=\left[\underset{\theta_d}{\operatorname{argmax}} \int_{\boldsymbol{\theta}_{-d}} p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta}_{-d}\right]_{d=1}^D \tag{34.22}
$$

这也称为**后验边缘最大化**（maximizer of posterior marginals，MPM）估计。注意，计算最大边缘概率涉及到边缘化与最大化，因此取决于整个分布；这通常比最大后验估计更具稳健性 [MMP87]。

例如，考虑一个估计一组二元向量的问题。图34.2展示了一个定义在 $\{0,1\}^{3}$ 上的分布，其中点按照它们之间的汉明距离排列，并与最近的邻居相连。标记为 $L$（对应配置 $(1,1,1)$） 的黑色状态（圆圈）具有概率 $p_{1}$，它对应着MAP估计。四个灰色状态具有概率 $p_{2} < p_{1}$；三个白色状态的概率为0。虽然黑色状态是最可能的，但它对于后验分布而言并不典型：它所有最近邻居的概率都是零，这意味着它非常孤立。相比之下，灰色状态虽然概率略低，但它们彼此相连，并且它们共同构成了总概率质量的绝大部分。

在图 34.2 的例子中，对于 $j=1:3$，有 $p(\theta_{j}=0)=3p_{2}$ 和 $p(\theta_{j}=1)=p_{2}+p_{1}$。如果 $2p_{2}>p_{1}$，则最大边缘概率构成的向量为 $(0,0,0)$。可以证明，这一 MPM 估计是一种**质心估计量**（centroid estimator），其意义在于它最小化了到后验均值（质心）的平方距离，同时（通常）能表示一个有效的配置，这与实际的后验均值不同（对于离散问题，分数形式的估计没有意义）。关于这一点的进一步讨论，请参阅 [CLO7]。

<img src="/assets/img/figures/book2/34.3.png" alt="image-20260102115318734" style="zoom:70%;" />

#### 34.1.5.5 结构化预测

在某些场景下，如自然语言处理或计算机视觉，预期的动作是返回一个输出对象 $y \in \mathcal{Y}$（例如一组标签或人体姿态）。该输出对象不仅需要在给定输入 $\boldsymbol{x}$ 时是可能的，而且输出内部本身也需要保持一致。例如，假设 $\boldsymbol{x}$ 是一个音素序列，而 $\boldsymbol{y}$ 是一个单词序列。尽管在逐词的基础上，$\boldsymbol{x}$ 可能听起来更像 $\boldsymbol{y}$ = "How to wreck a nice beach"，但如果我们考虑单词序列的整体性，那么我们可能会发现（在语言模型先验下）$\boldsymbol{y}$ = "How to recognize speech" 总体上更可能成立。（参考图 34.3。）我们可以使用**结构化预测模型**（structured prediction model），例如**条件随机场**（参见第 4.4 节），来捕捉这种在给定输入情况下，输出之间的依赖关系。

除了对 $p(\boldsymbol{y} \mid \boldsymbol{x})$ 的输出之间的依赖关系进行建模，我们可能更偏好某些特定的 $\hat{\boldsymbol{y}}$，这一点可以通过损失函数 $\ell(\boldsymbol{y}, \hat{\boldsymbol{y}})$ 进行实现。例如，如图 34.3所示，我们可能不愿意轻易地假设用户在步骤 $t$ 说的是 $\hat{y}_t$ = "nudist"，除非我们对该预测非常有信心，因为错误分类这个词的代价可能高于其他单词。

给定一个损失函数，我们可以使用**最小化贝叶斯风险**（minimum Bayes risk, MBR）来选择最优行动：

$$
\hat{\boldsymbol{y}}=\min _{\hat{\boldsymbol{y}} \in \mathcal{Y}} \sum_{\boldsymbol{y} \in \mathcal{Y}} p(\boldsymbol{y} \mid \boldsymbol{x}) \ell(\boldsymbol{y}, \hat{\boldsymbol{y}}) \tag{34.23}
$$

我们可以通过从后验预测分布中采样 $M$ 个解 $\boldsymbol{y}^m \sim p(\boldsymbol{y} \mid \boldsymbol{x})$ 来经验性地近似该期望值。（理想情况下，这些样本彼此之间是多样的。）我们使用同一组 $M$ 个样本来近似最小化，从而得到：

$$
\hat{\boldsymbol{y}} \approx \min _{\boldsymbol{y}^i, i \in\{1, \ldots, M\}} \sum_{j \in\{1, \ldots, M\}} p\left(\boldsymbol{y}^j \mid \boldsymbol{x}\right) \ell\left(\boldsymbol{y}^j, \boldsymbol{y}^i\right) \tag{34.24}
$$

上式被称为 **经验MBR**（empirical MBR）[Pre+17a]，被应用在计算机视觉问题中。一个类似的方法被应用在 [Fre+22]，被应用在神经机器翻译。

```text
MBR的直观解释 不要只选“自己最自信”的那个答案，而要选一个“让大家都觉得还行”的答案。
```

#### 34.1.5.6 公正性

使用最大似然估计（ML）训练的模型正越来越多地用于高风险应用，例如决定某人是否应从监狱释放等。在此类应用中，我们不仅要关注准确性，还必须关注**公平性**（fairness），这一点至关重要。关于公平性的含义，学界已提出了多种定义（例如，参见 [VR18]），其中许多定义所追求的目标是相互冲突的 [Kle18]。下面我们提及几种常见的定义，所有这些定义都可以从决策理论的角度来解读。

考虑一个二分类问题，真实标签为 $Y$，预测标签为 $\hat{Y}$，并包含一个**敏感属性**（sensitive attribute） $S$（例如性别或种族）。**机会均等**（equal opportunity） 的概念要求在不同子组间具有相等的**真正例率**，即 $p(\hat{Y}=1 \mid Y=1, S=0)=p(\hat{Y}=1 \mid Y=1, S=1)$。**几率均等**（equalodds） 要求在不同子组间不仅具有相等的真正例率，还要具有相等的**假正例率**，即 $p(\hat{Y}=1 \mid Y=0, S=0)=p(\hat{Y}=1 \mid Y=0, S=1)$。**统计平价**（statistical parity） 要求积极的预测结果不受受保护属性值的影响（无论真实标签如何），即 $p(\hat{Y}=1 \mid S=0)=p(\hat{Y}=1 \mid S=1)$。

更多细节参考 [KR19]。

![image-20251209002031821](/assets/img/figures/book2/34.4.png)

## 34.2 决策（影响）图表

在处理结构化的多阶段决策问题时，使用一种称为**影响图**（influence diagram）[HM81; KM08]（也称为**决策图**（decision diagram））的图形表示法非常有用。它通过添加**决策节点**（decision nodes）（也称为行动节点（action nodes），用矩形表示）和**效用节点**（utility nodes）（也称为价值节点（value nodes），用菱形表示）来扩展有向概率图模型（第四章）。原有的随机变量被称为**机会节点**（chance nodes），并像往常一样用椭圆形表示。

### 34.2.1 案例：石油勘探者

（以 [Rai68] 中的例子为例）考虑为一位石油"勘探者"所面临的决策问题建立一个模型。勘探者是指钻探"野猫井"（wildcatter）的人，所谓"野猫井"是指在尚未确认为油田的区域钻探的勘探井。

假设你必须决定是否在某个给定地点钻探一口油井。你有两种可能的行动：$d=1$ 表示钻探，$d=0$ 表示不钻探。假设自然界存在 3 种状态：$o=0$ 表示该井是干井（无油），$o=1$ 表示是湿井（有一些油），$o=2$ 表示是饱和井（有大量油）。我们可以用决策图来表示这个问题，如图 34.4(a) 所示。

现在假设先验分布为 $p(o)=[0.5,0.3,0.2]$，效用函数 $U(d,o)$ 定义为：

|       | $o=0$ | $o=1$ | $o=2$ |
| :---: | :---: | :---: | :---: |
| $d=0$ |   0   |   0   |   0   |
| $d=1$ |  -70  |  50   |  200  |

我们发现，如果你不钻探，你将没有损失，但也不会赚钱。如果你钻了一口枯井，将损失70元；如果钻的是湿井，将获得50元；如果钻的是饱和井，将获得200元。

在先验知识以外，如果没有任何信息，你会采取什么行为呢？你采取行为 $d$ 的先验期望效用为

$$
\mathrm{EU}(d)=\sum_{o=0}^2 p(o) U(d, o) \tag{34.25}
$$

我们发现 $\textrm{EU}(d=0)=0$ 和 $\textrm{EU}(d=1)=20$，所以最大期望效用是

$$
\mathrm{MEU}=\max \{\mathrm{EU}(d=0), \mathrm{EU}(d=1)\}=\max \{0,20\}=20 \tag{34.26}
$$

所以最优的行为是 钻井，即 $d^*=1$。

<img src="/assets/img/figures/book2/34.5.png" alt="image-20251209222224424" style="zoom:67%;" />

### 34.2.2 信息弧

现在让我们考虑对模型进行一个简单的扩展：假设我们可以获得一种测量信号（称为"声响"），它是关于油井状态的一个含噪声的提示。因此，我们在模型中增加了一条从 $O$ 指向 $S$ 的弧线。此外，假设在我们决定是否钻探之前，声响测试的结果是可用的；因此，我们增加了一条从 $S$ 指向 $D$ 的**信息弧**（information arc）。如图 34.4(b) 所示。请注意，效用取决于行动和世界的真实状态，但不取决于测量信号。

我们假设声响变量可以处于以下三种状态之一：$s=0$ 表示漫反射模式，暗示无油；$s=1$ 表示开放反射模式，暗示有一些油；$s=2$ 表示闭合反射模式，表明有大量石油。由于 $S$ 是由  $O$ 引起的，我们在模型中添加了一条从 $O$ 指向 $S$ 的弧。让我们使用 $p(S\mid O)$ 条件分布来对我们传感器的可靠性进行建模：

|       | $s=0$ | $s=1$ | $s=2$ |
| :---: | :---: | :---: | :---: |
| $o=0$ |  0.6  |  0.3  |  0.1  |
| $o=1$ |  0.3  |  0.4  |  0.3  |
| $o=2$ |  0.1  |  0.4  |  0.5  |

假设声响观测为 $s$。执行行为 $d$ 的后验期望效用为

$$
\mathrm{EU}(d \mid s)=\sum_{o=0}^2 p(o \mid s) U(o, d) \tag{34.27}
$$

针对上式，我们需要考虑每一种观测结果，$s \in\{0,1,2\}$，以及每一种行为，$d \in\{0,1\}$。如果 $s=0$，我们发现油井状态的后验分布为 $p(o \mid s=0)=[0.732,0.219,0.049]$，所以 $\operatorname{EU}(d=0 \mid s=0)=0$，$\mathrm{EU}(d=1 \mid s=0)=-30.5$。如果 $s=1$，我们发现 $\mathrm{EU}(d=0 \mid s=1)=0$，$\mathrm{EU}(d=1 \mid s=1)=32.9$。如果 $s=2$，我们有 $\mathrm{EU}(d=0 \mid s=2)=0$，$\operatorname{EU}(d=1 \mid s=2)=87.5$。所以最优的策略 $d^*(s)$ 为：如果 $s=0$，选择 $d=0$ 将得到 $0$元；如果 $s=1$，选择 $d=1$ 将得到 $32.9$ 元；如果 $s=2$，选择 $d=1$ 将得到 $87.5$ 元。

所以，在观察到试验性探查结果前，野猫井的最大期望效用为

$$
\operatorname{MEU}=\sum_s p(s) \mathrm{EU}\left(d^*(s) \mid s\right) \tag{34.28}
$$

其中先验的边际分布为 $p(s)=\sum_o p(o) p(s \mid o)=[0.41,0.35,0.24]$。所以MEU为

$$
\mathrm{MEU}=0.41 \times 0+0.35 \times 32.9+0.24 \times 87.5=32.2 \tag{34.29}
$$

如图34.5，将相关数值展示在了决策树的相关节点上。

### 34.2.3 信息价值

现在假设你可以选择是否进行这项测试，如图 34.4(c) 所示，我们在决策图中新增一个测试节点 $T$。如果 $T=1$，表示进行测试，并且 $S$ 可以进入 $\{0, 1, 2\}$ 的状态，由 $O$ 决定，与上述情况完全相同。如果 $T=0$，表示我们不进行测试，$S$ 则进入一个特殊的未知状态。同时，进行测试也会产生一定的成本。

进行测试是否值得？这取决于如果我们知道测试结果（即 $S$ 的状态），**最大期望效用** 会改变多少。如果不进行测试，根据公式 (34.26)，有 MEU = 20。如果进行测试，根据公式 (34.29)，有 MEU = 32.2。因此，如果你进行测试（并根据其结果采取最优行动），所带来的效用提升是 **12.2**。这被称为**完全信息价值**（value of perfect information，VPI）。因此，只要测试的成本低于 **12.2**，我们就应该进行测试。

在图模型的术语中，一个变量 $S$ 的 VPI 可以通过计算图 34.4(b) 中基础影响图 $G$ 的 MEU，然后计算在同一个影响图中添加了从 $S$ 到行动节点的信息弧后的 MEU，最后计算两者之差来确定。换句话说，

$$
\mathrm{VPI}=\operatorname{MEU}(\mathcal{G}+S \rightarrow D)-\operatorname{MEU}(\mathcal{G}) \tag{34.30}
$$

其中 $D$ 表示决策节点， $S$ 表示考察的变量。上式将告诉我们是否值得获取测量值 $S$。

### 34.2.4 计算最优策略

通常，给定一个影响图，我们可以通过修改变量消除算法（第9.5节）来自动计算最优策略，正如 [LN01; KM08] 中所解释的那样。其基本思想是从最终的行动开始向后倒推，假设所有后续行动都已是最优选择，来计算每一步的最优决策。当影响图具有简单的链式结构时，例如在马尔可夫决策过程（第34.5节）中，其结果等价于贝尔曼方程（第34.5.5节）。

## 34.3 A/B测试

假设你试图决定哪个版本的产品销量可能更好，或者哪个版本的药物疗效可能更佳。我们将可供选择的版本称为 $A$ 和 $B$；有时版本 $A$ 被称为**对照组**（control），版本 $B$ 被称为**处理组**（treatment）。（不同的选择也被称为“**选项**”(arms)。）

处理此类问题的一个常见方法是使用 **A/B 测试**。在这种方法中，你通过将不同行动（$A$或$B$）随机分配给总体样本中的不同子集，在一段时间内同时尝试这两种行动，然后测量每个行动所带来的累积**奖励**（reward），并最终选出优胜者。（这种方法有时被称为“**测试并推广**”(test and roll)，因为你首先测试出哪种行动最好，然后将其推广到剩余总体中。）

A/B 测试的关键问题之一是，在获得可能带有噪声的测试结果后，需要制定一个决策规则或策略来决定哪个行动是最优的。另一个问题是决定分配到处理组和对照组的测试人数 $n_1$和 $n_0$。背后的权衡在于：使用更大的 $n_1$ 和 $n_0$ 有助于收集更多数据，从而更有信心地选择最佳行动，但这会产生**机会成本**（opportunity cost），因为测试阶段执行的可能不是回报最高的行动。（这是**探索-利用权衡**的一个例子，我们将在第 34.4.3 节进一步讨论。）本节将沿用 [FB19] ，基于贝叶斯决策理论对该问题进行分析。关于 A/B 测试的更多细节可以在 [KTX20] 中找到。

### 34.3.1 贝叶斯方法

假设动作 $j$ 的第 $i$ 次奖励满足分布 $Y_{ij} \sim \mathcal{N}(\mu_j, \sigma_j^2)$ ，其中 $i = 1 : n_j$ 且 $j = 0 : 1$。 $j = 0$ 代表对照组（动作 $A$），$j = 1$ 代表处理组（动作 $B$），$n_j$ 表示从组 $j$ 收集的样本数量。参数 $\mu_j$ 表示动作 $j$ 的期望奖励；我们的目标是估计这些参数。（为简化起见，假设 $\sigma_j^2$ 已知。）

我们将采用一种特别适用于序列决策问题的**贝叶斯方法**。为简化，假设未知参数满足高斯先验 $\mu_j \sim \mathcal{N}(m_j, \tau_j^2)$，其中 $m_j$ 表示动作 $j$ 的先验期望奖励，$\tau_j$ 表示先验的置信度。我们假设先验参数已知。（实际上，我们可以使用经验贝叶斯方法，如第 34.3.2 节所述。）

#### 34.3.1.1 最优策略

首先假设实验的样本量（处理组的 $n_1$ 和对照组的 $n_0$ ）已知。我们的目标是计算最优策略或决策规则 $\pi(\boldsymbol{y}_1, \boldsymbol{y}_0)$，用于指定应部署哪个动作，其中 $\boldsymbol{y}_j = (y_{1j}, \ldots, y_{n_j, j})$ 表示来自动作 $j$ 的数据。

最优策略很简单：选择具有更大后验期望奖励的动作：

$$
\pi^*(\boldsymbol{y}_1, \boldsymbol{y}_0) =
\begin{cases}
1 & \text{如果 } \mathbb{E}[\mu_1 | \boldsymbol{y}_1] \geq \mathbb{E}[\mu_0 | \boldsymbol{y}_0] \\
0 & \text{如果 } \mathbb{E}[\mu_1 | \boldsymbol{y}_1] < \mathbb{E}[\mu_0 | \boldsymbol{y}_0]
\end{cases} \tag{34.31}
$$

剩下的就是计算未知参数 $\mu_j$ 的后验分布。应用高斯分布的贝叶斯法则（公式 (2.121)），对应的后验分布由下式给出：

$$
\begin{align}
p\left(\mu_j \mid \boldsymbol{y}_j, n_j\right) & =\mathcal{N}\left(\mu_j \mid \hat{m}_j, \widehat{\tau}_j^2\right) \tag{34.32} \\
1 / \hat{\tau}_j^2 & =n_j / \sigma_j^2+1 / \tau_j^2 \tag{34.33}\\
\hat{m}_j / \hat{\tau}_j^2 & =n_j \bar{y}_j / \sigma_j^2+m_j / \tau_j^2 \tag{34.34}
\end{align}
$$

不难发现，后验精度（方差的倒数）是先验精度与 $n_j$ 倍测量精度的加权和。后验精度的加权均值是先验精度的加权均值与测量精度的加权均值之和。

给定后验分布，我们可以将 $\hat{m}_j$ 代入公式 (34.31)。在完全对称的情况下（即 $n_1 = n_0$，$m_1 = m_0 = m$，$\tau_1 = \tau_0 = \tau$，以及 $\sigma_1 = \sigma_0 = \sigma$），我们发现最优的策略就是简单地“选择赢家”，即具有更高经验表现的选项：

$$
\pi^*\left(\boldsymbol{y}_1, \boldsymbol{y}_0\right)=\mathbb{I}\left(\frac{m}{\tau^2}+\frac{\bar{y}_1}{\sigma^2}>\frac{m}{\tau^2}+\frac{\bar{y}_0}{\sigma^2}\right)=\mathbb{I}\left(\bar{y}_1>\bar{y}_0\right) \tag{34.35}
$$

然而，当问题不对称时，我们需要考虑不同的样本量和/或不同的先验信念。

#### 34.3.1.2 最优样本数量

现在讨论计算每个选项的最优测试样本量，即 $n_0$ 和 $n_1$。假设总群体大小为 $N$，并且不能重复使用测试阶段的人员。

测试阶段的先验期望奖励为：

$$
\mathbb{E} [R_{\text{test}}] = n_0 m_0 + n_1 m_1 \tag{34.36}
$$

推广阶段的期望奖励取决于我们使用的决策规则 $\pi\left(\boldsymbol{y}_1, \boldsymbol{y}_0\right)$：

$$
\begin{align}
\mathbb{E}_\pi\left[R_{\text {roll }}\right] & =\int_{\mu_1} \int_{\mu_0} \int_{\boldsymbol{y}_1} \int_{\boldsymbol{y}_0}\left(N-n_1-n_0\right)\left(\pi\left(\boldsymbol{y}_1, \boldsymbol{y}_0\right) \mu_1+\left(1-\pi\left(\boldsymbol{y}_1, \boldsymbol{y}_0\right)\right) \mu_0\right) \tag{34.37}\\
& \times p\left(\boldsymbol{y}_0 \mid \mu_0\right) p\left(\boldsymbol{y}_1 \mid \mu_1\right) p\left(\mu_0\right) p\left(\mu_1\right) d \boldsymbol{y}_0 d \boldsymbol{y}_1 d \mu_0 d \mu_1 \tag{34.38}
\end{align}
$$

对于 $\pi = \pi^*$，可以证明上式等于：

$$
\mathbb{E} [R_{\text{roll}}] \triangleq \mathbb{E}_{\pi^*} [R_{\text{roll}}] = (N - n_1 - n_0) \left( m_1 + e \Phi \left( \frac{e}{v} \right) + v \phi \left( \frac{e}{v} \right) \right) \tag{34.39}
$$

其中 $\phi$ 表示高斯概率密度函数，$\Phi$ 表示高斯累积分布函数，$e = m_0 - m_1$，且

$$
v = \sqrt{\frac{\tau_1^4}{\tau_1^2 + \sigma_1^2 / n_1} + \frac{\tau_0^4}{\tau_0^2 + \sigma_0^2 / n_0}} \tag{34.40}
$$

在完全对称的情况下，公式 (34.39) 简化为：

$$
\mathbb{E}\left[R_{\text {roll }}\right]=\underbrace{(N-2 n) m}_{R_a}+\underbrace{(N-2 n) \frac{\sqrt{2} \tau^2}{\sqrt{\pi} \sqrt{2 \tau^2+\frac{2}{n} \sigma^2}}}_{R_b} \tag{34.41}
$$


上式有一种直观的解释。第一项 $R_a$ 表示在了解各个行动任何信息之前预期获得的先验收益。第二项$R_b$ 则是通过选择最优行动部署所能获得的收益。

让我们将 $R_b$ 写成 $R_b = (N - 2n)R_i$，其中 $R_i$ 表示增量收益。可以看到增量收益随着 $n$ 增加而增加，因为测试样本量越大，越可能选择正确的动作；然而，这一增量收益只能在更少的人身上获得，正如 $N - 2n$ 这个前置因子所示。（这是探索-利用权衡的结果。）

总期望奖励由公式 (34.36) 和公式 (34.41) 相加：

$$
\mathbb{E}[R] = \mathbb{E}[R_{\text{test}}] + \mathbb{E}[R_{\text{roll}}] = Nm + (N - 2n)\left(\frac{\sqrt{2}\tau^2}{\sqrt{\pi}\sqrt{2\tau^2 + \frac{2}{n}\sigma^2}}\right) \tag{34.42}
$$

（非对称情况下的方程见 [FB19]。）

通过最大化公式 (34.42) 的期望奖励，以找到测试阶段的最优样本量。根据对称性，最优解满足 $n_1^* = n_2^* = n^*$，并且由 $\frac{d}{dn} \mathbb{E}[R] = 0$ 推断 $n^*$ 满足：

$$
n^* = \sqrt{\frac{N}{4}u^2 + \left(\frac{3}{4}u^2\right)^2 - \frac{3}{4}u^2} \leq \sqrt{N}\frac{\sigma}{2\tau} \tag{34.43}
$$

其中 $u^2 = \frac{\sigma^2}{\tau^2}$。由此可见，最优样本量 $n^*$ 随着观测噪声 $\sigma$ 的增加而增加，因为需要收集更多数据才能对正确决策有信心。然而，最优样本量随着 $\tau$ 的增加而减少，因为先验信念认为效应大小（effect size） $\delta = \mu_1 - \mu_0$ 可能很大，这意味着需要更少的数据就能得出可靠的结论。

#### 34.3.1.3 遗憾

给定一个策略，很自然会想知道它有多好。我们将策略的**遗憾（regret）**定义为在掌握真实最佳行动的**完全信息**（perfect information，PI）下所能获得的期望奖励，与采用我们策略所获得的期望奖励之间的差值。最小化遗憾等价于使我们策略的期望奖励尽可能接近最佳可能的奖励（该奖励可能高或低，取决于具体问题）。

一个拥有完全信息（知晓哪个 $\mu_j$ 更大）的先知会选择得分最高的行动，从而获得期望奖励 $N\mathbb{E}[\max(\mu_1, \mu_2)]$。由于我们假设 $\mu_j \sim \mathcal{N}(m, \tau^2)$，可以得到：

$$
\mathbb{E}[R|PI] = N\left(m + \frac{\tau}{\sqrt{\pi}}\right) \tag{34.44}
$$

因此，最优策略的遗憾为：

$$
\mathbb{E}[R|PI] - (\mathbb{E}[R_{\text{test}}|\pi^*] + \mathbb{E}[R_{\text{roll}}|\pi^*]) = N\frac{\tau}{\sqrt{\pi}} \left( 1 - \frac{\tau}{\sqrt{\tau^2 + \frac{\sigma^2}{n^*}}} \right) + \frac{2n^*\tau^2}{\sqrt{\pi}\sqrt{\tau^2 + \frac{\sigma^2}{n^*}}} \tag{34.45}
$$

可以证明，遗憾是 $O(\sqrt{N})$ 阶的，当使用时间范围（总体大小）为 $N$ 时，这对于该问题是最优的 [AG13]。。

---

**公式 （34.44）的推理过程**：

关键是计算 $\mathbb{E}[\max(\mu_1, \mu_2)]$。推到过程是：令

$$
X_1=\frac{\mu_1-m}{\tau}, \quad X_2=\frac{\mu_2-m}{\tau} \tag{a}
$$

则 $X_1, X_2 \stackrel{\text { i.i.d. }}{\sim} \mathcal{N}(0,1)$ 是两个独立的标准正态分布。

已知 $\max \left(X_1, X_2\right)=\frac{X_1+X_2}{2}+\frac{\left|X_1-X_2\right|}{2}$，设 $Z=X_1-X_2$，则 $Z \sim \mathcal{N}(0,2)$，我们有：$\mathbb{E}\left[X_1+X_2\right]=0$，$\mathbb{E}[|Z|]=\sqrt{\frac{2}{\pi}} \cdot \sqrt{2}=\frac{2}{\sqrt{\pi}}$。因此

$$
\mathbb{E}\left[\max \left(X_1, X_2\right)\right]=0+\frac{1}{2} \cdot \frac{2}{\sqrt{\pi}}=\frac{1}{\sqrt{\pi}} \tag{b}
$$

接下来变换回原尺度，由于 $\mu_i=m+\tau X_i$，所以$\max \left(\mu_1, \mu_2\right)=m+\tau \cdot \max \left(X_1, X_2\right)$。两边取期望：

$$
\mathbb{E}\left[\max \left(\mu_1, \mu_2\right)\right]=m+\tau \cdot \mathbb{E}\left[\max \left(X_1, X_2\right)\right]=m+\tau \cdot \frac{1}{\sqrt{\pi}} \tag{c}
$$

最终的结果为：

$$
\mathbb{E}[R \mid P I]=N \cdot \mathbb{E}\left[\max \left(\mu_1, \mu_2\right)\right]=N\left(m+\frac{\tau}{\sqrt{\pi}}\right) \tag{d}
$$

---

#### 34.3.1.4 期望错误率

有时目标被设定为**最佳臂识别**（best arm identification），即确定 $\mu_1 > \mu_0$是否成立。也就是说，若定义 $\delta = \mu_1 - \mu_0$，我们想判断 $\delta > 0$ 还是 $\delta < 0$。这很自然地可表述为一个**假设检验**（hypothesis test）问题。然而，这或许是一个错误的目标，因为如果 $\delta$ 很小，通常不值得耗费大量成本去收集足够样本以确信 $\delta > 0$。相反，使用第34.3.1.1节的方法优化总期望收益更有意义。

尽管如此，我们可能想知道，如果使用第34.3.1.1节中的策略，我们选错臂的概率是多少。在对称情况下，由下式给出：

$$
\Pr(\pi(y_1, y_0) = 1 | \mu_1 < \mu_0) = \Pr(Y_1 - Y_0 > 0 | \mu_1 < \mu_0) = 1 - \Phi \left( \frac{\mu_1 - \mu_0}{\sigma \sqrt{\frac{1}{n_1} + \frac{1}{n_0}}} \right) \tag{34.46}
$$

上式假设 $\mu_j$ 已知。由于它们实际上未知，我们可以使用 $\mathbb{E}[\Pr(\pi(y_1, y_0) = 1 | \mu_1 < \mu_0)]$ 来计算期望错误率。根据对称性，量 $\mathbb{E}[\Pr(\pi(y_1, y_0) = 0 | \mu_1 > \mu_0)]$ 相同。可以证明，这两个量由下式给出：

$$
\text{错误概率} = \frac{1}{4} - \frac{1}{2\pi} \arctan \left( \frac{\sqrt{2\pi}}{\sigma} \sqrt{\frac{n_1 n_0}{n_1 + n_0}} \right) \tag{34.47}
$$

正如预期的那样，错误率随着样本量 $n_1$ 和 $n_0$ 的增加而降低，随着观测噪声 $\sigma$ 的增加而增加，并随着效应大小方差 $\tau$ 的增加而降低。因此，最小化分类错误的策略也将最大化期望收益，但它可能会选择过大的样本量，因为它没有考虑 $\delta$ 的大小。

![image-20251213114348517](/assets/img/figures/book2/34.6.png)

### 34.3.2 示例

本节介绍满足上述框架的一个简单示例。目标是进行**网站测试**（website testing）——比较两个不同版本网页的**点击率**（click through rate）。此时的观测数据是一个二元随机变量 $y_{ij} \sim \text{Ber}(\mu_j)$，因此自然地选择 Beta分布作为先验 $\mu_j \sim \text{Beta}(\alpha, \beta)$（见第 3.4.1 节）。然而，在这种情况下，最优样本量和决策规则更难计算（详情见 [FB19; Sta+17]）。作为一个简单的近似，可以假设 $\overline{y}_{ij} \sim \mathcal{N}(\mu_j, \sigma^2)$，其中 $\mu_j \sim \mathcal{N}(m, \tau^2)$，$m = \frac{\alpha}{\alpha + \beta}$，$\tau^2 = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$，且 $\sigma^2 = m(1 - m)$。

为了设置高斯先验，[FB19] 使用了约 2000 个先前 A/B 测试的经验数据。对于每个测试，他们观察了每个版本页面被展示的次数，以及用户点击每个版本的总次数。基于这些数据，他们使用分层贝叶斯模型推断出 $\mu_j \sim \mathcal{N}(m = 0.68, \tau = 0.03)$。这个先验意味着期望效应大小相当小：$\mathbb{E}[|\mu_1 - \mu_0|] = 0.023$，（这与 [Aze+20] 的结果一致，他们发现对 Microsoft Bing EXP 平台的大多数更改效果可以忽略不计，尽管偶尔会有一些“大成功”。）

在这个先验下，并假设总体 $N = 100,000$，公式 (34.43) 表明最优试验次数为 $n_1^* = n_0^* = 2284$。测试阶段的期望奖励（点击次数或**转化次数**）为：$\mathbb{E}[R_{\text{test}}] = 3106$，而部署阶段的期望奖励为：$\mathbb{E}[R_{\text{roll}}] = 66,430$，总奖励为 $69,536$。期望错误率为 $10\%$。

在图 34.6a 中，我们绘制了期望奖励随测试阶段规模 $n$ 变化的曲线。我们看到，奖励随着 $n$ 急剧增加到在 $n^* = 2284$ 处达到全局最大值，然后更缓慢地下降。这表明，进行略微过大的测试比进行同等程度的过小测试要好。（然而，当使用重尾模型时，[Aze+20] 发现进行许多更小的测试更好。）

在图 34.6b 中，我们绘制了选取错误动作的概率随 $n$ 变化的曲线。我们看到，大于最优规模的测试只能略微降低这个错误率。因此，如果你希望使误分类率很低，你可能需要一个很大的样本量，特别是当 $\mu_1 - \mu_0$ 很小时，因为那时将很难检测出真正的最佳动作。然而，在这种情况下，识别最佳动作的重要性也较低，因为两个动作的期望奖励非常相似。这就解释了为什么基于频率统计的 A/B 测试经典方法（使用假设检验方法来确定 A 是否优于 B）通常会推荐比必要大得多的样本量。（进一步讨论请参见 [FB19] 及其参考文献。）

## 34.4 上下文赌博机

在第 34.3 节中，我们讨论了 **A/B 测试**。在这种方法中，决策者以固定的次数（分别为 $n_1$ 和 $n_0$）尝试两种不同的行动 $a_1$ 和 $a_0$ ，测量由此产生的一系列奖励 $\boldsymbol{y}_1$ 和 $\boldsymbol{y}_0$，然后选择在剩余时间（或剩余总体中）使用最佳行动，以最大化期望奖励。

显然可以将其推广到两种以上的行动。更重要的是，我们可以将其推广到多阶段决策问题。具体来说，假设我们允许决策者尝试一个行动 $a_t$，观察奖励 $r_t$，然后在时间步 $t+1$ 就决定下一步做什么，而不是等到 $n_1+n_0$ 次实验全部完成。这种即时反馈使得**自适应策略**（adaptive policies）成为可能，这种策略可以带来更高的期望奖励（更低的遗憾）。这样，我们就把一个单次决策问题转变成了一个**序列决策问题**（sequential decision problem）。序列决策问题有很多种，但在本节中，我们考虑最简单的一种，称为**赌博机问题**（bandit problem）（参见 [LS19; Sli19]）。

### 34.4.1 赌博机类型

在**多臂赌博机**（multi-armed bandit）中，存在一个智能体（决策者），它在每一步根据某个**策略**（policy） $a_t \sim \pi_t$ 选择某个行动，并从环境中获得某个采样**奖励** $r_t \sim p_R\left(a_t\right)$，该奖励的期望为 $R(s, a)=\mathbb{E}[R \mid a]$。

想象赌场里的一个赌徒，他面对多台吃硬币的老虎机，每台机器的奖励支付率都不同。一台老虎机有时被称为"**独臂赌博机**"（one-armed bandit），一组 $K$ 台老虎机则被称为**多臂赌博机**（multi-armed bandit）；每个不同的行动对应于拉动一台不同的老虎机摇臂，即 $a_t \in\{1, \ldots, K\}$。我们的目标是尽快找出哪台机器支付的钱最多，然后一直玩那台机器，直到你变得尽可能富有。

通过定义**上下文老虎机**（contextual bandit）来扩展上述模型，其中每一步策略的输入是某个随机选择的状态或上下文 $s_t \in \mathcal{S}$，该状态按照某个过程随时间演变 $s_t \sim p\left(s_t\mid s_{1: t-1}\right)$，且演变过程与智能体的行动无关。此时的策略为 $a_t \sim \pi_t\left(a_t \mid s_t\right)$，奖励函数为 $r_t \sim p_R\left(r_t \mid s_t, a_t\right)$，奖励期望 $R(s, a)=\mathbb{E}[R \mid s, a]$。在每一步，智能体可以利用观测到的数据 $\mathcal{D}_{1: t}$（其中 $\mathcal{D}_t=\left(s_t, a_t, r_t\right)$）来更新策略，以最大化期望奖励。

在**有限时域** （finite horizon）的（上下文）赌博机中，目标是最大化期望**累积奖励**（cumulative reward）：

$$
J \triangleq \sum_{t=1}^T \mathbb{E}_{p_R\left(r_t \mid s_t, a_t\right) \pi_t\left(a_t \mid s_t\right) p\left(s_t \mid \boldsymbol{s}_{1: t-1}\right)}\left[r_t\right]=\sum_{t=1}^T \mathbb{E}\left[r_t\right] \tag{34.48}
$$

（注意，奖励在每个时间步都会累积，即使智能体正在更新其策略；这有时被称为“**边学边赚**（earning while learning）”）。在**无限时域**（infinite horizon）的设定中，即 $T=\infty$ 时，累积奖励可能是无限的。为防止 $J$ 无界，引入**折扣因子**（discount factor） $0<\gamma<1$，从而

$$
J \triangleq \sum_{t=1}^{\infty} \gamma^{t-1} \mathbb{E}\left[r_t\right] \tag{34.49}
$$

$\gamma$ 可以解释为智能体在任何时刻终止的概率（在这种情况下，它将停止累积奖励）。

另一种写法如下：

$$
J=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbb{E}\left[r_t\right]=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbb{E}\left[\sum_{a=1}^K R_a\left(s_t, a_t\right)\right] \tag{34.50}
$$

其中定义

$$
R_a\left(s_t, a_t\right)= \begin{cases}R\left(s_t, a\right) & \text { if } a_t=a \\ 0 & \text { otherwise }\end{cases} \tag{34.51}
$$

因此，我们在概念上评估所有臂的奖励，但只有实际被选择的那个（即 $a_t$）会给智能体带来非零值，即 $r_t$。

基础版赌博机问题有很多扩展。一种自然的扩展是允许智能体进行**多次选择**（multiple plays），即同时选择 $M \leq K$ 个不同的赌博机。令 $\boldsymbol{a}_t$ 为相应的动作向量，它指定了所选老虎机的编号。然后我们定义奖励为

$$
r_t=\sum_{a=1}^K R_a\left(s_t, \boldsymbol{a}_t\right) \tag{34.52}
$$

其中

$$
R_a\left(s_t, \boldsymbol{a}_t\right)= \begin{cases}R\left(s_t, a\right) & \text { if } a \in \boldsymbol{a}_t \\ 0 & \text { otherwise }\end{cases} \tag{34.53}
$$

这对于建模**资源分配**（resource allocation）问题很有用。

另一种变体被称为**不安分老虎机**（restless bandit）[Whi88]。这与多次选择设定相同，只不过我们还假设每个臂都有与之关联的自身状态向量 $s_t^a$，该状态根据某个随机过程演化，无论臂 $a$ 是否被选择。然后我们定义

$$
r_t=\sum_{a=1}^K R_a\left(s_t^a, \boldsymbol{a}_t\right) \tag{34.54}
$$

其中 $s_t^a \sim p\left(s_t^a \mid \boldsymbol{s}_{1: t-1}^a\right)$ 是某个任意分布，通常假定为马尔可夫的。（每个臂关联的状态即使在该臂未被选择时也会演化，这正是“不安分”一词的由来。）这可以用来建模每个臂所提供的奖励之间的序列依赖性。

### 34.4.2 应用

上下文老虎机模型有许多实际应用。例如，考虑一个**在线广告系统**（online advertising system）。在这种情况下，状态 $s_t$ 代表用户当前浏览网页的特征，而行动 $a_t$ 代表系统选择展示的广告。由于广告的相关性取决于网页内容，奖励函数的形式为 $R\left(s_t, a_t\right)$，因此这个问题是上下文相关的。目标是最大化期望奖励，这相当于最大化用户点击广告的期望次数；这被称为**点击率**（click through rate, CTR）。（关于此应用的更多信息，可参见 [Gra+10; Li+10; McM+13; Aga+14; Du+21; YZ22]。）

上下文老虎机的另一个应用出现在**临床试验**（clinical trials）中 [VBW15]。在这种情况下，状态 $s_t$ 是我们正在治疗的当前患者的特征，行动 $a_t$ 是医生选择给予他们的治疗（例如，一种新药或一种**安慰剂**（placebo））。我们的目标是最大化期望奖励，即被治愈的期望人数。（另一个替代目标是尽可能快地确定哪种治疗方法是最好的，而不是最大化期望奖励；这种变体被称为**最佳臂识别** （best-arm identification）[ABM10]。）

![image-20251210222932755](/assets/img/figures/book2/34.7.png)

### 34.4.3 探索-利用平衡

解决老虎机问题的根本困难被称为**探索-利用权衡**（exploration-exploitation tradeoff）。这指的是智能体需要尝试多种状态/行动组合（这被称为**探索**），以收集足够的数据来可靠地学习奖励函数 $R(s, a)$；然后，它可以通过为每个状态选择预测的最佳行动来**利用**其知识。如果智能体过早地开始利用一个不正确的策略模型，它将收集到次优的数据，并陷入一个负面的**反馈循环**（feedback loop），如图 34.7 所示。这与监督学习不同，在监督学习中数据是从一个固定分布中独立同分布地采样的（详见 [Jeu+19]）。

我们在下面讨论探索-利用问题的一些解决方案。

![image-20260102144145086](/assets/img/figures/book2/34.8.png)

### 34.4.4 最优方案

本节讨论探索-利用权衡的最优解，令奖励函数的参数满足后验分布 $\boldsymbol{b}_t=p\left(\boldsymbol{\theta} \mid \boldsymbol{h}_t\right)$，其中 $\boldsymbol{h}_t=\left\{s_{1: t-1}, a_{1: t-1}, r_{1: t-1}\right\}$ 表示观测历史；$\boldsymbol{b}_t$被称为 **信念状态**（belief state）或**信息状态**（information state），它是历史 $\boldsymbol{h}_t$ 的一个有限状态统计量。信念状态可以使用贝叶斯法则进行确定性的更新：

$$
\boldsymbol{b}_t=\operatorname{BayesRule}\left(\boldsymbol{b}_{t-1}, a_t, r_t\right) \tag{34.55}
$$

例如，考虑与上下文无关的**伯努利赌博机**（Bernoulli bandit），其中 $p_R(r|a) = \text{Ber}(r|\mu_a)$，并且 $\mu_a = p_R(r = 1|a) = R(a)$ 是采取动作 $a$ 的期望奖励。假设使用一个因式分解的 Beta 先验

$$
p_0(\boldsymbol{\theta})=\prod_a \operatorname{Beta}\left(\mu_a \mid \alpha_0^a, \beta_0^a\right) \tag{34.56}
$$

其中 $\boldsymbol{\theta}=\left(\mu_1, \ldots, \mu_K\right)$。我们可以计算后验的闭式解，正如我们在第 3.4.1 节所讨论的。具体来说，我们得到

$$
p\left(\boldsymbol{\theta} \mid \mathcal{D}_t\right)=\prod_a \operatorname{Beta}(\mu_a \mid \underbrace{\alpha_0^a+N_t^0(a)}_{\alpha_t^a}, \underbrace{\beta_0^a+N_t^1(a)}_{\beta_t^a}) \tag{34.57}
$$

其中

$$
N_t^r(a)=\sum_{s=1}^{t-1} \mathbb{I}\left(a_t=a, r_t=r\right) \tag{34.58}
$$

图 34.8 展示了一个双臂伯努利赌博机的情况。

我们可以使用类似的方法处理**高斯赌博机**（Gaussian bandit），其中 $p_R(r|a) = \mathcal{N}(r|\mu_a, \sigma^2_a)$，利用第 3.4.3 节的结果。对于上下文老虎机，问题变得更加复杂。如果我们假设是一个**线性回归赌博机**（linear regression bandit）$p_R(r \mid s, a ; \boldsymbol{\theta})=\mathcal{N}\left(r \mid \boldsymbol{\phi}(s, a)^{\top} \boldsymbol{\theta}, \sigma^2\right)$，我们可以使用贝叶斯线性回归来计算闭式解下的 $p\left(\boldsymbol{\theta} \mid \mathcal{D}_t\right)$，正如第 15.2 节所讨论的。如果我们假设是一个**逻辑回归赌博机**（logistic regression bandit）$p_R(r|s, a; \boldsymbol{\theta}) = \text{Ber}(r|\sigma(\boldsymbol{\phi}(s, a)^T\boldsymbol{\theta}))$，我们可以使用贝叶斯逻辑回归来计算 $p(\boldsymbol{\theta}|\mathcal{D}_t)$，正如第 15.3.5 节所讨论的。如果我们有一个形式为 $p_R(r|s, a; \boldsymbol{\theta}) = \text{GLM}(r|f(s, a; \boldsymbol{\theta}))$ 的**神经网络赌博机**（neural bandit），其中 $f$ 是一个非线性函数，那么后验推断会变得更具挑战性，正如我们在第 17 章讨论的。不过，标准的技术，例如扩展卡尔曼滤波器（第 17.5.2 节），仍然可以应用。（关于如何将此方法扩展到大型 DNN，参见 [DMKM22] 的“子空间神经老虎机”方法。）

无论算法细节如何，我们可以将信念状态的更新表示为：

$$
p\left(\boldsymbol{b}_t \mid \boldsymbol{b}_{t-1}, a_t, r_t\right)=\mathbb{I}\left(\boldsymbol{b}_t=\operatorname{BayesRule}\left(\boldsymbol{b}_{t-1}, a_t, r_t\right)\right) \tag{34.59}
$$

那么每一步观测到的奖励可以预测为

$$
p\left(r_t \mid \boldsymbol{b}_t\right)=\int p_R\left(r_t \mid s_t, a_t ; \boldsymbol{\theta}\right) p\left(\boldsymbol{\theta} \mid \boldsymbol{b}_t\right) d \boldsymbol{\theta} \tag{34.60}
$$

我们看到这是一种特殊形式的（受控）马尔可夫决策过程（第 34.5 节），称为**信念状态 MDP**（belief-state MDP）。

在上下文无关、臂数量有限的老虎机这一特殊情况下，这个信念状态 MDP 的最优策略可以通过动态规划计算（见第 34.6 节）；结果可以表示为每个步骤的动作概率表 $\pi_t(a_1, \ldots, a_K)$；这被称为**吉廷斯指数**（Gittins index） [Git89]。然而，为一般的上下文老虎机计算最优策略是棘手的 [PT87]，所以我们必须求助于近似方法，正如我们在下文所讨论的。

### 34.4.5 Upper confidence bounds (UCBs)

探索-利用问题的最优解决方案是难以处理的。然而，一种直观且合理的方案是遵循"**面对不确定性保持乐观**（optimism in the face of uncertainty）"的原则。该原则以贪婪的方式选择行动，但依据的是对其奖励的乐观估计。基于此原则的最重要的一类策略统称为**上置信界**（upper confidence bound，UCB）方法。

要使用UCB策略，智能体需要维护一个乐观的奖励函数估计 $\tilde{R}_t$，使得对于所有行动 $a$，$\tilde{R}_t\left(s_t, a\right) \geq R\left(s_t, a\right)$ 都以高概率成立，然后相应地选择贪婪行动：
$$
a_t=\underset{a}{\operatorname{argmax}} \tilde{R}_t\left(s_t, a\right) \tag{34.61}
$$
UCB可以被视为一种**探索奖励**（exploration bonus）的形式，其中乐观的估计鼓励了探索。通常，乐观的程度 $\tilde{R}_t-R$ 会随时间递减，因此智能体会逐渐减少探索。在构建恰当的乐观奖励估计的情况下，UCB策略已被证明能在许多赌博机问题的变体中实现近乎最优的遗憾 [LS19]。（我们将在第34.4.7节讨论遗憾。）

乐观函数 $\tilde{R}$ 可以通过不同的方式获得，有时能以闭合形式得到，如下文所述。

#### 34.4.5.1 频率学派方法

一种方法是使用**集中不等式**（concentration inequality） [BLM16] 来推导估计误差的高概率上界：$|\hat{R}_t(s, a) - R_t(s, a)| \leq \delta_t(s, a)$，其中 $\hat{R}_t$ 是 $R$ 的通常估计（一般是极大似然估计，MLE），$\delta_t$ 是适当选择的函数。然后通过设定 $\tilde{R}_t(s, a) = \hat{R}_t(s, a) + \delta_t(s, a)$得到**乐观奖励估计**。

作为示例，再次考虑上下文无关的伯努利多臂赌博机，$R(a) \sim \text{Ber}(\mu(a))$。其极大似然估计 $\hat{R}_t(a)=\hat{\mu}_t(a)$ 由执行行动 $a$ 后的观测奖励的经验均值给定：

$$
\hat{\mu}_t(a) = \frac{N_t^1(a)}{N_t(a)} = \frac{N_t^1(a)}{N_t^0(a) + N_t^1(a)} \tag{34.62}
$$

其中 $N_t^r(a)$ 表示到第 $t-1$ 步为止，尝试动作 $a$ 且观察到的奖励为 $r$ 的次数，$N_t(a)$ 表示尝试动作 $a$ 的总次数：

$$
N_t(a) = \sum_{s=1}^{t-1} \mathbb{I}(a_t = a) \tag{34.63}
$$

那么**切尔诺夫-霍夫丁不等式**（Chernoff-Hoeffding inequality） [BLM16] 可得 $\delta_t(a) = c / \sqrt{N_t(a)}$，其中 $c$ 为某个适当的常数，因此

$$
\tilde{R}_t(a) = \hat{\mu}_t(a) + \frac{c}{\sqrt{N_t(a)}} \tag{34.64}
$$

#### 34.4.5.2 贝叶斯方法

#### 34.4.5.3 案例

### 34.4.6 Thompson 采样

### 34.4.7 Regret

## 34.5 马尔可夫决策问题

在本节中，我们通过允许本质状态根据智能体所选择的行动而改变，来推广上下文赌博机问题。由此产生的模型被称为**马尔可夫决策过程**（Markov decision process,MDP），我们将在下文详细解释。该模型构成了**强化学习**（我们将在第35章讨论）的基础。

<img src="/assets/img/figures/book2/34.11.png" alt="image-20251217234821900" style="zoom:50%;" />

### 34.5.1 基础

**马尔可夫决策过程** [Put94] 可用于模拟智能体与环境的交互。它通常由一个元组 $\left\langle\mathcal{S}, \mathcal{A}, p_T, p_R, p_0\right\rangle$ 描述，其中：$\mathcal{S}$ 是环境状态的集合，$\mathcal{A}$ 是智能体可以采取的行动的集合，$p_T$ 是状态**转移模型**（transition model），$p_R$ 是**奖励模型**（reward model），$p_0$ 是初始状态分布。交互在时间 $t=0$ 开始，初始状态 $s_0 \sim p_0$。然后，在时间 $t \geq 0$，智能体观测到环境状态 $s_t \in \mathcal{S}$，并遵循一个策略 $\pi$ 来采取行动 $a_t \in \mathcal{A}$。作为响应，环境发出一个实值奖励信号 $r_t \in \mathcal{R}$ 并进入一个新状态 $s_{t+1} \in \mathcal{S}$。策略通常是随机的，$\pi(a \mid s)$ 表示在状态 $s$ 下选择行动 $a$ 的概率。如果策略是随机的，我们用 $\pi(s)$ 表示在 $\mathcal{A}$ 上的条件概率分布；如果策略是确定性的，则用它表示所选择的行动。每一步过程称为一次**状态转移**（transition）；在时间 $t$，它由元组 $\left(s_t, a_t, r_t, s_{t+1}\right)$ 构成，其中：$a_t \sim \pi\left(s_t\right)$，$s_{t+1} \sim p_T\left(s_t, a_t\right)$，$r_t \sim p_R\left(s_t, a_t, s_{t+1}\right)$。因此，在策略 $\pi$ 下，生成一条长度为 $T$ 的轨迹 $\boldsymbol{\tau}$ 的概率为：

$$
p(\boldsymbol{\tau})=p_0\left(s_0\right) \prod_{t=0}^{T-1} \pi\left(a_t \mid s_t\right) p_T\left(s_{t+1} \mid s_t, a_t\right) p_R\left(r_t \mid s_t, a_t, s_{t+1}\right) \tag{34.74}
$$

根据奖励模型 $p_R$ 定义的**奖励函数**（reward function）是很有用的，即在状态 $s$ 下采取行动 $a$ 时的平均即时奖励：

$$
R(s, a) \triangleq \mathbb{E}_{p_T\left(s^{\prime} \mid s, a\right)}\left[\mathbb{E}_{p\left(r \mid s, a, s^{\prime}\right)}[r]\right] \tag{34.75}
$$

在后续讨论中，消除对下一状态的依赖不会丧失一般性，因为我们关注的是沿着轨迹的总体（可加）期望奖励。因此，我们经常使用元组 $\left\langle\mathcal{S}, \mathcal{A}, p_T, R, p_0\right\rangle$ 来描述一个 MDP。

MDP的状态和行动集合可以是离散或连续的。当两个集合都是有限集时，我们可以将这些函数表示为查找表；这被称为**表格化表示**（tabular representation）。在这种情况下，我们可以将MDP表示为一个**有限状态机**（finite state machine），这是一种图结构，其中节点对应状态，边对应行动以及由此产生的奖励和下一状态。图34.11给出了一个具有3个状态和2个行动的MDP的简单示例。

与RL密切相关的**控制理论**（control theory）领域使用了略微不同的术语。具体来说，环境被称为**被控对象**（plant），智能体被称为**控制器**（controller）。状态用 $\boldsymbol{x}_t \in \mathcal{X} \subseteq \mathbb{R}^D$ 表示，行动用 $\boldsymbol{u}_t \in \mathcal{U} \subseteq \mathbb{R}^K$ 表示，奖励则用成本 $c_t \in \mathbb{R}$ 表示。除了这些记法上的差异，RL和控制理论这两个领域非常相似（例如，参见 [Son98; Rec19]），尽管控制理论倾向于专注于可证明最优的方法（通过做出较强的建模假设），而RL则倾向于使用启发式方法处理更困难的问题，对于这些问题通常难以获得最优性保证。

<img src="/assets/img/figures/book2/34.12.png" alt="image-20251220115003042" style="zoom:50%;" />

### 34.5.2 Partially observed MDPs

MDP框架的一个重要推广是放宽了智能体能直接看到隐藏世界状态 $s_t$ 的假设；相反，我们假设它只看到一个从隐藏状态生成的、可能带有噪声的观测值 $x_t \sim p\left(\cdot \mid s_t, a_t\right)$。由此产生的模型被称为**部分可观测马尔可夫决策过程**（partially observable Markov decision process, POMDP）。此时，智能体的策略是一个从所有可用数据到行动的映射，即 $a_t \sim \pi\left(\mathcal{D}_{1: t-1}, x_t\right), \mathcal{D}_t=\left(x_t, a_t, r_t\right)$。参见图34.12的图示。MDP是 $x_t=s_t$ 时的特例。

通常，POMDP比MDP难解决得多（例如，参见[KLC98]）。一种常见的近似方法是使用最近几个观测到的输入，例如长度为$h$的历史数据 $\boldsymbol{x}_{t-h: t}$，作为隐藏状态的代理，然后将其视为一个完全可观测的MDP来处理。

### 34.5.3 回合与回报 （Episodes and returns）

马尔可夫决策过程描述了轨迹 $\boldsymbol{\tau}=\left(s_0, a_0, r_0, s_1, a_1, r_1, \ldots\right)$ 是如何随机生成的。如果智能体可以与环境永远交互下去，我们称之为**持续型任务**（continuing task）。或者，如果交互在系统进入**终止状态**（terminal state）或**吸收状态**（absorbing state）后宣告结束，则智能体处于**分幕式任务**（episodic task）中；我们称状态 $s$ 是吸收状态，如果从 $s$ 出发的下一个状态总是 $s$ 自身且奖励为 0。进入终止状态后，我们可以从一个新的初始状态 $s_0 \sim p_0$ 开始一个新的**回合**（epsiode）。回合长度通常是随机的。例如，机器人到达目标所需的时间可能变化很大，这取决于它做出的决策以及环境中的随机性。请注意，我们可以通过重新定义吸收状态中的转移模型为初始状态分布 $p_0$ ，将分幕式 MDP 转换为持续型 MDP。最后，如果分幕式任务中的轨迹长度 $T$ 是固定且已知的，则称之为**有限时域问题**（finite horizon problem）。

设 $\boldsymbol{\tau}$ 是一条长度为 $T$ 的轨迹，如果任务是持续型的，则 $T$ 可以是 $\infty$。我们定义时间 $t$ 状态的**回报**（return）为从该时刻起获得的期望奖励之和，其中每个奖励都乘以一个**折扣因子**（discount factor） $\gamma \in[0,1]$：

$$
\begin{align}
G_t & \triangleq r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots+\gamma^{T-t-1} r_{T-1} \tag{34.76} \\
& =\sum_{k=0}^{T-t-1} \gamma^k r_{t+k}=\sum_{j=t}^{T-1} \gamma^{j-t} r_j \tag{34.77}
\end{align}
$$

$G_t$ 有时被称为"**剩余回报**"（reward-to-go）。对于在时间 $T$ 终止的分幕式任务，我们定义当 $t \geq T$ 时 $G_t=0$。显然，回报满足以下递归关系：

$$
G_t=r_t+\gamma\left(r_{t+1}+\gamma r_{t+2}+\cdots\right)=r_t+\gamma G_{t+1} \tag{34.78}
$$

折扣因子 $\gamma$ 扮演着两个角色。首先，即使 $T=\infty$（即无限时域），只要使用 $\gamma<1$ 且奖励 $r_t$ 是有界的，它就能确保回报是有限的。其次，它赋予短期奖励更大的权重，这通常会鼓励智能体更快实现其目标（示例见第34.5.5.1节）。然而，如果 $\gamma$ 太小，智能体会变得过于贪婪。在 $\gamma=0$ 的极端情况下，智能体完全是**短视的**（myopic），只试图最大化其即时奖励。通常，折扣因子反映了这样一种假设：交互在下一步终止的概率为 $1-\gamma$。对于已知 $T$ 的有限时域问题，我们可以设定 $\gamma=1$，因为我们事先知道智能体的生命周期。

### 34.5.4 价值函数

令 $\pi$ 为某个给定策略。我们定义 **状态-价值函数**（state-value function），简称**价值函数**（value function）（$\mathbb{E}_\pi[\cdot]$ 表示基于策略 $\pi$ 选择行动）：

$$
V_\pi(s) \triangleq \mathbb{E}_\pi\left[G_0 \mid s_0=s\right]=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0=s\right] \tag{34.79}
$$

这是在持续型任务（即 $T=\infty$）中，如果我们从状态 $s$ 开始并遵循策略 $\pi$ 来选择行动所能获得的期望回报。

类似地，我们定义**行动价值函数**（action-value function）（也称为 **Q 函数**）如下：

$$
Q_\pi(s, a) \triangleq \mathbb{E}_\pi\left[G_0 \mid s_0=s, a_0=a\right]=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0=s, a_0=a\right] \tag{34.80}
$$

该值表示如果我们从在状态 $s$ 采取行动 $a$ 开始，然后遵循策略 $\pi$ 选择后续行动，所能获得的期望回报。

最后，我们定义**优势函数**（advantage function）如下：

$$
A_\pi(s, a) \triangleq Q_\pi(s, a)-V_\pi(s) \tag{34.81}
$$

这告诉我们在状态 $s$ 下选择行动 $a$ 然后切换到策略 $\pi$（相对于始终遵循 $\pi$ 的基线回报）所带来的收益。请注意，$A_\pi(s, a)$ 既可以是正数也可以是负数，并且由于一个有用的等式$V_\pi(s)=\mathbb{E}_{\pi(a \mid s)}\left[Q_\pi(s, a)\right]$，我们有 $\mathbb{E}_{\pi(a \mid s)}\left[A_\pi(s, a)\right]=0$。

### 34.5.5 最优价值函数和策略

假设 $\pi_*$ 是某个策略，使得对于所有状态 $s \in \mathcal{S}$ 和所有策略 $\pi$，都有 $V_{\pi_*} \geq V_\pi$，那么它就是一个**最优策略**（optimal policy）。同一个 MDP 可能有多个最优策略，但根据定义，它们的价值函数必须相同，分别 $V_*$ 和 $Q_*$。我们称 $V_*$ 为**最优状态价值函数**（optimal state-value function），称 $Q_*$ 为**最优行动价值函数**（optimal action-value function）。此外，任何有限 MDP 都至少存在一个确定性的最优策略 [Put94]。

最优价值函数的一个基本结果是**贝尔曼最优性方程**（Bellman’s optimality equations）：

$$
\begin{align}
V_*(s) & =\max _a R(s, a)+\gamma \mathbb{E}_{p_T\left(s^{\prime} \mid s, a\right)}\left[V_*\left(s^{\prime}\right)\right]  \tag{34.82}\\
Q_*(s, a) & =R(s, a)+\gamma \mathbb{E}_{p_T\left(s^{\prime} \mid s, a\right)}\left[\max _{a^{\prime}} Q_*\left(s^{\prime}, a^{\prime}\right)\right] \tag{34.83}
\end{align}
$$

反之，最优价值函数是满足这些方程的唯一解。换句话说，尽管价值函数被定义为无限多个奖励之和的期望，但它可以通过*一个仅涉及 MDP 单步转移和奖励模型的递归方程* 来刻画。这种递归在本章后面看到的许多 RL 算法中起着核心作用。给定一个价值函数（不一定是最优的）（$V$ 或 $Q$），方程 (34.82) 和 (34.83) 左右两边之间的差异被称为**贝尔曼误差**（Bellman error）或**贝尔曼残差**（Bellman residual）。

此外，给定最优价值函数，我们可以使用以下方法推导出最优策略：

$$
\begin{align}
\pi_*(s) & =\underset{a}{\operatorname{argmax}} Q_*(s, a) \tag{34.84} \\
& =\underset{a}{\operatorname{argmax}}\left[R(s, a)+\gamma \mathbb{E}_{p_T\left(s^{\prime} \mid s, a\right)}\left[V_*\left(s^{\prime}\right)\right]\right] \tag{34.85}
\end{align}
$$

遵循这样的最优策略可以确保智能体从任何状态开始都能获得最大的期望回报。求解 $V_*$、$Q_*$ 或 $\pi_*$ 的问题被称为**策略优化**（policy optimization）。相比之下，对于给定的策略 $\pi$，求解 $V_\pi$ 或 $Q_\pi$ 被称为**策略评估**（policy evaluation），这构成了强化学习问题的一个重要子类，我们将在后续章节讨论。对于策略评估，我们有类似的贝尔曼方程，只需将方程 (34.82) 和 (34.83) 中的 $\max _a\{\cdot\}$ 替换为 $\mathbb{E}_{\pi(a \mid s)}[\cdot]$ 即可。

在方程 (34.84) 和 (34.85) 中，与贝尔曼最优方程一样，我们必须对 $\mathcal{A}$ 中的所有行动取最大值，这个使价值函数 $Q_*$ 或  $V_*$ 最大化的行动被称为**贪婪行动**（greedy action）。如果 $\mathcal{A}$ 是一个小的有限集，那么寻找贪婪行动在计算上是容易的。对于高维连续空间，我们可以将 $a$ 视为一个行动序列，且一次优化一个维度 [Met+17][Met+17]，或者使用无梯度优化器，例如交叉熵方法（第 6.7.5 节），正如 **QT-Opt** 方法中所使用的  [Kal+18a][Kal+18*a*]。最近，**连续行动 Q 学习**（continuous action Q-learning，CAQL，[Ryu+20][Ryu+20]）提出使用混合整数规划来解决 argmax 问题，利用了 Q 网络的 ReLU 结构。我们也可以在学习了最优 Q 函数后，通过训练一个策略 $a_*=\pi_*(s)$ 来分摊这种优化的成本。

![image-20251220132407821](/assets/img/figures/book2/34.13.png)

#### 34.5.5.1 案例

在本节中，我们通过一个简单示例来让价值函数等概念更加具体化。考虑如图34.13(a)所示的**一维网格世界**（grid world）。其中存在5个可能的状态，$S_{T1}$ 与 $S_{T2}$ 属于吸收状态，因为智能体一旦进入这些状态，交互即告终止。系统设置两个动作：向上（↑）与向下（↓）。奖励函数在除目标状态 $S_{T2}$ 外的所有位置均为零；进入 $S_{T2}$ 时可获得奖励 1。因此，在每个状态下的最优动作都是向下移动。

图34.13(b)展示了当 $\gamma = 0$ 时的$Q_*$函数。注意，我们只显示了非吸收状态下的函数值，因为根据定义，吸收状态中的最优$Q$值均为 0。可以看到，$Q_*(s_3, \downarrow) = 1.0$，这是因为如果智能体从 $s_3$ 向下移动，将在下一步获得奖励 1.0；然而，对于所有其他状态‑动作对，$Q_*(s,a) = 0 $，因为它们不会提供非零的即时奖励。这一最优 $Q$ 函数反映出：使用 $ \gamma = 0 $ 意味着完全短视，完全忽略了未来收益。

图34.13(c)展示了当 $\gamma = 1$ 时的 $Q_*$。在这种情况下，我们对所有未来奖励赋予同等权重。因此，对于所有状态‑动作对，$ Q_*(s,a) = 1$，因为智能体最终总能到达目标。这是无限远视的视角。然而，它并未给智能体提供任何关于如何行动的短期指导。例如，在状态 $s_2$ 中，由于向上和向下两个动作最终都能以相同的 $ Q_* $ 值到达目标，因此无法明确应该选择向上还是向下。

图34.13(d)展示了当 $ \gamma = 0.9$ 时的$Q_*$。这体现了一种偏好近期奖励，同时也考虑未来奖励的折中。这会激励智能体寻找到达目标的最短路径，而这通常是我们期望的行为。合适的 $\gamma$ 值需要由智能体设计者决定，就像奖励函数的设计一样，它必须反映出智能体的期望行为模式。

## 34.6 MDP中的规划问题

在本节中，我们讨论当 MDP 模型已知时，如何计算最优策略。这个问题被称为**规划**（planning），与之相对的是模型未知情况下的**学习**问题，后者通过强化学习（第35章）来解决。我们将讨论的规划算法基于**动态规划**（dynamic programming,DP） 和**线性规划**（linear programming,LP）。

出于简化，本节我们假设状态和行动的集合是离散的，且 $\gamma<1$。然而，最优策略的精确计算复杂度通常与状态空间 $S$ 和行动空间 $\mathcal{A}$ 的大小成多项式关系，当状态空间是多个有限集的笛卡尔积时，这个问题通常难以处理。这一挑战被称为**维度灾难**（curse of dimensionality）。因此，通常需要采用近似方法，例如使用价值函数或策略的参数化或非参数化表示，这既是为了计算上的可行性，也是为了将这些方法推广到处理具有一般状态和行动集的 MDP。在这种情况下，我们有了**近似动态规划**（approximate dynamic programming,ADP） 和**近似线性规划**（approximate linear programming,ALP） 算法（例如，参见 [Ber19]）。

<img src="/assets/img/figures/book2/34.14.png" alt="image-20251221144324670" style="zoom:70%;" />

### 34.6.1 价值迭代

一种普遍使用且有效的求解 MDP 的动态规划方法是**价值迭代**（value iteration, VI）。从初始价值函数估计 $V_0$ 开始，该算法通过以下方式迭代更新估计值：

$$
V_{k+1}(s)=\max _a\left[R(s, a)+\gamma \sum_{s^{\prime}} p\left(s^{\prime} \mid s, a\right) V_k\left(s^{\prime}\right)\right] \tag{34.86}
$$

请注意，该更新规则（有时称为**贝尔曼备份**（Bellman backup））正是贝尔曼最优性方程 (34.82) 的右边，只是用当前估计值 $V_k$ 替换了未知的 $V_*$。方程 (34.86) 的一个基本性质是更新结果是一个**收缩映射**（contraction）：可以证明

$$
\max _s\left|V_{k+1}(s)-V_*(s)\right| \leq \gamma \max _s\left|V_k(s)-V_*(s)\right| \tag{34.87}
$$

换句话说，每次迭代都会将价值函数的最大误差减少一个常数因子。由此可知，$V_k$ 将收敛到 $V_*$，之后便可以得到最优策略。在实践中，当 $V_k$足够接近 $V_*$ 时，我们通常便可以终止 VI，因为相对于 $V_k$ 的贪婪策略已经接近最优。价值迭代的思路同样可以用来学习最优行动价值函数 $Q_*$。

在价值迭代中需要考虑所有可能状态 $s$ 的 $V_*(s)$ 和 $\pi_*(s)$，并在每次迭代中对所有可能的下一状态 $s^\prime$ 取平均，如图 34.14（右）所示。然而，对于某些问题，我们可能只关心某些特殊起始状态的价值（和策略）。例如，在图的最短路径问题中，我们试图从当前状态找到到达目标状态的最短路径。这可以通过定义一个转移矩阵 $p_T\left(s^{\prime} \mid s, a\right)$ 建模为分幕式 MDP，其中从节点 $s$ 延边 $a$ 到达相邻节点 $s^\prime$ 的概率为 $1$ （确定性转移）。奖励函数定义为对所有状态 $s$（目标状态除外）有 $R(s, a)=-1$，目标状态被建模为吸收状态。

在此类问题中，我们可以使用一种称为**实时动态规划**（real-time dynamic programming，RTDP） 的方法 [BBS95]，来高效计算一个**最优部分策略**（optimal partial policy），该策略只规定对可达状态应采取的行动。RTDP 维护一个价值函数估计 $V$。在每一步，它通过 $V(s) \leftarrow \max _a \mathbb{E}_{p_T\left(s^{\prime} \mid s, a\right)}\left[R(s, a)+\gamma V\left(s^{\prime}\right)\right]$ 对当前状态 $s$ 执行一次贝尔曼备份。它可以（通常带有一些探索地）选择一个行动 $a$，到达下一个状态 $s^\prime$，并重复此过程。这可以看作是更一般的**异步价值迭代**（asynchronous value iteration）的一种形式，它将计算努力集中在状态空间中更可能从当前状态到达的部分，而不是在每次迭代中同步更新所有状态。

### 34.6.2 策略迭代

另一种计算 $\pi_*$ 的有效 DP 方法是**策略迭代**（policy iteration）。它是一种在确定性策略空间中进行搜索的迭代算法，直到收敛至最优策略。每次迭代包含两个步骤：**策略评估**（policy evaluation）和**策略改进**（policy improvement）。

如前所述，策略评估步骤计算当前策略的价值函数。令 $\pi$ 表示当前策略，$\boldsymbol{v}(s)=V_\pi(s)$ 表示按状态索引、以向量形式编码的价值函数，$\boldsymbol{r}(s)=\sum_a \pi(a \mid s) R(s, a)$ 表示奖励向量， $\mathbf{T}\left(s^{\prime} \mid s\right)=\sum_a \pi(a \mid s) p\left(s^{\prime} \mid s, a\right)$表示状态转移矩阵。策略评估的贝尔曼方程可以写成以下矩阵-向量形式：

$$
\boldsymbol{v}=\boldsymbol{r}+\gamma \mathbf{T} \boldsymbol{v} \tag{34.88}
$$

这是一个包含 $|\mathcal{S}|$ 个未知数的线性方程组。我们可以使用矩阵求逆来求解它：$\boldsymbol{v}=(\mathbf{I}-\gamma \mathbf{T})^{-1} \boldsymbol{r}$。或者，我们可以使用价值迭代，通过计算 $\boldsymbol{v}_{t+1}=\boldsymbol{r}+\gamma \mathbf{T} \boldsymbol{v}_t$ 直至接近收敛，或者使用某种计算上更高效的异步变体。

一旦我们评估了当前策略 $\pi$ 的 $V_\pi$，我们就可以用它来推导出一个更好的策略 $\pi^{\prime}$，因此得名策略改进。为此，我们只需计算一个确定性策略 $\pi^{\prime}$，该策略在每个状态下都相对于 $V_\pi$ 贪婪地行动；即 $\pi^{\prime}(s)=\operatorname{argmax}_a\left\{R(s, a)+\gamma \mathbb{E}\left[V_\pi\left(s^{\prime}\right)\right]\right\}$。我们可以保证 $V_{\pi^{\prime}} \geq V_\pi$。为了理解这一点，如前所述定义 $\boldsymbol{r}^{\prime}$，$\mathbf{T}^{\prime}$和 $\boldsymbol{v}^{\prime}$ ，但针对新策略 $\pi^{\prime}$。$\pi^{\prime}$ 的定义意味着 $\boldsymbol{r}^{\prime}+\gamma \mathbf{T}^{\prime} \boldsymbol{v} \geq \boldsymbol{r}+\gamma \mathbf{T} \boldsymbol{v}=\boldsymbol{v}$，其中的等式源于贝尔曼方程。重复相同的等式，我们有

$$
\begin{align}
\boldsymbol{v} & \leq \boldsymbol{r}^{\prime}+\gamma \mathbf{T}^{\prime} \boldsymbol{v} \leq \boldsymbol{r}^{\prime}+\gamma \mathbf{T}^{\prime}\left(\boldsymbol{r}^{\prime}+\gamma \mathbf{T}^{\prime} \boldsymbol{v}\right) \leq \boldsymbol{r}^{\prime}+\gamma \mathbf{T}^{\prime}\left(\boldsymbol{r}^{\prime}+\gamma \mathbf{T}^{\prime}\left(\boldsymbol{r}^{\prime}+\gamma \mathbf{T}^{\prime} \boldsymbol{v}\right)\right) \leq \cdots \tag{34.89} \\
& =\left(\mathbf{I}+\gamma \mathbf{T}^{\prime}+\gamma^2 \mathbf{T}^{\prime 2}+\cdots\right) \boldsymbol{r}=\left(\mathbf{I}-\gamma \mathbf{T}^{\prime}\right)^{-1} \boldsymbol{r}=\boldsymbol{v}^{\prime} \tag{34.90}
\end{align}
$$

从初始策略 $\pi_0$ 开始，策略迭代在策略评估 ($E$) 和改进 ($I$) 步骤之间交替进行，如下所示：

$$
\pi_0 \xrightarrow{E} V_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} V_{\pi_1} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} V_* \tag{34.91}
$$

该算法在迭代 $k$ 时停止，如果策略 $\pi_k$ 相对于其自身的价值函数 $V_{\pi_k}$ 是贪婪的。在这种情况下，该策略是最优的。由于最多有  $|\mathcal{A}|^{|\mathcal{S}|}$个确定性策略，并且每次迭代都严格改进了策略，因此算法必须在有限次迭代后收敛。

在 PI 中，我们在策略评估（涉及多次迭代，直到 $V_\pi$ 收敛）和策略改进之间交替进行。在 VI 中，我们在一次策略评估迭代和一次策略改进迭代（更新规则中的“max”算子）之间交替进行。在**广义策略改进**中，我们可以按任意顺序自由地混合任意数量的这些步骤。一旦策略相对于其自身的价值函数是贪婪的，该过程就会收敛。

请注意，策略评估计算的是 $V_\pi$，而价值迭代计算的是 $V_*$。这一区别在图 34.14 中使用**备份图**（backup diagram）进行了说明。这里根节点代表任意状态 $s$，下一层的节点代表状态-行动组合（实心圆），叶节点代表每个可能行动所导致的一系列可能的下一个状态 $s^{\prime}$。在前者（策略评估）中，我们根据策略对所有行动取平均，而在后者（价值迭代）中，我们对所有行动取最大值。

### 34.6.3 线性规划

## 34.7 主动学习

在本节中，我们将讨论**主动学习**（active learning，AL），在这种学习方式中，智能体可以自主选择它想要使用的数据，以便尽可能快地学习底层的预测函数，即使用最少数量的标注数据。如图 34.15 所示，这比使用随机收集的数据要高效得多。这在标注成本高昂的场景中非常有用，例如医学图像分类 [GIG17; Wal+20]。

主动学习有多种方法，正如 [Set12; Ren+21; ZSH22] 中所综述的那样。在本节中，我们仅讨论其中几种方法。

### 34.7.1 主动学习场景

最早的主动学习方法之一被称为**成员查询合成**（membership query synthesis） [Ang88]。在这种场景下，智能体可以生成一个任意的查询 $\boldsymbol{x} \sim p(\boldsymbol{x})$，然后向**预言机**请求其标签 $y=f(\boldsymbol{x})$。（"预言机"是指一个能知晓所有可能问题真实答案的系统。）这种场景主要具有理论意义，因为学习好的生成模型很困难，并且很少能按需访问预言机（尽管人力众包计算平台可以被视为高延迟的预言机）。

另一种场景是**基于流的选择性采样**（stream-based selective sampling） [ACL89]，其中智能体接收一个输入流 $\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots$，并且在每一步都必须决定是否请求该输入的标签。同样，这种场景也主要具有理论意义。

机器学习中最后一个且被广泛使用的设置是**基于池的采样**（pool-based-sampling） [LG94]，其中未标注样本的池 $\mathcal{X}$ 从一开始就是可用的。在每一步中，我们对批次中的每个候选样本应用一个**获取函数**(acquisition function)，以决定为哪个样本收集标签。然后我们收集该标签，用新数据更新模型，并重复此过程，直到我们耗尽池中的样本、用尽时间或达到某种期望的性能。在后续章节中，我们将只关注基于池的采样。

### 34.7.2 与其他形式的序列决策的关系

（基于池的）主动学习与贝叶斯优化（第6.6节）和上下文老虎机问题（第34.4节）密切相关。[Tou14] 详细讨论了它们之间的联系，但简而言之，这些方法之所以不同，是因为它们解决的目标函数略有不同，如表34.1所总结。具体来说：在**主动学习**中，我们的目标是识别一个函数 $f: \mathcal{X} \rightarrow \mathcal{Y}$，该函数在应用于随机输入 $\boldsymbol{x}$ 时能产生最小的期望损失。在**贝叶斯优化**中，我们的目标是识别一个输入点 $\boldsymbol{x}$，使得函数输出 $f(\boldsymbol{x})$ 在该点最大。在**老虎机问题**中，我们的目标是识别一个策略 $\pi: \mathcal{X} \rightarrow \mathcal{A}$，该策略在应用于随机输入（上下文）$\boldsymbol{x}$ 时能带来最大的期望奖励。（我们看到主动学习和老虎机的目标是相似的，但在老虎机问题中，智能体只能选择行动，而不能选择状态，因此对（奖励）函数的评估位置只有部分控制权。）

在这三个问题中，我们都希望用尽可能少的行动来找到最优解，因此必须解决探索-利用的权衡问题（第34.4.3节）。一种方法是使用诸如高斯过程（第18章）之类的方法来表示我们对函数的不确定性，这让我们能够计算 $p\left(f \mid \mathcal{D}_{1: t}\right)$。然后我们定义某个**获取函数** $\alpha(\boldsymbol{x})$，该函数在给定信念状态 $p\left(f \mid \mathcal{D}_{1: t}\right)$ 的情况下，评估在输入位置 $\boldsymbol{x}$ 查询函数的有用程度，并选择 $\boldsymbol{x}_{t+1}=\operatorname{argmax}_{\boldsymbol{x}} \alpha(\boldsymbol{x})$ 作为我们的下一个查询点。（在老虎机设置中，智能体无法选择状态 $\boldsymbol{x}$，但可以选择行动 $a$。）例如，在贝叶斯优化中，通常使用改进概率（第6.6.3.1节），而对于回归任务的主动学习，我们可以使用后验预测方差。主动学习的目标将导致智能体"四处"查询，而贝叶斯优化的目标将导致智能体"聚焦"于最有希望的区域，如图34.16所示。我们将在第34.7.3节讨论用于主动学习的其他获取函数。

### 34.7.3 获取策略

本节将讨论在AL中一些选择查询样本的启发式方法。

#### 34.7.3.1 不确定采样

一种直观的、用于选择下一个标注样本的启发式方法是：挑选模型当前最不确定的那个样本。这被称为**不确定性采样**。我们已经在图34.16的回归问题中说明了这一点，其中我们用后验方差来表示不确定性。

对于分类问题，我们可以用多种方式衡量不确定性。令 $\boldsymbol{p}_n=\left[p\left(y=c \mid \boldsymbol{x}_n\right)\right]_{c=1}^C$ 为每个未标注输入 $\boldsymbol{x}_n$ 的类别概率向量。令 $U_n=\alpha\left(\boldsymbol{p}_n\right)$ 为样本 $n$ 的不确定性，其中 $\alpha$ 是一个获取函数。一些常见的 $\alpha$ 选择包括：**熵采样**（entropy sampling） [SW87a]，使用 $\alpha(\boldsymbol{p})=-\sum_{c=1}^C p_c \log p_c$；**间隔采样**（margin sampling），使用 $\alpha(\boldsymbol{p})=p_2-p_1$，其中 $p_1$ 是最可能类别的概率，$p_2$ 是第二可能类别的概率；**最不置信采样**（least confident sampling），使用 $\alpha(\boldsymbol{p})=1-p_c$.，其中 $c^*=\operatorname{argmax}_c p_c$。这些策略之间的差异如图34.17所示。在实践中，通常发现间隔采样效果最好 [Chu+19]。

#### 34.7.3.2 委员会查询

在本节中，我们讨论如何将不确定性采样应用于诸如支持向量机之类的模型，这些模型仅返回点预测，而非概率分布。基本方法是创建一个由多个不同模型组成的集成，并利用模型预测之间的**分歧**作为一种不确定性的度量。（即使对于概率模型（如DNN），这种方法也很有用，因为正如我们在深度集成（第17.3.9节）中讨论的那样，模型不确定性通常可能大于参数不确定性。）

更详细地说，假设我们有 $K$ 个集成成员，令 $c_n^k$ 表示成员 $k$ 对输入 $\boldsymbol{x}_n$ 的预测类别。令 $v_{n c}=\sum_{k=1}^K \mathbb{I}\left(c_n^k=c\right)$ 为投给类别 $c$ 的票数，且 $q_{n c}=v_{n c} / C$ 为由此导出的分布。（类似的方法可用于回归模型，其中我们使用各成员预测的标准差。）然后，我们可以使用**间隔采样**或**熵采样**，但基于分布 $\boldsymbol{q}_n$ 进行计算。这种方法被称为**委员会查询**（query by committee） [SOS92]，并且通常能够超越使用单一模型的普通不确定性采样，正如我们在图34.18中所展示的。

#### 34.7.3.3 信息论方法

一种自然的获取策略是选择那些其标签能最大程度减少我们关于模型参数 $\boldsymbol{w}$ 的不确定性的数据点。这被称为**信息增益**（information gain）准则，最早由 [Lin56] 提出。其定义如下：

$$
\alpha(\boldsymbol{x}) \triangleq \mathbb{H}(p(\boldsymbol{w} \mid \mathcal{D}))-\mathbb{E}_{p(y \mid \boldsymbol{x}, \mathcal{D})}[\mathbb{H}(p(\boldsymbol{w} \mid \mathcal{D}, \boldsymbol{x}, y))] \tag{34.95}
$$

（注意，第一项相对于 $\boldsymbol{x}$ 是常数，但为了后续方便我们将其包含在内。）这等价于参数后验分布的期望变化，由下式给出：

$$
\alpha^{\prime}(\boldsymbol{x}) \triangleq \mathbb{E}_{p(y \mid \boldsymbol{x}, \mathcal{D})}\left[D_{\mathrm{KL}}(p(\boldsymbol{w} \mid \mathcal{D}, \boldsymbol{x}, y) \| p(\boldsymbol{w} \mid \mathcal{D}))\right] \tag{34.96}
$$

利用互信息的对称性，我们可以将方程 (34.95) 重写如下：

$$
\begin{align}
\alpha(\boldsymbol{x}) & =\mathbb{H}(\boldsymbol{w} \mid \mathcal{D})-\mathbb{E}_{p(y \mid \boldsymbol{x}, \mathcal{D})}[\mathbb{H}(\boldsymbol{w} \mid \mathcal{D}, \boldsymbol{x}, y)]  \tag{34.97}\\
& =\mathbb{I}(\boldsymbol{w}, y \mid \mathcal{D}, \boldsymbol{x}) \tag{34.98}\\
& =\mathbb{H}(y \mid \boldsymbol{x}, \mathcal{D})-\mathbb{E}_{p(\boldsymbol{w} \mid \mathcal{D})}[\mathbb{H}(y \mid \boldsymbol{x}, \boldsymbol{w}, \mathcal{D})] \tag{34.99}
\end{align}
$$

这种方法的优点在于，我们现在只需要考虑输出 $y$ 的预测分布的不确定性，而不必考虑参数 $\boldsymbol{w}$ 的不确定性。这种方法被称为**贝叶斯主动学习通过分歧** （Bayesian active learning by disagreement，BALD）。[Hou+12]。

方程 (34.99) 有一个有趣的解释。第一项偏好那些预测标签存在不确定性的样本 $\boldsymbol{x}$。仅使用此作为选择标准相当于我们上面讨论过的**不确定性采样**。然而，这对于本身存在歧义或错误标注的样本可能会有问题。通过添加第二项，我们惩罚了这种行为，因为我们对那些即使已知参数其预测分布熵值也很高的点添加了一个大的负权重。因此，我们忽略了**偶然性（固有的）不确定性**，而专注于**认知不确定性**。

### 34.7.4 Batch 主动学习

