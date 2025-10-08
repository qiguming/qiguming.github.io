---
title: 7 推理算法：综述
author: fengliang qi
date: 2025-10-07 11:33:00 +0800
categories: [BOOK-2, PART-II]
tags: [Inference]
math: true
mermaid: true
toc: true
comments: true

---

> 本章将开启 推理 这一部分的内容。
>

* TOC
{:toc}
## 7.1 引言

在机器学习的概率论视角下，所有的未知量——关于未来的预测，系统中的隐变量，或者模型的参数——被作为随机变量，且对应概率分布。**推理**（inference）过程就是计算这些未知量的后验分布，并以任意可获得的数据为条件。

进一步讲，令 $\boldsymbol{\theta}$ 表示未知变量，$\mathcal{D}$ 表示已知变量。已知似然 $p(\mathcal{D}\mid\boldsymbol{\theta})$ 和先验 $p(\boldsymbol{\theta})$ ，可以使用贝叶斯定理计算后验 $p(\boldsymbol{\theta} \mid \mathcal{D})$：

$$
p(\boldsymbol{\theta} \mid \mathcal{D})=\frac{p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})}{p(\mathcal{D})} \tag{7.1}
$$

上式的主要计算瓶颈是分母部分的归一化常数，该常数需要求解下面的高维积分：

$$
p(\mathcal{D})=\int p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta}) d \boldsymbol{\theta} \tag{7.2}
$$

这是为了将非归一化联合概率 $p(\boldsymbol{\theta}, \mathcal{D})$ 转换为归一化概率 $p(\boldsymbol{\theta}\mid\mathcal{D})$，该转换过程需考虑参数 $\boldsymbol{\theta}$ 所有可能的合理取值。

一旦得到后验概率，我们可以使用它计算某些函数的后验期望：

$$
\mathbb{E}[g(\boldsymbol{\theta}) \mid \mathcal{D}]=\int g(\boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta} \tag{7.3}
$$

通过定义不同的函数 $g$，我们可以计算很多感兴趣的量，比如：

$$
\begin{align}
\text { mean: } & g(\boldsymbol{\theta})=\boldsymbol{\theta} \tag{7.4}\\
\text { covariance: } & g(\boldsymbol{\theta})=(\boldsymbol{\theta}-\mathbb{E}[\boldsymbol{\theta} \mid \mathcal{D}])(\boldsymbol{\theta}-\mathbb{E}[\boldsymbol{\theta} \mid \mathcal{D}])^{\top} \tag{7.5}\\
\text { marginals: } & g(\boldsymbol{\theta})=p(\theta_1=\theta_1^* \mid \boldsymbol{\theta}_{2: D}) \tag{7.6}\\
\text { predictive: } & g(\boldsymbol{\theta})=p(\boldsymbol{y}_{N+1} \mid \boldsymbol{\theta}) \tag{7.7}\\
\text { expected loss: } & g(\boldsymbol{\theta})=\ell(\boldsymbol{\theta}, a) \tag{7.8}
\end{align}
$$

其中 $\boldsymbol{y}_{N+1}$ 表示在看到 $N$ 个样本后的下一次预测，后验期望损失使用损失函数 $\ell$ 和行为 $a$（34.1.3节）表示。最后，如果我们针对模型 $M$ 定义 $g(\boldsymbol{\theta})=p(\mathcal{D} \mid \boldsymbol{\theta}, M)$，我们还可以将边际似然（第3.8.3节）表述为关于先验的期望：

$$
\mathbb{E}[g(\boldsymbol{\theta}) \mid M]=\int g(\boldsymbol{\theta}) p(\boldsymbol{\theta} \mid M) d \boldsymbol{\theta}=\int p(\mathcal{D} \mid \boldsymbol{\theta}, M) p(\boldsymbol{\theta} \mid M) d \boldsymbol{\theta}=p(\mathcal{D} \mid M) \tag{7.9}
$$

由此可见，积分（及计算期望）是贝叶斯推断的核心，而微分则是优化过程的核心。

本章我们将对计算（近似）后验分布及其对应期望值的算法技术进行提纲挈领地概述，后续章节将展开更详细的讨论。需要强调的是，这些方法大多独立于具体模型——这使得问题解决者能够专注于为任务构建最优模型，而后依赖相应的推断算法完成剩余工作，这一过程常被称为“转动贝叶斯曲柄”。关于贝叶斯计算的更多细节，可参阅 [Gel+14a; MKL21; MFR20] 等文献。

![image-20251008140348870](/assets/img/figures/book2/7.1.png)

## 7.2 常见的推断模式

有很多不同的后验分布可以计算，但我们主要讨论其中的3种模式。不同的模式决定了不同的推理算法。

### 7.2.1 全局隐变量（Global latents）

第一种模式是模型中只包含**全局隐变量**（global latent variables），比如模型的参数 $\boldsymbol{\theta}$，这些参数被 $N$ 个已观测的训练样本共享。如图7.1a所示，该模式对应于监督学习或者判别式学习的设定，对应的联合概率分布为

$$
p(\boldsymbol{y}_{1: N}, \boldsymbol{\theta} \mid \boldsymbol{x}_{1: N})=p(\boldsymbol{\theta})[\prod_{n=1}^N p(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta})] \tag{7.10}
$$

我们的目标是计算后验分布 
$$p(\boldsymbol{\theta} \mid \boldsymbol{x}_{1: N}, \boldsymbol{y}_{1: N})$$。
第三部分讨论的大多数贝叶斯监督学习模型都遵循这种模式。

### 7.2.2 局部隐变量（Local latents）

第二种模式是模型中包含**局部隐变量**，例如当模型参数 $\boldsymbol{\theta}$ 已知时，我们需要推断隐藏状态 $\boldsymbol{z}_{1:N}$。这种情况如图7.1b所示，此时的联合分布具有如下形式：

$$
p(\boldsymbol{x}_{1: N}, \boldsymbol{z}_{1: N} \mid \boldsymbol{\theta})=[\prod_{n=1}^N p(\boldsymbol{x}_n \mid \boldsymbol{z}_n, \boldsymbol{\theta}_x) p(\boldsymbol{z}_n \mid \boldsymbol{\theta}_z)] \tag{7.11}
$$

推理的目标是对于每个 $n$ 计算后验分布 $p(\boldsymbol{z}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta})$。这是我们在第9章中考虑的大多数PGM（概率图模型）推理方法的设置。

若模型参数未知（大多数隐变量模型如混合模型均属此情况），我们可选择通过某种方法（例如最大似然估计）对其进行估计，然后代入参数的点估计值。此方法的优势在于：在给定参数 $\boldsymbol{\theta}$ 的条件下，所有隐变量均条件独立，因此我们可以跨数据并行执行推断。这使得我们可以使用**期望最大化算法**（第6.5.3节）等方法——在E步中同时推断所有 $n$ 对应的 $p(\boldsymbol{z}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}_t)$，随后在M步中更新 $\boldsymbol{\theta}_t$。若对 $\boldsymbol{z}_n$ 的推断无法精确求解，可采用变分推断方法，此组合策略被称为**变分EM算法**（第6.5.6.1节）。

另一种方法是使用小批量数据对似然函数进行近似，通过对小批量中每个样本的隐变量 $\boldsymbol{z}_n$ 进行边缘化处理，得到：

$$
\log p(\mathcal{D}_t \mid \boldsymbol{\theta}_t)=\sum_{n \in \mathcal{D}_t} \log [\sum_{\boldsymbol{z}_n} p(\boldsymbol{x}_n, \boldsymbol{z}_n \mid \boldsymbol{\theta}_t)] \tag{7.12}
$$

其中 $$\mathcal{D}_t $$
表示第 $t$ 步的小批量数据。若无法精确计算该边缘化过程，可采用变分推断方法，此组合策略被称为**随机变分推断**（第10.1.4节）。此外，我们还可以学习一个推断网络 $$q_\boldsymbol{\phi}(\boldsymbol{z} \mid \boldsymbol{x} ; \boldsymbol{\theta})$$ 
来替代在每个批次 $t$ 中为每个样本 $n$ 运行推断引擎的操作；学习参数 $\boldsymbol{\phi}$ 的成本可分摊到所有批次中。这种方法被称为**摊销随机变分推断**（参见第10.1.5节）。

### 7.2.3 全局和局部隐变量

第三种模式是模型中同时包含 **局部和全局隐变量**。如图7.1c所示，对应的联合概率分布为：

$$
p(\boldsymbol{x}_{1: N}, \boldsymbol{z}_{1: N}, \boldsymbol{\theta})=p(\boldsymbol{\theta}_x) p(\boldsymbol{\theta}_z)[\prod_{n=1}^N p(\boldsymbol{x}_n \mid \boldsymbol{z}_n, \boldsymbol{\theta}_x) p(\boldsymbol{z}_n \mid \boldsymbol{\theta}_z)] \tag{7.13}
$$

这本质上是图7.1b中隐变量模型的贝叶斯版本，其特点在于同时对局部变量 $\boldsymbol{z}_n$ 和共享全局变量 $\boldsymbol{\theta}$ 的不确定性进行建模。这种方法在机器学习社区中相对少见，因为通常认为参数 $\boldsymbol{\theta}$ 的不确定性相较于局部变量 $\boldsymbol{z}_n$ 的不确定性可以忽略不计——其根本原因在于：参数受到全部 $N$ 个数据点的共同约束，而每个局部隐变量 $\boldsymbol{z}_n$ 仅受单个数据点 $\boldsymbol{x}_n$ 的影响。然而，采用"完全贝叶斯"方法对局部与全局变量的不确定性同时建模仍具有显著优势，本书后续将呈现相关应用案例。

## 7.3 精确推理算法

在某些情况下，我们可以通过可处理的方式执行精确后验推断。具体而言，若**先验分布与似然函数共轭**，则后验分布将具有解析可解性。一般而言，当先验与似然同属指数族分布时（第2.4节），即可满足该条件。特别地，若未知变量由 $\boldsymbol{\theta}$ 表示，则我们假设：

$$
\begin{align}
p(\boldsymbol{\theta}) & \propto \exp (\boldsymbol{\lambda}_0^{\top} \mathcal{T}(\boldsymbol{\theta})) \tag{7.14}\\
p(\boldsymbol{y}_i \mid \boldsymbol{\theta}) & \propto \exp (\tilde{\boldsymbol{\lambda}}_i(\boldsymbol{y}_i)^{\top} \mathcal{T}(\boldsymbol{\theta})) \tag{7.15}
\end{align}
$$

其中 $\mathcal{T}(\boldsymbol{\theta})$ 表示充分统计量，$\boldsymbol{\lambda}$ 表示自然参数。只需要将自然参数相加，我们便可以计算后验分布：

$$
\begin{align}
p(\boldsymbol{\theta} \mid \boldsymbol{y}_{1: N}) & =\exp (\boldsymbol{\lambda}_*^{\top} \mathcal{T}(\boldsymbol{\theta}))  \tag{7.16}\\
\boldsymbol{\lambda}_* & =\boldsymbol{\lambda}_0+\sum_{n=1}^N \tilde{\boldsymbol{\lambda}}_n(\boldsymbol{y}_n) \tag{7.17}
\end{align}
$$

更多细节，参考3.4节。

另一种能精确计算后验分布的情形是：当所有 $D$ 个未知变量均为离散变量，且每个变量具有 $K$ 个状态时，归一化常数中的积分将转化为包含 $K^D$ 个项的求和运算。然而在多数情况下，$K^D$ 的规模会超出可计算范围。但若该概率分布满足特定的条件独立性性质（通过概率图模型表述），则我们可以将联合分布表示为若干局部项的乘积（参见第4章）。这使得我们可以运用动态规划方法实现高效计算（参见第9章）。

## 7.4 近似推断算法

对于大多数概率模型而言，我们无法精确计算边缘分布或后验分布，因此必须借助**近似推断**方法。现有多种算法可供选择，它们在速度、精度、简洁性与通用性之间各有权衡。下文将简要讨论部分算法，后续章节会展开详细论述（各类方法的综述可参阅[Alq22; MFR20]）。

### 7.4.1 MAP近似

最简单的近似推断方法是计算最大后验（MAP）估计

$$
\hat{\boldsymbol{\theta}}=\operatorname{argmax} p(\boldsymbol{\theta} \mid \mathcal{D})=\operatorname{argmax} \log p(\boldsymbol{\theta})+\log p(\mathcal{D} \mid \boldsymbol{\theta}) \tag{7.18}
$$

然后假设后验分布将全部概率质量归于MAP估计：

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) \approx \delta(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}}) \tag{7.19}
$$

这种方法的优势在于，我们可以利用多种优化算法来计算 MAP 估计，这些算法将在第 6 章讨论。然而，MAP 估计也存在一些缺陷，部分问题我们将在下文探讨。

#### 7.4.1.1 MAP无法给出不确定性

在许多统计应用（尤其是科学领域）中，了解一个参数估计值的可信度至关重要。显然，点估计无法传达任何不确定性信息。虽然可以从点估计中推导出频率学派的不确定性度量（参见第3.3.1节），但直接计算后验分布无疑是更自然的方式——通过后验分布我们可以推导出标准误差（参见第3.2.1.6节）和置信区域（参见第3.2.1.7节）等重要统计量。

在预测任务（机器学习的主要关注点）背景下，我们在第3.2.2节中看到，使用点估计会低估预测不确定性，这可能导致模型不仅做出错误预测，更是以高置信度做出错误预测。让预测模型“知晓自身认知局限”被普遍视为至关重要，而贝叶斯方法正是实现这一目标的有效策略。

![image-20251008143915027](/assets/img/figures/book2/7.2.png)

#### 7.4.1.2 MAP估计通常是后验分布的非典型代表

在某些情况下，我们可能不关注不确定性，而只希望获得后验分布的单一统计量。然而，后验分布的**众数**通常是一种较差的选择，因为与众数不同，均值或中位数更能代表分布的典型特征。图7.2(a)在一维连续空间中直观展示了这一点：众数位于孤立峰顶（黑线），远离大部分概率质量；相比之下，均值（红线）则位于分布的中心区域。

另一个示例如图7.2(b)所示：此处众数为0，但均值非零。这种偏态分布常见于估计方差参数时，尤其是在层次模型中。在此类情形下，MAP估计（以及MLE估计）显然会成为效果极差的估计量。

#### 7.4.1.3 MAP对重参数化不具备不变形

MAP估计还存在一个更微妙的问题：其结果依赖于概率分布的参数化方式，而这是非常不可取的。例如，在表示伯努利分布时，我们应当能够使用成功概率或对数几率进行参数化，且这种选择不应影响我们的实际认知。

假设 $$\hat{x}=\operatorname{argmax}_x p_x(x)$$
是 $x$ 的 MAP 估计。现令 $y = f(x)$ 为 $x$ 的一个变换。一般而言，
$$\hat{y}=\operatorname{argmax}_y p_y(y)$$ 
并不等于 $f(\hat{x})$。例如，设 
$$x \sim \mathcal{N}(6,1)$ 且 $y = f(x)$$，
其中 
$$f(x) = \frac{1}{1+\exp(-x+5)}$$。
我们可以利用变量变换公式（第2.5.1节）得到 
$$p_{y}(y) = p_{x}(f^{-1}(y))\mid\frac{df^{-1}(y)}{dy}\mid$$，
或采用蒙特卡洛近似法。结果显示在图2.12中：原始高斯分布 $p(x)$ 被S型非线性函数“挤压”变形，且变换后分布的众数不等于原分布众数的变换结果。

由此可见，MAP估计依赖于参数化选择。最大似然估计不受此影响，因为似然是函数而非概率密度；贝叶斯推断也不存在此问题，因为在参数空间积分时已考虑了测度变换。

![image-20251008145215601](/assets/img/figures/book2/7.3.png)

### 7.4.2 网格近似

若要刻画不确定性，就需要考虑参数 $\boldsymbol{\theta}$ 可能取一系列数值（每个取值都具有非零概率）的情况。实现这一特性的最简单方法是将参数的可能取值空间划分为有限个区域 $\boldsymbol{r}_1, \ldots, \boldsymbol{r}_K$，每个区域代表参数空间中一个以 $\boldsymbol{\theta}_k$ 为中心、体积为 $\Delta$ 的子空间。这种方法称为**网格近似法**。参数落在每个区域的概率由 $p(\boldsymbol{\theta} \in \boldsymbol{r}_k \mid \mathcal{D}) \approx p_k \Delta$ 给出，其中：

$$
\begin{align}
& p_k=\frac{\tilde{p}_k}{\sum_{k^{\prime}=1}^K \tilde{p}_{k^{\prime}}}  \tag{7.20}\\
& \tilde{p}_k=p(\mathcal{D} \mid \boldsymbol{\theta}_k) p(\boldsymbol{\theta}_k) \tag{7.21}
\end{align}
$$

随着 $K$ 的增加，每个网格的尺寸变小。上式的分母趋向于积分的一种简单的数值近似

$$
p(\mathcal{D})=\int p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta}) d \boldsymbol{\theta} \approx \sum_{k=1}^K \Delta \tilde{p}_k \tag{7.22}
$$

作为一个简单的例子，我们将使用近似贝塔-伯努利模型后验值的问题。具体来说，目标是近似

$$
p(\theta \mid \mathcal{D}) \propto[\prod_{n=1}^N \operatorname{Ber}(y_n \mid \theta)] \operatorname{Beta}(1,1) \tag{7.23}
$$

该例中数据集 $\mathcal{D}$ 包含10次正面与1次反面（观测总数 $N = 11$），并采用均匀先验分布。尽管我们可以通过第3.4.1节的方法精确计算此后验分布，但本例仍具有教学示范价值——我们可以将近似结果与精确解进行对比。此外，由于目标分布仅为一维，结果可视化也更为便捷。

图7.3a展示了网格近似法在我们的一维问题中的应用。可见该方法能有效捕捉后验分布的偏态特征（这源于10正1反的不平衡样本数据）。然而遗憾的是，该方法难以推广至超过2维或3维的问题，因为网格点的数量会随维度增加呈指数级增长。

### 7.4.3 Laplace (二次)近似

本节讨论使用一个多变量高斯分布来近似后验分布；这被称为 **拉普拉斯近似**（Laplace）或者**二次近似**（参考 [TK86;RMC09]）。

假设后验分布可以写成：

$$
p(\boldsymbol{\theta} \mid \mathcal{D})=\frac{1}{Z} e^{-\mathcal{E}(\boldsymbol{\theta})} \tag{7.24}
$$

其中 $\mathcal{E}(\boldsymbol{\theta})=-\log p(\boldsymbol{\theta}, \mathcal{D})$ 被称为能量函数，$Z=p(\mathcal{D})$ 被称为归一化常数。在众数 $\hat{\boldsymbol{\theta}}$ 周边进行泰勒展开，我们有

$$
\mathcal{E}(\boldsymbol{\theta}) \approx \mathcal{E}(\hat{\boldsymbol{\theta}})+(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}})^{\top} \boldsymbol{g}+\frac{1}{2}(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}})^{\top} \mathbf{H}(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}}) \tag{7.25}
$$

其中 $\boldsymbol{g}$ 表示在众数处的梯度，$\mathbf{H}$ 为对应的海森矩阵。考虑到 $\hat{\boldsymbol{\theta}}$ 是众数，所以梯度项等于0。所以

$$
\begin{align}
\hat{p}(\boldsymbol{\theta}, \mathcal{D}) & =e^{-\mathcal{E}(\hat{\boldsymbol{\theta}})} \exp [-\frac{1}{2}(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}})^{\top} \mathbf{H}(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}})] \tag{7.26} \\
\hat{p}(\boldsymbol{\theta} \mid \mathcal{D}) & =\frac{1}{Z} \hat{p}(\boldsymbol{\theta}, \mathcal{D})=\mathcal{N}(\boldsymbol{\theta} \mid \hat{\boldsymbol{\theta}}, \mathbf{H}^{-1}) \tag{7.27}\\
Z & =e^{-\mathcal{E}(\hat{\boldsymbol{\theta}})}(2 \pi)^{D / 2}\mid\mathbf{H}\mid^{-\frac{1}{2}} \tag{7.28}
\end{align}
$$

最后一行推导源于多元高斯分布的归一化常数。

拉普拉斯近似法易于实现，因为我们可以利用现有的优化算法计算 MAP 估计，随后只需计算该众数处的海森矩阵（在高维空间中可采用对角近似）。

图7.3b展示了该方法在我们的一维问题中的应用效果。但遗憾的是，该近似效果并不理想——这是因为后验分布具有偏态特征，而高斯分布是对称的。此外，目标参数被约束在区间 $\theta\in[0,1]$ 内，而高斯分布假设参数空间无约束 ($\boldsymbol{\theta}\in\mathbb{R}^{D}$)。幸运的是，我们可以通过变量变换解决后一个问题：本例中可对 $\alpha=\mathrm{logit}(\theta)$ 施加拉普拉斯近似。这种通过变换简化推断任务的技巧在实践中非常常用。

关于拉普拉斯近似的具体应用：第15.3.5节展示了其在贝叶斯逻辑回归中的应用，第17.3.2节介绍了在贝叶斯神经网络中的应用，第4.3.5.3节则阐述了在高斯马尔可夫随机场中的应用。

![image-20251008193322820](/assets/img/figures/book2/7.4.png)

### 7.4.4 变分推断

在第7.4.3节，我们讨论了Laplace近似，其中我们首先使用优化算法找到MAP估计，然后在该点使用海森矩阵近似后验分布的曲率。本节，我们讨论 **变分推断**（variational inference，VI），又被称为**变分贝叶斯**（variational Bayes，VB）。这是另一种基于优化的后验推断方法，但具有更大的建模灵活性（因此精度更高）。

VI 试图使用一个易处理的分布 $q(\boldsymbol{\theta})$ 来近似一个不易处理的概率分布 $p(\boldsymbol{\theta} \mid \mathcal{D})$，所以优化过程需要最小化两个分布之间的分歧 $D$：

$$
q^*=\underset{q \in \mathcal{Q}}{\operatorname{argmin}} D(q, p) \tag{7.29}
$$

其中 $\mathcal{Q}$ 表示易处理的概率分布族（如可完全因式分解的分布）。在具体实践中，我们并不是直接优化函数 $q$ 本身，而是优化函数 $q$ 的参数；我们称之为 **变分参数**（variational parameters）$\boldsymbol{\psi}$。

通常使用 KL 散度作为分歧的度量

$$
D(q, p)=D_{\mathrm{KL}}(q(\boldsymbol{\theta} \mid \boldsymbol{\psi}) \\mid p(\boldsymbol{\theta} \mid \mathcal{D}))=\int q(\boldsymbol{\theta} \mid \boldsymbol{\psi}) \log \frac{q(\boldsymbol{\theta} \mid \boldsymbol{\psi})}{p(\boldsymbol{\theta} \mid \mathcal{D})} d \boldsymbol{\theta} \tag{7.30}
$$

其中 $p(\boldsymbol{\theta} \mid \mathcal{D})=p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta}) / p(\mathcal{D})$。如此，推断问题退化成了如下的优化问题：

$$
\begin{align}
\boldsymbol{\psi}^* & =\underset{\boldsymbol{\psi}}{\operatorname{argmin}} D_{\mathbb{KL}}(q(\boldsymbol{\theta} \mid \boldsymbol{\psi}) \mid p(\boldsymbol{\theta} \mid \mathcal{D})) \tag{7.31}\\
& =\underset{\boldsymbol{\psi}}{\operatorname{argmin}} \mathbb{E}_{q(\boldsymbol{\theta} \mid \boldsymbol{\psi})}[\log q(\boldsymbol{\theta} \mid \boldsymbol{\psi})-\log (\frac{p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\mathcal{D})})] \tag{7.32}\\
& =\underset{\boldsymbol{\psi}}{\operatorname{argmin}} \underbrace{\mathbb{E}_{q(\boldsymbol{\theta} \mid \boldsymbol{\psi})}[-\log p(\mathcal{D} \mid \boldsymbol{\theta})-\log p(\boldsymbol{\theta})+\log q(\boldsymbol{\theta} \mid \boldsymbol{\psi})]}_{-\mathbf{Ł}(\boldsymbol{\psi})}+\log p(\mathcal{D}) \tag{7.33}
\end{align}
$$

考虑到 $\log p(\mathcal{D})$ 与 $\psi$ 无关，所以我们可以忽略它并聚焦到最大化

$$
\mathrm{Ł}(\boldsymbol{\psi}) \triangleq \mathbb{E}_{q(\boldsymbol{\theta} \mid \boldsymbol{\psi})}[\log p(\mathcal{D} \mid \boldsymbol{\theta})+\log p(\boldsymbol{\theta})-\log q(\boldsymbol{\theta} \mid \boldsymbol{\psi})] \tag{7.34}
$$

由于 $D_{\mathbb{KL}}(q \mid p) > 0$，我们有 $\mathrm{Ł}(\boldsymbol{\lambda}) \le \log p(\mathcal{D})$。其中 $\log p(\mathcal{D})$ 作为对数边际似然，亦被称为**证据**（evidence）。因此 $\mathrm{Ł}(\boldsymbol{\lambda})$ 被称为**证据下界**。通过最大化该下界，我们可以使变分后验逐渐逼近真实后验分布（详见第10.1节）。

我们可以自由选择任何形式的近似后验分布。例如，可采用高斯分布 $q(\theta\mid\boldsymbol{\psi}) = \mathcal{N}(\boldsymbol{\theta}\mid\boldsymbol{\mu}, \boldsymbol{\Sigma})$。这与拉普拉斯近似不同——在变分推断中，我们需要优化协方差矩阵 $\boldsymbol{\Sigma}$，而非将其等同于海森矩阵。若 $\boldsymbol{\Sigma}$ 为对角矩阵，则意味着后验分布可完全因子化，这被称为**平均场**（mean field）近似。

高斯近似并非适用于所有参数类型。例如在我们的一维示例中，参数存在 $\theta\in[0,1]$ 的约束条件。此时可采用 $q(\theta\mid\boldsymbol{\psi})=\text{Beta}(\theta\mid a,b)$ 形式的变分近似，其中 $\boldsymbol{\psi}=(a,b)$。然而，选择合适的变分分布形式需要相当的专业经验。为创建适用范围更广、更易于使用的“即插即用”方法，可采用**自动微分变分推断**（automatic differentiation variational inference，ADVI）[Kuc+16]。该方法通过变量变换将参数转换为无约束形式，再施以高斯变分近似，并利用自动微分技术推导变换变量密度所需的雅可比项（详见第10.2.2节）。

现将ADVI应用于我们的一维Beta-伯努利模型：令 $\theta=\sigma(z)$，将 $p(\theta\mid\mathcal{D})$ 替换为 $q(z\mid\boldsymbol{\psi})=\mathcal{N}(z\mid\mu,\sigma)$，其中 $\boldsymbol{\psi}=(\mu,\sigma)$。通过随机梯度下降优化ELBO的随机近似，结果如图7.4所示，其近似效果较为合理。

![image-20251008195357471](/assets/img/figures/book2/7.5.png)

### 7.4.5 马尔可夫链蒙特卡洛（MCMC）

尽管变分推断速度较快，但由于其被限制在特定的函数形式 $ q \in \mathcal{Q} $ 中，可能会对后验分布产生有偏近似。一种更灵活的方法是使用基于样本集的非参数近似：$ q(\boldsymbol{\theta}) \approx \frac{1}{S} \sum_{s=1}^{S} \delta(\boldsymbol{\theta} - \boldsymbol{\theta}^s) $，这被称为**蒙特卡洛近似**（Monte Carlo approximation）。其中的关键问题在于如何高效生成后验样本 $ \boldsymbol{\theta}^s \sim p(\boldsymbol{\theta}\mid\mathcal{D}) $，而无需计算归一化常数 $p(\mathcal{D})=\int p(\boldsymbol{\theta}, \mathcal{D}) d \boldsymbol{\theta}$。

针对低维场景，我们可以使用**重要性采样**（importance sampling）等方法（第11.5节将详细讨论）。然而对于高维问题，更常用的方法是**马尔可夫链蒙特卡洛**（Markov chain Monte Carlo，MCMC）。我们将在第12章详细展开，此处先作简要介绍。

最常用的MCMC方法是**Metropolis-Hastings算法**。其基本思想是：从参数空间的随机点出发，通过从**提议分布**（proposal distribution） $q(\boldsymbol{\theta}^{\prime} \mid \boldsymbol{\theta})$ 中采样新状态（参数）来执行随机游走。若 $ q $ 经过恰当选择，所得马尔可夫链的平稳分布将满足：在空间中访问到每个点的时间占比与该点的后验概率成正比。

其核心要点在于：决定是转移到新提议点 $\boldsymbol{\theta}'$ 还是停留在当前点 $\boldsymbol{\theta}$ 时，我们仅需计算未归一化的密度比：

$$
\frac{p(\boldsymbol{\theta} \mid \mathcal{D})}{p(\boldsymbol{\theta}^{\prime} \mid \mathcal{D})}=\frac{p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta}) / p(\mathcal{D})}{p(\mathcal{D} \mid \boldsymbol{\theta}^{\prime}) p(\boldsymbol{\theta}^{\prime}) / p(\mathcal{D})}=\frac{p(\mathcal{D}, \boldsymbol{\theta})}{p(\mathcal{D}, \boldsymbol{\theta}^{\prime})} \tag{7.35}
$$

这种方法避免了计算归一化常数 $ p(\mathcal{D}) $ 的需求（实践中通常使用对数概率替代联合概率以避免数值问题）。

可见，该算法的输入仅需一个计算对数联合密度 $\log p(\boldsymbol{\theta}, \mathcal{D})$ 的函数，以及一个用于决定下一步状态转移的提议分布 $q(\boldsymbol{\theta}^{\prime} \mid \boldsymbol{\theta})$。通常采用高斯分布作为提议分布 $q(\boldsymbol{\theta}^{\prime} \mid \boldsymbol{\theta})=\mathcal{N}(\boldsymbol{\theta}^{\prime} \mid \boldsymbol{\theta}, \sigma \mathbf{I})$，这被称为**随机游走Metropolis算法**（random walk Metropolis）。但该方法效率可能较低，因为其本质是在参数空间中盲目游走以寻找高概率区域。

对于具有条件独立结构的模型，通常可以逐个变量地计算其全条件分布 $p(\boldsymbol{\theta}_d \mid \boldsymbol{\theta}_{-d}, \mathcal{D})$并进行采样。这类似于坐标上升法的随机版本，被称为**吉布斯采样**（Gibbs sampling）（详见第12.3节）。

当所有未知变量均为连续变量时，我们通常可以计算对数联合密度的梯度 $\nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}, \mathcal{D})$。利用该梯度信息可引导提议分布向更高概率区域移动，该方法被称为**哈密尔顿蒙特卡洛**（Hamiltonian Monte Carlo, HMC）。由于其高效性，HMC已成为目前最广泛使用的MCMC算法之一（详见第12.5节）。

我们将HMC应用于Beta-伯努利模型的结果如图7.5所示（对参数进行了logit变换）。图(b)展示了4条并行马尔可夫链生成的样本，可见它们如预期般在真实后验分布附近波动。图(a)通过各链后验样本的核密度估计显示，其结果与图7.3的真实后验分布高度吻合。

### 7.4.6 序贯蒙特卡洛

MCMC类似于一种随机局部搜索算法，它通过在后验分布的状态空间中移动，不断比较当前值与邻近提议值。另一种方法则是使用一系列从简单到复杂的分布序列进行推断，最终分布即为目标后验分布，这被称为**序贯蒙特卡洛**（sequential Monte Carlo，SMC）。该方法更类似于树搜索而非局部搜索，相对MCMC具有多种优势（第13章将详细讨论）。

SMC的典型应用场景是**序贯贝叶斯推断**（sequential Bayesian inference），即以前馈方式递归计算后验分布 $p(\boldsymbol{\theta}_t \mid \mathcal{D}_{1: t})$，其中 $\mathcal{D}_{1: t}=\{(\boldsymbol{x}_n, y_n): n=1: t\}$ 表示截至当前时刻观测到的所有数据。该分布序列在接收全部数据后将收敛于完整批处理后验 $ p(\boldsymbol{\theta}\mid\mathcal{D}) $。此外，该方法同样适用于数据持续不断到达的场景（如状态空间模型，参见第29章）。SMC在此类动态模型中的应用被称为**粒子滤波**（particle filtering），具体原理详见第13.2节。

![image-20251008200334679](/assets/img/figures/book2/7.6.png)

### 7.4.7 具有挑战性的后验分布

在许多应用场景中，后验分布可能呈现高维且多峰的特性。对此类分布进行近似逼近具有较大挑战性。图7.6展示了一个简单的二维示例：我们比较了MAP估计（不捕获任何不确定性）、高斯参数化近似（如拉普拉斯近似或变分推断，见图b），以及基于样本的非参数化近似。若样本通过MCMC生成，则样本间存在序列相关性且可能仅探索局部模态（见图c）；但理想情况下，我们应能从分布的整个支撑集独立采样（见图d）。此外，我们还可以在每个样本点附近拟合局部参数化近似（参见第17.3.9.1节），从而获得后验的半参数化近似。

## 7.5 评估近似推断算法

现有多种不同的近似推断算法，它们在速度、精度、通用性、简洁性等维度各有权衡，这使得公平比较变得困难。

一种评估思路是通过与“真实”后验 $ p(\boldsymbol{\theta}\mid\mathcal{D}) $（通过离线“精确”方法计算）对比来衡量近似分布 $ q(\boldsymbol{\theta}) $ 的精度。我们通常关注精度与速度的权衡关系，可通过计算 $D_{\mathbb{KL}}(p(\boldsymbol{\theta} \mid \mathcal{D}) \\mid q_t(\boldsymbol{\theta}))$ 来量化（其中 $ q_t(\boldsymbol{\theta}) $ 表示经过 $ t $ 单位计算时间后的近似后验）。当然，也可采用其他分布相似性度量指标，如瓦瑟斯坦距离。

然而，真实后验 $ p(\boldsymbol{\theta}\mid\mathcal{D}) $ 通常无法计算。一种简单的替代方案是通过模型在未观测样本数据上的预测能力进行评估（类似于交叉验证）。更一般性地，如[KPS98; KPS99]所提出，我们可以比较不同后验分布的期望损失或贝叶斯风险（第34.1.3节）：

$$
R=\mathbb{E}_{p^*(\boldsymbol{x}, \boldsymbol{y})}[\ell(\boldsymbol{y}, q(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D}))] \text { where } q(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D})=\int p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}) q(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta} \tag{7.36}
$$

其中 $\ell(\boldsymbol{y}, q(\boldsymbol{y}))$ 为某种损失函数（例如对数损失）。或者，我们也可以按照[Far22]的建议，通过后验分布在特定下游任务（如持续学习或主动学习）中的表现来衡量其性能。

关于变分推断的专项评估方法可参阅[Yao+18b; Hug+20]，蒙特卡洛方法的评估准则参见[CGR06; CTM17; GAR16]。
