---
title: 21 变分自编码器(draft)
author: fengliang qi
date: 2023-07-01 11:33:00 +0800
categories: [BOOK-2, PART-IV]
tags: [vae, generative model]
math: true
mermaid: true
toc: true
comments: true

---

> 本章，我们介绍VAE
>

* TOC
{:toc}
## 21.1 简介

本章我们将讨论形式如下的生成模型：


$$
\begin{align}
\boldsymbol{z} & \sim p_{\boldsymbol{\theta}}(\boldsymbol{z}) \tag{21.1}\\
\boldsymbol{x} \mid \boldsymbol{z} & \sim \operatorname{Expfam}\left(x \mid d_{\boldsymbol{\theta}}(\boldsymbol{z})\right) \tag{21.2}
\end{align}
$$


其中 $p(\boldsymbol{z})$  表示隐变量 $\boldsymbol{z}$ 服从的某种先验分布，$d_{\boldsymbol{\theta}}(\boldsymbol{z})$ 表示深度神经网络，又被称为 **解码器**（decoder），$\operatorname{Expfam}(\boldsymbol{x} \mid \boldsymbol{\eta})$ 表示指数族分布——比如高斯分布或者伯努利分布的乘积，上述模型被称为 **深度隐变量模型**（deep latent variable model, DLVM）。如果先验分布属于常见的高斯分布，模型又被称为 **深度隐高斯模型**（deep latent Gaussian model, DLGM）。

该模型在进行后验推理时（即计算 $p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})$）存在计算层面的困难，因为需要计算边际似然


$$
p_{\boldsymbol{\theta}}(\boldsymbol{x})=\int p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{z}) d \boldsymbol{z} \tag{21.3}
$$


所以我们需要借助于近似推理的方法。在本章的大部分，我们将使用在10.3.6节讨论的**平摊推理** （amortized inference）。在该方法中，我们训练另一个模型 $q_\boldsymbol{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$ ——**识别网络** （recognition network） 或者 **推理网络** （inference network），同时配以一个生成模型来实现近似的后验推理。这种组合被称为**变分自编码器** （variational auto encoder, VAE），因为它可以被当作确定式自编码器（16.3.3节）的概率式版本。

本章，我们将介绍VAE的基本知识以及相关的拓展。需要注意的是，关于VAE的文献很多，我们只会讨论其中的一部分内容。

## 21.2 VAE基础

本节，我们将讨论变分自编码器的基础知识。

### 21.2.1 模型假设

在最基础的设定中，VAE定义了一个形如下式的生成模型


$$
p_{\boldsymbol{\theta}}(\boldsymbol{z}, \boldsymbol{x})=p_{\boldsymbol{\theta}}(\boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) \tag{21.4}
$$


其中 $p_{\boldsymbol{\theta}}(\boldsymbol{z})$ 通常为高斯分布，$p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})$ 通常为指数族分布（如高斯分布或伯努利分布的连乘），分布的参数由一个神经网络解码器 $d_{\boldsymbol{\theta}}(\boldsymbol{z})$ 决定。比如，如果观测值由多个二值变量组成（如手写数字图片），我们可以使用


$$
p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})=\prod_{d=1}^D \operatorname{Ber}\left(x_d \mid \sigma\left(d_{\boldsymbol{\theta}}(\boldsymbol{z})\right)\right. \tag{21.5}
$$


同时，VAE需要拟合一个识别网络


$$
q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})=q\left(\boldsymbol{z} \mid e_{\boldsymbol{\phi}}(\boldsymbol{x})\right) \approx p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}) \tag{21.6}
$$


来实现近似的后验推理。此处 $q_\boldsymbol{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$ 通常是一个高斯分布，该分布的参数由一个神经网络编码器 $e_\phi(\boldsymbol{x})$ 决定：


$$
\begin{align}
q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) & =\mathcal{N}(\boldsymbol{z} \mid \boldsymbol{\mu}, \operatorname{diag}(\exp (\boldsymbol{\ell}))) \tag{21.7} \\
(\boldsymbol{\mu}, \boldsymbol{\ell}) & =e_{\boldsymbol{\phi}}(\boldsymbol{x}) \tag{21.8}
\end{align}
$$


其中$\boldsymbol{\ell}=\log \boldsymbol{\sigma}$。该模型可以被认为是将输入 $\boldsymbol{x}$ 编码至随机隐变量 $\boldsymbol{z}$ , 再解码至输入的近似解，如图21.1 所示。

![21.1](/assets/img/figures/book2/21.1.png)

{: style="width: 100%;" class="center"}
图21.1: VAE的示意图。取自[Haf18][^Haf18]中的一幅图。获得了Danijar Hafner的友善许可后使用。
{:.image-caption}

训练一个推理网络来“反转”生成网络，而不是使用优化算法来推理隐变量的方法被称为**平摊推理**（amortized inference），我们在第10.3.6节中进行过讨论。这个方法首先在 **Helmholtz machine** 中被提出。然而，当时的文章并没有提出一个单独的统一目标函数来同时实现推理和生成，而是使用了 wake-sleep （10.6节）方法进行训练。相反，VAE 优化了对数似然的变分下确界，这意味着模型将确保收敛到一个局部最优MLE（极大似然估计）。

我们可以使用其他方法来拟合**深度隐高斯模型** (DLGM)。然而，通过学习一个推理网络来拟合DLGM通常更快，并且有一些正则化的收益。

### 21.2.2 模型拟合

我们可以使用平摊随机变分推理来拟合一个VAE，正如我们在第10.2.1.6节中所讨论的那样。 例如，假设我们使用一个VAE，其似然函数为对角伯努利分布，变分后验为完全协方差高斯分布。然后我们可以使用第10.2.1.2节中讨论的方法来推导出如下的训练算法。请参见算法21.1以获取相应的伪代码。

![21.1.5](/assets/img/figures/book2/21.1.5.png)

{: style="width: 100%;" class="center"}
{:.image-caption}

### 21.2.3 VAE和自编码器的对比

VAE 与确定性自编码器（AE）非常相似。存在两点主要区别：在AE中，目标函数只有关于重构的对数似然，而没有KL散度；此外，AE的编码器部分是确定式的，所以编码器只需要计算 $\mathbb{E}[\boldsymbol{z} \mid \boldsymbol{x}]$，而不需要计算 $\mathbb{V}[\boldsymbol{z} \mid \boldsymbol{x}]$。鉴于这些相似之处，可以使用相同的基础代码来实现这两种方法。那么相对于确定性自编器而言，VAE 的优点和潜在缺点是什么呢？

为了回答这个问题，我们将使用两种模型拟合 CelebA 数据集。两种模型具有相同的卷积结构，编码器中每个卷积层的隐藏通道数为: (32、64、128、256、512)。每层的分辨率为: (32、16、8、4、2)。最终的2x2x512特征图被reshape并通过线性层，生成随机隐向量的均值和 (边际) 方差，其大小为 256。解码器的结构是编码器的镜像。每个模型使用批次大小为 256 进行5个周期的训练，在GPU上大约需要20分钟。

VAE 相对于确定性自编码器的主要优点是它定义了一个合适的生成模型，可以通过解码先验样本 $\boldsymbol{z} \sim \mathcal{N}(0, \mathbf{I})$ 创建合理的新图像。相比之下，自编码器只知道如何解码从训练集派生的隐编码，因此在提供随机输入时表现不佳。这在图 21.2 中有所体现。

上述两个模型也可以用来重建给定的输入图像。在图 21.3 中，我们发现 AE 和 VAE 都可以合理地重建输入图像，尽管 VAE 的重建结果有些模糊，原因将在第 21.3.1 节中讨论。为了减少模糊度，可以使用权重 $\beta$ 控制 KL 惩罚项的重要性，这被称为 $\beta$-VAE，在第 21.3.1 节中将进行更详细的讨论。

![21.2](/assets/img/figures/book2/21.2.png)

{: style="width: 100%;" class="center"}

图21.2：使用在CelebA上训练的(V)AE来进行无条件图像生成的示意图。第一行：确定性自编码器。第二行：$\beta$-VAE，其中 $\beta$= 0.5。第三行：VAE（其中 $\beta$= 1）。由celeba_vae_ae_comparison.ipynb生成。

{:.image-caption}



![21.3](/assets/img/figures/book2/21.3.png)

{: style="width: 100%;" class="center"}

图21.3: 使用训练并应用于CelebA的（V）AE进行图像重构的示例。第一行：原始图像。第二行：确定性自编码器。第三行：具有$\beta$= 0.5的VAE。第四行：具有$\beta$ = 1的VAE。由celeba_vae_ae_comparison.ipynb生成。

{:.image-caption}

### 21.2.4* VAE 在一个增强空间中进行优化

本节，我们将推导几种ELBO表达式的变体，从而揭示 VAEs 的工作原理。

首先，我们需要定义一个联合*生成* 分布


$$
p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})=p_{\boldsymbol{\theta}}(\boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) \tag{21.9}
$$


从中我们可以得到关于生成数据的边际分布


$$
p_{\boldsymbol{\theta}}(\boldsymbol{x})=\int_{\boldsymbol{z}} p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) d \boldsymbol{z} \tag{21.10}
$$


以及关于生成数据的后验分布


$$
p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})=p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) / p_{\boldsymbol{\theta}}(\boldsymbol{x}) \tag{21.11}
$$


同时我们定义联合 *推理* 分布


$$
q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{z}, \boldsymbol{x})=p_{\mathcal{D}}(\boldsymbol{x}) q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \tag{21.12}
$$


其中


$$
p_{\mathcal{D}}(\boldsymbol{x})=\frac{1}{N} \sum_{n=1}^N \delta\left(\boldsymbol{x}_n-\boldsymbol{x}\right) \tag{21.13}
$$


为经验分布。从中我们可以推导出关于隐变量的边际分布，又被称为 **聚合后验** （aggregated posterior）：


$$
q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{z})=\int_{\boldsymbol{x}} q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z}) d \boldsymbol{x} \tag{21.14}
$$


以及推理似然函数


$$
q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{x} \mid \boldsymbol{z})=q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z}) / q_{\mathcal{D}, \phi}(\boldsymbol{z}) \tag{21.15}
$$


图21.4给出了示意图。

基于上述定义，我们可以推导出关于 ELBO 的不同变种，参考[ZSE19][^ZSE19]。首先注意，在全量数据上的 ELBO 平均值为：


$$
\begin{align}
\mathfrak{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}} & =\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]\right]-\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[D_{\mathbb{K} \mathbb{L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \| \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)\right] \tag{21.16}\\
& =\mathbb{E}_{q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})+\log p_{\boldsymbol{\theta}}(\boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right] \tag{21.17}\\
& =\mathbb{E}_{q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z})}\left[\log \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})}{q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z})}+\log p_{\mathcal{D}}(\boldsymbol{x})\right] \tag{21.18}\\
& =-D_{\mathbb{K L}}\left(q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right)+\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[\log p_{\mathcal{D}}(\boldsymbol{x})\right] \tag{21.19}
\end{align}
$$


如果我们定义 $\stackrel{c}{=}$ 为相等（上述式子）加上任意常数，我们可以将上述式子重写为：


$$
\begin{align}
\mathbb{E}_{\boldsymbol{\theta}, \boldsymbol{\phi}} & \stackrel{c}{=}-D_{\mathrm{KL}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right) \tag{21.20}\\
& \stackrel{c}{=}-D_{\mathrm{KL}}\left(p_{\mathcal{D}}(\boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x})\right)-\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})\right)\right] \tag{21.21}
\end{align}
$$


因此，最大化 ELBO 需要最小化两个 KL 项。第一个 KL 项通过 MLE 实现最小化，第二个 KL 项通过拟合真实后验分布实现最小化。因此，如果近似后验分布的解空间有限，这些优化目标之间可能存在冲突。

最后，我们注意到 ELBO 可以写成
$$
\mathfrak{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}} \stackrel{c}{=}-D_{\mathrm{KL}}\left(q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)-\mathbb{E}_{q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{z})}\left[D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{x} \mid \boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right)\right] \tag{21.22}
$$
从方程（21.59）可以看出，VAE 试图最小化推理边际和生成先验之间的差异 $D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)$，同时最小化重构误差 $D_{\mathbb{K} \mathbb{L}}\left(q_\phi(\boldsymbol{x}|z|) \| p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right)$。由于 $\boldsymbol{x}$ 的维度通常比 $\boldsymbol{z}$ 高得多，后者通常占主导地位。因此，如果这两个目标之间存在冲突 （例如，由于拟合能力有限），VAE 将优先考虑最小化重构误差而不是后验推断的准确性。因此，学习到的后验分布可能不是真实后验的很好逼近（更多讨论请参见[ZSE19][^ZSE19]）

![21.4](/assets/img/figures/book2/21.4.png)

{: style="width: 100%;" class="center"}

图21.4: 最大似然（ML）目标可以看作是最小化 $D_{\mathbb{K} L}\left(p_{\mathcal{D}}(\boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x})\right)$。（注意：在图中，$p_{\mathcal{D}}(\boldsymbol{x})$ 以 $q_D(\boldsymbol{x})$ 表示。）ELBO目标是最小化 $D_{\mathbb{K L}}\left(q_{\mathcal{D}, \phi}(\boldsymbol{x}, \boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right)$，它是 $D_{\mathbb{K L}}\left(q_{\mathcal{D}}(\boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x})\right)$的上确界。来源于[KW19a][^KW19a]的图2.4。在Durk Kingma的友善许可下使用。

{:.image-caption}

## 21.3 VAE的推广

本节，我们将讨论基础VAE模型的几种变体。

### 21.3.1 $\beta$-VAE

通常情况下，VAE 生成的图像有些模糊，如图 21.3、图 21.2 和图 20.9 所示。而对于那些优化目标为精确似然函数的模型（例如 pixelCNN（第 22.3.2 节）和流模型（第 23 章）），则不会出现这种情况。要了解为什么 VAE 不同，请考虑解码器是具有固定方差的高斯分布的情况，此时：


$$
\log p_{\boldsymbol{\theta}}(x \mid z)=-\frac{1}{2 \sigma^2}\left\|x-d_{\boldsymbol{\theta}}(z)\right\|_2^2+\text { const } \tag{21.23}
$$


令 $e_{\boldsymbol{\phi}}(\boldsymbol{x})=\mathbb{E}\left[q_\phi(\boldsymbol{z} \mid \boldsymbol{x})\right]$ 为输入 $\boldsymbol{x}$ 的编码函数，$$\mathcal{X}(\boldsymbol{z})=\left\{\boldsymbol{x}: e_\phi(\boldsymbol{x})=\boldsymbol{z}\right\}$$ 表示映射到同一个隐变量 $$\boldsymbol{z}$$ 的输入集合。对于参数冻结的推理网络，当使用平方重构损失时，生成器的最优解是确保 $$d_\theta(\boldsymbol{z})=\mathbb{E}[\boldsymbol{x}: \boldsymbol{x} \in \mathcal{X}(\boldsymbol{z})]$$。因此，解码器应该预测映射到该 $\boldsymbol{z}$ 的所有输入 $\boldsymbol{x}$ 的平均值，即模糊的图像。

我们可以通过增加后验分布的表达能力（避免将不同的输入映射到相同的隐变量中）或生成器的表达能力（通过添加缺失的信息）或两者同时来解决这个问题。然而，更简单的解决方案是减少 KL 项的惩罚强度，使模型更接近确定性自编码器：


$$
\mathcal{L}_\beta(\boldsymbol{\theta}, \boldsymbol{\phi} \mid \boldsymbol{x})=\underbrace{-\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]}_{\mathcal{L}_E}+\beta \underbrace{D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)}_{\mathcal{L}_R} \tag{21.24}
$$


其中 $\mathcal{L}_E$ 表示重构误差（负对数似然），$\mathcal{L}_R$ 表示 $\text{KL}$ 正则。这被称为 $\beta$-$\mathbf{VAE}$ 目标[Hig+17a][^Hig+17a]。如果令 $\beta=1$，我们恢复了标准 VAE 中使用的目标函数；如果我们设置$\beta=0$，我们恢复了标准自编码器中使用的目标函数。

通过将 $\beta$ 从 0 变化到正无穷，我们可以在失真率曲线上到达不同的点，如第 5.4.2 节中所讨论的。这些点在重构误差（失真）和隐变量中存储的关于输入的信息量（相应编码的比特率）之间做出不同的权衡。通过使用 $\beta\lt1$，我们可以存储更多关于每个输入的信息位，因此重构图像更清晰。如果我们使用 $\beta\gt1$，则可以获得更紧凑的隐变量。

#### 21.3.1.1 解耦表征

使用 $\beta\gt1$ 的一个优点是，它鼓励编码器学习一个“解耦”的隐变量。直观地讲，这意味着每个隐变量的维度代表输入中的不同变化因素。这通常是通过总相关性（第 5.3.5.1 节）来形式化的，其定义如下：


$$
\mathrm{TC}(\boldsymbol{z})=\sum_k \mathbb{H}\left(z_k\right)-\mathbb{H}(\boldsymbol{z})=D_{\mathbb{K} \mathbb{L}}\left(p(\boldsymbol{z}) \| \prod_k p_k\left(z_k\right)\right) \tag{21.25}
$$


当且仅当 $\boldsymbol{z}$ 的分量相互独立且解耦时，这个值为零。 [AS18][^AS18] 证明使用 $\beta\gt1$ 会降低 TC。

不幸的是，在 [Loc+18][^Loc+18] 中，他们证明非线性隐变量模型是不可识别的，因此对于任意的解耦表征，都存在一个具有完全相同似然的等效的完全缠结的表征。因此，仅仅通过调整 $\beta$ 是不足以恢复正确的潜在表征的，必须通过编码器、解码器、先验、数据集或学习算法选择合适的归纳偏差。详见第 32.4.1 节。

#### 21.3.1.2 与信息瓶颈的联系

在本节中，我们将展示 $\beta$-$\mathbf{VAE}$ 是信息瓶颈（IB）目标函数（第 5.6 节）的无监督版本。如果输入是 $\boldsymbol{x}$，瓶颈是 $\boldsymbol{z}$，目标输出为 $\tilde{\boldsymbol{x}}$，则无监督 IB 目标函数变为：


$$
\begin{align}
\mathcal{L}_{\mathrm{UIB}} & =\beta \mathbb{I}(\boldsymbol{z} ; \boldsymbol{x})-\mathbb{I}(\boldsymbol{z} ; \tilde{\boldsymbol{x}}) \tag{21.26}\\
& =\beta \mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{z})}\left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{x}) p(\boldsymbol{z})}\right]-\mathbb{E}_{p(\boldsymbol{z}, \tilde{\boldsymbol{x}})}\left[\log \frac{p(\boldsymbol{z}, \tilde{\boldsymbol{x}})}{p(\boldsymbol{z}) p(\tilde{\boldsymbol{x}})}\right] \tag{21.27}
\end{align}
$$


其中


$$
\begin{align}
& p(\boldsymbol{x}, \boldsymbol{z})=p_{\mathcal{D}}(\boldsymbol{x}) p(\boldsymbol{z} \mid \boldsymbol{x}) \tag{21.28}\\
& p(\boldsymbol{z}, \tilde{\boldsymbol{x}})=\int p_{\mathcal{D}}(\boldsymbol{x}) p(\boldsymbol{z} \mid \boldsymbol{x}) p(\tilde{\boldsymbol{x}} \mid \boldsymbol{z}) d \boldsymbol{x} \tag{21.29}
\end{align}
$$


直观地说，方程式（21.63）中的目标意味着我们应该选择一个能够可靠预测 $\tilde{\boldsymbol{x}}$ 的表示 $\boldsymbol{z}$，同时不要过多地记忆输入 $\boldsymbol{x}$ 的信息。这个权衡参数由 $\beta$ 控制。

从方程式（5.180）中，我们得到了这个无监督目标函数的以下变分上界：


$$
\mathcal{L}_{\mathrm{UVIB}}=-\mathbb{E}_{q_{\mathcal{D}, \boldsymbol{\phi}}(\boldsymbol{z}, \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(x \mid \boldsymbol{z})\right]+\beta \mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)\right] \tag{21.30}
$$


当对 $\boldsymbol{x}$ 取平均值时，这个公式与式子（21.24）相匹配。

### 21.3.2 InfoVAE

在第 21.2.4 节中，我们讨论了用于训练 VAE 的标准 ELBO 目标函数的一些缺点，即当解码器拟合能力强大时导致隐向量学习不足的倾向（第 21.4 节），以及由于数据空间和隐空间中 KL 项之间不匹配而学习到较差的后验近似的倾向（第 21.2.4 节）。我们可以通过使用以下形式的广义目标函数在某种程度上解决这些问题：


$$
\mathrm{Ł}(\boldsymbol{\theta}, \phi \mid x)=-\lambda D_{\mathbb{K} L}\left(q_\phi(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)-\mathbb{E}_{q_\phi(\boldsymbol{z})}\left[D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{x} \mid \boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right)\right]+\alpha \mathbb{I}_q(\boldsymbol{x} ; \boldsymbol{z}) \tag{21.31}
$$


其中，$\alpha \geq 0$ 控制 $\boldsymbol{x}$ 和 $\boldsymbol{z}$ 之间的互信息 $\mathbb{I}_q(\boldsymbol{x} ; \boldsymbol{z})$ 的权重，  $\lambda \geq 0$ 控制 $\boldsymbol{z}$ 空间 KL 和 $\boldsymbol{x}$ 空间 KL 之间的权衡。这被称为 InfoVAE 目标函数 [ZSE19][^ZSE19]。如果我们设置  $\alpha=0$ 和 $\lambda=1$，我们将得到标准 ELBO，如式（21.59）所示。

不幸的是，式子（21.31）中的目标函数无法按照给定的方式计算，因为互信息项是棘手的：


$$
\mathbb{I}_q(\boldsymbol{x} ; \boldsymbol{z})=\mathbb{E}_{q_\phi(\boldsymbol{x}, \boldsymbol{z})}\left[\log \frac{q_\phi(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{x}) q_\phi(\boldsymbol{z})}\right]=-\mathbb{E}_{q_\phi(\boldsymbol{x}, \boldsymbol{z})}\left[\log \frac{q_\phi(\boldsymbol{z})}{q_\phi (\boldsymbol{z} \mid \boldsymbol{x})}\right] \tag{21.32}
$$


然而，基于 $q_\phi(\boldsymbol{x} \mid \boldsymbol{z})=p_{\mathcal{D}}(\boldsymbol{x}) q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) / q_\phi(\boldsymbol{z})$ 的事实，我们可以将目标重写成：


$$
\begin{align}
\mathrm{Ł} & =\mathbb{E}_{q_\phi(\boldsymbol{x}, \boldsymbol{z})}\left[-\lambda \log \frac{q_{\boldsymbol{\phi}}(\boldsymbol{z})}{p_{\boldsymbol{\theta}}(\boldsymbol{z})}-\log \frac{q_\phi(\boldsymbol{x} \mid \boldsymbol{z})}{p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})}-\alpha \log \frac{q_{\boldsymbol{\phi}}(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\right] \tag{21.33}\\
& =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{z})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})-\log \frac{q_{\boldsymbol{\phi}}(\boldsymbol{z})^{\lambda+\alpha-1} p_{\mathcal{D}}(\boldsymbol{x})}{p_{\boldsymbol{\theta}}(\boldsymbol{z})^\lambda q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})^{\alpha-1}}\right] \tag{21.34}\\
& =\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]\right]-(1-\alpha) \mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[D_{\mathrm{KL}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)\right] \tag{21.35}\\
& -(\alpha+\lambda-1) D_{\mathrm{KL}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)-\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[\log p_{\mathcal{D}}(\boldsymbol{x})\right] \tag{21.35}
\end{align}
$$


其中最后的常数项可以忽略。可以使用重新参数化技巧优化前两项。不幸的是，最后一项需要计算 $q_\phi(z)=\int_x q_\phi(x, z) d x$，这是不可计算的。幸运的是，我们可以轻松地从这个分布中采样，通过从 $p_{\mathcal{D}}(\boldsymbol{x})$ 中采样 $\boldsymbol{x}$ 和从 $q_\phi(z \mid x)$ 中采样 $\boldsymbol{z}$。因此，$q_\phi(z)$ 是一个**隐式概率模型**（implicit probability model），类似于 GAN（参见第 26 章）。

只要我们使用严格的散度，即 $D(q, p)=0$ 当且仅当 $q=p$，那么可以证明这不会影响程序的最优性。特别是，[ZSE19][^ZSE19] 的命题2告诉我们以下内容：

定理1。假设  $\mathcal{X}$ 和 $\mathcal{Z}$ 是连续的空间，且 $\alpha<1$（用于限制互信息的上界），并且  $\lambda>0$。对于任何固定的 $\mathbb{I}_q(x ; z)$ 值，采用任何严格散度 $$D\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}), p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)$$ 的近似 InfoVAE 损失，在 $$p_{\boldsymbol{\theta}}(\boldsymbol{x})=p_{\mathcal{D}}(\boldsymbol{x})$$ 且 $$q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})=p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})$$ 的情况下进行全局优化。

#### 21.3.2.1 与 MMD VAE 的联系

如果设置 $\alpha=1$，则 InfoVAE 的优化目标简化为


$$
\mathrm{Ł} \stackrel{c}{=} \mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]\right]-\lambda D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right) \tag{21.36}
$$


MMD VAE3 将上述项中的 KL 散度替换为在第 2.7.3 节中定义的（平方的）最大均值差异或 MMD 散度。（根据上述定理，这是有效的。）这种方法相对于标准 InfoVAE 的优势在于所得到的目标是可计算的。特别地，如果我们设置 $\lambda=1$ 并改变符号，我们得到：


$$
\mathcal{L}=\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\left[\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[-\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]\right]+ \operatorname{MMD}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}), p_{\boldsymbol{\theta}}(\boldsymbol{z})\right) \tag{21.37}
$$


如我们在第 2.7.3 节中讨论的那样，可以按如下方式计算 MMD：


$$
\operatorname{MMD}(p, q)=\mathbb{E}_{p(\boldsymbol{z}), p\left(\boldsymbol{z}^{\prime}\right)}\left[\mathcal{K}\left(\boldsymbol{z}, \boldsymbol{z}^{\prime}\right)\right]+\mathbb{E}_{q(\boldsymbol{z}), q\left(\boldsymbol{z}^{\prime}\right)}\left[\mathcal{K}\left(\boldsymbol{z}, \boldsymbol{z}^{\prime}\right)\right]-2 \mathbb{E}_{p(\boldsymbol{z}), q\left(\boldsymbol{z}^{\prime}\right)}\left[\mathcal{K}\left(\boldsymbol{z}, \boldsymbol{z}^{\prime}\right)\right] \tag{21.38}
$$


其中，$\mathcal{K}()$ 是某个核函数，如 RBF 核函数，$\mathcal{K}\left(z, z^{\prime}\right)=\exp \left(-\frac{1}{2 \sigma^2}\left\|z-z^{\prime}\right\|_2^2\right)$。直观地说，MMD 衡量了来自先验分布和聚合后验分布的样本（在潜在空间中）的相似性。

在实践中，我们可以通过使用当前小批量中所有 $B$ 个样本的后验预测均值 $$\boldsymbol{z}_n=e_\boldsymbol{\phi}(\boldsymbol{x}_n)$$，并将其与来自 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 先验的 $B$ 个随机样本进行比较，来实现 MMD 目标。

如果我们使用一个固定方差的高斯解码器，则负对数似然仅为一个平方误差项：


$$
-\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})=\left\|\boldsymbol{x}-d_{\boldsymbol{\theta}}(\boldsymbol{z})\right\|_2^2 \tag{21.39}
$$


因此，整个模型是确定性的，只是在潜在空间和可见空间中预测均值。

#### 21.3.2.2 与 $\beta$-VAEs 的关系

如果我们设置 $\alpha=0$ 和 $\lambda=1$，则会恢复原始的ELBO。如果自由选择 $\lambda>0$，但是使用 $\alpha=1-\lambda$，则会得到$\beta$-VAE。

#### 21.3.2.3 与对抗自编码器的关系

如果我们设置$\alpha=1$ 和 $\lambda=1$，并且选择$D$为Jensen-Shannon散度（可以通过训练二元鉴别器来最小化，如第26.2.2节所解释），那么我们得到了一个称为**对抗自编码器**（adversarial autoencoder）的模型[Mak+15a]。

![21.5](/assets/img/figures/book2/21.5.png)

{: style="width: 100%;" class="center"}

图21.5: 

{:.image-caption}

### 21.3.3 多模态 VAEs

可以将 VAE 扩展到创建不同类型变量（如图像和文本）的联合分布，这有时被称为**多模态VAE**（multimodal VAE, MVAE）。假设有$M$个模态。我们假设它们在给定潜在编码的条件下是条件独立的，因此生成模型的形式如下：


$$
p_{\boldsymbol{\theta}}\left(x_1, \ldots, x_M, z\right)=p(z) \prod_{m=1}^M p_{\boldsymbol{\theta}}\left(x_m \mid z\right) \tag{21.40}
$$


其中 $p(z)$ 为一个固定的先验分布。图21.5(a)给出了示意图。

标准的 ELBO 为：


$$
\mathfrak{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{X})=\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \mathbf{X})}\left[\sum_m \log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_m \mid \boldsymbol{z}\right)\right]-D_{\mathbb{K} \mathbb{L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \mathbf{X}) \| p(\boldsymbol{z})\right) \tag{21.41}
$$


其中 $\mathbf{X}=\left(x_1, \ldots, x_M\right)$ 表示观测数据。然而不同的似然项 $p\left(x_m \mid z\right)$ 可能具备不同的取值范围（例如，对于像素，可以使用高斯概率密度函数，对于文本，可以使用分类概率质量函数），所以针对每个模态的似然函数，我们使用单独的权重项 $\lambda_m\ge0$。除此之外，令 $\beta \ge 0$ 可以控制 KL 正则项的强度。这样，我们便得到一个加权版本的 ELBO:


$$
\mathfrak{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{X})=\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \mathbf{X})}\left[\sum_m \lambda_m \log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_m \mid \boldsymbol{z}\right)\right]-\beta D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \mathbf{X}) \| p(\boldsymbol{z})\right) \tag{21.42}
$$


通常情况下，我们无法获得来自所有$M$个模态的大量配对（对齐）数据。例如，我们可能有很多图像（模态1）和很多文本（模态2），但是（图像，文本）配对非常少。因此，将损失函数推广到适应特征子集的边缘分布是很有用的。假设 $O_m=1$ 表示观察到模态$m$（即$\boldsymbol{x}_m$已知），如果模态$m$缺失或未观察到，则 $O_m=0$。令 $\mathbf{X}=\left\{x_m: O_m=1\right\}$ 表示可见特征。我们现在使用以下目标函数：


$$
\mathrm{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{X})=\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \mathbf{X})}\left[\sum_{m: O_m=1} \lambda_m \log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_m \mid \boldsymbol{z}\right)\right]-\beta D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \mathbf{X}) \| p(\boldsymbol{z})\right) \tag{21.43}
$$


关键问题是如何计算在给定不同特征子集时的后验概率 $q_\phi(z \mid \mathbf{X})$。一般来说，这可能很困难，因为推理网络是一个判别模型，它假设所有输入都是可用的。例如，如果它是在（图像，文本）配对上训练的，则如何仅针对图像计算后验概率 $$q_\phi\left(\boldsymbol{z} \mid \boldsymbol{x}_1\right)$$，或仅针对文本计算后验概率 $q_\phi\left(z \mid x_2\right)$（在VAE存在缺失输入时，这个问题通常会出现；我们将在第21.3.4节中讨论一般情况。）

幸运的是，在我们对模态之间的条件独立性假设的基础上，我们可以通过计算模型下的精确后验来计算给定一组输入的 $q_\phi(\boldsymbol{z} \mid \mathbf{X})$ 的最优形式，即：


$$
\begin{align}
p(\boldsymbol{z} \mid \mathbf{X}) & =\frac{p(\boldsymbol{z}) p\left(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_M \mid \boldsymbol{z}\right)}{p\left(x_1, \ldots, \boldsymbol{x}_M\right)}=\frac{p(\boldsymbol{z})}{p\left(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_M\right)} \prod_{m=1}^M p\left(\boldsymbol{x}_m \mid \boldsymbol{z}\right) \tag{21.44}\\
& =\frac{p(\boldsymbol{z})}{p\left(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_M\right)} \prod_{m=1}^M \frac{p\left(\boldsymbol{z} \mid \boldsymbol{x}_m\right) p\left(\boldsymbol{x}_m\right)}{p(\boldsymbol{z})} \tag{21.45}\\
& \propto p(\boldsymbol{z}) \prod_{m=1}^M \frac{p\left(\boldsymbol{z} \mid \boldsymbol{x}_m\right)}{p(\boldsymbol{z})} \approx p(\boldsymbol{z}) \prod_{m=1}^M \tilde{q}\left(\boldsymbol{z} \mid \boldsymbol{x}_m\right) \tag{21.46}
\end{align}
$$


这可以看作是多个专家模型的乘积（第24.1.1节），其中每个 $\tilde{q}\left(z \mid x_m\right)$ 对应第 $m$ 个模态的“专家”，而 $p(z)$ 是先验。我们可以通过修改对$m$的乘积来计算上述后验，用于我们具有数据的任何模态子集。如果我们对先验使用高斯分布 $p(z)=\mathcal{N}\left(z \mid \boldsymbol{\mu}_0, \boldsymbol{\Lambda}_0^{-1}\right)$ 和边缘后验比率 $\tilde{q}\left(\boldsymbol{z} \mid x_m\right)=\mathcal{N}\left(z \mid \mu_m, \boldsymbol{\Lambda}_m^{-1}\right)$ ，那么我们可以使用方程（2.154）的结果计算高斯分布的乘积：


$$
\prod_{m=0}^M \mathcal{N}\left(z \mid \mu_m, \Lambda_m^{-1}\right) \propto \mathcal{N}(z \mid \mu, \Sigma), \quad \boldsymbol{\Sigma}=\left(\sum_m \boldsymbol{\Lambda}_m\right)^{-1}, \quad \boldsymbol{\mu}=\boldsymbol{\Sigma}\left(\sum_m \boldsymbol{\Lambda}_m \boldsymbol{\mu}_m\right) \tag{21.47}
$$


因此，整体后验精度是各个专家后验精度的总和，整体后验均值是各个专家后验均值的精度加权平均值。请参见图21.5(b)进行说明。对于线性高斯（因子分析）模型，我们可以确保 $q\left(z \mid x_m\right)=p\left(z \mid x_m\right)$，在这种情况下，上述解是精确的后验分布[WN18]，但一般情况下它将是一个近似解。

我们需要训练各个专家识别模型 $q\left(z \mid x_m\right)$ 以及联合模型 $q(z \mid \mathbf{X})$，以便模型在测试时知道如何处理完全观察到的和部分观察到的输入。在[Ved+18]中，他们提出了一个相对复杂的“三重ELBO”目标。在[WG18]中，他们提出了更简单的方法，即针对完全观察到的特征向量、所有边缘分布和一组随机选择的联合模态优化ELBO：


$$
\mathrm{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{X})=\mathrm{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_M\right)+\sum_{m=1}^M \mathrm{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\boldsymbol{x}_m\right)+\sum_{j \in \mathcal{J}} \mathrm{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\mathbf{X}_j\right) \tag{21.48}
$$


这个方法很好地推广到半监督设置中，其中我们只有一些来自联合分布的对齐（“标记”）示例，但是有许多来自各个边缘分布的不对齐（“未标记”）示例。请参见图21.6(c)进行说明。 

请注意，上述方案只能处理固定数量的缺失模式；在第21.3.4节中，我们将推广以允许任意缺失情况。

### 21.3.4 半监督VAEs

在这一部分中，我们讨论如何将VAE扩展到半监督学习设置中，其中我们既有标记数据 $\mathcal{D}_L=\left\{\left(x_n, y_n\right)\right\}$，又有无标记数据 $\mathcal{D}_U=\left\{\left(x_n\right)\right\}$。我们专注于M2模型，该模型在[Kin+14a][^Kin+14a]中提出。

生成模型的形式为：


$$
p_{\boldsymbol{\theta}}(\boldsymbol{x}, y)=p_{\boldsymbol{\theta}}(y) p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid y)=p_{\boldsymbol{\theta}}(y) \int p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid y, \boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{z}) d \boldsymbol{z} \tag{21.49}
$$


其中 $z$ 表示某个隐变量，$p_{\boldsymbol{\theta}}(\boldsymbol{z})=\mathcal{N}(\boldsymbol{z} \mid \mathbf{0}, \mathbf{I})$ 为关于隐变量的先验分布，$p_{\boldsymbol{\theta}}(y)=\operatorname{Cat}(y \mid \pi)$ 表示关于标签的先验分布，$p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid y, \boldsymbol{z})=p\left(\boldsymbol{x} \mid f_{\boldsymbol{\theta}}(y, \boldsymbol{z})\right)$ 表示似然函数，比如高斯分布，其参数通过函数 $f$ 计算（即神经网路）。这种方法的主要创新是假设数据的生成既依赖于潜在类别变量 $y$，又依赖于连续潜在变量 $z$。对于有标注数据，类别变量 $y$ 是可观测的，而对于无标记数据，类别变量 $y$ 是未观测的。

为了计算有标注数据的似然值 $p_{\boldsymbol{\theta}}(\boldsymbol{x}, y)$，我们需要基于变量 $\boldsymbol{z}$ 求边际分布，我们可以使用如下形式的推理网络实现这一点：


$$
q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid y, \boldsymbol{x})=\mathcal{N}\left(\boldsymbol{z} \mid \boldsymbol{\mu}_{\boldsymbol{\phi}}(y, \boldsymbol{x}), \operatorname{diag}\left(\boldsymbol{\sigma}_{\boldsymbol{\phi}}(y, \boldsymbol{x})\right)\right. \tag{21.50}
$$


然后我们使用如下的变分下确界：


$$
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, y) \geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}, y)}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid y, \boldsymbol{z})+\log p_{\boldsymbol{\theta}}(y)+\log p_{\boldsymbol{\theta}}(\boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}, y)\right]=-\mathcal{L}(\boldsymbol{x}, y) \tag{21.51}
$$


上式与标准 VAEs 一样（21.2 节）。区别在于我们的观测变量包含两类数据： $\boldsymbol{x}$ 和 $y$ 。

为了计算无标注数据的似然 $p_{\boldsymbol{\theta}}(\boldsymbol{x})$，我们需要基于 $\boldsymbol{z}$ 和 $y$ 求解边际似然，我们可以使用如下的推理网络实现这一点


$$
\begin{align}
q_{\boldsymbol{\phi}}(\boldsymbol{z}, y \mid \boldsymbol{x}) & =q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) q_{\boldsymbol{\phi}}(y \mid \boldsymbol{x}) \tag{21.52}\\
q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) & =\mathcal{N}\left(\boldsymbol{z} \mid \boldsymbol{\mu}_{\boldsymbol{\phi}}(\boldsymbol{x}), \operatorname{diag}\left(\boldsymbol{\sigma}_{\boldsymbol{\phi}}(\boldsymbol{x})\right)\right. \tag{21.53}\\
q_{\boldsymbol{\phi}}(y \mid \boldsymbol{x}) & =\operatorname{Cat}\left(y \mid \boldsymbol{\pi}_{\boldsymbol{\phi}}(\boldsymbol{x})\right) \tag{21.54}
\end{align}
$$


其中 $q_\phi(y \mid \boldsymbol{x})$ 的作用类似于一个判别式分类器，用于补全确实的数据。我们接下来使用如下的变分下确界：


$$
\begin{align}
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) & \geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}, y \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid y, \boldsymbol{z})+\log p_{\boldsymbol{\theta}}(y)+\log p_{\boldsymbol{\theta}}(\boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z}, y \mid \boldsymbol{x})\right] \tag{21.55}\\
& =-\sum_y q_{\boldsymbol{\phi}}(y \mid \boldsymbol{x}) \mathcal{L}(\boldsymbol{x}, y)+\mathbb{H}\left(q_{\boldsymbol{\phi}}(y \mid \boldsymbol{x})\right)=-\mathcal{U}(\boldsymbol{x}) \tag{21.56}
\end{align}
$$


需要注意的是判别式分类器 $q_\phi(y \mid \boldsymbol{x})$ 仅仅用来计算无标注数据的对数似然，这一点并不是最优的。因此我们可以在有标注数据上添加一个额外的分类损失，从而得到最终的目标函数：


$$
\mathcal{L}(\boldsymbol{\theta})=\mathbb{E}_{(\boldsymbol{x}, y) \sim \mathcal{D}_L}[\mathcal{L}(\boldsymbol{x}, y)]+\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_U}[\mathcal{U}(\boldsymbol{x})]+\alpha \mathbb{E}_{(\boldsymbol{x}, y) \sim \mathcal{D}_L}\left[-\log q_{\boldsymbol{\phi}}(y \mid \boldsymbol{x})\right] \tag{21.57}
$$


其中 $\mathcal{D}_L$ 表示有标注数据，$\mathcal{D}_U$ 表示无标注数据，$\alpha$ 为控制生成器和判别器学习的权重超参数。

### 21.3.5 含序列编码/解码器的VAEs

在本节中，我们讨论用于文本和生物序列等序列数据的变分自编码器（VAE），其中数据 $\boldsymbol{x}$ 是一个可变长度的序列，但我们有一个固定大小的隐变量 $\boldsymbol{z} \in \mathbb{R}^K$。（我们在第29.13节中考虑了更一般的情况，其中 $\boldsymbol{z}$ 是一个变长序列的隐变量，称为序列VAE或动态VAE。）我们只需要修改解码器 $p(\boldsymbol{x} \mid \boldsymbol{z})$ 和编码器 $q(\boldsymbol{z} \mid \boldsymbol{x})$ 来处理序列数据。

![21.6](/assets/img/figures/book2/21.6.png)

{: style="width: 100%;" class="center"}

图21.6: 展示了一个具有双向RNN编码器和单向RNN解码器的VAE的示意图。输出生成器可以使用GMM和/或softmax分布。来自[HE18][^HE18]的第2图。在David Ha的友善许可下使用。

{:.image-caption}



#### 21.3.5.1 模型

如果我们在VAE的编码器和解码器中使用RNN，我们得到的模型被称为VAE-RNN，正如[Bow+16a][^Bow+16a]中提出的那样。更详细地说，生成模型是 $$p\left(\boldsymbol{z}, \boldsymbol{x}_{1: T}\right)=p(\boldsymbol{z}) \operatorname{RNN}\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{z}\right)$$，其中 $\boldsymbol{z}$ 可以作为RNN的初始状态注入，或作为每个时间步的输入。推断模型是 $$q\left(\boldsymbol{z} \mid \boldsymbol{x}_{1: T}\right)=\mathcal{N}(\boldsymbol{z} \mid \boldsymbol{\mu}(\boldsymbol{h}), \boldsymbol{\Sigma}(\boldsymbol{h}))$$，其中 $$\boldsymbol{h}=\left[\boldsymbol{h}_T^{\vec{T}}, \boldsymbol{h}_1^{\leftarrow}\right]$$ 是应用于 $x_{1: T}$ 的双向RNN的输出。请参见图21.6进行说明。

最近，人们尝试将Transformer与VAE结合起来。例如，在[Li+20][^Li+20]的Optimus模型中，他们使用BERT模型作为编码器。更详细地说，编码器 $q(z \mid x)$ 是从与输入序列 $\boldsymbol{x}$ 对应的“类别标签”所附加的虚拟令牌的嵌入向量中派生的。解码器是一个标准的自回归模型（类似于GPT），但有一个额外的输入，即隐变量 $\boldsymbol{z}$。他们考虑了两种注入潜变量的方法。最简单的方法是在解码步骤中将 $\boldsymbol{z}$ 添加到每个令牌的嵌入层中，通过定义 $\boldsymbol{h}_i^{\prime}=\boldsymbol{h}_i+\mathbf{W} \boldsymbol{z}$ 来实现，其中 $\boldsymbol{h}_i \in \mathbb{R}^H$ 是第 $i$ 个令牌的原始嵌入，$\mathbf{W} \in \mathbb{R}^{H \times K}$ 是一个解码矩阵，其中 $K$ 是隐变量的大小。然而，通过让解码器的所有层都关注潜变量代码 $\boldsymbol{z}$，他们在实验中取得了更好的结果。一个简单的方法是定义记忆向量 $h_m=\mathbf{W} \boldsymbol{z}$，其中 $\mathbf{W} \in \mathbb{R}^{L H \times K}$，$L$ 是解码器中的层数，然后将 $\boldsymbol{h}_m \in \mathbb{R}^{L \times H}$ 附加到每一层的所有其他嵌入中。

另一种替代方法被称为Transformer VAE，在[Gre20][^Gre20]中提出。该模型使用漏斗Transformer作为编码器，并使用T5条件Transformer作为解码器。此外，它还使用了一个MMD VAE（第21.3.2.1节）来避免后验坍塌现象。

![21.7](/assets/img/figures/book2/21.7.png)

{: style="width: 100%;" class="center"}

图21.7: 

{:.image-caption}



![21.8](/assets/img/figures/book2/21.8.png)

{: style="width: 100%;" class="center"}

图21.8: 

{:.image-caption}

#### 21.3.5.2 应用

本节，我们将讨论一些对于针对序列数据的一些 VAE 的应用。

##### 文本

在[Bow+16b]中，他们将VAE-RNN模型应用于自然语言句子。（有关相关工作，还可以参考[MB16; SSB17]。）尽管从标准的困惑度测量（在给定前面的单词的情况下预测下一个单词）来看，这并没有改进性能，但它确实提供了一种推断句子的语义表示的方法。然后，这可以用于潜空间插值，如第20.3.5节所讨论的。使用VAE-RNN进行这种插值的结果如图21.7a所示。（类似的结果在[Li+20]中使用VAE-transformer也得到了展示。）与此相反，如果我们使用标准的确定性自编码器，具有相同的RNN编码器和解码器网络，我们学习到的空间含义要少得多，如图21.7b所示。原因是确定性自编码器在其潜空间中有“空洞”，这些空洞解码为无意义的输出。然而，由于RNN（和transformer）是强大的解码器，我们需要解决后验坍塌问题，我们在第21.4节中讨论了这个问题。避免这个问题的一种常见方法是使用KL退火，但更有效的方法是使用第21.3.2节中的InfoVAE方法，其中包括对抗性自编码器（在[She+20]中与RNN解码器一起使用）和MMD自编码器（在[Gre20]中与transformer解码器一起使用）。

##### 素描

在[HE18]中，他们将VAE-RNN模型应用于生成各种动物和手写字符的素描（线条绘画）。他们称其模型为sketch-rnn。训练数据记录了（x；y）笔的位置序列，以及笔是接触纸张还是离开纸张。发射模型使用GMM表示实值位置偏移，并使用分类Softmax分布表示离散状态。

图21.8展示了各种类条件模型的一些样本。我们通过改变发射模型的温度参数ϴ来控制生成器的随机性。（更确切地说，在重新归一化之前，我们将GMM的方差乘以ϴ，并将离散概率除以ϴ。）当温度较低时，模型尽可能地重构输入。然而，当输入不典型于训练集（例如，有三只眼睛的猫，或者牙刷）时，重构会“规范化”为具有两只眼睛的典型猫，同时保留一些输入的特征。

##### 分子设计

在[GB+18]中，他们使用VAE-RNN模型来建模分子图结构，该结构使用SMILES表示法表示为字符串。此外，还可以学习从潜空间到某个感兴趣的标量量（如分子的溶解度或药效）的映射。然后，我们可以在连续潜空间中进行基于梯度的优化，尝试生成最大化该量的新图形。请参见图21.9以了解这种方法的概要。 主要问题是确保潜空间中的点解码为有效的字符串/分子。有各种解决方案，包括使用语法VAE，其中RNN解码器被随机上下文无关语法所替代。有关详细信息，请参阅[KPHL17]。

## 21.4 避免后验坍塌

如果解码器 $p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})$ 足够强大（比如，pixel-CNN 或者text-RNN），那么 VAE 就不需要使用隐编码 $\boldsymbol{z}$ 。这被称为 **后验坍塌** （posterior\ collapse）或者 **变分过剪枝**（variational overpruning）（参见例如，[Che+17b; Ale+18; Hus17a; Phu+18; TT17; Yeu+17; Luc+19; DWW19; WBC21]）。要了解为什么会发生这种情况，请考虑方程（21.21）。如果存在一个生成器参数 $\boldsymbol{\theta}^*$ 设置，使得对于每个 $\boldsymbol{z}$，$$p_{\boldsymbol{\theta}^*}(\boldsymbol{x} \mid \boldsymbol{z})=p_{\mathcal{D}}(\boldsymbol{x})$$，则可以使得 $$D_{\mathbb{K} \mathbb{L}}\left(p_{\mathcal{D}}(\boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{x})\right)=0$$。由于生成器与潜变量无关，我们有 $$p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})=p_{\boldsymbol{\theta}}(\boldsymbol{z})$$。先验 $$p(\boldsymbol{z})$$ 通常是一个简单的分布，如高斯分布，因此我们可以找到推断参数的设置，使得 $$q_{\boldsymbol{\phi}^*}(\boldsymbol{z} \mid \boldsymbol{x})=p_{\boldsymbol{\theta}}(\boldsymbol{z})$$，从而确保 $D_{\mathbb{K} \mathbb{L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})\right)=0$。因此，我们成功地最大化了ELBO，但我们没有学习到任何有用的数据潜变量表示，而这正是潜变量建模的目标之一。下面我们将讨论一些解决后验坍塌问题的方法。

### 21.4.1 KL 退火

解决这个问题的常见方法是在[BoW+16a][^BoW+16a]中提出的KL退火方法，其中ELBO中的KL惩罚项被缩放为参数 $\beta$，从 0.0（对应于自编码器）增加到 1.0（对应于标准的MLE训练）。（值得注意的是，在第21.3.1节中的$\beta$-VAE模型中使用的是 $\beta \gt 1$。）

 KL退火可以有效地工作，但需要调整 $\beta$ 的时间表。一个标准的做法是使用**循环退火**（cyclical annealing），即多次重复增加 $\beta$ 的过程。这样做可以逐步学习到更有意义的潜变量编码，通过利用在先前循环中学习到的良好表示作为优化的初始点。

### 21.4.2 **Lower bounding the rate**

另一种方法是坚持使用原始未修改的ELBO目标，但通过限制 $q$ 的灵活性，防止速率（即 $D_{\mathbb{K K L}}(q \| p)$ 项）收敛到0。例如，[XD18; Dav+18]使用von Mises-Fisher先验和后验（第2.2.5.3节），而不是高斯分布，并且他们将后验约束为具有固定的集中度，$q(\boldsymbol{z} \mid \boldsymbol{x})=\operatorname{vMF}(\boldsymbol{z} \mid \boldsymbol{\mu}(\boldsymbol{x}), \kappa)$。这里参数 $\kappa$ 控制着代码的速率。而 $\delta$-VAE方法[Oor+19]使用高斯自回归先验和对角高斯后验。我们可以通过调整自回归先验的回归参数来确保速率至少为 $\delta$。

### 21.4.3 Free bits

在本节中，我们讨论**自由比特**（free bits）方法[Kin+16]，这是另一种用于下界速率的方法。为了解释这一点，考虑一个完全分解的后验，其中KL惩罚项的形式为：


$$
\mathcal{L}_R=\sum_i D_{\mathbb{K K L}}\left(q_{\boldsymbol{\phi}}\left(z_i \mid \boldsymbol{x}\right) \| p_{\boldsymbol{\theta}}\left(z_i\right)\right) \tag{21.58}
$$


其中zi是z的第i维。我们可以将其替换为一个铰链损失（hinge loss），这将放弃对已经低于目标压缩率的维度进一步减小KL的追求：


$$
\mathcal{L}_R^{\prime}=\sum_i \max \left(\lambda, D_{\mathbb{K K L}}\left(q_{\boldsymbol{\phi}}\left(z_i \mid \boldsymbol{x}\right) \| p_{\boldsymbol{\theta}}\left(z_i\right)\right)\right) \tag{21.59}
$$


因此，KL非常小的比特“是自由的”，因为模型不必根据先验编码它们而“付费”。

### 21.4.4 增加跨连

潜变量崩溃的一个原因是潜变量z与观测数据x之间的连接不足。一个简单的解决方案是通过修改生成模型的架构来添加跳跃连接，类似于残差网络（第16.2.4节），如图21.10所示。这被称为跳跃VAE（Skip-VAE）[Die+19a]。

### 21.4.5 改进的变分推断

后验崩溃问题部分是由于对后验的近似不足引起的。在[He+19]中，他们提议保持模型和VAE目标不变，但在每个生成模型拟合步骤之前更积极地更新推理网络。这使得推理网络能更准确地捕捉当前的真实后验，从而鼓励生成器在有用时使用潜在代码。 

然而，这仅解决了后验崩溃中由摊销间隔[CLD18]引起的部分问题，而没有解决更基本的变分修剪问题。在变分修剪中，如果模型的后验与先验偏离太远，KL项将对其进行惩罚，而先验通常过于简单无法匹配聚合后验。 

改善变分修剪的另一种方法是使用比传统ELBO（第10.5.1节）更紧的下界，或更准确的后验近似（第10.4节），或更准确的（分层）生成模型（第21.5节）。

### 21.4.6 可替代的优化目标

除了上述方法之外，另一种选择是用其他目标替换ELBO目标，例如第21.3.2节讨论的InfoVAE目标，其中包括对抗自编码器和最大均值差异自编码器作为特例。InfoVAE目标包含一个项，明确强制x和z之间的互信息非零，从而有效解决了后验崩溃的问题。

## 21.5 包含层次结构的 VAEs

我们定义了一个具有L个随机层的分层VAE或HVAE，其生成模型如下所示：


$$
p_{\boldsymbol{\theta}}\left(\boldsymbol{x}, \boldsymbol{z}_{1: L}\right)=p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_L\right)\left[\prod_{l=L-1}^1 p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_l \mid \boldsymbol{z}_{l+1}\right)\right] p_{\boldsymbol{\theta}}\left(\boldsymbol{x} \mid \boldsymbol{z}_1\right) \tag{21.60}
$$


我们可以通过使模型非马尔科夫化来改进上述模型，即让每个zl依赖于所有更高级别的随机变量zl+1:L，而不仅仅是前一个级别，即


$$
p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})=p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_L\right)\left[\prod_{l=L-1}^1 p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_l \mid \boldsymbol{z}_{l+1: L}\right)\right] p_{\boldsymbol{\theta}}\left(\boldsymbol{x} \mid \boldsymbol{z}_{1: L}\right) \tag{21.61}
$$


需要注意的是，现在似然是p(x|z1:L)，而不仅仅是p(x|z1)。这类似于从所有前一个变量到所有子变量添加跳跃连接。我们可以通过使用确定性的“骨干”残差连接来轻松实现此操作，该连接累积所有随机决策并将其传播到链条下方，如图21.11（左）所示。我们将在下面讨论如何在这样的模型中进行推理和学习。

### 21.5.1  Bottom-up vs top-down 推断

要在分层VAE中进行推理，我们可以使用以下形式的自底向上推理模型：


$$
q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})=q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_1 \mid \boldsymbol{x}\right) \prod_{l=2}^L q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_l \mid \boldsymbol{x}, \boldsymbol{z}_{1: l-1}\right) \tag{21.62}
$$


然而，更好的方法是使用以下形式的自顶向下推理模型：


$$
q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})=q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_L \mid \boldsymbol{x}\right) \prod_{l=L-1}^1 q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_l \mid \boldsymbol{x}, \boldsymbol{z}_{l+1: L}\right) \tag{21.63}
$$


zl的推理将来自x的自下而上信息与来自更高层的自上而下信息z>l = zl+1:L相结合。请参见图21.11（右）进行说明。

使用上述模型，ELBO可以表示为以下形式（使用KL的链式法则）：


$$
\begin{align}
\text {Ł}(\boldsymbol{\theta}, \boldsymbol{\phi} \mid \boldsymbol{x}) & =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]-D_{\mathbb{K K L}}\left(q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_L \mid \boldsymbol{x}\right) \| p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_L\right)\right) \tag{21.64}\\
& -\sum_{l=L-1}^1 \mathbb{E}_{q_\phi\left(\boldsymbol{z}_{>l} \mid \boldsymbol{x}\right)}\left[D_{\mathbb{K K L}}\left(q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_l \mid \boldsymbol{x}, \boldsymbol{z}_{>l}\right) \| p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_l \mid \boldsymbol{z}_{>l}\right)\right)\right] \tag{21.65}
\end{align}
$$


其中


$$
q_\phi\left(z_{>l} \mid x\right)=\prod_{i=l+1}^L q_\phi\left(z_i \mid x, z_{>i}\right) \tag{21.66}
$$


为 $l$ 层以上的近似后验分布（即 $z_l$ 的父节点）。

自顶向下推理模型更好的原因是它更接近给定层的真实后验，其表达式为：


$$
p_{\boldsymbol{\theta}}\left(z_l \mid x, z_{l+1: L}\right) \propto p_{\boldsymbol{\theta}}\left(z_l \mid z_{l+1: L}\right) p_{\boldsymbol{\theta}}\left(x \mid z_l, z_{l+1: L}\right) \tag{21.67}
$$


因此，后验结合了自顶向下的先验项 $p_\theta\left(z_l \mid z_{l+1: L}\right)$ 和自下而上的似然项 $p_{\boldsymbol{\theta}}\left(\boldsymbol{x} \mid z_l, z_{l+1: L}\right)$。我们可以通过定义如下来近似该后验：


$$
q_\phi\left(z_l \mid x, z_{l+1: L}\right) \propto p_\theta\left(z_l \mid z_{l+1: L}\right) \tilde{q}_\phi\left(z_l \mid x, z_{l+1: L}\right) \tag{21.68}
$$


其中 $\tilde{q}_\phi\left(z_l \mid x, z_{l+1: L}\right)$ 是对自下而上似然的学习高斯近似。如果先验和似然都是高斯的，我们可以在闭合形式中计算出这个乘积，正如梯度网络论文[Sn+16; Søn+16]中所提出的。更灵活的方法是让 $q_\phi\left(z_l \mid x, z_{l+1: L}\right)$ 进行学习，但是强制它与学习的先验 $p_{\boldsymbol{\theta}}\left(z_l \mid z_{l+1: L}\right)$ 共享一些参数，正如[Kin+16]中所提出的。这减少了模型中的参数数量，并确保后验和先验始终保持相对接近。

### 21.5.2 举例：非常深的VAE

已经有很多论文探索了不同类型的HVAE模型（例如，查阅[Kin+16; Sn+16; Chi21a; VK20a; Maa+19]），我们没有足够的空间来讨论它们。在这里，我们关注[Chi21a]的“非常深的VAE”或VD-VAE模型，因为它简单但能产生最先进的结果（截至撰写本文时）。 

该架构是一个简单的卷积VAE，具有双向推理，如图21.12所示。对于每个层，先验和后验都是对角高斯分布。作者发现，与转置卷积相比，最近邻上采样（在解码器中）效果要好得多，并且避免了后验崩溃。这使得可以使用普通的VAE目标进行训练，而不需要在第21.5.4节中讨论的任何技巧。

层次结构中低分辨率的潜变量（位于层次结构的顶部）捕捉了每个图像的大部分全局结构；剩余的高分辨率潜变量仅用于填充细节，使图像看起来更加逼真，并提高似然。这表明该模型可能对有损压缩有用，因为许多低级别的细节可以从先验（即“产生幻觉”）中绘制出来，而不必由编码器发送。 

我们还可以在多个分辨率上使用该模型进行无条件采样。图21.13展示了一个在FFHQ-256数据集上训练的具有78个随机层的模型的示例。

### 21.5.3 与自回归模型连接

直到最近，大多数分层VAE只有少量的随机层。因此，它们生成的图像看起来没有其他模型（如自回归PixelCNN模型，参见第22.3.2节）生成的图像好，也没有那么高的似然度。然而，通过赋予VAE更多的随机层，可以在似然度和样本质量方面超过AR模型，同时使用更少的参数和更少的计算资源[Chi21a; VK20a; Maa+19]。 

为了了解为什么这是可能的，注意到我们可以将任何AR模型表示为退化的VAE，如图21.14（左）所示。这个想法很简单：编码器通过设置 $z_{1: D}=x_{1: D}$（因此 $q_\phi\left(z_i=x_i \mid z_{>i}, \boldsymbol{x}\right)=1$）将输入复制到潜空间，然后模型学习一个自回归先验 $p_{\boldsymbol{\theta}}\left(z_{1: D}\right)=\prod_d p\left(z_d \mid z_{1: d-1}\right)$，最后，似然函数只是将潜向量复制到输出空间，所以 $p_{\boldsymbol{\theta}}\left(x_i=z_i \mid \boldsymbol{z}\right)=1$。由于编码器计算的是准确（虽然是退化的）的后验，我们有 $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})=p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})$，因此ELBO是紧束的，并且简化为对数似然度。


$$
\log p_{\boldsymbol{\theta}}(\boldsymbol{x})=\log p_{\boldsymbol{\theta}}(\boldsymbol{z})=\sum_d \log p_{\boldsymbol{\theta}}\left(x_d \mid \boldsymbol{x}_{<d}\right) \tag{21.69}
$$


因此，只要VAE至少具有D个随机层（其中D是观察数据的维数），我们就可以用VAE来模拟任何AR模型。 

在实践中，数据通常存在于较低维度的流形中（例如，参见[DW19]），这可以允许更紧凑的潜变量编码。例如，图21.14（右）显示了一种分层代码，其中在较低层的潜因子在给定较高层的条件下是独立的，因此可以并行生成。这种类似树的结构可以以O(logD)的时间生成样本，而自回归模型始终需要O(D)的时间。（请记住，对于图像来说，D是像素数，因此随着图像分辨率的增加，它呈二次增长。例如，即使是一个小小的32x32图像也有D = 3072。）

除了速度外，分层模型还需要比“平面”模型少得多的参数。用于生成图像的典型架构是多尺度方法：模型从一组小型的、空间排列的潜变量开始，每个后续层将空间分辨率增加（通常增加一倍）。这使得高级层可以捕捉全局的、长距离的相关性（例如，面部的对称性或整体的肤色），同时让较低层捕捉细节。

### 21.5.4 变分剪枝

分层VAE的一个常见问题是高级潜变量层经常被忽略，因此模型无法学习有趣的高级语义。这是由变分修剪引起的。这个问题类似于我们在第21.4节讨论的潜变量崩溃的问题。

缓解这个问题的常见启发式方法是使用KL平衡系数[Che+17b]，以确保每个层中编码了相同数量的信息。也就是说，我们使用以下惩罚项：


$$
\sum_{l=1}^L \gamma_l \mathbb{E}_{q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_{>l} \mid \boldsymbol{x}\right)}\left[D_{\mathbb{K} \mathbb{L}}\left(q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_l \mid \boldsymbol{x}, \boldsymbol{z}_{>l}\right) \| p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_l \mid \boldsymbol{z}_{>l}\right)\right)\right] \tag{21.70}
$$


当KL惩罚项很小（在当前小批量中）时，将平衡项l设置为一个较小的值，以鼓励使用该层；当KL项较大时，将平衡项l设置为一个较大的值（这仅在“热身期间”执行）。具体而言，[VK20a]建议将平衡系数l设为与层的大小sl和平均KL损失成比例的值：


$$
\gamma_l \propto s_l \mathbb{E}_{\boldsymbol{x} \sim \mathcal{B}}\left[\mathbb{E}_{q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_{>l} \mid \boldsymbol{x}\right)}\left[D_{\mathbb{K L L}}\left(q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_l \mid \boldsymbol{x}, \boldsymbol{z}_{>l}\right) \| p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_l \mid \boldsymbol{z}_{>l}\right)\right)\right]\right] \tag{21.71}
$$


其中 $\mathcal{B}$ 为当前的minibatch。

### 21.5.5 其他的优化困难

在训练（分层）VAE时，常见的问题是损失可能变得不稳定。这主要原因是KL项没有界限（可能变得无限大）。在[Chi21a]中，他们通过两种方式解决了这个问题。首先，确保每个残差瓶颈块中最后一个卷积层的初始随机权重按 $1 / \sqrt{L}$ 进行缩放。其次，如果损失的梯度范数超过某个阈值，则跳过更新步骤。 

在[VK20a]的Nouveau VAE方法中，他们使用了一些更复杂的措施来确保稳定性。首先，他们使用批归一化，但进行了一些调整。其次，他们使用编码器的谱正则化。具体地，他们添加了惩罚项 $\beta \sum_i \lambda_i$，其中 $\lambda_i$ 是第 $i$ 个卷积层的最大奇异值（使用单次幂迭代估计），而 $\beta \geq 0$ 则为调整参数。第三，他们在每个层中使用逆自回归流（第23.2.4.3节），而不是对角高斯近似。第四，他们使用残差表示来表示后验。特别地，假设第 $l$ 层中第 $i$ 个变量的先验分布为：


$$
p_{\boldsymbol{\theta}}\left(z_l^i \mid \boldsymbol{z}_{>l}\right)=\mathcal{N}\left(z_l^i \mid \mu_i\left(\boldsymbol{z}_{>l}\right), \sigma_i\left(\boldsymbol{z}_{>l}\right)\right) \tag{21.72}
$$


他们提出了如下的近似后验：


$$
q_{\boldsymbol{\phi}}\left(z_l^i \mid \boldsymbol{x}, \boldsymbol{z}_{>l}\right)=\mathcal{N}\left(z_l^i \mid \mu_i\left(\boldsymbol{z}_{>l}\right)+\Delta \mu_i\left(\boldsymbol{z}_{>l}, \boldsymbol{x}\right), \sigma_i\left(\boldsymbol{z}_{>l}\right) \cdot \Delta \sigma_i\left(\boldsymbol{z}_{>l}, \boldsymbol{x}\right)\right) \tag{21.73}
$$


其中，$\Delta$ 项是由编码器计算得出的相对变化。相应的KL惩罚项可以简化为以下形式（为简洁起见，省略了 $l$ 的下标）：


$$
D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}\left(z^i \mid \boldsymbol{x}, \boldsymbol{z}_{>l}\right) \| p_{\boldsymbol{\theta}}\left(z^i \mid \boldsymbol{z}_{>l}\right)\right)=\frac{1}{2}\left(\frac{\Delta \mu_i^2}{\sigma_i^2}+\Delta \sigma_i^2-\log \Delta \sigma_i^2-1\right) \tag{21.74}
$$


只要 $\sigma_i$ 从下方有界，就可以通过调整编码器参数轻松控制KL项。

## 21.6 矢量量化VAE

在这个部分中，我们描述了VQ-VAE，它代表“向量量化VAE”[OVK17; ROV19]。这类似于标准的VAE，但它使用了一组离散的潜在变量。

### 21.6.1 二进制编码的自编码器

解决这个问题的最简单方法是构建一个标准的VAE，但在编码器的末尾添加一个离散化层，即令 $z_e(x) \in\{0, \ldots, S-1\}^K$，其中$S$是状态的数量，$K$是离散隐变量的数量。例如，我们可以将隐向量（使用$S = 2$）二值化，通过将$z$截取到$\{0,1\}^K$的范围内。这对于数据压缩可能很有用（参见例如[BLS17][^BLS17]）。

假设关于隐变量的先验分布是均匀分布。由于编码器是确定性的，KL散度将退化为一个常数，等于$\log K$。这避免了后验坍缩问题（第21.4节）。不幸的是，编码器的不连续量化操作禁止了直接使用基于梯度的优化方法。[OVK17][^OVK17]中提出的解决方案是使用直通估计器，我们在第6.3.8节中讨论。我们在图21.15中展示了这种方法的一个简单示例，其中我们使用了高斯似然函数，因此损失函数的形式为：


$$
\mathcal{L}=\|\boldsymbol{x}-d(e(\boldsymbol{x}))\|_2^2 \tag{21.75}
$$


其中 $e(x) \in\{0,1\}^K$ 表示编码器，$d(z) \in \mathbb{R}^{28 \times 28}$ 表示解码器。

### 21.6.2 VQ-VAE 模型

通过使用离散隐变量的3d张量，$z \in \mathbb{R}^{H \times W \times K}$，我们可以获得更具表现力的模型，其中$K$是每个隐变量的离散值数量。与其仅对连续向量 $z_e(x)_{i j}$ 进行二值化，我们将其与嵌入向量的代码书 $\left\{\boldsymbol{e}_k: k=1: K, \boldsymbol{e}_k \in \mathbb{R}^L\right\}$ 进行比较，然后将$z_{ij}$设置为最接近代码书条目的索引：


$$
q\left(z_{i j}=k \mid x\right)= \begin{cases}1 & \text { if } k=\operatorname{argmin}_{k^{\prime}}\left\|z_e(x)_{i, j,:}-e_{k^{\prime}}\right\|_2 \\ 0 & \text { otherwise }\end{cases} \tag{21.76}
$$


当重构输入时，我们将每个离散代码索引替换为相应的实值代码书向量：


$$
\left(z_q\right)_{i j}=e_k \text { where } z_{i j}=k \tag{21.77}
$$


然后，将这些值像往常一样传递给解码器 $p\left(x \mid z_q\right)$。请参见图21.16，以了解整体架构的示意图。请注意，尽管$z_q$是从代码书向量的离散组合生成的，但使用分布式编码使得模型非常具有表现力。例如，如果我们使用一个32×32的网格，其中$K = 512$，那么我们可以生成$512^{32×32} = 2^{9216}$个不同的图像，这是非常巨大的数量。

为了适应这个模型，我们可以使用直通估计器最小化负对数似然（重构误差），就像以前一样。这意味着将梯度从解码器输入zq(x)传递到编码器输出ze(x)，绕过方程式（21.76），如图21.16中的红色箭头所示。不幸的是，这意味着代码书条目将不会获得任何学习信号。为了解决这个问题，作者提出了一个额外的损失项，称为代码书损失，它鼓励代码书条目e与编码器的输出匹配。我们将编码器ze(x)视为一个固定的目标，通过添加一个stop gradient操作符来实现；这确保ze在前向传递中被正常处理，但在反向传递中具有零梯度。修改后的损失（省略空间索引i和j）如下：


$$
\mathcal{L}=-\log p\left(x \mid z_q(x)\right)+\left\|\operatorname{sg}\left(z_e(x)\right)-e\right\|_2^2 \tag{21.78}
$$


其中$e$是分配给$z_e(x)$的代码书向量，sg是停止梯度操作符。

更新代码书向量的另一种方法是使用移动平均值。为了了解这是如何工作的，首先考虑批处理设置。设 $\left\{z_{i, 1}, \ldots, z_{i, n_i}\right\}$ 为距离词典项$\boldsymbol{e}_i$最近的来自编码器的$n_i$个输出。我们可以更新$\boldsymbol{e}_i$以最小化均方误差（MSE）：


$$
\sum_{j=1}^{n_i}\left\|z_{i, j}-e_i\right\|_2^2 \tag{21.79}
$$


这个更新有闭式形式：


$$
e_i=\frac{1}{n_i} \sum_{j=1}^{n_i} z_{i, j} \tag{21.80}
$$


这就像在拟合GMM的均值向量时EM算法的M步骤。在小批量设置中，我们用指数移动平均替换上述操作，如下所示：


$$
\begin{align}
N_i^t & =\gamma N_i^{t-1}+(1-\gamma) n_i^t \tag{21.81}\\
\boldsymbol{m}_i^t & =\gamma \boldsymbol{m}_i^{t-1}+(1-\gamma) \sum_j z_{i, j}^t \tag{21.82}\\
\boldsymbol{e}_i^t & =\frac{\boldsymbol{m}_i^t}{N_i^t} \tag{21.83}
\end{align}
$$


作者发现使用 $\gamma =0.9$ 效果更好。

上述过程将学会更新代码书向量，以使其与编码器的输出相匹配。然而，确保编码器不太频繁地“改变主意”使用哪个代码书值也很重要。为了防止这种情况发生，作者提出向损失中添加第三项，称为承诺损失，鼓励编码器的输出接近代码书的值。因此，我们得到最终的损失：


$$
\mathcal{L}=-\log p\left(x \mid z_q(x)\right)+\left\|\operatorname{sg}\left(z_e(x)\right)-e\right\|_2^2+\beta\left\|z_e(x)-\operatorname{sg}(e)\right\|_2^2 \tag{21.84}
$$



作者们发现 $f = 0.25$ 的取值效果不错，尽管这个值当然取决于重构损失（NLL）项的规模。关于这个损失的概率解释可以在 [Hen+18][^Hen+18] 中找到。总体来说，解码器只优化第一项，编码器优化第一项和最后一项，而嵌入层优化中间的项。

### 21.6.3 学习先验分布

在训练了VQ-VAE模型之后，有可能学习一个更好的先验来匹配聚集后验。为了做到这一点，我们只需将编码器应用于一组数据，$\{x_n\}$，从而将它们转换为离散序列，$\{z_n\}$。然后我们可以使用任何类型的序列模型来学习联合分布$p(z)$。在原始的VQ-VAE论文[OVK17][^OVK17]中，他们使用了因果卷积的PixelCNN模型（第22.3.2节）。更近期的工作使用了变换器解码器（第22.4节）。然后可以使用VQ-VAE模型的解码器部分来解码来自这个先验的样本。我们在下面的章节中给出了一些这方面的例子。

### 21.6.4 层次化拓展（VQ-VAE-2）

在[ROV19][^ROV19]中，他们通过使用分层潜在代码扩展了原始的VQ-VAE模型。该模型在图21.17中进行了说明。他们将这种模型应用于256×256×3大小的图像。第一层潜在层将其映射到64×64大小的量化表示，第二层潜在层将其映射到32×32大小的量化表示。这种分层方案允许顶层专注于图像的高层语义，将细微的视觉细节，如纹理，留给较低层。（关于分层VAE的更多讨论，请参见第21.5节。）

在拟合了VQ-VAE之后，他们使用一个增加了自注意力机制（第16.2.7节）的PixelCNN模型来学习顶层代码的先验，以捕捉长距离的依赖关系。（这种混合模型被称为PixelSNAIL [Che+17c][^Che+17c]。）对于较低层的先验，他们仅使用标准的PixelCNN，因为使用注意力机制会太昂贵。然后可以使用VQ-VAE解码器来解码模型的样本，如图21.17所示。

### 21.6.5 离散化VAE

在VQ-VAE中，我们对潜在变量使用独热编码，$q(z=k \mid \boldsymbol{x})=1$ 当且仅当 $k=\operatorname{argmin}_k\left\|z_e(\boldsymbol{x})-e_k\right\|_2$， 然后设置$z_q=e_k$ 。这并没有捕捉潜在代码中的任何不确定性，并且需要使用直通估计器（straight-through estimator）进行训练。

已经研究了各种其他拟合具有离散潜在代码的VAE的方法。在DALL-E论文（第22.4.2节）中，他们使用了一种相当简单的方法，基于使用Gumbel-softmax松弛方法来处理离散变量（见第6.3.6节）。简而言之，让 $q(z=k \mid \boldsymbol{x})$ 是输入 $\boldsymbol{x}$ 被分配到代码本条目 $k$ 的概率。我们可以通过计算$w_k=\operatorname{argmax}_k g_k+\log q(z=k \mid \boldsymbol{x})$  从中精确抽样$w_k \sim q(z=k \mid \boldsymbol{x})$ ，其中每个 $g_k$ 都来自一个Gumbel分布。我们可以通过使用一个温度 $\tau>0$ 的softmax来“松弛”这个过程，并计算


$$
w_k=\frac{\exp \left(\frac{g_k+\log q(z=k \mid \boldsymbol{x})}{\tau}\right)}{\sum_{j=1}^K \exp \left(\frac{g_j+\log q(z=j \mid \boldsymbol{x})}{\tau}\right)} \tag{21.85}
$$


我们现在将潜在代码设置为代码本向量的加权和：


$$
z_q=\sum_{k=1}^K w_k e_k \tag{21.86}
$$


在温度τ趋近于0的极限情况下，权重w的分布会收敛到一个独热分布，在这种情况下，z就会等于代码本中的某个条目。但是对于有限的τ，我们会“填充”向量之间的空间。

这使我们能够以通常的可微分方式表达ELBO（证据下界）：


$$
\mathcal{L}=-\mathbb{E}_{q(\boldsymbol{z} \mid \boldsymbol{x})}[\log p(\boldsymbol{x} \mid \boldsymbol{z})]+\beta D_{\mathbb{K K} L}(q(\boldsymbol{z} \mid \boldsymbol{x}) \| p(\boldsymbol{z})) \tag{21.87}
$$


其中 $\beta>0$ 控制正则化的量。（与VQ-VAE不同，KL项不是常数，因为编码器是随机的。）此外，由于Gumbel噪声变量是从独立于编码器参数的分布中采样的，我们可以使用重参数化技巧（第6.3.5节）来优化这一过程。

### 21.6.6 VQ-GAN

VQ-VAE的一个缺点是它在重构损失中使用了均方误差，这可能导致模糊的样本。在VQ-GAN论文[ERO21][^ERO21]中，他们用（块状的）GAN损失（参见第26章）替换了这一点，并加入了感知损失；这导致了更高的视觉保真度。此外，他们使用变换器（参见第16.3.5节）来模拟潜在代码的先验。参见图21.18以了解整体模型的可视化。在[Yu+21][^Yu+21]中，他们用变换器替换了VQ-GAN模型的CNN编码器和解码器，取得了改进的结果；他们将这称为VIM（向量量化图像建模）。

## 结束（以下内容暂时无效）

### 21.3.4 含缺失值的 VAEs

有时我们可能会有缺失的数据，其中数据向量x在RD中的某些部分可能是未知的。在第21.3.3节中，我们讨论了多模态变分自编码器的一种特殊情况。在本节中我们允许任意缺失模式。

为了对缺失数据建模，假设 $\boldsymbol{m} \in\{0,1\}^D$ 是一个二进制向量，其中$m_j=1$表示$x_j$缺失，$m_j=0$表示$x_j$不缺失。令$\mathbf{X}=\left\{\boldsymbol{x}^{(n)}\right\}$和$\mathbf{M}=\left\{\boldsymbol{m}^{(n)}\right\}$为$N \times D$ 矩阵。此外，令$\mathbf{X}_o$为$\mathbf{X}$的观测部分，$\mathbf{X}_h$为$\mathbf{X}$的隐藏部分。如果我们假设$p\left(\mathbf{M} \mid \mathbf{X}_o, \mathbf{X}_h\right)=p(\mathbf{M})$，则称数据是**完全随机缺失**（missing completely at random，MCAR），因为缺失与隐藏或观测特征无关。如果我们假设 $p\left(\mathbf{M} \mid \mathbf{X}_o, \mathbf{X}_h\right)=p\left(\mathbf{M} \mid \mathbf{X}_o\right)$，则称数据是**随机缺失**（missing at random, MAR），因为缺失与隐藏特征无关，但可能与可见特征有关。如果以上两个假设都不成立，则称数据**不是随机缺失**（not missing at random，NMAR）。

在MCAR和MAR情况下，我们可以忽略缺失机制，因为它对隐藏特征没有任何信息。然而，在NMAR情况下，我们需要对缺失数据机制进行建模，因为缺乏信息可能具有信息量。例如，某人在调查中未回答敏感问题（例如，“您是否感染COVID-19？”）可能对潜在值具有信息。有关缺失数据模型的更多信息，请参考[LR87; Mar08]等文献。

在VAE的上下文中，我们可以将缺失值视为潜在变量来建模MCAR场景。如图21.7(a)所示。由于有向图模型中的缺失叶节点不会影响它们的父节点，因此在计算后验概率$p\left(\boldsymbol{z}^{(i)} \mid \boldsymbol{x}_o^{(i)}\right)$时，我们可以在计算时简单地忽略它们，其中$\boldsymbol{x}_o^{(i)}$是示例$i$的观测部分。但是，当使用分块推断网络时，处理缺失输入可能会很困难，因为通常会对模型进行训练以计算$p\left(\boldsymbol{z}^{(i)} \mid \boldsymbol{x}_{1: d}^{(i)}\right)$。其中，一种解决方法是使用在第21.3.3节中在多模态VAE的上下文中讨论的专家乘积方法。然而，此方法仅适用于整个块（对应于不同的模态）缺失的情况，并且对于任意的缺失模式（例如由于遮挡或镜头上的划痕而导致的像素丢失）效果不佳。此外，这种方法对于NMAR情况也无法奏效。

在[CNW20]中提出的另一种方法是将缺失指示器明确地纳入模型中，如图21.7(b)所示。我们假设模型总是生成每个$\boldsymbol{x}_j$ ($j=1:d$)，但我们只能看到“损坏”的版本 $\tilde{\boldsymbol{x}}_j$。如果$m_j=0$，则$\tilde{\boldsymbol{x}}_j=\boldsymbol{x}_j$，但如果$m_j=1$，则$\tilde{x}_j$是一个特殊值，如 0，与 $x_j$无关。我们可以通过使用另一个潜在变量$\boldsymbol{z}_m$来建模缺失元素（$\boldsymbol{m}$的分量）之间的任何相关性。这个模型可以很容易地扩展到NMAR情况，方法是让$\boldsymbol{m}$依赖于观测数据的潜在因素$\boldsymbol{z}$以及通常的缺失潜在因素$\boldsymbol{z}_m$，如图21.7(c)所示。

我们修改VAE，使其对缺失模式进行条件化，因此VAE解码器的形式为p(xo|z;m)，编码器的形式为q(z|xo;m)。但是，我们像往常一样假设先验分布p(z)与m无关。我们可以计算给定缺失性的观测数据的对数边际似然的下界，计算方法如下：
$$
\begin{align}
\log p\left(\boldsymbol{x}_o \mid \boldsymbol{m}\right) & =\log \iint p\left(\boldsymbol{x}_o, \boldsymbol{x}_m \mid \boldsymbol{z}, \boldsymbol{m}\right) p(\boldsymbol{z}) d \boldsymbol{x}_m d \boldsymbol{z} \tag{21.86}\\
& =\log \int p\left(\boldsymbol{x}_o \mid \boldsymbol{z}, \boldsymbol{m}\right) p(\boldsymbol{z}) d \boldsymbol{z} \tag{21.87}\\
& =\log \int p\left(\boldsymbol{x}_o \mid \boldsymbol{z}, \boldsymbol{m}\right) p(\boldsymbol{z}) \frac{q(\boldsymbol{z} \mid \tilde{\boldsymbol{x}}, \boldsymbol{m})}{q(\boldsymbol{z} \mid \tilde{\boldsymbol{x}}, \boldsymbol{m})} d \boldsymbol{z} \tag{21.88}\\
& =\log \mathbb{E}_{q(\boldsymbol{z} \mid \tilde{\boldsymbol{x}}, \boldsymbol{m})}\left[p\left(\boldsymbol{x}_o \mid \boldsymbol{z}, \boldsymbol{m}\right) \frac{p(\boldsymbol{z})}{q(\boldsymbol{z} \mid \tilde{\boldsymbol{x}}, \boldsymbol{m})}\right] \tag{21.89}\\
& \geq \mathbb{E}_{q(\boldsymbol{z} \mid \tilde{\boldsymbol{x}}, \boldsymbol{m})}\left[\log p\left(\boldsymbol{x}_o \mid \boldsymbol{z}, \boldsymbol{m}\right)\right]-D_{\mathbb{K} \mathbb{L}}(q(\boldsymbol{z} \mid \tilde{\boldsymbol{x}}, \boldsymbol{m}) \| p(\boldsymbol{z})) \tag{21.90}
\end{align}
$$
我们可以按照通常的方式拟合这个模型。







### 21.2.2 证据下确界（Evidence lower bound, ELBO）

拟合模型的目标是最大化边际似然
$$
p_{\boldsymbol{\theta}}(\boldsymbol{x})=\int p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{z}) d \boldsymbol{z} \tag{21.9}
$$
不幸的是，计算上式的积分是很棘手的。然而，我们可以使用一个推理网络来计算一个近似后验 $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$ ，进而得到一个关于边际似然的下确界。这个方法在10.1.2节被讨论，但我们在此处复习一遍，只是在符号上有点不同。

首先，我们有如下的分解式：
$$
\begin{align}
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) & =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x})\right] \tag{21.10}\\
& =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \left(\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})}{p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})}\right)\right] \tag{21.11}\\
& =\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \left(\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})} \frac{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}{p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})}\right)\right] \tag{21.12}\\
& =\underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \left(\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\right)\right]}_{\ell_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\boldsymbol{x})}+\underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \left(\frac{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}{p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})}\right)\right]}_{D_{\mathrm{KL}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})\right)} \tag{21.13}\\

\end{align}
$$
式 (21.13) 中的KL散度是非负项，所以我们有：
$$
\text{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\boldsymbol{x}) \leq \log p_{\boldsymbol{\theta}}(\boldsymbol{x}) \tag{21.14}
$$
其中 $\log p_{\boldsymbol{\theta}}(\boldsymbol{x})$ 表示对数边际似然，又被称为 **证据**（evidence）。所以 $\text{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\boldsymbol{x})$ 被称为 **证据下确界**（evidence lower bound, ELBO）。

我们可以将 ELBO 写成如下的几种形式：
$$
\begin{align}
\text{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\boldsymbol{x}) & =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right] \tag{21.15}\\
& =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})+\log p_{\boldsymbol{\theta}}(\boldsymbol{z})\right]+\mathbb{H}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right) \tag{21.16}\\
& =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]-D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right) \tag{21.17}
\end{align}
$$
我们将最后一种形式理解为对数似然的期望值加上一个正则项，这个正则项确保 （每个样本的）后验分布是“表现良好的”（与先验分布偏离得不会太多）。

证据下确界的紧凑程度由**变分差距**（variational gap）决定，即$D_{\mathbb{K} \mathbb{L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})\right)$。一个更好的后验近似结果将导致一个更紧凑的下确界。当 KL 为 0，则后验近似是精确的，那任何 ELBO 的提升将直接导致关于数据的似然值增加，这一点与 EM 算法十分相似 （6.6.3节）。

另一个最大化ELBO的替代方案是最小化负ELBO，又被称为 **变分自由能**（variational free energy）：
$$
\mathcal{L}(\boldsymbol{\theta}, \phi ; \boldsymbol{x})=\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[-\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]+D_{\mathbb{K L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right) \tag{21.18}
$$

### 21.2.3 计算 ELBO

为了近似地求解 ELBO，通过从后验分布中采样单个样本 $\boldsymbol{z}_s \sim q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})$，然后使用蒙特卡洛估计方法求取期望值
$$
\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right] \approx \frac{1}{S} \sum_{s=1}^S \log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}, \boldsymbol{z}_s\right) \tag{21.19}
$$
我们通常可以计算出 $\mathbb{H}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right)$ 的解析解，这取决于变分分布（即 $q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})$）的形式。举例而言，如果使用一个高斯后验分布，$q(\boldsymbol{z} \mid \boldsymbol{x})=\mathcal{N}(\boldsymbol{z} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$，我们可以使用式(5.95) 计算熵：
$$
\mathbb{H}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right)=\frac{1}{2} \ln |\boldsymbol{\Sigma}|+\text { const } \tag{21.20}
$$
类似的，我们也可以计算 KL项 $D_{\mathbb{K} \mathbb{L}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z})\right)$ 的解析解。举例而言，如果假设一个对角高斯先验分布 $p(\boldsymbol{z})=\mathcal{N}(\boldsymbol{z} \mid \mathbf{0}, \mathbf{I})$ 和一个对角高斯后验 $q(\boldsymbol{z} \mid \boldsymbol{x})=\mathcal{N}(\boldsymbol{z} \mid \boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}))$ ，我们可以使用式 (5.81) 计算 KL 的封闭形式：
$$
D_{\mathbb{K} \mathbb{L}}(q \| p)=-\frac{1}{2} \sum_{k=1}^K\left[\log \sigma_k^2-\sigma_k^2-\mu_k^2+1\right] \tag{21.21}
$$
其中 $K$ 表示隐变量的维度。

在某些情况下，我们可能无法得到熵或者KL的封闭解，此时我们只能使用蒙特卡洛方法近似所有项：
$$
\text{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\boldsymbol{x}) \approx \frac{1}{S} \sum_{s=1}^S\left[\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x} \mid \boldsymbol{z}_s\right)+\log p_{\boldsymbol{\theta}}\left(\boldsymbol{z}_s\right)-\log q_{\boldsymbol{\phi}}\left(\boldsymbol{z}_s \mid \boldsymbol{x}\right)\right] \tag{21.22}
$$
为简单起见，我们可以使用 $S=1$。



### 21.2.4 优化 ELBO

单个数据点 $\boldsymbol{x}$ 的 ELBO 由式 (21.17) 给定。对于整个数据集，需要考虑缩放因子 $N=|\mathcal{D}|$，即样本的数量：
$$
\mathrm{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathcal{D})=\frac{1}{N} \sum_{\boldsymbol{x}_n \in \mathcal{D}} \mathrm{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\boldsymbol{x}_n\right)=\frac{1}{N} \sum_{\boldsymbol{x}_n \in \mathcal{D}}\left[\mathbb{E}_{q_{\boldsymbol{\phi}}\left(\boldsymbol{z} \mid \boldsymbol{x}_n\right)}\left[\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_n \mid \boldsymbol{z}\right)+\log p_{\boldsymbol{\theta}}(\boldsymbol{z})-\log q_{\boldsymbol{\phi}}\left(\boldsymbol{z} \mid \boldsymbol{x}_n\right)\right]\right] \tag{21.23}
$$
我们的目标是关于 $\theta$ 和 $\phi$ 最大化上式：第一项鼓励模型更好地拟合数据本身，第二项鼓励缩小近似后验分布与真实后验分布之间的KL散度。

通过采样样本 $\boldsymbol{x}$ ，我们可以构建一个关于上式的无偏估计，然后便可以基于某个给定的样本计算目标函数值。所以现在我们聚焦在某个固定的样本 $\boldsymbol{x}$，为简单起见，我们省略了公式中关于 $\boldsymbol{x}_n$ 求和的符号。

关于生成模型的参数 $\boldsymbol{\theta}$ 的梯度是比较容易计算的，因为我们可以将导数项置于期望项内部，然后使用单点蒙特卡洛采样：
$$
\begin{align}
\nabla_{\boldsymbol{\theta}} \mathfrak{Ł}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\boldsymbol{x}) & =\nabla_{\boldsymbol{\theta}} \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right] \tag{21.24}\\
& =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\nabla_{\boldsymbol{\theta}}\left\{\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right\}\right] \tag{21.25}\\
& \approx \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}, \boldsymbol{z}^s\right) \tag{21.26}
\end{align}
$$
其中 $\boldsymbol{z}^s \sim q_\boldsymbol{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$ 。这是一个关于梯度的无偏估计，所以可以使用SGD进行求解。

然而关于推理网络参数 $\boldsymbol{\phi}$ 的梯度却很难计算，因为
$$
\begin{align}
\nabla_{\boldsymbol{\phi}} \mathrm{E}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\boldsymbol{x}) & =\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})\right] \tag{21.27}\\
& \neq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\nabla_{\boldsymbol{\phi}}\left\{\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \tag{21.28}\mid \boldsymbol{x})\right\}\right]
\end{align}
$$
然而，我们可以使用重参数技巧（在21.2.5节讨论）。（我们也可以使用 blackbox VI, 10.3.2节进行过讨论。）



### 21.2.5 使用重参数技巧计算 ELBO 梯度

本节，我们将讨论重参数技巧，用于计算关于分布参数的梯度。我们在 6.5.4 节讨论了相关细节，但此处总结一些基本原理。

重参数的技巧关键在于将随机变量 $\boldsymbol{z} \sim q_\boldsymbol{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$ 重写成另一个随机变量 $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$ 经过某个可微（可逆)）变换 $r$ 的结果，同时新的随机变量 $\boldsymbol{\epsilon}$ 不依赖于参数 $\boldsymbol{\phi}$ ，换句话说，我们假设：
$$
\boldsymbol{z}=r(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \boldsymbol{x}) \tag{21.29}
$$
比方说，
$$
\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma})) \Longleftrightarrow \boldsymbol{z}=\boldsymbol{\mu}+\boldsymbol{\epsilon} \odot \boldsymbol{\sigma}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{21.30}
$$
使用上式，我们有：
$$
\mathbb{E}_{q_\boldsymbol{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}[f(\boldsymbol{z})]=\mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\boldsymbol{z})] \text { s.t. } \boldsymbol{z}=r(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \boldsymbol{x}) \tag{21.31}
$$
其中我们定义
$$
f(\boldsymbol{z})=\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})-\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \tag{21.32}
$$
所以
$$
\nabla_\boldsymbol{\phi} \mathbb{E}_{q_\boldsymbol{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}[f(\boldsymbol{z})]=\nabla_\boldsymbol{\phi} \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\boldsymbol{z})]=\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_\boldsymbol{\phi} f(\boldsymbol{z})\right] \tag{21.33}
$$
上式可以通过单点蒙特卡洛采样实现近似。这使我们能够将梯度沿着函数 $f$ 传回，并进入用于计算$\boldsymbol{z}=r(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \boldsymbol{x})$ 的 DNN 转换函数 $r$。图21.2 给出了示例。

由于我们现在正在处理随机变量 $\boldsymbol{\epsilon}$，因此需要使用变量转换公式来计算
$$
\log q_\phi(\boldsymbol{z} \mid \boldsymbol{x})=\log p(\boldsymbol{\epsilon})-\log \left|\operatorname{det}\left(\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}}\right)\right| \tag{21.34}
$$
其中 $\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}}$ 为雅各比矩阵：
$$
\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}}=\left(\begin{array}{ccc}
\frac{\partial z_1}{\partial \epsilon_1} & \cdots & \frac{\partial z_1}{\partial \epsilon_k} \\
\vdots & \ddots & \vdots \\
\frac{\partial z_k}{\partial \epsilon_1} & \cdots & \frac{\partial z_k}{\partial \epsilon_k}
\end{array}\right) \tag{21.35}
$$
通过设计变换函数 $\boldsymbol{z}=r(\boldsymbol{\epsilon})$ 使得雅各比矩阵容易计算。接下来我们给出几个案例。



#### 21.2.5.1 完全因式分解的高斯分布（**Fully factorized Gaussian**）

假设我们有一个完全因式分解的高斯后验分布：
$$
\begin{align}
\boldsymbol{\epsilon} & \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{21.36}\\
\boldsymbol{z} & =\boldsymbol{\mu}+\boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \tag{21.37} \\
(\boldsymbol{\mu}, \log \boldsymbol{\sigma}) & =e_{\boldsymbol{\phi}}(\boldsymbol{x}) \tag{21.38}
\end{align}
$$
则雅各比矩阵为 $\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}}=\operatorname{diag}(\sigma)$，所以
$$
\log q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})=\sum_{k=1}^K \log \mathcal{N}\left(\epsilon_k \mid 0,1\right)-\log \sigma_k=\sum_{k=1}^K-\frac{1}{2} \log (2 \pi)-\frac{1}{2} \epsilon_k^2-\log \sigma_k \tag{21.39}
$$

#### 21.2.5.2 完全协方差高斯分布（**Full covariance Gaussian**）

现在考虑一个完全协方差高斯后验分布：
$$
\begin{align}
& \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{21.40} \\
& \boldsymbol{z}=\boldsymbol{\mu}+\mathbf{L}\boldsymbol{\epsilon} \tag{21.41}
\end{align}
$$
其中 $\mathbf{L}$ 是一个下三角矩阵，且在对角线上存在非零项，满足 $\boldsymbol{\Sigma}=\mathbf{L L}^{\top}$。该变换的雅各比矩阵为 $\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}}=\mathbf{L}$。因为 $\mathbf{L}$ 是一个三角矩阵，所以它的行列式是主对角元素的乘积，所以
$$
\log \left|\operatorname{det} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}}\right|=\sum_{k=1}^K \log \left|L_{k k}\right| \tag{21.42}
$$
我们需要使转换函数 $r$ 的参数成为关于输入 $\boldsymbol{x}$ 的函数。一种方法是定义：
$$
\begin{align}
\left(\boldsymbol{\mu}, \log \sigma, \mathbf{L}^{\prime}\right) & =e_{\boldsymbol{\phi}}(\boldsymbol{x}) \tag{21.43} \\
\mathbf{L} & =\mathbf{M} \odot \mathbf{L}^{\prime}+\operatorname{diag}(\boldsymbol{\sigma}) \tag{21.44}
\end{align}
$$
其中 $\mathbf{M}$ 是一个掩码矩阵，在对角线及以上全为元素0，在对角线一下为元素1。通过这个构造，矩阵 $\mathbf{L}$ 的对角线项由 $\boldsymbol{\sigma}$ 给定，所以
$$
\log \left|\operatorname{det} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}}\right|=\sum_{k=1}^K \log \left|L_{k k}\right|=\sum_{k=1}^K \log \sigma_k \tag{21.45}
$$
算法 21.1 给出了对应的计算重参数 ELBO 的伪代码。

```伪代码

```



#### 21.2.5.3 逆自回归流 (**Inverse autoregressive flows**)

在第10.4.3 节，我们将讨论如何使用反自回归流来学习更具表现力的后验分布 $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$，利用这种非线性变换雅可比的可计算性。

