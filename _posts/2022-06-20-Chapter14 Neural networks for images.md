---
title: 14 面向图像数据的神经网络
author: fengliang qi
date: 2022-06-20 11:33:00 +0800
categories: [MLAPP-CN, PART-III]
tags: [ml, dnn，cnn]
math: true
mermaid: true
toc: true
comments: true
---

> 本章，我们将开始介绍处理图像类型的数据的神经网络——卷积神经网络(CNN)。
>

* TOC
{:toc}
## 14.1 引言

> <table><tr><td bgcolor=blue>名词对照表</td></tr></table> 
>- 1
> - 2
> - 3
> - 4
> - 
> - 
> - 
> - 
> 
><table><tr><td bgcolor=blue>内容导读</td></tr></table> 
> - CNN 的结构入门
>- 

在多层感知机 ($\textrm{MLP}$) 模型中，每一个隐含层的核心操作是计算 $\mathbf{z}=\varphi(\mathbf{W} \mathbf{x})_{j}$。然而，如果输入是一张大小不定的图像 $\mathbf{x} \in \mathbb{R}^{W H C}$，其中 $W$ 表示图像的宽度，$H$ 表示高度，$C$ 表示输入的通道数 (例如, $C=3$ 表示 $\textrm{RGB}$ 三个颜色通道)。此时我们需要针对每一种输入的尺寸学习大小不同的权重矩阵。除此之外，哪怕输入的尺寸是固定的，对于一张大小合理的图像，所需的参数数量也是令人望而却步的。

![kernel-pattern](/assets/img/figures/kernelPattern.png)

{: style="width: 100%;" class="center"}
图 $14.1$：我们可以通过寻找出现在正确（相对）位置的某些判别特征 (图像模板) 来对数字进行分类。 来自[Cho][^Cho]的图$5.1$。 经 $\textrm{Francois Chollet}$ 许可后使用。
{:.image-caption}

本章，我们将讨论**卷积神经网络**（$\textrm{convolutional neural networks, CNN}$)。我们将原有的矩阵乘操作替换成一个卷积操作，我们将在 $14.2$ 节解释这部分内容，但其基本思想可以简单归纳为——我们将图片切分成若干互相重叠的 $\textrm{2d}$ **图像块** ($\textrm{image patches}$)，然后将每一个图像块与若干尺寸较小的权重矩阵进行匹配，这些权重矩阵又被称为 **滤波器** ($\textrm{filters}$)，表示一个目标的某个部分，上述过程如图 $14.1$ 所示。我们可以将卷积操作称之为 **模板匹配** ($\textrm{template matching}$)。关键之处在于我们可以从数据中学习这些模板，也可以在特征空间——而不仅仅是在像素空间——中进行匹配。这些模板可以作用于输入的任意位置，从而减少了参数的数量，同时也确保模型对于输入具有**平移不变性** ($\textrm{translation-invariant}$)。这个性质对于图像分类任务——判断图像中任意位置是否存在某一类目标——十分有用。除了图像分类，$\textrm{CNNs}$ 还有很多其他领域的应用，正如我们将在本章进行讨论的——$\textrm{1d}$ 输入 ($15.3$ 节) 和 $\textrm{3d}$ 输入；然而，本章我们重点讨论 $\textrm{2d}$ 输入的场景。

![conv-matrix-vector](/assets/img/figures/convMatrixVector.png)

{: style="width: 100%;" class="center"}
图 $14.2$：使用 $\mathbf{w}=[5,6,7]$ 对 $\mathbf{x}=[1,2,3,4]$ 进行离散卷积并返回 $\mathbf{z}=[5,16,34,52,45,28]$。我们发现这个操作包括“翻转” $\mathbf{w}$ ， "遍历" $\mathbf{x}$，逐元素相乘，以及对结果求和。
{:.image-caption}

![1d-cross-correlation](/assets/img/figures/1dcrossCorrelation.png)

{: style="width: 100%;" class="center"}
图 $14.3$：一维交叉关联。改编自[Zha+19a][Zha19a]的图$13.10.1$。
{:.image-caption}



## 14.2 基本原理

本节，我们将讨论 $\textrm{CNNs}$ 的基本原理。

### 14.2.1 一维卷积

两个函数之间的卷积 $f,g:\mathbb{R}^D\rightarrow \mathbb{R}$，定义为：
$$
[f \circledast g](\mathbf{z})=\int_{\mathbb{R}^{D}} f(\mathbf{u}) g(\mathbf{z}-\mathbf{u}) d \mathbf{u} \tag{14.1}
$$
现在假设我们用长度有限的向量替换函数，换句话说，我们可以将其视为定义在有限数量的点集上的函数。举个例子，假设只考虑函数 $f$ 在点 $\{-L,-L+1,...,0,1,...,L\}$ 处的响应值，并构成权重向量 (又被称为 **滤波器** 或者 **核** ($\textrm{kernel}$)) —— $w_{-L}=f(-L)$ ... $w_L=f(L)$。现在令函数 $g$ 在点 $\{-N,...,N\}$ 处的特征向量表示为 —— $x_{-N}=g(-N)$ ... $x_N=g(N)$。然后上述公式将转换成：
$$
[\mathbf{w} \circledast \mathbf{x}](i)=w_{-L} x_{i+L}+\cdots+w_{-1} x_{i+1}+w_{0} x_{i}+w_{1} x_{i-1}+\cdots+w_{L} x_{i-L} \tag{14.2}
$$
(我们稍后会讨论边界条件)。不难发现，我们将权重向量 $\mathbf{w}$ 进行了翻转 (因为它的下标索引是倒序的)，然后将其在向量 $\mathbf{x}$ 的元素上进行滑动遍历，并且在每一个位置上对局部视窗内的元素进行求和，整个过程如图 $14.2$ 所示。

有一个十分类似的操作，其中我们不对权重向量 $\mathbf{w}$ 进行翻转：
$$
[\mathbf{w} * \mathbf{x}](i)=w_{-L} x_{i-L}+\cdots+w_{-1} x_{i-1}+w_{0} x_{i}+w_{1} x_{i+1}+\cdots+w_{L} x_{i+L} \tag{14.3}
$$
这被称为**交叉关联** ($\textrm{cross correlation}$)；如果权重向量是对称的 (且通常是这种情况)，那么交叉关联与卷积则是相同的操作。在深度学习的文献中，术语“卷积”通常是指交叉关联；我们将遵循这一惯例。

我们也可以不考虑下标索引的负数部分——权重 $\mathbf{w}$ 和特征 $\mathbf{x}$ 分别只考虑定义域 $\{0,1,...,L-1\}$ 和 $\{0,1,...,N-1\}$ 处的计算。此时上式将变成：
$$
[\mathbf{w} \circledast \mathbf{x}](i)=\sum_{u=0}^{L-1} w_{u} x_{i+u} \tag{14.4}
$$
图 $14.3$ 给出了相应的示例。

{:.image-caption}

![2d-cross-correlation](/assets/img/figures/2d_crosscorr.png)

{: style="width: 100%;" class="center"}
图 $14.4$：二维交叉关联。改编自[Zha+19a][Zha19a]的图$6.2.1$。
{:.image-caption}

{:.image-caption}

![3times3conv](/assets/img/figures/14_5.png)

{: style="width: 100%;" class="center"}
图 $14.5$：使用一个 $3\times3$ 的卷积核(中)对 $2d$ 图片(左)进行卷积并生成一张 $2d$ 响应图(右)。响应图中高亮的区域对应图片中包含对角线模式的右上与左下区域。图片引用自 [Cho][^Cho] 的图 $5.3$。经  Francois Chollet 许可后使用。
{:.image-caption}



### 14.2.2 二维卷积

对于二维场景，式 $(14.4)$ 变成
$$
[\mathbf{W} \circledast \mathbf{X}](i, j)=\sum_{u=0}^{H-1} \sum_{v=0}^{W-1} w_{u, v} x_{i+u, j+v} \tag{14.5}
$$
其中二维滤波器 $\mathbf{W}$ 的尺寸为 $H\times W$。举例来说，考虑使用一个大小为 $2\times2$ 的核 $\mathbf{W}$ 对一个 $3\times3$ 的输入 $\mathbf{X}$ 进行卷积，将得到一个大小为 $2\times2$ 的输出 $\mathbf{Z}$：
$$
\begin{align}
\mathbf{Z} &=\left(\begin{array}{ll}
w_{1} & w_{2} \\
w_{3} & w_{4}
\end{array}\right) *\left(\begin{array}{lll}
x_{1} & x_{2} & x_{3} \\
x_{4} & x_{5} & x_{6} \\
x_{7} & x_{8} & x_{9}
\end{array}\right) \tag{14.6} \\
&=\left(\begin{array}{ll}
\left(w_{1} x_{1}+w_{2} x_{2}+w_{3} x_{4}+x_{4} x_{5}\right) & \left(w_{1} x_{2}+w_{2} x_{3}+w_{3} x_{5}+x_{4} x_{6}\right) \\
\left(w_{1} x_{4}+w_{2} x_{5}+w_{3} x_{7}+x_{4} x_{8}\right) & \left(w_{1} x_{5}+w_{2} x_{6}+w_{3} x_{8}+x_{4} x_{9}\right)
\end{array}\right) \tag{14.7}
\end{align}
$$
图 $14.4$ 展示了这个过程。

我们可以将二维卷积视为 **模板匹配** ($\textrm{template matching}$)，因为以坐标 $(i,j)$ 为中心的图像块与矩阵 $\mathbf{W}$ 越相似，则 $(i,j)$ 处的响应值越大。如果模板 $\mathbf{W}$ 对应于一个有向边，那么使用该卷积核对输入进行卷积，则输出的**热图** ($\textrm{heat map}$) 中，那些包含与模板方向一致的边的区域将会被“点亮”，如图 $14.5$ 所示。更一般地来讲，我们将卷积视作 **特征检测** ($\textrm{feature detection}$) 的一种形式，最终的输出 $\mathbf{Z}=\mathbf{W} \circledast \mathbf{X}$ 因此被称为 **特征图** ($\textrm{feature map}$)。

### 14.2.3 卷积的矩阵-向量乘形式

既然卷积是一个线性操作，我们便可以将其表示为矩阵乘。举例来说，考虑公式 ($14.7$)，只需要将二维矩阵 $\mathbf{X}$ 平展为一维向量 $\mathbf{x}$，并且乘上一个类常对角 ($\textrm{Toeplitz-like}$) 矩阵，我们便可以将其重写成如下的矩阵-向量乘形式：
$$
\begin{align}
\mathbf{z} &=\mathbf{C x}=\left(\begin{array}{ccc|ccc|ccc}
w_{1} & w_{2} & 0 & w_{3} & w_{4} & 0 & 0 & 0 & 0 \\
0 & w_{1} & w_{2} & 0 & w_{3} & w_{4} & 0 & 0 & 0 \\
0 & 0 & 0 & w_{1} & w_{2} & 0 & w_{3} & w_{4} & 0 \\
0 & 0 & 0 & 0 & w_{1} & w_{2} & 0 & w_{3} & w_{4}
\end{array}\right)\left(\begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \\
x_{5} \\
x_{6} \\
x_{7} \\
x_{8} \\
x_{9}
\end{array}\right) \tag{14.8} \\
&=\left(\begin{array}{l}
w_{1} x_{1}+w_{2} x_{2}+w_{3} x_{4}+w_{4} x_{5} \\
w_{1} x_{2}+w_{2} x_{3}+w_{3} x_{5}+w_{4} x_{6} \\
w_{1} x_{4}+w_{2} x_{5}+w_{3} x_{7}+w_{4} x_{8} \\
w_{1} x_{5}+w_{2} x_{6}+w_{3} x_{8}+w_{4} x_{9}
\end{array}\right) \tag{14.9}
\end{align}
$$
通过将 $4\times1$ 的向量 $\mathbf{z}$ 进行 $\textrm{reshape}$，我们可以回溯 $2\times2$ 的输出矩阵 $\textbf{Z}$。所以我们发现 $\textrm{CNNs}$ 与 $\textrm{MLPS}$ 是类似的，只不过前者的权重矩阵是一个特殊的稀疏结构，并且同一些矩阵元素被分布在整个矩阵空间。这一特点实现了平移不变性的思想，于此同时，相较于一个标准的全连接层或者稠密层的权重矩阵，这种方式大幅减少了参数的数量。

{:.image-caption}

![padding_and_stride](/assets/img/figures/14_6.png)

{: style="width: 100%;" class="center"}
图 $14.6$：$2d$ 卷积中 $padding$ 和 $stride$ 的示意图。 $(a)$ 我们使用一个 $3\times3$ 的卷积核对一个大小为 $5\times7$ 的输入进行 "$\textit{same convolution}$" ，并得到大小为 $5\times 7$ 的输出。$(b)$ 我们将步长设置为 $2$，此时的输出大小为 $3\times 4$。图像改自 [Ger19][^Ger19] 的图 $14.3-14.4$。
{:.image-caption}

### 14.2.4 边界条件和步长

在公式 $(14.7)$中 ，我们发现如果使用一个 $2\times2$ 的卷积核对 $3\times 3$ 的图像进行卷积，最终将得到一个大小为 $2\times 2$ 的输出。通常来说，如果图像的大小为 $x_h\times x_w$ ，卷积核大小为 $f_h \times f_w$，则最终的输出大小为 $(x_h-f_h+1)\times (x_w-f_w+1)$，这被称为 **有效卷积** ($\textrm{valid convolution}$)，因为只是将卷积核应用于图像的“有效”部分，换句话说，我们不让它 “从两端滑出”。如图 $14.6a$ 所示，如果我们希望输出的大小与输入保持一致，我们可以使用 $\textbf{zero-padding}$，意味着我们可以在图像的边界进行补 $0$ 操作，这被称为 $\textbf{same convolution}$。

考虑到每一个输出像素其实是它所属**感受野** ($\textrm{receptive field}$，与卷积核大小有关) 的加权组合，相邻的输出在数值上将会十分相似，因为它们的输入是重合的。通过每隔 $s$ 步进行一次卷积的方式，我们可以缓解这种冗余性 (同时可以加速计算)，这被称为 **步长卷积** ($\textrm{strided convolution}$)。如图 $14.6a$ 所示，我们使用一个大小为 $3\times3$ 的卷积核对一个大小为 $5\times7$ 的图片进行卷积，所使用的步长为 $2$，最终得到的输出大小为 $3\times4$。

一般情况下，如果输入的大小为 $x_h\times x_w$，卷积核大小为 $f_h\times f_w$，并且在宽高两个维度的补零的大小分别为 $p_h$ 和 $p_w$，使用的步长分别为 $s_h$ 和 $s_w$，然后最终的输出大小为 [$\textrm{DV16}$][^DV16]：
$$
\left\lfloor\frac{x_{h}+2 p_{h}-f_{h}+s_{h}}{s_{h}}\right\rfloor \times\left\lfloor\frac{x_{w}+2 p_{w}-f_{w}+s_{w}}{s_{w}}\right\rfloor \tag{14.10}
$$
举例来说，考虑图 $14.6(a)$。我们有 $p=1, f=3, s=1, x_{h}=5$ 以及 $x_w = 7$，所以输出的大小为
$$
\left(\left\lfloor\frac{5+2-3+1}{1}\right\rfloor,\left\lfloor\frac{7+2-3+1}{1}\right\rfloor\right)=\left(\left\lfloor\frac{5}{1}\right\rfloor,\left\lfloor\frac{7}{1}\right\rfloor\right)=5 \times 7 \tag{14.11}
$$
通常情况下，如果我们使用的补全大小是卷积核大小的一半，即 $p=\lfloor f / 2\rfloor$，(其中 $f$ 是奇数)，那么输出的大小与输入的大小保持一样。

现在考虑图 $14.6(b)$，其中我们将步长设置为 $s=2$。此时的输出尺寸将小于输入：
$$
\left(\left\lfloor\frac{5+2-3+2}{2}\right\rfloor,\left\lfloor\frac{7+2-3+2}{2}\right\rfloor\right)=\left(\left\lfloor\frac{6}{2}\right\rfloor,\left\lfloor\frac{4}{1}\right\rfloor\right)=3 \times 4 \tag{14.12}
$$

[^DV16]:

{:.image-caption}

![multi_channels_conv](/assets/img/figures/14_7.png)

{: style="width: 100%;" class="center"}
图 $14.7$：将 $2d$ 卷积应用于 $2$ 个通道的输入。图片改自 [Zha+19a][^Zha+19a]的 $6.4.1$ 。
{:.image-caption}

{:.image-caption}

![multi_conv_layers](/assets/img/figures/14_8.png)

{: style="width: 100%;" class="center"}
图 $14.8$：包含 $2$ 个卷积层的 $CNN$。输入包含 $3$ 个颜色通道。中间的特征图包含多个通道。圆柱体对应于超列，表示某个特定位置的特征向量。图片改自 [Ger19][^Ger19] 的图$14.6$。
{:.image-caption}



#### 14.2.4.1 多输入与多输出通道

在图 $14.5$，输入是一张灰度图。通常情况下，输入存在多个**通道** ($\textrm{channels}$)（例如， $\textrm{RGB}$ 或者卫星图像的高光谱波段），通过对每一个通道都定义一个卷积核，我们便可以应对这种情况；所以此时的卷积核 $\mathbf{W}$ 是一个 $\textrm{3d}$ 权重矩阵或者**张量** ($\textrm{tensor}$)。在计算输出的过程中，我们使用核 $\mathbf{W}_{:,:,c}$ 来对通道 $c$ 进行卷积，然后沿着通道进行求和：
$$
z_{i, j}=b+\sum_{u=0}^{H-1} \sum_{v=0}^{W-1} \sum_{c=0}^{C-1} x_{s i+u, s j+v, c} w_{u, v, c} \tag{14.13}
$$
其中 $s$ 表示步长 (为简单起见，我们假设在宽和高两个方向的步长相同)， $b$ 表示偏置项，图 $14.7$ 给出了示意。

每一个权重矩阵可以检测单个类型的特征。如图 $14.1$ 所示，通常情况下，我们是希望检测多种类型的特征。为实现这一点，我们可以将 $\mathbf{W}$ 替换成一个 $\textrm{4d}$ 权重矩阵。用于检测通道 $c$ 的第 $d$ 种特征的滤波器存储在 $\mathbf{W}_{:,:,c,d}$ 中。在这种情况下，我们可以将卷积的定义拓展为：
$$
z_{i, j, d}=b_{d}+\sum_{u=0}^{H-1} \sum_{v=0}^{W-1} \sum_{c=0}^{C-1} x_{s i+u, s j+v, c} w_{u, v, c, d}
$$
如图 $14.8$ 所示，每一个垂直的圆柱形的列表示在一个给定位置 $\mathbf{Z}_{i, j, 1: D}$ 处的输出特征；这有时也被称为一个**超列** ($\textrm{hypercolumn}$)。每个元素都是下层每个特征图的感受野中 $C$ 个特征的不同加权组合。

{:.image-caption}

![pointwise_conv](/assets/img/figures/14_9.png)

{: style="width: 100%;" class="center"}
图 $14.9$：$1\times1\times3\times2$ 卷积示意图。图片改自 [Zha+19a][^Zha+19a] 的图 $6.4.2$。
{:.image-caption}

#### 14.2.4.2 $1\times1$ $\textrm{(pointwise)}$ 卷积

有些时候我们只需要对一个给定的位置上的特征进行加权融合，而不需要考虑跨空间区域的元素。这一点可以通过使用 $\mathbf{1\times 1}$ 卷积来实现，也被称为 **逐点卷积** ($\textrm{pointwise convolution}$)，图 $14.9$ 展示了具体过程。该卷积将通道数由 $C$ 转换成 $D$，而没有改变空间维度的大小：
$$
z_{i, j, d}=b_{d}+\sum_{c=0}^{C-1} x_{i, j, c} w_{0,0, c, d} \tag{14.15}
$$
这可以被认为并行地对每一个特征列使用一个单层的 $\textrm{MLP}$。

{:.image-caption}

![max_pooling](/assets/img/figures/14_10.png)

{: style="width: 100%;" class="center"}
图 $14.10$：最大池化示意图。图像改自 [Zha+19a][^Zha+19a] 的图 $6.5.1$。
{:.image-caption}

#### 14.2.5 池化层

卷积可以保留输入特征的位置信息 (同时降低分辨率)，该性质被称为 $\textbf{equivariance}$。在某些情况下，我们希望关于位置具有 $\textrm{invariant}$。举个例子，在进行图像分类的过程中，我们只想知道一个感兴趣的目标 (比如，一张脸) 在图片的任意位置是否存在。

一种简单的实现方法被称为 **最大池化** ($\textrm{max pooling}$)，它只计算输入值中的最大值，如图 $14.10$ 所示。另一个替代方案是使用 **平均池化** ($\textrm{average pooling}$)，它将最大值替换成期望值。在任何一种情况下，无论输入模式出现在其感受野内的哪个位置，输出神经元都具有相同的响应 (请注意，我们将池化应用到每个特征通道上)。

如果我们对特征图中的所有位置进行平均，则该方法称为**全局平均池化** ($\textrm{global average pooling}$)。 因此，我们可以将 $H\times W\times D$ 的特征图转换为 $1 \times 1 \times D$ 维的特征图，后者可以重新整形为 $D$ 维向量，可以将其传递到全连接层中，以在传递到 $\textrm{softmax}$ 输出之前将其映射到 $C$ 维向量。 使用全局平均池化意味着我们可以将分类器应用于任何大小的图像，因为最终的特征图在映射到 $C$ 类上的分布之前总是会转换为固定的 $D$ 维向量。

{:.image-caption}

![normalization](/assets/img/figures/14_11.png)

{: style="width: 100%;" class="center"}
图 $14.11$：$CNN$ 中不同激活归一化方法示意图。每一个子图展示了一张特征图张量，其中 $N$ 表示 $batch$ 维度， $C$ 表示通道维度， $(H,W)$ 表示空间维度。蓝色的像素表示使用同样的期望和方差进行归一化。从左到右： $batch\ norm, layer\ norm, instance\ norm$ 和 $group\ norm$ (包含 $2$ 个 $3$ 通道组)。图像引用自 [WH18][^WH18]。经 $Kaiming\ He$ 允许后使用。
{:.image-caption}

### 14.2.6 $\textbf{Normalization}$ 层

在 $13.4.5$ 节我们讨论了 $\textbf{batch normalization}$ (批量归一化)，它将给定特征通道内的所有激活标准化为零均值和单位方差。 这可以极大地帮助模型训练。 然而，尽管批量归一化效果很好，但当批量较小时，它会遇到困难，因为估计的均值和方差参数可能不可靠。

一种解决方案是通过汇集张量其他维度的统计数据来计算均值和方差，而不是批次中的其他样本。 更准确地说，令 $z_i$ 表示张量的第 $i$ 个元素； 在二维图像的情况下，索引 $i$ 有 $4$ 个分量，分别表示批次、通道、高度和宽度：$i = (i_N, i_C, i_H, i_W )$。 我们计算每个索引 $z_i$ 的均值和标准差，如下所示：
$$
\mu_{i}=\frac{1}{\left|\mathcal{S}_{i}\right|} \sum_{k \in \mathcal{S}_{i}} z_{k}, \sigma_{i}=\sqrt{\frac{1}{\left|\mathcal{S}_{i}\right|} \sum_{k \in \mathcal{S}_{i}}\left(z_{k}-\mu_{i}\right)^{2}+\epsilon} \tag{14.16}
$$
式中 $\mathcal{S}_i$ 表示被平均的元素集合。我们然后计算 $\hat{z}_{i}=\left(z_{i}-\mu_{i}\right) / \sigma_{i}$ 和 $\tilde{z}_{i}=\gamma_{c} \hat{x}_{i}+\beta_{c}$，其中 $c$ 表示对应于索引 $i$ 的通道。

在 $\textrm{batch norm}$中，我们将批次、高度、宽度汇集在一起，因此 $\mathcal{S}_i$ 是张量中与 $i$ 的通道索引匹配的所有位置的集合。 为了避免小批量的问题，我们可以改为对通道、高度和宽度进行融合，但在批量索引上进行匹配。 这被称为**层归一化** ($\textrm{layer normalization}$) [$\textrm{BKH16}$][^BKH16]。 或者，我们可以为批处理中的每个样本和每个通道设置单独的归一化参数。 这被称为**实例归一化** ($\textrm{instance normalization}$) [$\textrm{UVL16}$][^UVL16]。

上述方法的自然推广称为**组归一化** ($\textrm{group normalization}$) [$\textrm{WH18}$][^WH18]，我们将所有通道与 $i$ 所在组中的所有位置汇集在一起。 如图 $14.11$ 所示。 层归一化是一种特殊情况，其中有一个包含所有通道的组。 实例归一化是一种特殊情况，其中有 $C$ 个组，每个通道一个。 在 [$\textrm{WH18}$][^WH18] 中，他们通过实验表明，使用大于单个通道但小于所有通道的组可能会更好（在训练速度以及训练和测试精度方面）。

{:.image-caption}

![normalization](/assets/img/figures/14_12.png)

{: style="width: 100%;" class="center"}
图 $14.12$：一个简单的用于分类 $MNIST$ 图片的 $CNN$。模型包含 $2$ 个卷积层， $2$ 个全连接层和$1$ 个 $\textrm{softmax}$ 输出。其中的方形框表示感受野。两个卷积核的大小为 $5\times5\times1\times n_1$ 和 $5\times 5 \times n_1 \times n_2$。两个全连接层的矩阵大小为 $(16n_2)\times n_3$ 和 $n_3 \times 10$。我们使用 $5 \times 5$ 的卷积核进行“$valid$” 卷积，意味着我们将输入的大小从 $28$ 像素每边降低到 $28-5+1=24$ (见$14.2.4$)。通过使用步长为 $2$ ，大小为 $2\times 2$ 的最大池化，我们将 $24$ 进一步降低到 $12$。第二个卷积进一步将输入大小降低到 $12-5+1=8$ 像素每边，第二个池化层将其降低到 $4$。在降低每一层的空间大小时，我们通常会增加特征的通道数 (所以 $n_2 \approx 2n_1$)。更多细节参照 $14.1$ 节。图片引用自 https: // bit. ly/ 2YB9o0H 。经  $Sumit\ Saha$ 允许后使用。
{:.image-caption}

### 14.2.7 组装在一起

一种常见的设计模式是通过交替卷积层和最大池化层来创建 CNN，最后是最终的线性分类层。 如图 $14.12$ 所示。 （在这个例子中我们省略了归一化层，因为模型深度很小）。这种设计模式首先出现在 $\textrm{Fukushima}$ 的 $\textbf{neocognitron}$ [$\textrm{Fuk75}$][^Fuk75] 中，并受到 $\textrm{Hubel}$ 和 $\textrm{Wiesel}$ 的人类视觉皮层中简单和复杂细胞模型的启发 [$\textrm{HW62}$][^HW62] 。 $1998$ 年，$\textrm{Yann LeCun}$ 在他的同名 $\textbf{LeNet}$ 模型 [$\textrm{LeC+98}$][^LeC+98] 中使用了类似的设计，该模型使用反向传播和 $\textrm{SGD}$ 来估计参数。 这种设计模式在视觉对象识别的神经启发模型 [$\textrm{RP99}$][^RP99] 以及各种实际应用中继续流行（参见第 $14.3$ 节和第 $14.4$ 节）。

[^BKH16]:
[^UVL16]:
[^WH18]:
[^FuK75]:
[^HW62]:
[^LeC+98]:
[^RP99]:

-------

## 14.3 使用 $\textbf{CNNs}$ 实现图像分类

我们一般使用 $\textrm{CNNs}$ 进行图像分类，该任务需要估计函数 $f: \mathbb{R}^{H \times W \times K} \rightarrow\{0,1\}^{C}$，其中 $K$ 表示输入的通道数 (比如， $\textrm{RGB}$ 图像的 $K = 3$)， $C$ 表示类别标签的数量。本节，我们将讨论集中常见的针对该任务的数据集和模型。

### 14.3.1 常用数据集

机器学习文献经常使用各种标准标注的图像数据集来对模型和算法进行基准测试。 我们将在下面讨论其中的一些。

#### 14.3.1.1 小规模数据集

最简单和最广泛使用的一种数据集称为 $\textbf{MNIST}$ [LeC+98][^LeC98]; [YB19][^YB19]。 这是一个包含 $\textrm{60k}$ 张大小为 $28 \times 28 \times 1$ 的训练图像的数据集，包含来自 $10$ 个类别的手写数字； 参见图 $3.19a$ 的说明。

如今，由于 $\textrm{MNIST}$ 分类被认为“太容易”，因此已经收集了其他数据集。 $\textrm{MNIST}$ 的一个简单替代品称为 $\textrm{FashionMNIST}$ [XRV17][^XRV17]。 这是一个包含 $\textrm{60k}$ 张大小为 $28 \times 28 \times 1$ 的图像的数据集，包含了 $\textrm{10}$ 个类别的服装项目； 参见图 $\textrm{14.13a}$ 的说明。 另一个灰度数据集是 $\textrm{EMNIST}$ [Coh+17][^Coh17]，它是一个包含 $\textrm{730k}$ 张大小为$28 \times 28 \times 1$ 的图像的数据集，展示了来自 $\textrm{47}$ 个类别的手写数字和字母。

对于小规模的彩色图像，最常用的数据集是 $\textbf{CIFAR}$ [KH09][^KH09]。该数据集包含约 $\textrm{50k}$ 张大小为 $32 \times 32 \times 3$ 的图片，其内容包含来自于日常目标中的 $10$ 个或 $100$ 个类别；图 $14.13b$ 给出了相关说明。

#### 14.3.1.2 $\textbf{ImageNet}$ 竞赛

小数据集对于模型设计很有用，但在更大的数据集上测试方法也很重要，无论是图像大小还是标注样本的数量。 这种类型最广泛使用的数据集称为 $\textbf{ImageNet}$ [Rus+15][Rus15]。 这是一个包含 $1400$ 万张大小为 $256 \times 256 \times 3$ 的图像的数据集，包含来自 $20,000$ 个类别的各种对象； 一些例子见图 $14.14a$ 所示。

$\textrm{ImageNet}$ 数据集被用作 $\textrm{ImageNet Large Sclae Visual Recognition Challenge (ILSRVC)}$ 的基础，该挑战赛从 $2010$ 年举办到 $2018$ 年，其中使用了来自 $1000$ 个类别的 $130$ 万张图像的子集。 在比赛过程中，社区取得了重大进展，如图 $14.14b$ 所示。 特别是，$2015$ 年标志着 $\textrm{CNN}$ 在对来自 $\textrm{ImageNet}$ 的图像进行分类的任务中可以胜过人类（或至少一个人类，即 $\textrm{Andrej Karpathy}$）的第一年。 请注意，这并不意味着 $\textrm{CNN}$ 在视觉方面比人类更好（参见例如图 $14.15$ 中的一些失败案例）。 相反，它很可能反映了数据集做出了许多细粒度的分类区别的事实——例如“老虎”和“虎猫”之间的区别——人类难以理解。 相比之下，足够灵活的 $\textrm{CNN}$ 可以学习任意模式，包括随机标签 [Zha+17a][Zha17a]。

#### 14.3.1.3 其他数据集

$\textrm{ML}$ 研究论文中常用的还有许多其他图像数据集。 例如，在研究生成模型 (相对于图像分类器) 时，通常使用 $\textbf{CelebA}$ 数据集 [Liu+15][^Liu15]，其中包含 $10,177$ 位名人的 $202,599$ 张图像。每张图像都用 $40$ 个二进制属性进行注注，例如“戴眼镜”或“卷发”。 有关该数据集的一些案例，请参见图 $14.16$，我们在第 $20.2.6.1$ 和 $20.3.5.2$ 节中使用了这些示例。(数据集的高分辨率版本，称为 $\textbf{CelebA-HQ}$，可在 [Kar+18][^Kar18] 中获得。)

### 14.3.2 常用模型

在本节中，我们将简要回顾多年来为解决图像分类任务而开发的各种 $\textrm{CNN}$。 有关图像分类器的更广泛综述，请参见例如 [Gu+18b][^Gu18b]。

#### 14.3.2.1 LeNet

最早的 $\textrm{CNN}$ 之一创建于 $1998$ 年，被称为 $\textbf{LeNet}$ [LeC+98][^LeC98]，以其创建者 $\textrm{Yann LeCun}$ 的名字命名。 它旨在对手写数字的图像进行分类，并在 $3.7.2$ 节中介绍的 $\textrm{MNIST}$ 数据集上进行了训练。 该模型如图 $14.17a$ 所示。 该模型的一些预测如图 $14.18$ 所示。 仅仅 $1$ 个 $\textrm{epoch}$ 之后，测试准确率就已经达到了 $98.8\%$。 相比之下，$13.2.4.2$ 中的 $\textrm{MLP}$ 在 1 个 $\textrm{epoch}$ 后的准确率为 $95.9\%$。 更多轮次的训练可以进一步提高准确性，使其性能与标签噪声无法区分。

当然，对孤立数字进行分类的适用性有限：在现实世界中，人们通常会写出一串数字或其他字母。 这需要分割和分类。 $\textrm{LeCun}$ 及其同事设计了一种将卷积神经网络与类似于条件随机场的模型相结合的方法来解决这个问题。 该系统由美国邮政局部署。 有关该系统的更详细说明，请参见 [LeC+98][^Lec98]。

#### 14.3.2.2 AlexNet

虽然 $\textrm{CNN}$ 已经存在很多年了，但直到 $2012$ 年 [KSH12][^KSH12] 的论文，主流计算机视觉研究人员才开始关注它们。 在那篇论文中，作者展示了如何将 $\textrm{ImageNet}$ 挑战（第 $14.3.1.2$ 节）的 (前 $5$ 名) 错误率从之前的 $26\%$ 降低到 $15\%$，这是一个巨大的进步。 该模型被称为 $\textbf{AlexNet}$ 模型，以其创建者 $\textrm{Alex Krizhevsky}$ 的名字命名。

图 $14.17b(b)$ 显示了架构。 它与图 $14.17a$ 所示的 $\textrm{LeNet}$ 非常相似，但有以下区别：它更深（$8$ 层可调参数（即不包括池化层）而不是 $5$ 层）； 它使用 $\textrm{ReLU}$ 非线性而不是$\textrm{tanh}$（请参阅第 $13.2.3$ 节了解为什么这很重要）； 它使用 $\textrm{dropout}$（第 $13.5.4$ 节）进行正则化而不是权重衰减； 并且它将几个卷积层堆叠在一起，而不是在卷积和池化之间严格交替。 将多个卷积层堆叠在一起的优点是，随着一层的输出被馈送到另一层，感受野变得更大（例如，连续三个 $3\times3$ 滤波器的感受野大小为 $7\times7$）。 这比使用具有更大感受野的单层要好，因为多层之间也存在非线性。此外，三个  $3\times3$ 滤波器的参数少于一个 $7\times7$ 滤波器。

请注意，$\textrm{AlexNet}$ 有 $\textrm{60M}$ 的自由参数（比 $\textrm{1M}$ 标记的示例多得多），主要是由于输出端的三个全连接层。 拟合这个模型依赖于使用两个 $\textrm{GPU}$（由于当时 $\textrm{GPU}$ 的内存有限），并且被广泛认为是工程杰作。 图 $14.14a$ 显示了模型对来自 $\textrm{ImageNet}$ 的一些图像所做的一些预测。

#### 14.3.2.3 GoogLeNet (Inception)

谷歌开发了一种称为 $\textbf{GoogLeNet}$ [Sze+15a][^Sze15a] 的模型 (这个名字是 $\textrm{Google}$ 和 $\textrm{LeNet}$ 的双关语)。与早期模型的主要区别在于 $\textrm{GoogLeNet}$ 使用了一种新的 $\textrm{block}$，称为 $\textbf{inception block}$ [^5]，它采用多个并行路径，每个路径都有一个不同尺寸的卷积滤波器。 有关说明，请参见图 $14.19$。 这让模型可以学习每一层的最佳滤波器的大小。 整个模型由 $9$ 个 $\textrm{inception block}$ 组成，然后是全局平均池化层。 请参见图 $14.20$ 中的说明。 自从该模型首次问世以来，就提出了各种扩展； 详情可见[IS15][^IS15]; [Sze+15b][^Sze15b];[SIV17][^SIV17]。

#### 14.3.2.4 ResNet

2015 年 $\textrm{ImageNet}$ 分类挑战赛的获胜者是微软的一个团队，他们提出了一个名为 $\textbf{ResNet}$ [He+16a][He16a] 的模型。其核心思想是将 $\mathbf{x}_{l+1}=\mathcal{F}_{l}\left(\mathbf{x}_{l}\right)$ 替换为
$$
\mathbf{x}_{l+1}=\varphi\left(\mathbf{x}_{l}+\mathcal{F}_{l}\left(\mathbf{x}_{l}\right)\right) \tag{14.17}
$$
这被称为 $\textbf{residual block}$，因为 $\mathcal{F}_l$ 只需要学习输入与输出之间的残差或者差异——这是个更简单的任务。在 [He+16a][^He16a] 中， $\mathcal{F}$ 的结构为 $\textrm{conv-BN-relu-conv-BN}$，其中 $\textrm{conv}$ 是一个卷积层， $\textrm{BN}$ 是一个 $\textrm{batch norm}$ 层 ($13.4.5$ 节)。(如果需要，我们可以在跨连路径上添加 $1\times1$ 卷积，以补偿 $\mathbf{x}_{l}$ 和 $\mathcal{F}\left(\mathbf{x}_{l}\right)$ 之间不同数量的通道)。图 $\textrm{14.21}$ 为对应示意图。

$\textrm{residual block}$ 的使用使我们能够训练非常深的模型。 例如，[He+16a][^He16a] 在  $\textrm{ImageNet}$ 上训练了一个 $152$ 层的 $\textrm{ResNet}$。 这可能的原因是梯度可以通过跳过连接直接从输出层到更浅的层，原因在 $13.4.4$ 节中解释。

在 [He+16b][^He16b] 中，他们展示了如何对上述方案进行小幅修改，从而使我们能够训练多达 $1001$ 层的模型。 关键的思路是，由于在加法步骤 $\mathbf{x}_{l+1}=\varphi\left(\mathbf{x}_{l}+\mathcal{F}_{l}\left(\mathbf{x}_{l}\right)\right)$ 之后使用了非线性激活函数，跳过连接上的信号仍在衰减。 他们表明，以下类型的 $\textrm{residual block}$ 效果更好：
$$
\mathbf{x}_{l+1}=\mathbf{x}_{l}+\mathcal{F}_{l}^{\prime}\left(\mathbf{x}_{l}\right) \tag{14.18}
$$
其中 $\mathcal{F}^{\prime}\left(\mathbf{x}_{l}\right)$ 是 $\textrm{Conv-BN-Relu-Conv-BN-Relu}$。这称为$\textbf{preactivation resnet}$ 或简称 $\textbf{PreResnet}$，因为它在通过非线性传递函数 $\varphi$ 之前返回激活。

非常深的 $\textrm{ResNet}$ 模型的问题在于，没有什么可以强制模型使用残差块，因此可能会浪费很多层。 $\textbf{wide resnet}$ 模型 [ZK16][^ZK16] 通过使用较少数量的“更宽”残差层（即具有更多卷积通道）来修改  $\textrm{ResNet}$。

## 14.4 使用 CNNs 解决其他判别式视觉任务

在本节中，我们将简要讨论如何使用 $\textrm{CNN}$ 处理各种其他视觉任务。 针对每个不同任务，在原有的模型基础构建模块库内引入了新的创新结构。 有关用于计算机视觉的 $\textrm{CNN}$ 的更多详细信息，请参见 [Bro19][^Bro19]。

### 14.4.1 Image tagging

图像分类将单个标签与整个图像相关联，即假设输出是互斥的。 在许多问题中，一张图像中可能存在多个目标，我们要标记所有目标——这被称为**图像标记**（$\textrm{image tagging}$），是多标签分类的一种应用。 在这种情况下，我们将输出空间定义为 $\mathcal{Y}=\{0,1\}^{C}$ ，其中 $C$ 是标签类型的数量。 由于输出的每个槽位是独立的（给定图像），我们应该用 $C$ 个逻辑单元替换最终的 $\textrm{softmax}$。

$\textrm{Instagram}$ 等社交媒体网站的用户经常为他们的图片创建哈希标签； 因此，这提供了一种创建大型有标注数据集的“免费”方式。 当然，很多标签的使用可能相当稀疏，它们的含义在视觉上也可能没有很好的定义 (例如，有人可能会在接受 $\textrm{COVID}$ 测试后给自己拍照并将图像标记为 $\textrm{“#covid”}$；但是，在视觉上它与其他图像并无二异）。因此这种用户生成的标签通常被认为含噪的。 然而，正如 [Mah+18][^Mah18] 中所讨论的，它对于模型“预训练”很有用。

最后，值得注意的是，图像标记通常是比图像分类更符合直觉的任务，因为许多图像中包含多个目标，并且很难知道我们应该标记哪个目标。 事实上，在 $\textrm{ImageNet}$ 上创建 “$\textrm{human performance benchmark}$” 的 $\textrm{Andrej Karpathy}$ 指出[^6]:

```markdown
Both [CNNs] and humans struggle with images that contain multiple ImageNet classes (usually many more than five), with little indication of which object is the focus of the image. This error is only present in the classification setting, since every image is constrained to have exactly one correct label. In total, we attribute 16% of human errors to this category.
```

### 14.4.2 Object detection

在某些情况下，我们希望模型产生可变数量的输出，对应于图像中可能存在的可变数量的感兴趣目标。(这是一个**开放世界** ($\textrm{open world}$) 问题的案例——对象数量未知)。

一个典型的例子是**目标检测** ($\textrm{object detection}$)，我们必须返回一组表示感兴趣目标位置的边界框，以及它们的类别标签。一个特殊情况是**人脸检测** ($\textrm{face detection}$)，其中图像中只有一类感兴趣目标，如图 $14.22a$ 所示[^7]。

解决此类检测问题的最简单方法是将其转换为封闭世界问题，其中任何对象都可以存在有限数量的候选位置（和方向）。这些候选位置称为**锚框** ($\textrm{anchor boxes}$)。 我们可以在多个位置、比例和纵横比上创建锚框，如图 $14.22b$ 所示。 对于每个锚框，我们训练系统预测它包含的对象类别（如果有的话)；我们还可以使用回归来预测对象位置与锚点中心的偏移量 (这些残差回归项允许更加精细的空间定位)。

抽象地，我们正在学习形式的函数
$$
f_{\theta}: \mathbb{R}^{H \times W \times K} \rightarrow[0,1]^{A \times A} \times\{1, \ldots, C\}^{A \times A} \times\left(\mathbb{R}^{4}\right)^{A \times A}\tag{14.19}
$$
其中 $A$ 是每个维度中锚框的数量，$C$ 是对象类型（类标签）的数量。 对于每个框位置 $(i, j)$，我们预测三个输出：对象存在概率 $p_{ij} \in [0, 1]$，对象类别 $y_{ij}\in\{1, . . . , C\}$ 和两个 $2d$ 偏移向量 $\bold{δ}_{ij}\in \mathbb{R}^4$，可以与锚框的质心相加以获得左上角和右下角坐标。

业内已经提出了几种这种类型的模型，包括 [Liu+16][^Liu16] 的 $\textbf{single shot detector}$ 和 [Red+16][^Red16] 的 $\textbf{YOLO}$（$\textrm{you only look once}$）模型。 多年来，人们提出了许多其他的目标检测方法。 这些模型在速度、准确性、简单性等方面做出了不同的权衡。参见 [Hua+17][^Hua17] 进行的经验比较，参见 [Zha+18][^Zha18] 获得最近的综述。

### 14.4.3 Human pose estimation

我们可以训练一个物体检测器来检测人。 我们还可以训练它来预测它们的 $2d$ 形状——由一组固定的骨骼关键点的位置表示，例如头部或手部的位置。 这称为**人体姿态估计** ($\textrm{human pose estimation}$)，如图 $14.23a$ 所示。 有几种技术，例如 $\textbf{PersonLab}$ [Pap+18][^Pap18] 和 $\textbf{OpenPose}$ [Cao+18][^Cao18]。 有关最近的评论，请参阅 [Bab19][^Bab19]。

我们还可以预测每个检测到的对象的 $3d$ 属性。 主要限制是收集足够标记训练数据的能力，因为人类注释者很难在 $3d$ 中标记事物。 但是，我们可以使用计算机图形引擎创建具有无限真实 $3d$ 注释的模拟图像（参见 [GNK18][^GNK18]）。

### 14.4.4 Image segmentation

我们可以为每个输入像素创建一个输出标签，以二维网格排列。这可以用于**语义分割** ($\textrm{semantic segmentation}$)，我们必须为每个像素预测一个类标签 $y_{i} \in\{1, \ldots, C\}$ (这不应与我们在 $14.4.4.3$ 节中讨论的实例分割相混淆)。 

处理语义分割的一种常见方法是使用**编码器-解码器** ($\textrm{encoder-decoder}$)架构，如 $14.4.4$ 节所示。编码器使用标准卷积将输入映射到一个小的 $2d$ $\textrm{bottleneck}$，它以粗略的空间分辨率捕获输入的高级属性。这使用了一种称为空洞卷积的技术，我们在 $14.4.4.1$ 节中进行了解释。解码器使用我们在第 $14.4.4.2$ 节中解释的称为转置卷积的技术将小的 $2d$ $\textrm{bottleneck}$映射回全尺寸输出图像。由于 $\textrm{bottleneck}$ 会丢失信息，我们还可以添加从输入层到输出层的旁路连接。整体结构类似于字母 $\textrm{U}$，因此也称为 $\textbf{U-net}$ [RFB15][^RFB15]。 

类似的架构可用于其他**密集预测** ($\textrm{dense prediction}$)或**图像到图像** ($\textrm{image-to-image}$) 的任务，例如**深度预测** ($\textrm{depth prediction}$)（对于每个像素 $i$, 预测与相机的距离，$z_i\in\mathbb{R}$）、**表面法线预测** ($\textrm{surface normal prediction}$)（对每一个图像块，预测表面的方向, $\mathbf{z}_i \in \mathbb{R}^3$) 等等。我们当然可以训练一个模型来同时解决所有这些任务，使用多个输出头，如图 $14.25$ 所示 (详见 [Kok17][^Kok17]）。

#### 14.4.4.1 Dilated convolution

卷积是一种组合局部邻域中的像素值的操作。 通过使用跨步，并将多层卷积堆叠在一起，我们可以扩大每个神经元的感受野，这是每个神经元响应的输入空间区域。 然而，我们需要很多层来为每个神经元提供足够的上下文来覆盖整个图像（除非我们使用非常大的过滤器，这会很慢并且需要太多参数）。

作为替代方案，我们可以使用带孔的卷积 [Mal99]，有时以法语术语 à trous 算法而闻名，最近更名为扩张卷积 [YK16]。 这种方法在执行卷积时只取每个 r'th 输入元素，其中 r 被称为速率或膨胀因子。 例如，在 1d 中，使用速率 r = 2 与滤波器 w 进行卷积等价于使用滤波器 w~ = [w1, 0, w2, 0, w3] 进行常规卷积，其中我们插入了 0 来扩展感受野（因此 术语“带孔的卷积”）。 这使我们能够在不增加参数数量或计算量的情况下获得增加的感受野的好处。

更准确地说，2d 中的空洞卷积定义如下：
$$
z_{i, j, d}=b_{d}+\sum_{u=0}^{H-1} \sum_{v=0}^{W-1} \sum_{c=0}^{C-1} x_{i+r u, j+r v, c} w_{u, v, c, d} \tag{14.20}
$$
为简单起见，我们假设高度和宽度的比率 r 相同。 将此与方程式进行比较。 (14.14)，其中步幅参数使用 xsi+u,sj+v,c。

#### 14.4.4.2  Transposed convolution

在第 14.4.4.1 节中，我们讨论了扩张卷积，其中我们在过滤器权重中放置 0（或“孔”）（以扩大其视野），然后应用常规卷积。 我们也可以想象做相反的事情，我们在输入图像中放置 0 以使其更大，然后应用常规卷积。 这称为转置卷积，因为对应的矩阵运算因为它从小输入映射到大输出。 请注意，转置卷积有时也称为反卷积，但这是对该术语的错误用法：反卷积是“取消”卷积效果的过程
过滤器，例如模糊过滤器。

#### 14.4.4.3 Instance segmentation

在语义分割中，我们为每个像素附加一个标签，但我们没有将像素分组为对象。 因此，输出的大小与输入的大小相同。 相比之下，在实例分割中，目标是预测图像中每个对象实例的标签和 2d 形状，如图 14.23b 所示。 在这种情况下，输出的数量是可变的。 我们可以将“东西”的语义分割和“事物”的实例分割组合成一个连贯的框架，称为“全景分割”[Kir+19]。



在本书的部分 $\mathrm{II}$， 我们讨论了线性模型在回归和分类任务的应用。在第 $\textrm{11}$ 章，我们讨论了线性回归模型$p(y|\mathbf{x}, \mathbf{w})=\mathcal{N}\left(y|\mathbf{w}^{\top}\mathbf{x}, \sigma^{2}\right)$  。在第 $\textrm{10}$ 章，我们讨论了逻辑回归，其中对于二分类情况，模型定义为 $p(y|\mathbf{x}, \mathbf{w})={\rm{Ber}}(y|\sigma(\mathbf{w}^{\top}\mathbf{x}))$；在多分类任务中，模型定义为 $ p(y|\mathbf{x},\mathbf{w})=\operatorname{Cat}(y|\mathcal{S}(\mathbf{W} \mathbf{x}))$。在第 $\textrm{12}$ 章，我们进一步讨论了广义线性模型，定义为:
$$
p(\mathbf{y}|\mathbf{x};\pmb{\theta})=p(\mathbf{y}|g^{-1}(f(\mathbf{x};\pmb{\theta}))) \tag{13.1}
$$
其中 $p(\mathbf{y}|\pmb{\mu})$ 表示均值为 $\pmb{\mu}$ 的指数族分布，$g^{-1}()$ 表示该指数族分布对应的逆连接函数。式中:
$$
f(\mathbf{x};\pmb{\theta})=\mathbf{Wx} + \mathbf{b} \tag{13.2}
$$
表示关于输入 $\mathbf{x}$ 的一个线性（仿射）变换函数， 其中 $\mathbf{W}$ 表示 **权重** ($\textrm{weights}$)，$\mathbf{b}$ 表示 **偏置** ($\textrm{biases}$)。

---

在线性模型中，我们依然假设输出与输入 $\mathbf{x}$ 呈线性关系，该假设具有很强的局限性。为了增加此类线性模型的灵活性，一种简单方法是使用特征变换，即利用 $\phi(\mathbf{x})$ 替代 $\mathbf{x}$。比如我们可以使用多项式变换。具体而言，对于 $\textrm{1}$ 维数据，特征变换函数可以定义为 $\phi(x)=[1,x,x^2,x^3,...]$，我们在 $\textrm{1.2.2.2}$ 节中对该方法进行了讨论。这种方法有时被称为 **基函数拓展** ($\textrm{basis function expansion}$)。在特征变换的基础上，式 $13.2$ 可以定义为:
$$
f(\mathbf{x}; \pmb{\theta})=\mathbf{W}\mathbf{\phi}(\mathbf{x}) + \mathbf{b} \tag{13.3}
$$
需要注意的是，上述模型的输出关于参数 $\pmb{\theta}=(\mathbf{W}, \mathbf{b})$ 依然是线性关系，这样可以降低了模型的拟合难度。然而，手动设计的特征变换函数依然具有很强的局限性。

---

一个很自然的泛化是为特征提取器赋予自己的参数 $\pmb{\theta}^\prime$， 即:
$$
f(\mathbf{x};\pmb{\theta},\pmb{\theta}^\prime)=\mathbf{W}\phi(\mathbf{x};\pmb{\theta}^\prime)+\mathbf{b} \tag{13.4}
$$
我们可以递归地重复上述过程，从而构造一个越来越复杂的函数。如果我们组合 $L$ 个函数，即
$$
f(\mathbf{x};\pmb{\theta})=f_L(f_{L-1}(...(f_1(\mathbf{x}))...))\tag{13.5}
$$
其中 $f _l(\mathbf{x})=f(\mathbf{x};\pmb{\theta} _l)$ 为第 $l$ 层的函数。这便是 **深度神经网络** ($\textrm{deep neural networks, DNNs}$) 背后的关键思想。

---

术语 “$\textrm{ DNN}$ ” 实际上包含了一大类模型，这类模型的特点在于它们都是将多个可微函数组合成任何类型的 $\textrm{DAG}$（有向无环图），从而实现输入到输出的映射函数的建模， 式 $13.5$ 是其中的最简单的一个例子，它的 $\textrm{DAG}$ 是一个链式结构。“$\textrm{ DNN}$ ”又被称为 **前馈神经网络**（$\textrm{feedforward neural network, FFNN}$）或 **多层感知机**（$\textrm{multilayer perceptron, MLP}$）。

$\textrm{MLP}$ 假定输入是一个维度固定的矢量，即 $\mathbf{x} \in \mathbb{R}^D$。 我们称此类数据为“**非结构化数据**” ($\textrm{unstructured data}$)，因为我们没有对输入的形式作任何假设。 但是，$\textrm{MLP}$ 很难应用于具有可变大小或形状的输入。 在第 $\textrm{14}$ 章中，我们将讨论**卷积神经网络**（$\textrm{convolutional neural networks, CNN}$），用于处理可变大小的图像。 在第 $\textrm{15}$ 章中，我们将讨论**递归神经网络**（$\textrm{recurrent neural networks, RNN}$），用于处理可变长度的序列。 在第 $\textrm{23}$ 章中，我们将讨论**图神经网络**（$\textrm{graph neural networks, GNN}$），用于处理可变形状的图数据。 有关 $\textrm{DNN}$ 的更多信息，可参考其他书籍 [$\textrm{HG20}$][^HG20], [$\textrm{Zha19a}$][^Zha19a], [$\textrm{Ger19}$][^Ger19]。

[^HG20]: $\textrm{[HG20]}$: J. Howard and S. Gugger. Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD. en. 1st ed. O’Reilly Media, Aug. 2020.
[^Zha19a]: $\textrm{[Zha19a]}$ : A. Zhang, Z. Lipton, M. Li, and A. Smola. Dive into deep learning. 2019.
[^Ger19]: $\textrm{[Ger19]}$ : A. Géron. Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques for Building Intelligent Systems (2nd edition). en. O’Reilly Media, Incorporated, 2019.

***

---

## 13.2 多层感知机

> <table><tr><td bgcolor=blue>名词对照表</td></tr></table> 
>- 不可微: non-differentiable
> - 优化器：optimizer
> 
><table><tr><td bgcolor=blue>内容导读</td></tr></table> 
> - 传统感知机模型并不能有效解决 $\textrm{XOR}$ 这一模式识别问题; 
>- 多层感知机 $\textrm{MLP}$ 可以有效解决上述难点，但其所包含的不可微的单位阶跃激活函数限制了其被大规模使用的可能;
> - 通过使用可微的非线性激活函数，可以有效改善上述问题，但应避免使用存在饱和区的激活函数（会导致梯度消失的问题）。

在第 $\textrm{10.2.5}$ 节，我们介绍了 **感知机** ($\textrm{perceptron}$) 模型， 它实际上就是逻辑回归模型的一个确定性版本（$\textrm{deterministic version}$）。具体而言，它是一个具备如下形式的映射函数:
$$
f(\mathbf{x};\mathbf{\theta})=\mathbb{I}(\mathbf{w}^{\rm{T}}\mathbf{x}+b\ge0)=H(\mathbf{w}^{\rm{T}}\mathbf{x}+b) \tag{13.6}
$$
其中 $H(a)$ 表示 **单位阶跃函数**（$\textrm{heaviside step function}$）， 又被称为 **线性阈值函数** ($\textrm{linear threshold function}$）。由于感知机模型的决策边界依然是线性的，所以表达能力十分有限。$\textrm{1969}$ 年，$\textrm{Marvin Minsky}$ 和 $\textrm{Seymour Papert}$ 出版了一本名为 $\textrm{《 Perceptrons》}$ [$\textrm{MP69}$][^MP69] 的著作，书中列举了许多感知机无法解决的模式识别问题。 在讨论如何解决这些问题之前，我们首先分析其中的一个具体示例。

[^MP69]: $\textrm{[MP69]}$ M. Minsky and S. Papert. Perceptrons. MIT Press, 1969.

| $x_1$ | $x_2$ | $y$  |
| ----- | ----- | :--: |
| 0     | 0     |  0   |
| 0     | 1     |  1   |
| 1     | 0     |  1   |
| 1     | 1     |  0   |

{: style="width: 100%;" class="center"}
表$13.1$： 抑或问题的真值表，$y=x _{1} \underline{\vee} x _{2}$。
{:.image-caption}



![xor-heaviside](/assets/img/figures/xor-heaviside.png)

{: style="width: 100%;" class="center"}
图 $13.1$：$\textrm{(a)}$ 抑或函数无法实现线性可分，但基于单位阶跃函数构建的两层模型可以将数据分开。程序由 $xor-heaviside.py$ 生成。 $\textrm{(b)}$ 包含一个隐藏层的神经网络，其中的权重由人工设计，该网络实现了抑或函数。 $h_1$ 表示 $AND$ 函数，$h_2$ 表示 $OR$ 函数。 偏置项表示为常数节点（值为 $\textrm{1}$）的连接权重。
{:.image-caption}

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

$\textrm{《 Perceptrons》}$书中最著名的例子之一就是 $\textrm{XOR}$ 问题。 在该示例中，我们的目标是学习一个函数，用于计算两个二进制输入的异或值。 表 $\textrm{13.1}$ 给出了该函数的真值表。 我们在图 $\textrm{13.1a}$ 中对真值表进行了可视化。 显然，其中的数据并不是线性可分离的，因此感知机模型无法实现对该映射函数的建模。

但是，我们可以通过叠加多个感知机模型来克服这个难点，即 **多层感知机**（$\textrm{multilayer perceptron, MLP}$）。 例如，要解决 $\textrm{XOR}$ 问题，我们可以使用图 $\textrm{13.1b}$  所示的 $\textrm{MLP}$。 它由 $\textrm{3}$ 个感知机组成，分别为 $h_1$，$h_2$ 和 $y$。 节点 $x$ 表示输入，节点 $1$ 表示常数项， 节点 $h_1$ 和 $h_2$ 被称为 **隐藏单元** （$\textrm{hidden units}$），因为在训练数据中未观察到它们的真值。

第一个隐藏单元利用合理设置的权重来计算 $h_{1}=x_{1} \wedge x_{2}$（ $\wedge$ 表示 $\rm{AND}$ 操作）。具体而言，它的输入为 $x_1$和 $x_2$，且权重均为 $\textrm{1.0}$，同时具有 $\textrm{-1.5}$ 的偏置项（通过虚设一个常量节点 $\textrm{1}$ 来模拟偏置项）。 因此，如果 $x_1$ 和 $x_2$ 都等于 $\textrm{1}$，则 $h_1$ 将被激活，因为
$$
\mathbf{w}_1^{\rm{T}}\mathbf{x}-b_1=[1.0, 1.0]^{\rm{T}}[1, 1] - 1.5 =0.5 > 0 \tag{13.7}
$$
类似的，第二个隐藏单元计算 $h_{2}=x_{1} \vee x_{2}$，其中 $\vee$ 为 $\rm{OR}$ 操作，第三个节点计算输出 $y=\overline{h_1} \wedge h_2$，其中 $\bar{h}=\neg h$ 为 $\rm{NOT}$ 操作 （ 逻辑非）。 所以节点 $y$ 表示为
$$
y=f\left(x_{1}, x_{2}\right)=\overline{\left(x_{1} \wedge x_{2}\right)} \wedge\left(x_{1} \vee x_{2}\right) \tag{13.8}
$$
上式等价于 $\rm{XOR}$ 函数。

将上述案例一般化， $\textrm{MLP}$ 可以用来表示任何逻辑函数。 但是，我们显然希望避免手动设置权重和偏置。 在本章的其余部分，我们将讨论从数据中学习这些参数的方法。

---

### 13.2.2 可微多层感知机

我们在第 $\textrm{13.2.1}$ 节中讨论的 $\textrm{MLP}$ 被定义为多个感知机的叠加，每个感知机都包含不可微的 $\textrm{Heaviside}$ 函数。 这使得模型很难训练，这就是为什么它们从未被广泛使用的原因。然而，如果我们将阶跃函数 $H:\mathbb{R}\rightarrow \{0,1\}$ 替换为一个可微的 **激活函数** （$\textrm{activation function}$）$\varphi:\mathbb{R} \rightarrow \mathbb{R}$ 。：更准确地说，我们将每层 $l$ 的隐藏单元 $\mathbf{z}_l$ 定义为通过这个激活函数进行逐元素传递的前一层的隐藏单元的线性变换的结果。

> <font color=red>More precisely, we define the hidden units $\mathbf{z}_l$ at each layer $l$ to be a linear transformation of the hidden units at the previous layer passed elementwise through this activation function.</font>


$$
\mathbf{z}_l=f_l(\mathbf{z}_{l-1})=\varphi(\mathbf{b}_l+\mathbf{W}_l\mathbf{z}_{l-1}) \tag{13.9}
$$


或者，以标量的形式表示为：


$$
z_{kl}=\varphi_l \left( b_{kl}+\sum_{j=1}^{K_{l-1}}w_{jkl}z_{jl-1} \right) \tag{13.10}
$$

如式 $ 13.5$ 所示，如果我们现在将 $L$ 个诸如此类的激活函数叠加在一起，然后使用链式规则，计算输出关于每一层中参数的梯度，也称为 **反向传播** （$\textrm{backpropagation}$），如我们在第 $\textrm{13.3}$ 节中所解释的。 （这对于任何一种可微的激活函数都是可行的，尽管某些类型的激活函数要比其他类型的函数更适用，正如我们在第 $\textrm{13.2.3}$ 节中讨论的那样）。然后，我们可以将梯度传递给优化器，从而最小化某些训练目标，正如我们在 $\textrm{13.4}$ 节讨论的那样。 因此，术语“$\textrm{ MLP}$”几乎总是指可微的模型，而不是指具有不可微的线性阈值单位的历史版本。

---

| $\textrm{Name}$                    | $\textrm{Definition}$                                        | $\textrm{Range}$     | $\textrm{Reference}$            |
| ---------------------------------- | ------------------------------------------------------------ | -------------------- | ------------------------------- |
| $\textrm{Sigmoid}$                 | $\sigma_{a}=\frac{1}{1+e^{-a}}$                              | $[0, 1]$             |                                 |
| $\textrm{Hyperbolic tangent}$      | $\tanh(a)=2\sigma(2a)-1$                                     | $[-1, 1]$            |                                 |
| $\textrm{Softplus}$                | $\sigma_{+}(a)=\log(1+e^a)$                                  | $[0, \infty]$        | [GBBB11][^GBB11]                |
| $\textrm{Rectified linear unit}$   | $\operatorname{ReLU}(a)=\max (a, 0)$                         | $[0, \infty]$        | [GBB11][^GBB11];[KSH12][^KSH12] |
| $\textrm{Leaky ReLU}$              | $\max (a, 0)+\alpha \min (a, 0)$                             | $[-\infty, +\infty]$ | [MHN13][^MHN13]                 |
| $\textrm{Exponential linear unit}$ | $\max (a, 0)+\min \left(\alpha\left(e^{a}-1\right), 0\right)$ | $[-\infty, +\infty]$ | [CUH16][^CUH16]                 |
| $\textrm{Swish}$                   | $a \sigma(a)$                                                | $[-\infty, +\infty]$ | [RZL17][^RZL17]                 |

{: style="width: 100%;" class="center"}
表 $\textrm{13.2}$：神经网络中常用的一些激活函数
{:.image-caption}



[^GBB11]: $\textrm{[GBB11]}$: X. Glorot, A. Bordes, and Y. Bengio. “Deep Sparse Rectifer Neural Networks”. In: AISTATS. 2011.
[^KSH12]: $\textrm{[KSH12]}$: A. Krizhevsky, I. Sutskever, and G. Hinton. “Imagenet classification with deep convolutional neural networks”. In: NIPS. 2012.
[^MHN13]: $\textrm{[MHN13]}$: A. L. Maas, A. Y. Hannun, and A. Y. Ng. “Rectifier Nonlinearities Improve Neural Network Acoustic Models". In: ICML. Vol. 28. 2013.
[^CUH16]: $\textrm{[CUH16]}$: D.-A. Clevert, T. Unterthiner, and S. Hochreiter. “Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)”. In: ICLR. 2016.
[^RZL17]: $\textrm{[RZL17]}$: P. Ramachandran, B. Zoph, and Q. V. Le. “Searching for Activation Functions”. In: (Oct. 2017). arXiv: 1710.05941 [cs.NE].

![activations](/assets/img/figures/activations.png)

{: style="width: 100%;" class="center"}
图 $\textrm{13.2}$：$\textrm{(a)}$ 对于 $sigmoid$ 函数而言，当输入在 $0$ 附近时，输出与输入呈线性关系，但对于较大的正值或负值输入，则输出存在饱和区。图形由程序 生成。 $\textrm{(b)}$ 一些常用的非饱和激活函数的可视化。图形由程序 生成。
{:.image-caption}

### 13.2.3 激活函数

我们可以在每一层使用任何一种可微的激活函数。然而，如果我们使用 *线性* （$\textrm{linear}$）激活函数  $\varphi_l(a)=c_la$，那整个模型将退化为一个常规的线性模型。以式 $13.5$ 为例，该模型将退化为：


$$
f(\mathbf{x};\mathbf{\theta})=\mathbf{W}_Lc_L(\mathbf{W}_{L-1}c_{L-1}(...(\mathbf{W}_1\mathbf{x})...)) \propto \mathbf{W}_L\mathbf{W}_{L-1}...\mathbf{W}_1\mathbf{x}=\mathbf{W}^\prime\mathbf{x} \tag{13.11}
$$


在上式中，为了符号上的简洁性，我们丢弃了偏置项。基于上述原因，使用非线性激活函数就显得十分重要。

在神经网络发展的早期阶段，通常会使用 $\textrm{S}$ 型 ($\textrm{logistic}$) 激活函数，该函数可以看做是单位阶跃函数的平滑近似版本。然而，如图 $\textrm{13.2a}$ 所示，对于较大的正值输入，$\textrm{S}$ 形函数存在饱和输出 $\textrm{1}$；对于较大的负值输入，$\textrm{S}$ 形函数存在饱和输出 $\textrm{0}$。 $\textrm{tanh}$ 激活函数具有相似的形状，但其饱和输出分别为 $\textrm{-1}$ 和 $\textrm{+1}$。 在这些饱和区域，输出关于输入的斜率将接近于零。因此，如我们在第 $\textrm{13.4.2}$  节中所讨论的，来自深层网络的任何梯度信号都将“消失”。

---

要想成功训练一个非常深的神经网络模型，一个关键因素是使用 **非饱和激活函数** （$\textrm{non-saturating activation functions}$）。几种不同的激活函数如表 $\textrm{13.2}$ 所示。其中最常用的是 **整流线性单元** （$\textrm{rectifled linear unit, ReLU}$）。定义为

$$
{\rm{ReLU}}(a)=\max(a,0)=a\mathbb{I}(a>0) \tag{13.12}
$$


该 $\textrm{ReLU}$ 函数简单地将负值输入置零，并保持正值输入保持不变。如 $\textrm{13.3.3.2}$ 节所介绍的，这种形式至少保证对于正值输入，梯度的值为 $\textrm{1}$，从而缓解了梯度消失的问题。

不幸的是，对于负值输入，$\textrm{ReLU}$ 的梯度依然为 $\textrm{0}$，因此，该单元将永远无法获得任何反馈信号来帮助其摆脱当前的参数设置 （**译者注：**即无法逃离负值区域）； 这被称为 “$\pmb{\textrm{dying ReLU}}$” 问题。

一种简单的解决方法是使用 [$\textrm{MHN13}$][^MHN13] 中提出的 $\pmb{\textrm{leaky ReLU}}$。 定义为


$$
{\rm{LReLU}}(a;\alpha)=\max(\alpha a,a) \tag{13.13}
$$

其中 $0 \lt \alpha \lt 1$。该函数对于正值输入的斜率为 $\textrm{1}$，对于负值输入的斜率为 $\alpha$， 所以可以确保当输入为负值时，依然可以有信号可以从更深的网络层中反传回来。如果我们允许参数 $\alpha$ 是可学习而非固定的， $\pmb{\textrm{leaky ReLU}}$ 将被称为  $\pmb{\textrm{parametric ReLU}}$。

另一个普遍的选择是 [$\textrm{CUH16}$][^CUH16] 中提出的 $\textrm{ELU}$，定义为


$$
{\rm{ELU}}(a;\alpha) = \begin{cases}
\alpha(e^a-1) & \text{if } a\le0\\
a & \text{if } a \gt 0
\end{cases} \tag{13.14}
$$

与 $\pmb{\textrm{leaky ReLU}}$ 相比，它具有平滑函数的优点。

在 [$\textrm{Kla+17}$][^Kla17] 中提出了一种 $\textrm{ELU}$ 的变体，称为 $\pmb{\textrm{SELU}}$（$\textrm{self-normalizing ELU}$）。 定义为


$$
{\rm{SELU}}(a;\alpha,\lambda)=\lambda{\rm{ELU}}(a;\alpha) \tag{13.15}
$$

[^Kla17]: $\textrm{[Kla+17]}$: G. Klambauer, T. Unterthiner, A. Mayr, and S.Hochreiter. “Self-Normalizing Neural Networks”. In: NIPS. 2017.

令人惊讶的是，他们证明了通过精心为 $\alpha$ 和 $\lambda$ 设置合理的值，即使不使用 $\textrm{batchnorm}$ 技术（见 $\textrm{13.4.5}$节），也可以确保通过激活函数来使每个网络层的输出是被标准化的（假设输入也已标准化），从而加快模型的拟合速度。

作为手动发现良好的激活函数的替代方法，我们可以使用黑盒优化方法来对激活函数空间进行搜索。 [$\textrm{RZL17}$][^RZL17] 使用这种方法发现了被称为 $\mathrm{swish}$ 的函数，该函数在某些图像分类数据集上似乎表现很好。该函数定义为


$$
{\rm{swish}}(a;\beta)=a\sigma(\beta a)\tag{13.16}
$$

有关这些激活函数的可视化对比，请参见图 $\textrm{13.2b}$。 我们发现，不同的激活函数主要是在处理负值输入的方式上存在差异。

---

---

### 13.2.4 案例模型

> <table><tr><td bgcolor=blue>名词对照表</td></tr></table> 
>- 嵌入矩阵: embedding matrix
> - 词嵌入: word embedding
> - 微调: fine-tune
> - 分类分布： categorical distribution
> 
><table><tr><td bgcolor=blue>内容总结</td></tr></table>
> - 介绍了 $\textrm{MLP}$ 在不同类型数据以及不同任务上的应用实例。

$\textrm{MLPs}$ 可以对很多类型的数据进行分类和回归，接下来我们将给出一些案例。

![mlpIris](/assets/img/figures/mlpIris.jpg)

{: style="width: 100%;" class="center"}
图 $\textrm{13.3}$：一个简单的用于鸢尾花分类问题的 $2$ 层 $\textrm{MLP}$。隐藏层中的节点代表隐藏单元 $h _{1,i}$ 和 $h _{2,j}$，节点间的边代表权重 $w _{2,i,j}$ ，其中 $2$ 表示第二层。 （第一层的权重矩阵对应于 $\textrm{input-to-hidden}$ 映射。）所以$\mathbf{z}  _{2}=\varphi _{2}(\mathbf{W} _{2} \mathbf{z} _{1}+\mathbf{b} _{1})$ ，尽管为了表达上的简洁，我们省略了偏置项。
{:.image-caption}

#### 13.2.4.1 MLP用于表格数据分类

图 $\textrm{13.3}$ 给出了一个包含两个隐藏层的 $\textrm{MLP}$ 示意图，将该 $\textrm{MLP}$ 应用于 $\textrm{1.2.1.1}$ 节中的表格鸢尾花数据集，该数据集具有 $\textrm{4}$ 个特征和 $\textrm{3}$ 个类别。 该模型具有如下形式


$$
\begin{align}
p(y|\mathbf{x};\mathbf{\theta})=& {\rm{Cat}}(y|f_3(\mathbf{x};\mathbf{\theta})) \tag{13.17}\\
f_3(\mathbf{x};\mathbf{\theta})=&\mathcal{S}(\mathbf{W}_3f_2(\mathbf{x};\mathbf{\theta})+\mathbf{b}_3) \tag{13.18} \\
f_2(\mathbf{x};\mathbf{\theta})=&\varphi_2(\mathbf{W}_2f_1(\mathbf{x};\mathbf{\theta})+\mathbf{b}_2) \tag{13.19} \\
f_1(\mathbf{x};\mathbf{\theta})=&\varphi_1(\mathbf{W}_1f_0(\mathbf{x};\mathbf{\theta})+\mathbf{b}_1) \tag{13.19} \\
f_0(\mathbf{x};\mathbf{\theta})=&\mathbf{x} \tag{13.21}
\end{align}
$$

其中 $\mathbf{\theta}=(\mathbf{W}_3,\mathbf{b}_3,\mathbf{W}_2,\mathbf{b}_2,\mathbf{W}_1,\mathbf{b}_1)$ 为模型中的参数，对应于 $\textrm{3}$ 组可调节权重的边。我们看到最终（输出）层的激活函数为 $\textrm{softmax}$ 函数，$\textrm{softmax}$ 函数是分类分布的反向连接函数。对于隐藏层，我们可以自由选择 $13.2.3$ 节中介绍的不同形式的激活函数。

---

![mlpMnist](/assets/img/figures/mlpMnist.jpg)

{: style="width: 100%;" class="center"}
图 $\textrm{13.4}$：用于$\textrm{MNIST}$ 分类的 $\textrm{MLP}$ 结构。需要注意的是模型的参数量 $100,480=(784+1)\times128$, $16,512=(128+1)\times 128$。
{:.image-caption}

![mnistResult](/assets/img/figures/mnistResult.png)

{: style="width: 100%;" class="center"}
图 $\textrm{13.5}$：基于 $\textrm{MLP}$（包含$2$个隐藏层和$1$个输出层，隐藏层含$128$个节点，输出层包含$10$个节点 ）  对一些 $\textrm{MNIST}$ 图片的分类结果。红色为错误预测，蓝色为正确预测。$\textrm{(a)}$ 训练 $1$ 个周期。 $\textrm{(b)}$ 训练 $2$ 个周期。
{:.image-caption}

#### 13.2.4.2 MLP用于图像分类

要将 $\textrm{MLP}$ 应用于图像分类，我们需要将 $\textrm{2d}$ 输入“**展开**”（$\textrm{flatten}$）为 $\textrm{1d}$ 向量。 然后，我们可以使用类似于第$\textrm{13.2.4.1}$ 节中所述的前馈网络。 例如，考虑构建一个 $\textrm{MLP}$ 以对 $\textrm{MNIST}$ 数字进行分类（第$\textrm{3.7.2}$ 节）。 这些数字图片表示为 $28\times28 = 784$ 维的向量。 如果我们使用 $\textrm{2}$ 个具有 $\textrm{128}$ 个单元的隐藏层，和 $\textrm{1}$ 个包含 $\textrm{10}$ 个输出单元的 $\textrm{softmax}$ 层，将得到如图 $\textrm{13.4}$ 所示的模型。

我们在图 $\textrm{13.5}$ 中展示了一些该模型的预测结果。 我们对训练集仅训练两个“**周期**”（$\textrm{epochs}$, 遍历数据集的次数），但是该模型已经具备了较好的性能，测试集的准确率为 $\textrm{97.1％}$。 此外，预测错误的案例似乎也是可以理解的，例如将 $\textrm{9}$ 误分类为 $\textrm{3}$。训练更多的时间可以进一步提高模型的测试精度。

在第 $\textrm{14}$ 章中，我们将讨论另一种称为卷积神经网络的模型，该模型更适用于图像数据的处理。 通过利用与图像数据空间结构相关的先验知识，它可以获得更好的性能，并使用更少的参数。相比之下，$\textrm{MLP}$ 对输入的排列具有不变性。 换句话说，我们可以随机地对像素进行排列，并且可以获得相同的结果（前提是我们对所有的输入使用相同的随机排列算法）。

><font color=red>In Chapter $14$ we discuss a different kind of model, called a convolutional neural network, which is better suited to images. This gets even better performance and uses fewer parameters, by exploiting prior knowledge about the spatial structure of images.   By contrast, the MLP is invariant to a permutation of its inputs. Put another way, we could randomly shuffle the pixels and we would get the same result (assuming we use the same random permutation for all inputs).  </font>

---

![mlpIMDB](/assets/img/figures/mlpIMDB.jpg)

{: style="width: 100%;" class="center"}
图 $\textrm{13.6}$：用于 $\textrm{IMDB}$ 分类的 $\textrm{MLP}$ 结构。语料库大小为 $V=1000$，嵌入层大小为 $E = 16$。嵌入矩阵 $\mathbf{W}_1$ 大小为 $10,000\times16$，隐藏层（$\textrm{dense}$）的权重矩阵 $\mathbf{W}_2$ 大小为 $16 \times 16$ ，偏置 $\mathbf{b}_2$ 大小为 $16$（$16 \times 16 + 16 = 272$），最后一层 （$\textrm{dense_1}$）的权重矩阵 $\mathbf{w}_3$ 大小为 $16$，偏置 $b_3$ 大小为 $1$。全局平均池化不包含参数。
{:.image-caption}

#### 13.2.4.3 MLP用于电影评论的情感分析

[$\textrm{Maa+11}$][^Maa11] 中提出的 $\textrm{IMDB}$ 电影评论数据集（$\textrm{IMDB}$ 表示 $\textrm{internet movie database}$）被称为 “文本分类界的 $\textrm{MNIST}$”。该数据集包含 $\textrm{25k}$ 带有标签的样本用于训练，另外 $\textrm{25k}$ 的样本用于测试。 每个样本都有一个二进制标签，代表积极或消极的评分。 此任务被称为（二进制）**情感分析** （$\textrm{sentiment analysis}$）。 例如，以下是训练集中的两个样本：

> 1. this film was just brilliant casting location scenery story direction everyone’s really suited the part they played robert \<UNK\> is an amazing actor ...
> 2. big hair big boobs bad music and a giant safety pin these are the words to best describe this terrible movie i love cheesy horror movies and i’ve seen hundreds...

显然，第一个样本的标签为正例（积极评价），第二个为负例（消极评价）。

[^Maa11]: $\textrm{[Maa+11]}$: A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. “Learning Word Vectors for Sentiment Analysis”. In: Proc. ACL. 2011, pp. 142–150.

我们可以设计一个 $\textrm{MLP}$ 实现情感分析。不妨假设输入是一个包含 $T$ 个符号（$\textrm{token}$）的序列 $\textbf{x} _{1:T}$，其中 $\textbf{x} _t$ 是一个长度为 $ V $ 的 $ \textrm{one-hot} $ 向量， $V$ 为语料库的大小，我们将这种编码方式称为无序的单词袋（第 $10.4.3.1$ 节）。模型的第一层为 $E\times V$  的嵌入矩阵 $\textbf{W} _1 $，该层将每一个稀疏的 $V $ 维向量映射到一个稠密的 $E $ 维向量 $\mathbf{e} _t=\mathbf{W} _1 \mathbf{x} _t $（ $19.5$ 节介绍了更多关于词嵌入的细节）。接着我们使用 **全局平均池化**（$\textrm{global average pooling}$）将 $T\times D $ 的序列嵌入向量转化为一个固定长度的向量 $\overline{\mathbf{e}}=\frac{1}{T}\sum _{t=1}^T \mathbf{e} _t $ 。最后我们将该向量传入一个非线性隐藏层，得到一个 $K $ 维向量 $\mathbf{h} $，并将其传入最后的线性 $\textrm{logistic} $ 层。 综上所述，模型定义如下：
$$
\begin{align}
p(y|\mathbf{x};\mathbf{\theta}) = & {\rm{Ber}}(y|\sigma(\mathbf{w}_3^{\rm{T}}\mathbf{h}+b_3)) \tag{13.22} \\
\mathbf{h}= & \varphi(\mathbf{W}_2{\rm{\bar{\mathbf{e}}}}+\mathbf{b}_2) \tag{13.23} \\
{\bar{\mathbf{e}}}=& \frac{1}{T}\sum_{t=1}^T\mathbf{e}_t \tag{13.24} \\
\mathbf{e}_t=& \mathbf{W}_1\mathbf{x}_t \tag{13.25}
\end{align}
$$


如果我们令语料库大小为 $V = 1000$，嵌入向量维度为 $E = 16$，隐藏层的维度为 $\textrm{16}$，则得到的模型如图 $\textrm{13.6}$ 所示。 最终的模型在验证集的准确度为 $\textrm{86％}$。

我们发现模型中大多数参数都分布在嵌入矩阵中，这可能会导致过拟合问题。 幸运的是，正如我们在第 $\textrm{19.5}$ 节中讨论的那样，我们可以使用无监督预训练方法得到的词嵌入，然后我们只需要在特定任务上对输出层的参数进行微调即可。

---

![mlpHeter](/assets/img/figures/mlpHeter.png)

{: style="width: 100%;" class="center"}
图 $\textrm{13.7}$：$\textrm{MLP}$ 包含一个共享的主干网络和两个输出头，一个用于预测期望，另一个用于预测方差。
{:.image-caption}

![heterResult](/assets/img/figures/heterResult.png)

{: style="width: 100%;" class="center"}
图 $\textrm{13.8}$：使用 $\textrm{MLE}$ 的 $\textrm{MLP}$ 对 $1d$ 数据进行回归，该数据的噪声水平不断提高。$\textrm{(a)}$ 输出方差与输入有关，如图 $13.7$ 所示；$\textrm{(b)}$ 均值使用与 $\textrm{(a)}$ 一样的模型，但方差被当作是一个固定的参数 $\sigma^2$，该值使用 $\textrm{MLE}$ 估计，如 $11.2.3.6$ 所述。
{:.image-caption}

#### 13.2.4.4 MLP用于异方差回归

我们还可以使用 $\textrm{MLP}$ 实现回归任务。 图 $\textrm{13.7}$ 展示了如何为异方差 （$\textrm{heteroskedastic}$）非线性回归任务建模。 （术语“异方差”仅表示预测的输出方差与输入有关，如第 $\textrm{3.3.3}$ 节中所述。）该函数具有两个输出，分别表示 $f_\mu(\mathbf{x})=\mathbb{E}[y|\mathbf{x},\mathbf{\theta}]$ 和 $f_\sigma(\mathbf{x})=\sqrt{\mathbb{V}[y|\mathbf{x},\mathbf{\theta}]}$。如图 $\textrm{13.7}$ 所示，通过使用一个共享的**主干网络**（$\textrm{backbone}$）和两个输出**头网络**（$\textrm{heads}$）， 我们可以在这两个函数之间共享大部分层（也包括其中的参数）。对于 $\mu$ 输出头，我们使用一个线性激活函数 $\varphi(a)=a$。对于 $\sigma$ 头，我们使用 $\textrm{softplus}$ 激活函数 $\varphi(a)=\sigma_{+}(a)$。如果我们使用线性输出头和一个非线性主干网络，整个模型定义为
$$
p(y|\mathbf{x},\mathbf{\theta})=\mathcal{N}(y|\mathbf{w}_\mathbf{\mu}^{\rm{T}}f(\mathbf{x};\mathbf{w}_{\rm{shared}}), \sigma_{+}(\mathbf{w}_{\mathbf{\sigma}}^{\rm{T}}f(\mathbf{x};\mathbf{w}_{\rm{shared}})))\tag{13.26}
$$
图$\textrm{13.8}$ 显示了这种模型在某些数据集上的优势，在该数据集中，预测的期望值随时间线性增长，并且随季节波动，与此同时，数据的方差呈二次方增加趋势。（这是 **随机波动率模型** （$\textrm{stochastic volatility model}$）的一个简单示例；它可以用于对财务数据以及地球的全球温度进行建模，其中地球温度的（由于气候变化）均值和方差不断增加。）我们发现将输出方差 $\sigma^2$ 视为固定（与输入无关）参数的回归模型置信度有时会比较低，因为模型需要适应整体的噪声水平，并且无法适应输入空间中每个点的噪声水平。

---

---

![plane_decomposition](/assets/img/figures/plane_decomposition.png)

{: style="width: 100%;" class="center"}
图 $\textrm{13.9}$：以 $ \textrm{ReLU} $ 为激活函数的 $\textrm{MLP}$ 将二维平面划分为有限个线性决策区域。$(a)$ 只有一个隐藏层，其中包含 $25$ 个节点； $(b)$ 含两个隐藏层。 图形来源 [HAB19][^HAB19].
{:.image-caption}

[^HAB19]: $\textrm{[HAB19]}$

### 13.2.5 网络深度的重要性

研究表明包含一个隐藏层的 $\textrm{MLP}$ 是一个**通用函数逼近器**（$\textrm{universal function approximator}$），这意味着只要给定足够的隐藏单元，$\textrm{MLP}$ 就可以逼近任何平滑函数，并能够达到任何所需的精度水平[HSW89][^HSW89]; [cyb89][^cyb89]; [Hor91][^Hor91]。直观地讲，背后的原因在于每个隐藏层中的单元都可以指定一个超半平面，并且这些单元的足够大的组合可以“划分”空间的任何区域，我们可以将其与任何响应相关联（这在使用分段线性激活函数时最容易看到，如图 $13.9$ 所示。

但是，实验和理论上的各种观点（[Has87][^Has87]; [Mon+14][^Mon14]; [Rag+17][^Rag17]; [Pog+17][^Pog17]）都表明，深层网络比浅层网络更有效。 原因是更深的网络层可以利用浅层网络的所学习的特征。 也就是说，该函数是以组合或分层的方式定义的。 例如，假设我们要对 $\textrm{DNA}$ 字符串进行分类，并且正例与正则表达式  $\textbf{\*AA??CGCG??AA\* }$相关联。 尽管我们可以使用含单个隐藏层的模型对数据进行拟合，但是从直观上来说，如果模型首先学会使用第 $\textrm{1}$ 层中的隐藏单元来检测 $\textrm{AA}$ 和 $\textrm{CG}$ “基元”（$\textrm{motifs}$），然后使用这些特征在第$2$层定义一个简单的线性分类器，类似于我们在$\textrm{13.2.1}$ 节中解决 $\textrm{XOR}$ 问题的思路。

[^HSW89]: $\textrm{[HSW89]}$: 
[^cyb89]: $\textrm{[cyb89]}$:
[^Hor91]: $\textrm{[Hor91]}$: 
[^Has87]: $\textrm{[Has87]}$: 
[^Mon14]: $\textrm{[Mon+14]}$:
[^Rag17]: $\textrm{[Rag+17]}$:
[^Pog17]: $\textrm{[Pog+17]}$:

#### 13.2.5.1 深度学习革命

尽管 $\textrm{DNN}$ 背后的思想可以追溯到几十年前，但直到 $\textrm{2010}$ 年代，它们才开始被广泛使用。 突破性的时刻发生在$\textrm{2012}$ 年，当时 [KSH12][^KSH12] 表明深层的 $\textrm{CNN}$ 可以在具有挑战性的 $\textrm{ImageNet}$ 图像分类数据集上显著提高性能，在一年内将错误率从$\textrm{26％}$降低到$\textrm{16％}$（见图 $\textrm{14.14b}$）； 与之前每年约 $\textrm{2％}$ 的下降幅度相比，这是一个巨大的飞跃。 大约在同一时间，[DHK13][^DHK13] 表明，在各种语音识别任务上，深度神经网络可以大大优于现有技术。

$\textrm{DNN}$ 的“爆发”有几个促成因素。 一是廉价的 $\textrm{GPU}$（图形处理单元）使用称为可能。 它们最初是为了加快视频游戏的图像渲染速度而开发的，但是它们也可以大大减少大型 $\textrm{CNN}$ 的训练时间，其中包含相似的矩阵矢量计算。 另一个是大规模含标签数据集的增长，这使得我们能够拟合包含大量参数的复杂函数，同时避免过拟合的风险。 （例如，$\textrm{ImageNet}$ 具有$\textrm{130}$ 万张含标签的图像，可以用于拟合具有数百万个参数的模型）。的确，如果将深度学习系统视为“火箭”，那么大规模数据集就被称为燃料[^1]。

[^1]: This popular analogy is due to Andrew Ng, who mentioned it in a keynote talk at the GPU Technology Conference (GTC) in 2015. His slides are available at https://bit.ly/38RTxzH.
由于 $\textrm{DNN}$ 在经验上取得了巨大的成功，许多公司开始对该技术产生兴趣，并开发了高质量的开源软件库，例如$\textrm{Tensorflow}$（由 $\textrm{Google}$ 开发），$\textrm{PyTorch}$（由 $\textrm{Facebook}$ 开发）和 $\textrm{MXNet}$（由亚马逊开发）。 这些库支持复杂的微分函数的自动微分（请参考 $\textrm{13.3}$ 节）和可扩展的基于梯度的优化（请参见 $\textrm{5.4}$ 节）。 在本书的不同地方，我们将使用其中的一些库来实现各种模型，而不仅仅是 $\textrm{DNN}$。

有关“深度学习革命”历史的更多详细信息，请参考[Sej18][^Sej18]。

[^KSH12]: $\textrm{[KSH12]}$: 
[^DHK13]: $\textrm{[DHK13]}$: 
[^Sej18]: $\textrm{[Sej18]}$: 

![neurons](/assets/img/figures/neurons.jpg)

{: style="width: 100%;" class="center"}
图 $13.10$: 在“电路”中连接在一起的两个神经元的图示。左侧神经元的输出轴突与右侧细胞的树突形成突触连接，电荷以离子流的形式存在，使细胞得以交流。来自 https://en.wikipedia.org/wiki/Neuron。在维基百科作者BruceBlaus的许可下使用。
{:.image-caption}

![networkSize](/assets/img/figures/networkSize.png)

{: style="width: 100%;" class="center"}
图 $\textrm{13.11}$：神经网络大小随时间的变化趋势图。模型 $1,2,3$ 和 $4$ 表示感知机 [Ros58][^Ros58]，自适应线性单元 （$\textrm{adaptive linear unit}$）[WH60][^WH60]，神经认知机（$\textrm{neocognitron}$）[Fuk80][^Fuk80] 和第一个利用反向传播算法训练的 $\textrm{MLP}$ [RHW86][^RHW86]。图形来自 [GBC16][^GBC16]的图 $1.11$。
{:.image-caption}

[^Ros58]:$\textrm{[Ros58]}$:
[^ WH60]:$\textrm{[WH60]}$:
[^Fuk80]:$\textrm{[Fuk80]}$:
[^RHW86]:$\textrm{[RHW86]}$:
[^GBC16]: $\textrm{[GBC16]}$: 

### 13.2.6 与生物学的联系

在本节中，我们将讨论上文介绍的各种神经网络（称为人工神经网络, $\textrm{artificial neural networks, ANN}$）与实际神经网络之间的联系。 真实的生物大脑工作细节非常复杂（参见 [Kan+12][^Kan12]），但是我们可以给出一个简单的“卡通”（$\textrm{cartoon}$）。

我们首先考虑单个神经元的模型。 近似地讲，我们可以说神经元 $k$ 是否激活（用 $h_{k} \in\{0,1\}$ 表示）取决于其输入的行为（用 $\mathbf{x} \in \mathbb{R}^{D}$表示）以及连接的强度（用 $\mathbf{w} _k \in \mathbb{R}^D$表示）。我们可以使用 $a _{k}=\mathbf{w} _{k}^{\top} \mathbf{x}$ 来计算输入的加权和。 这些权重可以看作是将输入 $x _d$ 连接到神经元 $h _k$ 的“电线”， 类似于真实神经元中的树突（见图$\textrm{13.10} $）。该加权和接着被用于与阈值 $b _k $ 进行比较，如果激活超过阈值，则神经元触发； 这类似于神经元发出电输出或动作电位。 因此，我们可以使用 $h _{k}(\mathbf{x})=H\left(\mathbf{w} _k^\top \mathbf{x}-b _k\right) $ 来模拟神经元的行为，其中 $H(a)=\mathbb{I}(a>0)$ 是$\textrm{Heaviside}$函数。，这称为神经元的 $\textrm{McCulloch-Pitts}$ 模型，于 $\textrm{1943}$ 年提出 [MP43][^MP43]。

[^Kan12]: $\textrm{[Kan+12]}$:
[^MP43]: $\textrm{[MP43]}$: 

我们可以将多个这样的神经元组合在一起以构成一个人工神经网络，最终的结果有时被视为大脑的模型。 但是，人工神经网络在许多方面与生物大脑存在差异，主要包括以下方面：

- 大多数 $\textrm{ANN}$ 使用反向传播来修改其连接强度（请参见第 $\textrm{13.3}$ 节）。 但是，真正的大脑不会使用反向传播，因为无法沿着轴突向后发送信息 [Ben+15b][^Ben15b]; [BS16][^BS16];[KH19][^KH19]。 相反，他们使用局部更新规则来调整突触强度。

- 大多数 $\textrm{ANN}$ 都是严格的前馈，但是真实的大脑有很多反馈连接。 据信这种反馈的作用类似于先验，它们可以与来自感官系统的自下而上的似然结合起来，计算出已经包含经验信息的隐藏状态的后验，然后可以将其用于最佳决策（参考[Doy+07][^Doy07]）。

  > <font color=red>Most ANNs are strictly feedforward, but real brains have many feedback connections. It is believed that this feedback acts like a prior, which can be combined with bottom up likelihoods from the sensory system to compute a posterior over hidden states of the world, which can then be used for optimal decision making.</font>

- 大多数人工神经网络使用简化的神经元，该神经元由处理线性组合的非线性单元组成，但实际的生物神经元具有复杂的树状结构（见图 $\textrm{13.10}$），具有复杂的时空动态。

- 大多数人工神经网络的大小和连接数均小于生物大脑（见图$\textrm{13.11}$）。 当然，在各种新型硬件加速器（例如$\textrm{GPU}$ 和 $\textrm{TPU}$（张量处理单元）等）的推动下，人工神经网络每周都会变得越来越大。但是，即使人工神经网络在单元数量上与生物大脑相匹配，这种比较也具有误导性，因为生物神经元的处理能力远高于人工神经元（见上文）。

- 大多数 $\textrm{ANN}$ 被设计为对单个函数建模，例如将图像映射到类别标签，或将一个单词序列映射到另一个单词序列。 相比之下，生物大脑是非常复杂的系统，由多个专门的交互模块组成，这些模块实现不同种类的功能或行为，例如感知，控制，记忆，语言等（请参见 [Sha88][^Sha88];  [Kan+12][^Kan12]）。

[^Ben15b]: $\textrm{[Ben+15b]}$:
[^BS16]: $\textrm{[BS16]}$:
[^KH19]: $\textrm{[KH19]}$:
[^Doy07]: $\textrm{[Doy+07]}$:
[^Sha88]: $\textrm{[Sha88]}$:
[^Kan12]: $\textrm{[Kan+12]}$:

当然，我们正在努力建立逼真的生物大脑模型（例如，蓝脑计划, $\textrm{Blue Brain Project}$, [Mar06][^Mar06]; [Yon19][^Yon19]）。但是，一个有趣的问题是，以这种细致程度研究大脑是否对 “解决$\textrm{AI}$” 有用？通常认为，如果我们的目标是建造“智能机器”，那么生物大脑的浅层细节并不重要，就像飞机不会拍打自己的机翼一样。但是，“ $\textrm{AI}$”大概将遵循与智能生物代理类似的 “智能定律”（$\textrm{laws of intelligence}$），就像飞机和鸟类遵循相同的空气动力学定律一样。 

不幸的是，我们尚不知道什么是“智能定律”，或者甚至是否存在这样的法则。在本书中，我们假设任何智能代理都应遵循信息处理和贝叶斯决策理论的基本原理，这是在不确定性下做出决策的最佳方法（请参见第 $\textrm{8.4.2}$ 节）。 

当然，生物代理受到许多约束（例如，计算，生态），这通常需要算法“捷径”才能获得最佳解决方案。这可以解释人们在日常推理中使用的许多启发式方法。[KST82][^KST82]; [GTA00][^GTA00]; [Gri20][^Gri20]。随着我们希望机器解决的任务变得越来越困难，我们也许能够从神经科学和认知科学的其他领域获得见识（例如，参见[MWK16][^MWK16]; [Has+17][^Has17]; [Lak+17][^Lak17]）。

> >  Of course, there are efforts to make realistic models of biological brains (e.g., the Blue Brain Project [Mar06; Yon19]). However, an interesting question is whether studying the brain at this level of detail is useful for “solving AI”. It is commonly believed that the low level details of biological brains do not matter if our goal is to build “intelligent machines”, just as aeroplanes do not flap their wings. However, presumably “AIs” will follow similar “laws of intelligence” to intelligent biological agents, just as planes and birds follow the same laws of aerodynamics.
>
> > Unfortunately, we do not yet know what the “laws of intelligence” are, or indeed if there even are such laws. In this book we make the assumption that any intelligent agent should follow the basic principles of information processing and Bayesian decision theory, which is known to be the optimal way to make decisions under uncertainty (see Sec. 8.4.2).
>
> > Of course, biological agents are subject to many constraints (e.g., computational, ecological) which often require algorithmic “shortcuts” to the optimal solution; this can explain many of the heuristics that people use in everyday reasoning [KST82; GTA00; Gri20]. As the tasks we want our machines to solve become harder, we may be able to gain insights from other areas of neuroscience and cognitive science (see e.g., [MWK16; Has+17; Lak+17]).

[^Mar06]: $\textrm{[Mar06]}$:
[^Yon19]: $\textrm{[Yon19]}$:
[^KST82]: $\textrm{[KST82]}$:
[^GTA00]: $\textrm{[GTA00]}$:
[^Gri20]: $\textrm{[Gri20]}$:
[^MWK16]: $\textrm{[MWK16]}$:
[^Has17]: $\textrm{[Has+17]}$:
[^Lak17]: $\textrm{[Lak+17]}$

## 13.3 反向传播

在本节，我们将介绍著名的 **反向传播算法**（$\textrm{backpropagation algorithm}$），正如第 $\textrm{13.4}$ 节讨论的，该算法可用于计算损失函数关于网络每一层参数的梯度，并将该梯度传递给基于梯度的优化算法。

反向传播算法最初由[BH69][^BH69]提出，同时在[Wer74][^Wer74]中被独立发现。 但是，该算法引起“主流”机器学习社区的注意，还要归功于[RHW86][^RHW86]。 关于该算法的更多历史信息，请参考  $\textrm{Wikipedia page}$[^3]。

为了方便分析，我们首先假设计算图是一个简单的类似于 $\textrm{MLP}$ 的线性链，该线性链中的每一层都是一个线性映射函数。 在这种情况下，反向传播等价于链式法则的重复使用（参考式（$\textrm{B.42}$））。 但是，正如我们在第 $\textrm{13.3.4}$ 节中将要讨论的，该方法可以推广到任意的有向无环图（$\textrm{DAG}$）模型。 整个反向传播算法的过程通常被称为 **自动微分**（$\textrm{automatic differentiation, autodiff}$）。

[^BH69]: 
[^Wer74]: 
[^RHW86]: 

[^3]: https://en.wikipedia.org/wiki/Backpropagation#History

### 13.3.1 前向与反向模式微分

考虑一个形式为 $\mathbf{o}=\mathbf{f}(\mathbf{x})$ 的映射函数，其中 $\mathbf{x}\in \mathbb{R}^n$, $\mathbf{o}\in \mathbb{R}^m$。假设函数 $\mathbf{f}$ 由多个子函数组合而成：
$$
\mathbf{f}=\mathbf{f}_4 \circ \mathbf{f}_3 \circ \mathbf{f}_2 \circ \mathbf{f}_1 \tag{13.27}
$$
其中 $\mathbf{f}_1:\mathbb{R}^n \rightarrow \mathbb{R}^{m_1}$, $\mathbf{f}_2:\mathbb{R}^{m_1} \rightarrow \mathbb{R}^{m_2}$, $\mathbf{f}_3:\mathbb{R}^{m_2} \rightarrow \mathbb{R}^{m_3}$, $\mathbf{f}_4:\mathbb{R}^{m_3} \rightarrow \mathbb{R}^{m}$。 为了得到最终的结果 $\mathbf{o}=\mathbf{f}(\mathbf{x})$,  需要依次计算中间过程 $\mathbf{x}_2=\mathbf{f}_1(\mathbf{x})$, $\mathbf{x}_3=\mathbf{f}_2(\mathbf{x}_2)$, $\mathbf{x}_4=\mathbf{f}_3(\mathbf{x}_3)$, $\mathbf{o}=\mathbf{f}_4(\mathbf{x}_4)$。

我们可以使用链式法则计算雅各比 （$\textrm{Jacobian}$） $\mathbf{J}_\mathbf{f}(\mathbf{x})=\frac{\partial \mathbf{o}}{\partial \mathbf{x}}\in \mathbb{R}^{m\times n}$：


$$
\begin{align}
\frac{\partial \mathbf{o}}{\partial \mathbf{x}} = & \frac{\partial \mathbf{o}}{\partial \mathbf{x}_4}\frac{\partial \mathbf{x}_4}{\partial \mathbf{x}_3}\frac{\partial \mathbf{x}_3}{\partial \mathbf{x}_2}\frac{\partial \mathbf{x}_2}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}_4(\mathbf{x}_4)}{\partial \mathbf{x}_4}\frac{\partial \mathbf{f}_3(\mathbf{x}_3)}{\partial \mathbf{x}_3}\frac{\partial \mathbf{f}_2(\mathbf{x}_2)}{\partial \mathbf{x}_2}\frac{\partial \mathbf{f}_1(\mathbf{x})}{\partial \mathbf{x}} \tag{13.28} \\
= & \mathbf{J}_{\mathbf{f}_4}(\mathbf{x}_4)\mathbf{J}_{\mathbf{f}_3}(\mathbf{x}_3)\mathbf{J}_{\mathbf{f}_2}(\mathbf{x}_2)\mathbf{J}_{\mathbf{f}_1}(\mathbf{x}_1) \tag{13.29}
\end{align}
$$


我们现在讨论如何高效地计算雅各比 $\mathbf{J}_\mathbf{f}(\mathbf{x})$。回顾雅各比的定义：


$$
\mathbf{J}_\mathbf{f}(\mathbf{x}) = \frac{\partial \mathbf{f}(\mathbf{x})}{\partial \mathbf{x}}= 
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix} 
=
\begin{pmatrix}
\nabla f_1(\mathbf{x})^{\rm{T}} \\
\vdots \\
\nabla f_m(\mathbf{x})^{\rm{T}}
\end{pmatrix}
=
\begin{pmatrix}
\frac{\partial \mathbf{f}}{\partial x_1}, & \cdots, \frac{\partial \mathbf{f}}{\partial x_n}
\end{pmatrix}
 \in \mathbb{R}^{m\times n} \tag{13.30}
$$


其中 $\nabla f_i(\mathbf{x})^{\rm{T}} \in \mathbb{R} ^ {1 \times n}$ 为第 $i$ 行 （ $i = 1:m$ ） ，$\frac{\partial \mathbf{f}}{\partial x_ j} \in \mathbb{R}^m$ 为第 $j$ 列 （$j = 1:n$）。值得注意的是， 当 $m = 1$ 时， 梯度表示为 $\nabla \mathbf{f}(\mathbf{x})$ ，其形状与 $\mathbf{x}$ 相同，即是一个列向量，然而，此时 $\mathbf{J}_ {\mathbf{f}}(\mathbf{x})$ 是一个行向量，在这种情况下，我们令 $\nabla\mathbf{f}(\mathbf{x})=\mathbf{J} _{\mathbf{f}}(\mathbf{x})^{\top} $。

我们可以通过向量雅各比乘（$\textrm{vector Jacobian product，VJP}$）$\mathbf{e} _i^{\rm{T}}\mathbf{J} _\mathbf{f}(\mathbf{x})$ 从 $\mathbf{J} _\mathbf{f}(\mathbf{x})$ 中提取第 $i$ 行，其中 $\mathbf{e} _i \in \mathbb{R}^m$ 表示单位基向量。类似地，我们可以使用雅各比向量乘（$\textrm{Jacobian vector product, JVP}$）$\mathbf{J} _\mathbf{f}(\mathbf{x})\mathbf{e} _j$ 从 $\mathbf{J} _\mathbf{f}(\mathbf{x})$ 中提取第 $j$ 列，其中 $\mathbf{e} _j \in \mathbb{R}^n$。 因此 $\mathbf{J} _\mathbf{f}(\mathbf{x})$ 的计算可以退化为 $\textrm{n}$ 个 $\textrm{JVPs}$ 或 $\textrm{m}$ 个 $\textrm{VJPs}$。[^译者注1]

[^译者注1]: 复杂度计算可参考 https://math.stackexchange.com/questions/2195377/reverse-mode-differentiation-vs-forward-mode-differentiation-where-are-the-be

如果 $n \lt m$，计算 $\mathbf{J}_\mathbf{f}(\mathbf{x})$ 的高效方法是，对每一列 $j=1:n$ 使用 $\textrm{JVPs}$， 并在计算的过程中使用从右到左的计算顺序。右乘列向量的形式为：


$$
\mathbf{J}_{\mathbf{f}}(\mathbf{x}) \mathbf{v}=\underbrace{\mathbf{J}_{\mathbf{f}_{4}}\left(\mathbf{x}_{4}\right)}_{m \times m_{3}} \underbrace{\mathbf{J}_{\mathrm{f}_{3}}\left(\mathbf{x}_{3}\right)}_{m_{3} \times m_{2}} \underbrace{\mathbf{J}_{\mathrm{f}_{2}}\left(\mathbf{x}_{2}\right)}_{m_{2} \times m_{1}} \underbrace{\mathbf{J}_{\mathbf{f}_{1}}\left(\mathbf{x}_{1}\right)}_{m_{1} \times n} \underbrace{\mathbf{v}}_{n \times 1} \tag{13.31}
$$


上式可以通过**前向模式微分**（$\textrm{forward mode differentiation}$）得到；算法 $\textrm{5}$ 给出了伪代码。假设 $m=1$，同时 $n=m_1=m_2=m_3$，计算 $\mathbf{J}_\mathbf{f}(\mathbf{x})$ 的时间复杂度为 $O(n^3)$。

> $\textrm{Algorithm 5}$: 前向模式微分
>
> 1. $\mathbf{x}_{1}:=\mathbf{x}$
>
> 2. $\mathbf{v}_j:=\mathbf{e}_j \in \mathbb{R}^n$ $\textrm{for}$ $j=1: n$
>
> 3. $\textbf{for}$ $k=1:K$ $\textbf{do}$
>
> 4. > $\mathbf{x}_{k+1}=\mathbf{f}_k\left(\mathbf{x}_k\right)$
>
> 5. > $\mathbf{v} _j:=\mathbf{J} _{\mathbf{f}_k}(\mathbf{x} _k)\mathbf{v} _j \textrm{ for }j=1: n$
>
> 6. $\textrm{Return}$ $\mathbf{o}=\mathbf{x} _{K+1},\left[\mathbf{J} _{\mathbf{f}}(\mathbf{x})\right] _{:, j}=\mathbf{v} _j\textrm{ for }j=1: n$



如果 $n \gt m$ （比如说，输出是一个标量），高效的计算方式是，对每一行 $i = 1 : m$ 使用 $\textrm{VJPs}$， 并采用从左到右的计算方式。左乘行向量 $\mathbf{u}^{\rm{T}}$ 的形式为
$$
\mathbf{u}^{\top} \mathbf{J}_{\mathbf{f}}(\mathbf{x})=\underbrace{\mathbf{u}^{\top}}_{1 \times m} \underbrace{\mathbf{J}_{\mathrm{f}_{4}}\left(\mathbf{x}_{4}\right)}_{m \times m_{3}} \underbrace{\mathbf{J}_{\mathbf{f}_{3}}\left(\mathbf{x}_{3}\right)}_{m_{3} \times m_{2}} \underbrace{\mathbf{J}_{\mathbf{f}_{2}}\left(\mathbf{x}_{2}\right)}_{m_{2} \times m_{1}} \underbrace{\mathbf{J}_{\mathbf{f}_{1}}\left(\mathbf{x}_{1}\right)}_{m_{1} \times n} \tag{13.32}
$$
上式可以通过使用**反向模式微分** （$\textrm{reverse mode differentiation}$）。算法 $6$ 给出了伪代码。假设 $m = 1$， 同时 $n = m_1=m_2=m_3$，其计算复杂度为 $O(n^2)$。



>$\textrm{Algorithm 6}$: 反向模式微分
>
>1. $\mathbf{x}_{1}:=\mathbf{x}$
>
>2. $\textbf{for}$ $k=1:K$ $\textbf{do}$
>
>3. > $\mathbf{x} _{k+1}=\mathbf{f} _{k}\left(\mathbf{x} _{k}\right)$
>
>4. $\mathbf{u} _{i}:=\mathbf{e} _{i} \in \mathbb{R}^{m}$ $\textrm{for}$ $i=1: m$
>
>5. $\textbf{for}$ $k=K:1$ $\textbf{do}$
>
>6. > $\mathbf{u} _{i}^{\top}:=\mathbf{u} _{i}^{\top} \mathbf{J} _{\mathbf{f} _{k}}\left(\mathbf{x} _{k}\right)$ $\textbf{for}$ $i=1: m$
>
>7. $\textrm{Return}$ $\mathbf{o}=\mathbf{x} _{K+1},\left[\mathbf{J} _{\mathbf{f}}(\mathbf{x})\right] _{i,:}=\mathbf{u} _{i}^{\top}$ $\textrm{for}$  $i=1: m$



![feedforward_model_with_4_layers](/assets/img/figures/feedforward_model_with_4_layers.png)

> 图 $13.12$: 一个简单的包含 $4$ 层的线性链式前馈网络。其中 $\mathbf{x}$ 表示输入， $\mathbf{o}$ 表示输出。

### 13.3.2 用于多层感知机的反向模式微分

在之前的章节中，我们仅仅考虑了一个简单的线性链式前馈网络，其中的每一层都不包含可学习的参数。本节，所考虑的网络中每一层包含参数 $\mathbf{\theta}_1,...,\mathbf{\theta}_4$，如图 $13.12$ 所示。我们集中讨论最终输出为标量的情况：$\mathcal{L}:\mathbb{R}^n\rightarrow \mathbb{R}$。举例来说，考虑作用于包含一个隐藏层的  $\textrm{MLP}$ 的 $l_2$ 损失函数：
$$
\mathcal{L}((\mathbf{x}, \mathbf{y}), \boldsymbol{\theta})=\frac{1}{2}\left\|\mathbf{y}-\mathbf{W}_{2} \varphi\left(\mathbf{W}_{1} \mathbf{x}\right)\right\|_{2}^{2} \tag{13.33}
$$
我们可以将上式表示为如下的前馈网络模型：
$$
\begin{align}
\mathcal{L} &=\mathbf{f}_{4} \circ \mathbf{f}_{3} \circ \mathbf{f}_{2} \circ \mathbf{f}_{1}  \tag{13.34} \\ 
\mathbf{x}_{2} &=\mathbf{f}_{1}\left(\mathbf{x}, \boldsymbol{\theta}_{1}\right)=\mathbf{W}_{1} \mathbf{x} \tag{13.35} \\
\mathbf{x}_{3} &=\mathbf{f}_{2}\left(\mathbf{x}_{2}, \emptyset\right)=\varphi\left(\mathbf{x}_{2}\right) \tag{13.36} \\
\mathbf{x}_{4} &=\mathbf{f}_{3}\left(\mathbf{x}_{3}, \boldsymbol{\theta}_{3}\right)=\mathbf{W}_{2} \mathbf{x}_{3} \tag{13.37} \\
\mathcal{L} &=\mathbf{f}_{4}\left(\mathbf{x}_{4}, \mathbf{y}\right)=\frac{1}{2}\left\|\mathbf{x}_{4}-\mathbf{y}\right\|^{2} \tag{13.38}
\end{align}
$$
我们用符号 $\mathbf{f}_k(\mathbf{x}_k,\boldsymbol{\theta}_k)$ 表示第 $k$ 层的函数，其中 $\mathbf{x}_k$ 为上一层的输出， $\boldsymbol{\theta}_k$ 表示该层的可选参数。

在这个例子中，最后一层的输出为一个标量，因为它对应于一个损失函数 $\mathcal{L}\in \mathbb{R}$。所以使用反向模式微分计算梯度向量的效率更高。

我们首先讨论如何计算标量输出关于每一层中参数的梯度。我们可以使用矢量微积分直接计算 $\frac{\partial L}{\partial \boldsymbol{\theta} _{4}}$。对于中间项，我们使用链式法则：


$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_{3}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{4}} \frac{\partial \mathbf{x}_{4}}{\partial \boldsymbol{\theta}_{3}} \tag{13.39}\\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_{2}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{4}} \frac{\partial \mathbf{x}_{4}}{\partial \mathbf{x}_{3}} \frac{\partial \mathbf{x}_{3}}{\partial \boldsymbol{\theta}_{2}} \tag{13.40} \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_{1}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{4}} \frac{\partial \mathbf{x}_{4}}{\partial \mathbf{x}_{3}} \frac{\partial \mathbf{x}_{3}}{\partial \mathbf{x}_{2}} \frac{\partial \mathbf{x}_{2}}{\partial \boldsymbol{\theta}_{1}} \tag{13.41}
\end{align}
$$


其中 $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_ {k}}=\left(\nabla_ {\boldsymbol{\theta}_ {k}} \mathcal{L}\right)^{\top}$ 是一个 $d _k$ 维的行向量，$d _k$ 为第 $k$ 层的参数数量。我们发现这些值可以通过递归方式进行计算，即将第 $k$ 层的梯度行向量与大小为 $n _{k} \times n _{k-1}$ 的雅各比 $\frac{\partial \mathbf{x} _{k}}{\partial \mathbf{x} _{k-1}}$ 相乘，其中 $n _k$ 为第 $k$ 层隐藏节点的数量。算法 $7$ 给出了伪代码。



> $\textrm{Algorithm 7}$: 含 $K$ 层的 $\textrm{MLP}$ 的反向传播算法
>
> 1. $\textrm{// Forward pass}$
>
> 2. $\mathbf{x}_{1}:=\mathbf{x}$
>
> 3. $\textbf{for}$ $k=1:K$ $\textbf{do}$
>
> 4. > $\mathbf{x} _{k+1}=\mathbf{f} _{k}\left(\mathbf{x} _{k}, \pmb{\theta} _k\right)$
>
> 5. $\textrm{// Backward pass}$
>
> 6. $\mathbf{u}_{K+1} := \mathbf{1}$
>
> 7. $\textbf{for}$ $k=K:1$ $\textbf{do}$
>
> 8. > $\mathbf{g} _{k}:=\mathbf{u} _{k+1}^{\top} \frac{\partial \mathbf{f} _{k}\left(\mathbf{x} _{k}, \boldsymbol{\theta} _{k}\right)}{\partial \boldsymbol{\theta} _{k}}$
>
> 9. > $\mathbf{u} _{k}^{\top}:=\mathbf{u} _{k+1}^{\top} \frac{\partial \mathbf{f} _{k}\left(\mathbf{x} _{k}, \boldsymbol{\theta} _{k}\right)}{\partial \mathbf{x} _{k}}$
>
> 10. $\textrm{// Output}$
>
> 11. $\textrm{Return}$ $\mathcal{L}=\mathrm{x} _{K+1}$, $\nabla _{\mathbf{x}} \mathcal{L}=\mathbf{u} _{1}$, $\left\\{\nabla _{\boldsymbol{\theta} _{k}} \mathcal{L}=\mathrm{g} _{k}: k=1: K\right\\}$



该算法计算出损失关于每一层参数的梯度。同时还计算了损失关于输入的梯度 $\nabla_{\mathbf{x}} \mathcal{L} \in \mathbb{R}^{n}$ ，其中 $n$ 表述输入的维度。后一项对于参数的更新并不需要，但对于需要生成输入 $\mathbf{x}$ 的模型来说却是有用的（见 $ 14.5$ 节）。

剩下的部分就是如何计算具体层的向量雅各比乘（$\textrm{VJP}$）。其中的细节取决于每一层映射函数的具体形式。我们在下文讨论一些具体的例子。

### 13.3.3 常规层的向量雅各比乘

回顾形式为 $\mathbf{f}:\mathbb{R}^n\rightarrow \mathbb{R}^m$ 的网络层的雅各比矩阵：


$$
\mathbf{J}_{\mathbf{f}}(\mathbf{x})=\frac{\partial \mathbf{f}(\mathbf{x})}{\partial \mathbf{x}}=\left(\begin{array}{ccc}
\frac{\partial f_{1}}{\partial x_{1}} & \cdots & \frac{\partial f_{1}}{\partial x_{n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_{m}}{\partial x_{1}} & \cdots & \frac{\partial f_{m}}{\partial x_{n}}
\end{array}\right)=\left(\begin{array}{c}
\nabla f_{1}(\mathbf{x})^{\top} \\
\vdots \\
\nabla f_{m}(\mathbf{x})^{\top}
\end{array}\right)=\left(\frac{\partial \mathbf{f}}{\partial x_{1}}, \cdots, \frac{\partial \mathbf{f}}{\partial x_{n}}\right) \in \mathbb{R}^{m \times n} \tag{13.42}
$$


其中 $\nabla f_{i}(\mathbf{x})^{\top} \in \mathbb{R}^{n}$ 表示第 $i$ 行 （$i = 1:m$），$\frac{\partial \mathbf{f}}{\partial x_{j}} \in \mathbb{R}^{m}$ 为第 $j$ 列（$j=1:n$）。本节，我们将描述如何计算常规层的 $\textrm{VJP}$ $\mathbf{u}^{\top} \mathbf{J}_{\mathbf{f}}(\mathbf{x})$ 。

#### 13.3.3.1 交叉熵层

考虑一个交叉熵损失层，其输入为 $\textrm{logits}$ $\mathbf{x}$ 和真实标签 $\mathbf{y}$， 损失层返回一个标量：
$$
z=f(\mathbf{x})=\text { CrossEntropyWithLogits }(\mathbf{y}, \mathbf{x})=-\sum_{c} y_{c} \log p_{c} \tag{13.43}
$$
其中 $\mathbf{p}=\mathcal{S}(\mathbf{x})=\frac{e^{x_{c}}}{\sum_{c^{\prime}=1}^{C} e^{x_{c^{\prime}}}}$ 表示预测的类别概率值， $\mathbf{y}$ 为标签的 $\textrm{one-hot}$ 编码。（所以 $\mathbf{p}$ 和 $\mathbf{y}$ 都是维度为 $C$ 的概率单纯形（$\textrm{simplex}$）。）输出关于输入的雅各比为：
$$
\mathbf{J}=\frac{\partial z}{\partial \mathbf{x}}=(\mathbf{p}-\mathbf{y})^{\top} \in \mathbb{R}^{1 \times C} \tag{13.44}
$$
为了说明这一点，假设目标的真实类别为 $c$。 我们有
$$
z=f(\mathbf{x})=-\log \left(p_{c}\right)=-\log \left(\frac{e^{x_{c}}}{\sum_{j} e^{x_{j}}}\right)=\log \left(\sum_{j} e^{x_{j}}\right)-x_{c} \tag{13.45}
$$
所以
$$
\frac{\partial z}{\partial x_{i}}=\frac{\partial}{\partial x_{i}} \log \sum_{j} e^{x_{j}}-\frac{\partial}{\partial x_{i}} x_{c}=\frac{e^{x_{i}}}{\sum_{j} e^{x_{j}}}-\frac{\partial}{\partial x_{i}} x_{c}=p_{i}-\mathbb{I}(i=c) \tag{13.46}
$$
定义 $\mathbf{y}=[\mathbb{I}(i=c)]$ ，我们将获得式 ($13.44$)。需要注意的是该层的雅各比是一个行向量，因为输出是一个标量。  对应的 $\textrm{VJP}$ 为 $\mathbf{u}^{\top} \mathbf{J}$ ，其中 $\mathbf{u} \in \mathbb{R}$ 。

#### 13.3.3.2 逐元素非线性 

考虑使用逐元素非线性的网络层 $\mathbf{z}=\mathbf{f}(\mathbf{x})=\varphi(\mathbf{x})$ ，所以 $z_{i}=\varphi\left(x_{i}\right)$ 。雅各比的 $(i,j)$ 元素值为：
$$
\frac{\partial z_{i}}{\partial x_{j}}=\left\{\begin{array}{ll}
\varphi^{\prime}\left(x_{i}\right) & \text { if } i=j \\
0 & \text { otherwise }
\end{array}\right. \tag{13.47}
$$
其中 $\varphi^{\prime}(a)=\frac{d}{d a} \varphi(a)$。换句话说， 输出关于输入的雅各比为
$$
\mathbf{J}=\frac{\partial \mathbf{f}}{\partial \mathbf{x}}=\operatorname{diag}\left(\varphi^{\prime}(\mathbf{x})\right) \tag{13.48}
$$
对于任意一个向量 $\mathbf{u}$， 我们可以将 $\mathbf{J}$ 的对角元素与向量 $\mathbf{u}$ 进行逐元素相乘，得到 $\mathbf{u}^{\top} \mathbf{J}$。举例来说，假设
$$
\varphi(a)=\operatorname{ReLU}(a)=\max (a, 0) \tag{13.49}
$$
我们有
$$
\varphi^{\prime}(a)=\left\{\begin{array}{ll}
0 & a<0 \\
1 & a>0
\end{array}\right. \tag{13.50}
$$
其中 $a=0$ 处的亚梯度（$\textrm{subderivative}$）（见 $\textrm{B.4.4}$）为 $[0,1]$ 区间的任意值。 通常情况下等于 $0$。所以
$$
\operatorname{ReLU}^{\prime}(a)=H(a) \tag{13.51}
$$
其中 $H$ 为单位阶跃函数。

#### 13.3.3.3 线性层

现在考虑线性层，$\mathbf{z}=\mathbf{f}(\mathbf{x}, \mathbf{W})=\mathbf{W} \mathbf{x}$ ，其中  $\mathbf{W} \in \mathbb{R}^{m \times n}$ , 所以 $\mathbf{x} \in \mathbb{R}^{n}$ ，$\mathbf{z} \in \mathbb{R}^{m}$。我们可以计算关于输入向量的雅各比， $\mathbf{J}=\frac{\partial \mathbf{z}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$ 。注意到
$$
z_{i}=\sum_{k=1}^{n} W_{i k} x_{k} \tag{13.52}
$$
所以雅各比的 $(i,j)$ 项为
$$
\frac{\partial z_{i}}{\partial x_{j}}=\frac{\partial}{\partial x_{j}} \sum_{k=1}^{n} W_{i k} x_{k}=\sum_{k=1}^{n} W_{i k} \frac{\partial}{\partial x_{j}} x_{k}=W_{i j} \tag{13.53}
$$
因为 $\frac{\partial}{\partial x_{j}} x_{k}=\mathbb{I}(k=j)$ 。所以关于输入的雅各比为
$$
\mathbf{J}=\frac{\partial \mathbf{z}}{\partial \mathbf{x}}=\mathbf{W} \tag{13.54}
$$
$\mathbf{u}^{\top} \in \mathbb{R}^{1 \times m}$ 与 $\mathbf{J} \in \mathbb{R}^{m \times n}$ 的 $\textrm{VJP}$ 为
$$
\mathbf{u}^{\top} \frac{\partial \mathbf{z}}{\partial \mathbf{x}}=\mathbf{u}^{\top} \mathbf{W} \in \mathbb{R}^{1 \times n} \tag{13.55}
$$
我们现在考虑关于权重矩阵的雅各比 $J=\frac{\partial \mathbf{z}}{\partial \mathbf{W}}$ 。它可以表示为一个 $m\times(m\times n)$的矩阵，处理起来会比较麻烦。所以取而代之的，我们关注对单个权重 $W_{ij}$ 的梯度。这个就比较容易计算了，因为 $\frac{\partial \mathbf{z}}{\partial W_{i j}}$ 是一个向量。为了计算该值，注意到
$$
\begin{align}
z_{k} &=\sum_{l=1}^{m} W_{k l} x_{l} \tag{13.56} \\
\frac{\partial z_{k}}{\partial W_{i j}} &=\sum_{l=1}^{m} x_{l} \frac{\partial}{\partial W_{i j}} W_{k l}=\sum_{l=1}^{m} x_{l} \mathbb{I}(i=k \text { and } j=l) \tag{13.57}
\end{align}
$$
所以
$$
\frac{\partial \mathbf{z}}{\partial W_{i j}}=\left(\begin{array}{lllllll}
0 & \cdots & 0 & x_{j} & 0 & \cdots & 0
\end{array}\right)^{\top} \tag{13.58}
$$
其中非零项处在位置 $i$。$\mathbf{u}^{\top} \in \mathbb{R}^{1 \times m}$ 与 $\frac{\partial \mathbf{z}}{\partial \mathbf{W}} \in \mathbb{R}^{m \times(m \times n)}$ 之间的 $\textrm{VJP}$ 可以表示为一个形状为 $1 \times(m \times n)$ 的矩阵。需要注意的是
$$
\mathbf{u}^{\top} \frac{\partial \mathbf{z}}{\partial W_{i j}}=\sum_{k=1}^{m} u_{k} \frac{\partial z_{k}}{\partial W_{i j}}=u_{i} x_{j} \tag{13.59}
$$
所以
$$
\left[\mathbf{u}^{\top} \frac{\partial \mathbf{z}}{\partial \mathbf{W}}\right]_{1,:}=\mathbf{u x}^{\top} \in \mathbb{R}^{m \times n} \tag{13.60}
$$

#### 13.3.3.4 将它们放在一起

练习 $13.1$ 需要将它们放在一起。

![computation_graph](/assets/img/figures/computation_graph.png)

> 图 $13.13$: 包含 $2$ 个（标量）输入和 $1$ 个（标量）输出的计算图。

![auto_diff](/assets/img/figures/auto_diff.png)

> 图 $13.14$：计算图中节点 $j$ 的自动微分的图示。

### 13.3.4 计算图

$\textrm{MLPs}$ 是一种简单的 $\textrm{DNN}$ 模型，其中每一层的输出直接输入到下一层，从而形成一个链式结构，如图 $13.12$ 所示。然而，最新的 $\textrm{DNN}$ 模型可以以更加复杂的形式组合可微的部件，从而构成一个 **计算图**（$\textrm{computation graph}$），这一点类似于程序员将初等函数组合成更加复杂的函数。（的确，有些人也建议将 “深度学习” 称为 “**可微编程**”（$\textrm{differentiable programming}$））唯一的约束在于最终的计算图对应于一个**有向无环图**（$\textrm{directed ayclic graph, DAG}$），其中每一个节点都是关于输入的可微函数。

举个例子，考虑函数
$$
f\left(x_{1}, x_{2}\right)=x_{2} e^{x_{1}} \sqrt{x_{1}+x_{2} e^{x_{1}}} \tag{13.61}
$$
我们可以使用图 $13.13$ 中的 $\textrm{DAG}$ 进行计算，其中包含如下的中间函数：
$$
\begin{align}{l}
x_{3}& =f_{3}\left(x_{1}\right)=e^{x_{1}} \tag{13.62} \\
x_{4}& =f_{4}\left(x_{2}, x_{3}\right)=x_{2} x_{3}  \tag{13.63} \\
x_{5}& =f_{5}\left(x_{1}, x_{4}\right)=x_{1}+x_{4} \tag{13.64}\\
x_{6}& =f_{6}\left(x_{5}\right)=\sqrt{x_{5}} \tag{13.65} \\
x_{7}& =f_{7}\left(x_{4}, x_{6}\right)=x_{4} x_{6} \tag{13.66}
\end{align}
$$
值得注意的是我们已经按照拓扑结构对节点进行了编号（父节点在子节点之前）。在反向传播期间，因为计算图不再是链式结构，我们需要对多条路径的梯度进行求和。举例来说，因为 $x_4$ 影响 $x_5$ 和 $x_7$，我们有
$$
\frac{\partial \mathrm{o}}{\partial \mathrm{x}_{4}}=\frac{\partial \mathrm{o}}{\partial \mathrm{x}_{5}} \frac{\partial \mathrm{x}_{5}}{\partial \mathrm{x}_{4}}+\frac{\partial \mathrm{o}}{\partial \mathrm{x}_{7}} \frac{\partial \mathrm{x}_{7}}{\partial \mathrm{x}_{4}} \tag{13.67}
$$
通过逆拓扑顺序的计算方式，我们可以避免重复计算
$$
\begin{align}
\frac{\partial \mathbf{o}}{\partial \mathbf{x}_{7}}= & \frac{\partial \mathbf{x}_{7}}{\partial \mathbf{x}_{7}}=\mathbf{I}_{m} \tag{13.68} \\
\frac{\partial \mathbf{o}}{\partial \mathbf{x}_{6}}= & \frac{\partial \mathbf{o}}{\partial \mathbf{x}_{7}} \frac{\partial \mathbf{x}_{7}}{\partial \mathbf{x}_{6}} \tag{13.69}\\
\frac{\partial \mathbf{o}}{\partial \mathbf{x}_{5}}= & \frac{\partial \mathbf{o}}{\partial \mathbf{x}_{6}} \frac{\partial \mathbf{x}_{6}}{\partial \mathbf{x}_{5}} \tag{13.70} \\
\frac{\partial \mathbf{o}}{\partial \mathbf{x}_{4}}= & \frac{\partial \mathbf{o}}{\partial \mathbf{x}_{5}} \frac{\partial \mathbf{x}_{5}}{\partial \mathbf{x}_{4}}+\frac{\partial \mathbf{o}}{\partial \mathbf{x}_{7}} \frac{\partial \mathbf{x}_{7}}{\partial \mathbf{x}_{4}} \tag{13.71}
\end{align}
$$
总而言之，我们使用
$$
\frac{\partial \mathrm{o}}{\partial \mathrm{x}_{j}}=\sum_{k \in \text { children }(j)} \frac{\partial \mathrm{o}}{\partial \mathrm{x}_{k}} \frac{\partial \mathrm{x}_{k}}{\partial \mathrm{x}_{j}} \tag{13.72}
$$
其中我们对节点 $j$ 的所有子节点 $k$ 进行求和，如图 $13.14$ 所示。对于每个子节点 $k$ 的梯度向量 $\frac{\partial \mathbf{o}}{\partial \mathbf{x}_{k}}$ 已经被计算，并被称为**共轭矩阵**（$\textrm{adjoint}$），该值用于与每个子节点的雅各比 $\frac{\partial \mathbf{x}_{k}}{\partial \mathbf{x}_{j}}$ 相乘。

通过使用 $\textrm{API}$ 的定义静态图，可以提前计算出计算图（ $\textrm{Tensorflow 1}$ 的工作原理。）或者，可以通过跟踪函数在输入上的执行情况来“及时”计算计算图（这就是 $\textrm{Tensorflow-eager}$ 模式以及 $\textrm{JAX}$ 和 $\textrm{PyTorch}$ 的工作原理。）后一种方法使处理动态图变得更容易，动态图的形状可以根据函数计算的值而改变。

## 13.4 训练神经网络

本节，我们将讨论如何基于数据对 $\textrm{DNNs}$ 进行训练。 最标准的方式是使用最大似然估计，通过最小化负对数似然：
$$
\mathcal{L}(\theta)=-\log p(\mathcal{D} \mid \theta)=-\sum_{n=1}^{N} \log p\left(\mathbf{y}_{n} \mid \mathbf{x}_{n} ; \theta\right) \tag{13.73}
$$
通常情况下，也会增加一个正则项（比如负对数先验），正如我们在 $13.5$ 节讨论的那样。

原则上，我们可以使用 $\textrm{backprop}$ 算法（第 $13.3$ 节）来计算这种损失的梯度，并将其传递给在第 $5$ 章中所讨论的现成的优化器。（第 $5.4.6.3$ 节中的 $\textrm{Adam}$ 优化器是一个普遍的选择，因为它能够扩展到大型数据集（由于是 $\textrm{SGD}$ 类型的算法），并且能够相当快速地收敛（由于使用对角预处理和动量）。）但是，在实践中，也有可能无法取得较好的结果。在本节中，我们将讨论可能出现的各种问题，以及一些解决方案。有关 $\textrm{DNN}$ 训练策略的更多详细信息，请参阅其他书籍[HG20][^HG20]；[Zha+19a][^Zha19a]；[Ger19][^Ger19]。

除了具体的实践问题，还有重要的理论问题。特别地，我们注意到 $\textrm{DNN}$ 损失不是一个凸目标，所以通常我们无法找到全局最优解。尽管如此，$\textrm{SGD}$ 总能收敛到出人意料的好的结果。具体的原因仍在研究当中，可以参考[Bah+20][^Bah20]对一些最近工作的回顾。

[^HG20]:$\textrm{[HG20]}$: J. Howard and S. Gugger. Deep Learning for Coders with Fastai and   PyTorch: AI Applications Without a PhD. en. 1st ed. O’Reilly Media, Aug. 2020.
[^Zha19a]:
[^Ger19]:
[^Bah20]:

![lr_sche](/assets/img/figures/lr_sche.png)

> 图$13.15$: 不同的学习率启发方式。$(a)$ 指数下降方案；$(b)$ 分段常数方案；$(c)$ 余弦退火方案（又被称为 **含重启器的随机梯度**（$\textrm{stochastic gradient with restarts}$））[Smi17][^Smi17],[LH17][^LH17]；$(d)$ 单周期方案[Smi18][^Smi18]。

![lr_finder](/assets/img/figures/lr_finder.png)

> 图$13.16$: 训练损失与学习率的关系，在 $FashionMNIST$ 数据集上，使用原生的 $SGD$ 拟合一个小的 $MLP$ 模型。（蓝色为原始损失，橘黄色为 $EWMA$ 版本）。

[^Smi17]: 
[^LH17]:
[^Smi18]:

### 13.4.1 调整学习率

在用 $\textrm{SGD}$ 训练 $\textrm{DNN}$ 时，调整学习速率对于取得良好的性能非常重要。在第 $5.4.3$ 节中，我们讨论了 $\textrm{SGD}$ 的学习率若要收敛到局部最优值必须满足的必要条件，但是这些条件并没有精确地指出要使用什么样的**学习率调节方案**（$\textrm{learning rate schedule}$）。

在深度学习文献中，提出了许多启发式的方案，其中一些方法如图 $13.15$ 所示。包括分段常数策略（图$\textrm{13.15b}$）、余弦或周期策略（图$\textrm{13.15c}$）、单周期策略（图$\textrm{13.15d}$）等（后者首先从一个较小的学习率开始，以防止参数“爆炸”，然后逐步增加，直到找到一个比较好的学习率，接着再逐步下降，以找到局部最小值）

除了选择学习率的衰减策略外，还需要选择初始学习率 $\eta_{0}$。[BCN18][^BCN18]提出了一种常见的启发式方法，该方法在[Smi18][^Smi18]中被同时发现（并称之为“**学习速率发现器**”（$\textrm{learning rate finder}$）），该方法定义为：从一个较小的学习率开始，计算小批量数据的训练或验证损失，然后依次尝试更大的（每一步增加 $10$ 倍）学习率，直到损失在 $\eta_{\max }$“爆炸”。 举例来说，在图 $13.16$ 中，我们发现 $\eta_{\max } \approx 0.1$ 。然后我们将 $\eta_{0}$ 设置为比 $\eta_{\mathrm{max}}$ 稍小的值 （如小于$10$倍大小）。

[^BCN18]:
[^Smi18]:

![sigmoid_act](/assets/img/figures/sigmoid_act.png)

> 图 $13.17$: $(a)$ $\textrm{sigmoid}$ 激活函数。 $(b)$ 对应的导数。

![relu_act](/assets/img/figures/relu_act.png)

> 图 $13.18$: $(a)$ $\textrm{ReLU}$ 激活函数。 $(b)$ 对应的导数。

### 13.4.2 梯度消失问题

在某些 $\textrm{DNN}$ 中，梯度信号在通过网络传播回来时会变为 $0$，进而阻止了学习过程的继续。这就是所谓的**梯度消失问题**（$\textrm{vanishing gradient problem}$）[GB10][^GB10]。

为了了解它为什么会发生，让我们考虑一下 $\textrm{sigmoid}$ 激活函数
$$
\varphi(a)=\sigma(a)=\frac{1}{1+\exp (-a)} \tag{13.74}
$$
上式的导数为
$$
\varphi^{\prime}(a)=\sigma(a)(1-\sigma(a)) \tag{13.75}
$$
现在考虑网络层 $\mathbf{z}=\sigma(\mathbf{W} \mathbf{x})$ 。假设这是最后一层，所以 $\delta=\frac{\partial \mathbf{f}(\mathbf{x}, \boldsymbol{\theta})}{\partial \mathbf{x}} = \mathbf{z}(1-\mathbf{z})$ 。使用 $13.3.3$ 节中的结果，我们发现损失函数关于 $\mathbf{x}$ 的梯度为：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}}=\delta^{\top} \mathbf{W}=\mathbf{W}^{\top} \mathbf{z}(1-\mathbf{z}) \tag{13.76}
$$
并且
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}}=\delta \mathbf{x}^{\top}=\mathbf{z}(1-\mathbf{z}) \mathbf{x}^{\top} \tag{13.77}
$$
如果权重被初始化为一个很大的值（正或负），那么 $\mathbf{W}{\mathrm{x}}$ 的（某些分量）很容易取很大的值，因此 $\mathbf{z}$ 将接近 $\textrm{sigmoid}$ 的饱和值 $0$ 或 $1$，如图 $\textrm{13.17a}$ 所示。在任何一种情况下，我们都可以看到梯度将变为 $0$，如图 $\textrm{13.17b}$ 所示。

解决梯度消失问题的标准方案是使用 $\textrm{ReLU}$ 激活函数。所以假设我们使用 $\mathbf{z}=\operatorname{ReLU}\left(\mathbf{Wx}\right)$，其中
$$
\operatorname{ReLU}(a)=\max (a, 0) \tag{13.78}
$$
梯度为
$$
\operatorname{ReLU}^{\prime}(a)=\mathbb{I}(a>0) \tag{13.79}
$$
假设这是最后一层，所以 $\delta=\frac{\partial \mathbf{f}(\mathbf{x}, \boldsymbol{\theta})}{\partial \mathbf{x}}=\mathbb{I}(\mathbf{z}>\mathbf{0})$ 。使用 $13.3.3$ 节的结果，发现激活函数的局部梯度为
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}}=\delta^{\top} \mathbf{W}=\mathbf{W}^{\top} \mathbb{I}(\mathbf{z}>\mathbf{0}) \tag{13.80}
$$
并且
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}}=\delta \mathbf{x}^{\top}=\mathbb{I}(\mathbf{z}>0) \mathbf{x}^{\top} \tag{13.81}
$$
如果权重初始化为较大的负值，会容易导致 $\mathbf{Wx}$ 变成较大的负值，进而导致 $\mathbf{z}=0$，如图 $\textrm{13.18a}$  所示。这将会导致关于权重的梯度等于 $0$，如图 $\textrm{13.18b}$ 所示。算法将永远无法摆脱该局部解，所以对应的激活单元将永远被关闭。这被称为 "$\textrm{dead relu}$" 问题。这个问题可以通过使用 $\operatorname{ReLU}$ 的非饱和变体进行解决，正如我们在 $13.2.3$ 节中所讨论的那样。

### 13.4.3 训练深度模型的困难

当我们训练非常深的模型的时候，梯度往往会趋向于变得过小（**梯度消失问题**，$\textrm{vanishing gradient problem}$）或过大（**梯度爆炸问题**，$\textrm{exploding gradient problem}$），因为误差信号在经过一系列层的时候会被放大或者抑制。（$\textrm{RNNs}$ 应用于较长序列时，也会出现这种问题，我们将会在 $15.2.5$ 节解释。）

为了更加深入地理解这个问题，考虑损失关于第 $l$ 层中某个节点的梯度：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{l}}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{l+1}} \frac{\partial \mathbf{z}_{l+1}}{\partial \mathbf{z}_{l}}=\mathbf{J}_{l} \mathbf{g}_{l+1} \tag{13.82}
$$
其中 $\mathbf{J} _{l}=\frac{\partial \mathbf{z} _{l+1}}{\partial \mathbf{z} _{l}}$ 为雅各比矩阵，$\mathbf{g} _{l+1}=\frac{\partial \mathcal{L}}{\partial \mathbf{z} _{l+1}}$ 表示下一层的梯度。如果 $\mathbf{J} _l$ 在不同的层都是一个常数，那么显然最后一层的梯度 $\mathbf{g} _L$ 对第 $l$ 层的贡献为 $\mathbf{J}^{L-l} \mathbf{g} _{L}$ 。所以系统的行为将依赖于 $\mathbf{J}$ 的特征向量。

尽管 $\mathbf{J}$ 是一个实数矩阵，但它不是（一般情况下）对称的， 所以它的特征值和特征向量可能是复数， 其中的虚数部分对应着振荡的行为。令 $\lambda$ 表示 $\mathbf{J}$ 的 **谱半径**（$\textrm{spectral radius}$），即特征值的绝对值的最大值。 如果它大于 $1$， 梯度将发生爆炸； 如果小于 $1$， 梯度将会消失。 （类似地， $\mathbf{W}$ 的谱半径， 连接着 $\mathbf{z} _l$ 和 $\mathbf{z} _{l+1}$， 决定了在前向模式下动态系统的稳定性。）

梯度爆炸问题可以通过 **梯度截断**（$\textrm{gradient clipping}$） 来进行改善，当梯度过大时，我们将它的幅值进行截断， 举例来说， 使用
$$
\mathrm{g}^{\prime}=\min \left(1, \frac{c}{\|\mathrm{~g}\|}\right) \mathrm{g} \tag{13.83}
$$
通过这种方式，$\mathrm{g}^{\prime}$ 的范数将不会超过 $c$，但其更新方向始终与 $\mathrm{g}$ 相同。

然而，梯度消失问题却很难解决。有如下几种解决方案：

- 更改模型结构，从而使得梯度的更新通过相加而不是相乘的形式；见 $13.4.4$ 节。
- 更改模型结构，使每一层的激活值标准化，从而使整个数据集上的激活值的分布在训练期间保持一致； 见 $13.4.5$ 节。
- 小心地选择参数的初始化值； 见 $13.4.6$ 节。

![res_block](/assets/img/figures/res_block.png)

> 图$13.19$: $(a)$ 残差模块的示意图。$(b)$ 残差模块有利于深度网络训练的原因。图形来自于 [Ger19][^Ger19] 的图 $14.16$。

### 13.4.4 残差连接

对于 $\textrm{DNNs}$ 而言，一种解决梯度消失问题的方案是使用 **残差网络**（$\textrm{residual network, ResNet}$）[He+16a][He16a]。该前向网络中的每一层的形式是一个**残差模块**（$\textrm{residual block}$），定义为
$$
\mathcal{F}_{l}^{\prime}(\mathrm{x})=\mathcal{F}_{l}(\mathrm{x})+\mathrm{x} \tag{13.84}
$$
其中 $\mathcal{F} _{l}$ 是一个很浅的非线性映射函数 （比如： 线性层—激活层—线性层）。 $\mathcal{F} _{l}$ 函数计算需要添加到输入  $\mathrm{x}$  中以生成所需输出的残差项或增量；相较于直接让网络学习如何预测输出，学习在输入的基础上产生的小扰动通常更加容易。（如第 $14.3.2.4$ 节所述，残差连接通常与 $\textrm{CNN}$ 一起使用，但也可用于 $\textrm{MLP}$。）

含有残差连接的模型与没有残差连接的模型具有相同的参数数量，但是训练起来比较容易。原因是梯度可以直接从输出传递到浅层，如图 $\textrm{13.19b}$ 所示。要说明这一点，值得注意的是，输出层的激活可以通过使用任意浅层 $l$ 的输出得到：
$$
\mathrm{z}_{L}=\mathrm{z}_{l}+\sum_{i=l}^{L-1} \mathcal{F}_{i}\left(\mathrm{z}_{i} ; \boldsymbol{\theta}_{i}\right) \tag{13.85}
$$
所以我们可以计算损失关于 $l$ 层的梯度：
$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_{l}} &=\frac{\partial \mathbf{z}_{l}}{\partial \boldsymbol{\theta}_{l}} \frac{\partial \mathcal{L}}{\partial \mathbf{z}_{l}} \tag{13.86} \\
&=\frac{\partial \mathbf{z}_{l}}{\partial \boldsymbol{\theta}_{l}} \frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L}} \frac{\partial \mathbf{z}_{L}}{\partial \mathbf{z}_{l}} \tag{13.87} \\
&=\frac{\partial \mathbf{z}_{l}}{\partial \boldsymbol{\theta}_{l}} \frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L}}\left(1+\sum_{i=l}^{L-1} \frac{\partial f\left(\mathbf{z}_{i} ; \boldsymbol{\theta}_{i}\right)}{\partial \mathbf{z}_{l}}\right) \tag{13.88} \\
&=\frac{\partial \mathbf{z}_{l}}{\partial \boldsymbol{\theta}_{l}} \frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L}}+\text { otherterms } \tag{13.89}
\end{align}
$$
所以我们发现第 $l$ 层的梯度可以直接取决于第 $L$ 层的梯度，并与网络的深度无关。

### 13.4.5 Batch normalization

对 $\textrm{DNN}$ 结构的另一个普遍的修改是添加一个层，该层可以确保层内激活值的分布均值为 $0$ 和方差为 $1$。这被称为**批处理规范化** （$\mathrm{batch\ normalization, BN}$）[IS15][^IS15]。

更准确地说，我们将样本 $n$ （在某一层）的激活向量 $\mathbf{z} _n$ （或者是预激活向量 $\mathbf{a} _n$）替换为 $\tilde{\mathbf{z}} _{n}$ ，计算方式为：


$$
\begin{align}
\tilde{\mathbf{z}}_{n} & =\pmb{\gamma} \odot \hat{\mathbf{z}}_{n}+\pmb{\beta} \tag{13.90} \\
\hat{\mathbf{z}}_{n} & =\frac{\mathbf{z}_{n}-\pmb{\mu}_{\mathcal{B}}}{\sqrt{\pmb{\sigma}_{\mathcal{B}}^{2}+\epsilon}} \tag{13.91} \\
\pmb{\mu}_{\mathcal{B}} & =\frac{1}{|\mathcal{B}|} \sum_{\pmb{\mathbf{z}} \in \mathcal{B}} \mathbf{z} \tag{13.92} \\
\pmb{\sigma}_{\mathcal{B}}^{2} & =\frac{1}{|\mathcal{B}|} \sum_{\pmb{\mathbf{z}} \in \mathcal{B}}\left(\mathbf{z}-\pmb{\mu}_{\mathcal{B}}\right)^{2} \tag{13.93}
\end{align}
$$
其中 $\mathcal{B}$ 为包含样本 $n$ 的批次，$\pmb{\mu} _\mathcal{B}$ 为该批次数据激活值的均值[^4]，$\pmb{\sigma} _{\mathcal{B}}^{2}$ 为对应的方差， $\hat{\mathbf{z}} _{n}$ 表示标准化后的激活向量，$\tilde{\mathbf{z}} _{n}$ 表示经过平移和缩放后的结果 （$\mathrm{BN}$ 层的输出），$\pmb{\beta}$ 和 $\pmb{\gamma}$ 表示该层的可学习参数， $\epsilon>0$ 是一个值很小的常数。考虑到 $\mathrm{BN}$ 是可微的， 我们可以很容易地将梯度反传到该层的输入和 $\mathrm{BN}$ 的参数 $\pmb{\beta}$ 和 $\pmb{\gamma}$。

[^4]:  当应用于卷积层时，我们平均跨空间位置和跨示例，但不跨通道（因此 $\pmb{\mu}$ 的长度是通道数）。当应用于一个完全连接的层时，我们只需对示例进行平均（因此 $\pmb{\mu}$ 的长度就是层的宽度）。

对于输入层， $\mathrm{batch\ normalization}$ 等价于我们在 $10.2.8$ 节讨论的常规的标准化过程。值得注意的是， 输入层的均值和方差只需要计算一次，因为数据是静态的。然而，中间层的经验均值和方差是不断改变的，因为参数一直在更新。（这通常被称为 “**内协变量漂移**”（$\mathrm{internal\ covariate\ shift}$）。这就是我们需要对每一个批次重新计算 $\pmb{\mu}$ 和 $\pmb{\sigma}^2$ 的原因。

$\mathrm{BN}$ 的作用（在训练速度和稳定性方面）是非常显著的，尤其是对于深度 $\textrm{CNNs}$。具体的原因还不是很清楚， 但 $\mathrm{BN}$ 似乎使优化曲面变得更加平滑 [San+18][^San18]。同时它也降低了对学习率的敏感性 [ALL18][^ALL18]。除了计算方面的优势，它还具备统计上的优势。特别地， $\mathrm{BN}$ 更像一个正则器；事实上，它可以被证明是相当于一种形式的近似贝叶斯推理 [TAS18][^TAS18];[Luo+19][^Luo19]。

然而，依赖于小批量数据会导致几个问题。首先，在小批量训练时，它可能会导致参数估计不稳定，尽管该方法的最新版本**批量重规范化**（$\textrm{batch renormalization}$）[Iof17][^Iof17]部分地解决了这个问题。其次，$\mathrm{BN}$ 在推理阶段需要进行一些调整， 因为在测试阶段时的 $\textrm{batch size}$ 可能是 $1$。 具体的测试过程为：在训练之后， 计算训练集中所有样本在第 $l$ 层的 $\pmb{\mu}_l$ 和 $\pmb{\sigma}_l^2$， 然后”冻结“这些参数， 并将这些值添加到该层其他参数的列表中， 即 $\pmb{\beta}_l$ 和 $\pmb{\gamma}_l$ 。在测试阶段， 我们利用这些冻结的参数计算 $\pmb{\mu}_l$ 和 $\pmb{\sigma}_l^2$， 而不是从测试批次中计算统计量。（所以在使用包含 $\mathrm{BN}$ 的模型中， 我们需要指定模型是用于训练还是测试。）

> The standard approach to this is as follows: after training, compute $\pmb{\mu}_l$ and $\pmb{\sigma}_l^2$ for layer $l$ across all the examples in the training set, and then “freeze” these parameters, and add them to the list of other parameters for the layer, namely $\pmb{\beta}_l$ and $\mathbf{\gamma}_l$. At test time, we then use these frozen training values for  $\pmb{\mu}_l$ and $\pmb{\sigma}_l^2$ , rather than computing statistics from the test batch. (Thus when using a model with $\textrm{BN}$, we need to specify if we are using it for inference or training.)

为了提高推理速度，我们可以将冻结后的批处理规范层与前一层结合起来。 特别地， 假设前一层计算 $\mathbf{XW}+\mathbf{b}$；将其与 $\mathrm{BN}$ 组合 $\pmb{\gamma}\ \odot(\mathbf{XW+b-\pmb{\mu}})/\pmb{\sigma}\ +\ \pmb{\beta}$。如果我们定义 $\mathbf{W}^{\prime}=\gamma \odot \mathbf{W} / \sigma$ 和 $\mathbf{b}^{\prime}=\gamma \odot(\mathbf{b}-\boldsymbol{\mu}) / \sigma+\boldsymbol{\beta}$ ， 然后我们可以将组合的层写成 $\mathbf{X} \mathbf{W}^{\prime}+\mathbf{b}^{\prime}$。 这被称为 **融合批规范**（$\mathrm{fused\ batchnorm}$）。在训练过程中，可以开发类似的技巧来加速 $\mathrm{BN}$ 计算[Jun+19][^Jun19]。

[^IS15]:
[^San18]:
[^ALL18]:
[^ TAS18]: text
[^ Luo19]: text
[^ Iof17]: text
[^ Jun19]: text

### 13.4.6 参数初始化

由于 $\textrm{DNN}$ 训练的目标函数是非凸的， $\textrm{DNN}$ 参数的初始化方式对我们最终能够得到的优化解以及训练优化过程的难易程度（即 ，信息在前向与反向传播过程中的有益程度）有很大的影响。 在本节的剩余部分，我们将介绍一些用于初始化参数的常见的启发式方法。

#### 13.4.6.1 启发式

在 [GB10][^GB10] 中，他们表明从标准正态中采样得到的参数会导致输出的方差远大于输入的方差，从而导致梯度爆炸。为了解决这个问题，他们提出从均值为 $0$ ，方差为 $\sigma^{2}=1 / \text{fan} _{\text{avg}}$ 的高斯分布中采样参数，其中$ \text{fan} _\text{avg} = (\text{fan} _\text{in} + \text{fan} _\text{out})/2$，其中 $\text{fan} _\text{in}$ 是一个单元的扇入（传入连接数），$\text{fan} _\text{out}$是单元的扇出（传出连接数）。这种方法被称为 $\textbf{Xavier}$ 初始化或 $\textbf{Glorot}$ 初始化，以 [GB10][^GB10] 的第一作者的名字命名。如果我们使用  $\sigma^{2}=1 / \text{fan} _{\text{in}}$，我们会得到一种称为 $\textbf{LeCun}$ 初始化的方法，以 $\textrm{Yann LeCun}$ 的名字命名，他于 $1990$ 年代提出该方法，等效于 $\text{fan} _\text{in} = \text{fan} _\text{out} $ 时的 $\textbf{Glorot}$ 初始化。如果我们用 $\sigma^{2}=2 / \text{fan} _{\text{in}}$，该方法称为 $\textbf{He}$初始化，得名于$\text{Kaiming He}$，他在[He+15][^He15]中提出了该方法。初始化方法的最佳选择取决于所使用的激活函数。对于线性、$\text{tanh}$、$\text{logistic}$ 和 $\text{softmax}$ 激活函数，推荐使用 $\textbf{Glorot}$。对于 $\text{ReLU}$ 及其变体，推荐使用 $\text{He}$。对于 $\text{SELU}$，推荐 $\text{LeCun}$。有关更多启发式的方法，请参考 [Gér19][^Ger19]。

```typo
Ximing He ——> Kaiming He
```

我们也可以使用一种数据驱动的参数初始化方法。举例来说，[MM16][^MM16] 提出一种简单而且有效的策略，被称为 $\textbf{layer-sequential unit-variance (LSUV)}$ 初始化，工作原理如下。首先我们对每一层（全连接或卷积层）的权重初始化一个 [SMG14][^SMG14] 中提出的正交矩阵。（该正交矩阵可以通过如下方式获得，首先从分布 $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 中进行采样， 并将 $\mathbf{w}$  $\text{reshape}$ 为矩阵 $\mathbf{W}$，然后使用 $\textrm{QR}$ 或者 $\textrm{SVD}$ 分解获取矩阵的正交基。）接着，对于每一层 $l$，我们计算一个 $\textrm{minibatch}$ 的激活值的方差 $v_l$；然后使用 $\mathrm{W} _{l}:=\mathrm{W} _{l} / \sqrt{v _{l}}$ 进行重缩放。该策略可以看作是正交初始化与 $\textrm{batch normalization}$ 的组合，而后者只需要针对第一个 $\textrm{mini-batch}$ 进行计算。实验表明，这样的 $\textrm{normalization}$ 已经足够，并且该方法要快于完全使用 $\textrm{batch normalization}$。

另一种方法被称为 $\mathbf{fixup}$ [ZDM19][^ZDM19]。该方法可以用于训练没有 $\textrm{batchnorm}$ 的非常深的残差网络。

[^GB10]: 
[^He15]:
[^Ger19]:
[^MM16]:
[^SMG14]:
[^ZDM19]:

#### 13.4.6.2 一种函数空间视角

考虑一个含有 $L$ 个隐藏层和 $1$ 个线性输出层的 $\textrm{MLP}$：
$$
f(\mathbf{x} ; \boldsymbol{\theta})=\mathbf{W}_{L}\left(\cdots \varphi\left(\mathbf{W}_{2} \varphi\left(\mathbf{W}_{1} \mathbf{x}+\mathbf{b}_{1}\right)+\mathbf{b}_{2}\right)\right)+\mathbf{b}_{L} \tag{13.94}
$$
我们可以假设模型参数分别服从如下分布：
$$
\mathbf{W}_{\ell} \sim \mathcal{N}\left(0, \alpha_{\ell}^{2} \mathbf{I}\right), \mathbf{b}_{\ell} \sim \mathcal{N}\left(0, \beta_{\ell}^{2} \mathbf{I}\right) \tag{13.95}
$$
我们可以对上述分布进行重参数化：
$$
\mathbf{W}_{\ell}=\alpha_{\ell} \eta_{\ell}, \eta_{\ell} \sim \mathcal{N}(0, \mathbf{I}), \mathbf{b}_{\ell}=\beta_{\ell} \epsilon_{\ell}, \epsilon_{\ell} \sim \mathcal{N}(0, \mathbf{I}) \tag{13.96}
$$
所以每一种先验超参都指定了一个如下的随机函数：
$$
f(\mathrm{x} ; \boldsymbol{\alpha}, \boldsymbol{\beta})=\alpha_{L} \eta_{L}\left(\cdots \varphi\left(\alpha_{1} \eta_{1} \mathrm{x}+\beta_{1} \epsilon_{1}\right)\right)+\beta_{L} \epsilon_{L} \tag{13.97}
$$
为了理解这些超参数的影响，我们可以从这些先验中采样得到 $\textrm{MLP}$ 的参数，并且绘制出最终的随机函数。我们使用 $\textrm{sigmoid}$ 非线性函数，所以 $\varphi(a)=\sigma(a)$。 我们考虑 $L=2$ 层的网络， 所以 $\mathbf{W}_1$ 对应 $\text{input-to-hidden}$ 的权重， $\mathbf{W}_2$ 对应 $\text{hidden-to-output}$ 的权重。我们假设输入和输出为标量，所以我们最终可以随机生成非线性映射 $f: \mathbb{R} \rightarrow \mathbb{R}$。

图 $13.20(a)$ 展示了一些采样得到的函数，其中 $\alpha_{1}=5, \beta_{1}=1, \alpha_{2}=1, \beta_{2}=1$ 。在图 $13.20(b)$ 中我们增加了 $\alpha_1$；这使得第一层的权重变得更大，导致 $\textrm{S}$ 型函数的形状更陡 （与图 $10.2$ 相比）。在图 $13.20(c)$ 中，我们增加 $\beta_1$；这使得第一层的偏置更大，使得函数的中心更多地向左右两侧移动。图 $13.20(d)$中，我们增加 $\alpha_2$，使得第二层的线性层权重更大，导致函数变得更加 “扭曲”(wiggly) （对输入的变化更加敏感，所以导致更大的动态范围）

上述结果只是针对 $\textrm{sigmoidal}$ 激活函数的情况。 $\textrm{ReLU}$ 函数的结果可能不一样。举例来说， [WI20, App.E][^WI20] 表明，对于包含 $\textrm{ReLU}$ 激活单元的 $\text{MLPs}$ ，如果我们令 $\beta_l=0$，那么所有的偏置项都为 $0$， 那么改变 $\alpha_l$ 的影响只是对输出进行尺度缩放。为了说明这一点，注意到公式 $(13.97)$ 可以简化为
$$
\begin{align}
f(\mathrm{x} ; \alpha, \beta=0) &=\alpha_{L} \eta_{L}\left(\cdots \varphi\left(\alpha_{1} \eta_{1} \mathrm{x}\right)\right)=\alpha_{L} \cdots \alpha_{1} \eta_{L}\left(\cdots \varphi\left(\eta_{1} \mathrm{x}\right)\right) \tag{13.98} \\
&=\alpha_{L} \cdots \alpha_{1} f(\mathrm{x} ;(\alpha=1, \beta=0)) \tag{13.99}
\end{align}
$$
其中我们使用了 $\text{ReLU}$ 的规律， $\varphi(\alpha z)=\alpha \varphi(z)$ 对任意正数 $\alpha$ 都成立，$\varphi(\alpha z)=0$ 对于任意的负数 $\alpha$ 都成立 （因为预激活值 $z \gt 0$）。一般情况下， 输入信号在一个随机初始化的网络中进行前向和反向传播时到底会发生什么情况，通常由 $\alpha$ 和 $\beta$ 决定，更多细节可参考 [Bah+20][^Bah20]。

从上面的分析中我们发现， $\text{DNN}$ 中参数的不同分布对应于函数的不同分布。所以模型的随机初始化更像是从先验知识中进行采样。当神经网络趋向于无限宽时，我们可以推导出该先验分布的解析解：这被称为 **神经网络高斯过程**（$\textrm{neural network Gaussian process}$）,我们将会在本书的第二册进行解释 [Mur22][^Mur22]。

[^WI20]:
[^Bah20]:
[^Mur22]:

## 13.5 正则化

在 $13.4$ 节，我们从计算角度讨论了在训练（大型）神经网络过程中的一些问题。本节，我们会从统计学的角度对这个问题展开讨论。特别地，我们将集中讨论避免过拟合的方法。关于这一点十分重要，因为较大的神经网络很容易具有百万级的参数。

### 13.5.1 提前终止 (Early stopping)

或许避免过拟合最好的方式就是**提前终止** ($\textrm{early stopping}$)，这是一种启发式的方法，当模型在验证集上的错误率开始增加时终止对模型的训练（见图 $4.7$）。该方法之所以奏效，是因为我们限制了优化算法将训练样本中的信息传递到模型参数的能力，正如 [AS19][^AS19] 所解释的。

[^AS19]: 

### 13.5.2 权重衰减 (Weight decay)

一种常见的避免过拟合的方式是为参数赋予一个先验分布，然后使用 $\textrm{MAP}$ 估计。通常情况下对模型的权重使用高斯先验 $\mathcal{N}\left(\mathbf{w} \mid \mathbf{0}, \alpha^{2} \mathbf{I}\right)$， 对偏置使用 $\mathcal{N}\left(\mathbf{b} \mid \mathbf{0}, \beta^{2} \mathbf{I}\right)$。（参考 $13.4.6.2$ 节对该先验的讨论）这等价于目标函数的 $l_2$ 正则。在神经网络的文献中，被称为 **权重衰减** （$\textrm{weight decay}$），因为该方法鼓励值小的权重，即更加简单的模型，正如岭回归的原理一样 （$11.3$ 节）。

### 13.5.3 稀疏化 DNNs

考虑到神经网络中包含很多权重，通常鼓励稀疏化是有益的。这将允许我们进行**模型压缩**（$\textrm{model compression}$），从而可以节省存储和时间。为了实现这一点，我们可以使用 $l_1$ 正则（$11.5$ 节），或者 $\textrm{ARD}$ （$11.6.6$ 节），或者 $\textrm{spike and slab}$ 先验（$11.6.4.1$ 节），或者 $\textrm{horseshoe}$ 先验（$11.6.4.3$ 节）等。（[GEH19][^GEH19] 给出了相关知识的综述）。

考虑一个具体例子，图 $13.21$ 展示了一个 $5$ 层 $\textrm{MLP}$，该模型已经对 $1$ 维数据进行了拟合，并且对权重使用了 $l_1$ 正则。我们发现最终的图形拓扑结构是稀疏的。当然，存在很多稀疏化估计的方法也是可行的。

尽管直觉上稀疏化的拓扑结构很具有吸引力，但实际上这种方法却很少被普遍使用，因为主流的 $\textrm{GPUs}$ 只针对 *稠密* （$\textit{dense}$）矩阵乘法作了优化，而稀疏矩阵的乘法并没有什么计算上的优势。然而，如果我们使用能够鼓励 *组* ($\textit{group}$) 稀疏化的方法，我们对模型中的整个层都进行减枝操作。这将导致 *块稀疏化*（$\textit{block sparse}$） 权重矩阵，这将有利于计算上的加速和存储上的节约（参考 [Sca+17][^Sca17];[Wen+16][^Wen16];[MAV17][^MAV17];[LUW17][^LUW17]）。

[^GEH19]:
[^Sca17]:
[^Wen16]:
[^MAV17]:
[^LUW17]:

### 13.5.4 Dropout

假设我们依概率 $p$ 随机的将每个节点的输出连接关闭，如图 $13.22$ 所示。该技术被称为 $\textbf{dropout}$ [Sri+14][^Sri14]。

$\textrm{Dropout}$ 可以显著地降低过拟合风险并且被广泛使用。直觉上，该技术之所以有效的原因在于，它阻止了隐藏节点间的复杂的互适应性。换句话说，每一个节点必须表现良好，及时某些其他的节点被随机的取消。这将阻止节点之间学习到复杂但脆弱的依赖关系[^5]。一个更加正式的解释，从高斯尺度混合先验角度进行分析，该解释可参考[NHLS19][^NHLS19]。

[^5]:  Geoff Hinton, who invented dropout, said he was inspired by a talk on sexual reproduction, which encourages genes to be individually useful (or at most depend on a small number of other genes), even when combined with random other genes. 

我们可以将该技术视作是一种对权重的含噪估计 $\theta_{l i j}=w_{l i j} \epsilon_{l i}$，其中 $\epsilon_{l i} \sim \operatorname{Ber}(1-p)$ 为伯努利噪声项。（所以如果我们采样 $\epsilon_{l i}=0$ , 那么所有连接 $l-1$ 层的节点 $i$ 与 $l$ 层的 $j$ 节点的权重将被设置为 $0$。）在测试阶段，我们通常将噪声关闭，这等价于令 $\epsilon_{l i}=1$ 。为了获得无噪声权重的估计值，我们应该使用 $w_{l i j}=\theta_{l i j} / \mathbb{E}\left[\epsilon_{l i}\right]$，所以测试阶段权重的期望值与训练阶段的期望值保持一致。对于伯努利噪声，我们有 $\mathbb{E}[\epsilon]=1-p$ ，所以在测试阶段我们应该除以 $1-p$。

```typo
th enoise -> the noise
```

然而，我们也可以在测试阶段使用 $\textrm{dropout}$。其最终的结果为 网络的$\textrm{ensemble}$，每一个子网络对应一个稀疏的图结构。这被称为 $\textbf{Monte Carlo dropout} $[GG16][^GG16];[KG17][^KG17]，具体形式为:
$$
p(\mathbf{y} \mid \mathbf{x}, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^{S} p\left(\mathbf{y} \mid \mathbf{x}, \hat{\mathbf{W}} \epsilon^{s}+\hat{\mathbf{b}}\right) \tag{13.100}
$$
其中 $S$ 为采样样本的数量，$\hat{\mathbf{W}} \epsilon^{s}$ 表明我们将所有估计的矩阵与一个经随机采样得到的噪声向量进行了矩阵乘。这种方式通常可以提供一个对贝叶斯后验预测分布 $p(\mathbf{y} \mid \mathbf{x}, \mathcal{D})$ 的好的近似，尤其是在噪声比是被优化过的情况下 [GHK17][^GHK17]。

[^Sri14]:
[^NHLS19]:
[^GG16]:
[^KG17]:
[^GHK17]:



### 13.5.5 贝叶斯神经网络

主流的 $\textrm{DNNs}$ 通常使用（含惩罚项）最大似然估计的目标函数来搜寻一种参数的配置。然而，对于特别大的模型，其参数量往往大于数据量，所以可能存在多种可能的模型，它们对训练集的拟合情况都很好，但在泛化性能上具有差异性。通过获取后验分布中的不确定性通常是有用的。我们可以关于模型参数计算边缘概率分布实现这一点
$$
p(\mathbf{y} \mid \mathbf{x}, \mathcal{D})=\int p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta} \tag{13.101}
$$
上述结果被称为 $\textbf{Bayesian neural network}$ 或者 $\textbf{BNN}$。它可以被认识是一个含不同权重的神经网络的无限集成。通过对参数求边缘分布，我们可以避免过拟合[Mac95][^Mac95]。贝叶斯边缘化对于大型神经网络具有挑战性，但同样可以获得大幅的性能提升 [WI20][^WI20]。关于$\textbf{Bayesian deep learning}$ 的更多细节，可以参考书籍 [Mur22][^Mur22]。

[^Mac95]:
[^WI20]:
[^Mur22]:

## 13.6 其他形式的前馈网络

### 13.6.1 径向基函数网络

考虑单层神经网络，其中的隐藏层的输入定义为：
$$
\boldsymbol{\phi}(\mathbf{x})=\left[\mathcal{K}\left(\mathbf{x}, \boldsymbol{\mu}_{1}\right), \ldots, \mathcal{K}\left(\mathbf{x}, \boldsymbol{\mu}_{K}\right)\right] \tag{13.102}
$$
其中 $\boldsymbol{\mu} _{k} \in \mathcal{X}$ 为 $K$ 个**质心**（$\textrm{centroids}$）或 **代理**（$\textrm{exemplars}$），其中 $\mathcal{K}(\mathbf{x}, \boldsymbol{\mu}) \geq 0$ 为 **核函数**（$\textrm{kernel function}$）。我们将在 $17.2$ 节讨论关于核函数的细节。此处我们只给出关于 **高斯核**（$\textrm{Gaussian kernel}$） 的例子
$$
\mathcal{K}_{\text {gauss }}(\mathbf{x}, \mathbf{c}) \triangleq \exp \left(-\frac{1}{2 \sigma^{2}}\|\mathbf{c}-\mathbf{x}\|_{2}^{2}\right) \tag{13.103}
$$
参数 $\sigma$ 被称为核的 **带宽**（$\textrm{bandwidth}$）。需要注意的是高斯核具有平移不变性，意味着它只是关于距离 $r=\|\mathbf{x}-\mathbf{c}\|_{2}$ 的函数，所以我们可以等价地写成
$$
\mathcal{K}_{\text {gauss }}(r) \triangleq \exp \left(-\frac{1}{2 \sigma^{2}} r^{2}\right) \tag{13.104}
$$
所以它又被称为 **径向基函数核**（$\textrm{ radial basis function kernel, RBF kernel}$）。

如果一个单层神经网络使用式 $13.102$ 定义形式作为隐藏层（包含 $\textrm{RBF}$ 核），则该神经网络被称为 $\textbf{RBF network}$ [BL88][^BL88]。形式定义为
$$
p(y\mid\mathbf{x} ; \boldsymbol{\theta})=p\left(y\mid\mathbf{w}^{\top} \boldsymbol{\phi}(\mathbf{x})\right) \tag{13.105}
$$
其中 $\boldsymbol{\theta}=(\boldsymbol{\mu}, \mathbf{w})$ 。如果质心 $\boldsymbol{\mu}$ 是固定的，我们可以使用（含正则项的）最小二乘方法求解权重 $\mathbf{w}$ 的最优解，如第 $11$ 章讨论的。如果质心是未知的，我们使用无监督聚类方法对质心进行估计，如 $\textrm{K-means}$（$21.3$ 节）。作为替代方案，我们可以为训练集中的每一个数据关联一个质心，即 $\boldsymbol{\mu} _{n}=\mathbf{x} _{n}$，此时 $K=N$，这是**非参数化模型**（$\textrm{ non-parametric model}$）的例子，因为参数的数量随着数据的数量增加（这种情况下呈线性增长关系），而非独立于数据量 $N$。如果 $K=N$，模型可以完美地拟合数据，同时也有可能发生过拟合。然而，通过确保输出的权重矩阵 $\mathbf{w}$ 稀疏化，模型可以只使用输入的样本中的有限的子集，这被称为 **稀疏核机器**（$\textrm{sparse kernel machine}$），具体细节将在 $17.6.1$ 和 $17.5$ 节讨论。另一种避免过拟合的方式是使用贝叶斯方式，通过对权重 $\mathbf{w}$ 积分；这将引出另一个模型，被称为 **高斯过程**（$\textrm{Gaussian process}$），我们将在 $17.3$ 节讨论该模型的更多细节。

#### 13.6.1.1 RBF网络用于回归

我们可以使用 $\textrm{RBF}$ 网络完成回归任务，定义为 $p(y \mid \mathbf{x}, \boldsymbol{\theta})=\mathcal{N}\left(\mathbf{w}^{T} \boldsymbol{\phi}(\mathbf{x}), \sigma^{2}\right)$ 。举例来说，图 $13.24$ 展示了 $\textrm{RBF}$ 拟合 $1$ 维数据的结果，其中我们使用了 $K=10$ 个均匀分布的 $\textrm{RBF}$ 代理，但相应的带宽从小到大变化。越小的带宽导致越畸变的函数，因为只有当点 $\mathbf{x}$  接近某个代理 $\boldsymbol{\mu}_{k}$ 时，预测的函数值才会是非零值。如果带宽非常大，设计矩阵将退化为元素值为 $1$ 的常数矩阵，因为每个点与每个代理的距离相同，所以最终的函数只是一条直线。

#### 13.6.1.2 RBF网络用于分类

我们可以使用 $\textrm{RBF}$ 网络完成二分类任务，定义为 $p(y \mid \mathbf{x}, \boldsymbol{\theta})=\operatorname{Ber}\left(\mathbf{w}^{T} \boldsymbol{\phi}(\mathbf{x})\right)$ 。作为一个例子，考虑数据来源于异或函数。这是一个具有二个位输入的二值函数。真实标签如图 $13.23(a)$ 所示。在图 $13.1(b)$ 中，我们展示了一些通过抑或函数标记的数据，但我们已经对数据进行了 $\textbf{jitter}$ [^6]，使可视化更加清晰。我们发现我们没有办法将数据分开，哪怕使用阶数为 $10$ 的多项式拟合。然而，使用 $\textrm{RBF}$ 核以及 $4$ 个代理，可以轻松地解决这个问题，如图 $13.1(c)$ 所示。

[^6]:Jittering is a common visualization trick in statistics, wherein points in a plot/display that would otherwise land on top of each other are dispersed with uniform additive noise

### 13.6.2 混合专家

当我们在考虑回归任务时，通常情况下假设输出是一个单峰分布，比如高斯分布或者学生分布，其中的期望和方差是关于输入的某个函数，举例，
$$
p(\mathbf{y} \mid \mathbf{x})=\mathcal{N}\left(\mathbf{y} \mid f_{\mu}(\mathbf{x}), \operatorname{diag}\left(\sigma_{+}\left(f_{\sigma}(\mathbf{x})\right)\right)\right) \tag{13.106}
$$
其中函数 $f$ 可能是 $\textrm{MLPs}$ （可能包含一些共享的隐藏层单元，如图 $13.7$ 所示）。然而，这种假设对于 $\textbf{one-to-many}$ 函数可能并不奏效，在这种问题中，每个输入可能对应多种可能的输出。

图 $13.25a$ 给出了类似于这个函数的例子。我们发现在图形的中间存在某些 $x$ 对应着两个同样可能的 $y$ 值。在现实世界中，也存在很多类似于这样的问题，比如， 从一个图片中预测一个人的$3d$ 姿态 [Bo+08][^Bo08]， 对一张黑白图片进行着色 [Gua+17][^Gua17]，预测视频序列的未来帧 [VT17][^VT17]，等等。任务使用单峰输出密度函数并使用最大似然进行训练的模型——哪怕是灵活的非线性模型，比如神经网络——在 $\textrm{one-to-many}$ 问题中都变现得不好，因为这类模型只是输出一个模糊的平均输出。

为了防止回归均值的问题，我们可以使用 **条件混合模型**（$\textrm{ conditional mixture model }$）。也就是说，我们假设输出是 $K$ 个不同输出的加权混合，对应于每个输入 $\mathbf{x}$ 输出分布的不同峰值。在高斯分布的情况下，即
$$
\begin{align}
p(\mathbf{y} \mid \mathbf{x}) &=\sum_{k=1}^{K} p(\mathbf{y} \mid \mathbf{x}, z=k) p(z=k \mid \mathbf{x}) \tag{13.107} \\
p(\mathbf{y} \mid \mathbf{x}, z=k) &=\mathcal{N}\left(\mathbf{y} \mid f_{\mu, k}(\mathbf{x}), \operatorname{diag}\left(f_{\sigma, k}(\mathbf{x})\right)\right) \tag{13.108} \\
p(z=k \mid \mathbf{x}) &=\operatorname{Cat}\left(z \mid \mathcal{S}\left(f_{z}(\mathbf{x})\right)\right) \tag{13.109}
\end{align}
$$
其中 $f_{\mu, k}$ 为第 $k$ 个高斯分布的期望，$f_{\sigma, k}$ 为对应的方差项，$f_z$ 预测使用混合元素中的哪一个。该模型被称为 **混合专家**（$\textrm{mixture of experts, MoE}$）[Jac+91][^Jac91]；[JJ94][^JJ94]; [YWG12][^YWG12]; [ME14][^ME14]。其思想在于，第 $k$ 个子模型 $p(\mathbf{y} \mid \mathbf{x}, z=k)$ 被认为是输入空间中某个区域的“专家”。 函数 $p(z=k \mid \mathbf{x})$ 被称为 **门函数**（$\textrm{gating function}$）决定着使用哪一个专家，该函数依赖于输入值。通过为某个特定输入 $\mathbf{x}$ 指定最有可能的专家， 我们可以只用“激活”模型的一个子集。这是 **条件计算**（$\textrm{conditional computation}$）的一个例子，因为我们根据门控网络的早期计算结果决定运行哪个专家 [Sha+17][^Sha17]。

我们可以使用 $\textrm{SGD}$ 训练模型，或者使用 $\textrm{EM}$ 算法（$5.7.3$介绍了后一种方法的更多细节）。

#### 13.6.2.1 混合线性专家

在这一节中，我们考虑一个简单的例子，其中我们使用线性回归专家和线性分类门控函数，即，模型具有形式：
$$
\begin{align}
p(y \mid \mathbf{x}, z=k, \boldsymbol{\theta}) &=\mathcal{N}\left(y \mid \mathbf{w}_{k}^{\top} \mathbf{x}, \sigma_{k}^{2}\right) \tag{13.110} \\
p(z \mid \mathbf{x}, \boldsymbol{\theta}) &=\operatorname{Cat}(z \mid \mathcal{S}(\mathbf{V} \mathbf{x})) \tag{13.111}
\end{align}
$$
单独的权重项 $p(z=k \mid \mathbf{x})$ 被称为专家 $k$ 对输入 $\mathbf{x}$ 的**责任**（$\textrm{ responsibility }$）。在图 $13.25b$中，我们发现门控网络如何平滑地将输入空间划分给 $K=3$个专家。

每个专家 $p(y \mid \mathbf{x}, z=k)$ 对应一个线性回归模型，且包含不同的参数。如图 $13.25c$ 所示。

如果我们将专家的加权组合作为输出，我们将得到图 $13.25a$ 中的红色曲线，显然是一个不好的预测结果。如果我们仅仅使用最有可能的专家进行预测（比如，具有最高责任的专家），我们将得到黑色的不连续曲线，显然是更好的预测器。

#### 13.6.2.2 混合密度网络

门控函数和专家可以是任何类型的条件概率模型，而不仅仅是一个线性模型。如果我们都使用 $\textrm{DNNs}$，最终将得到一个被称为 **混合密度网络**（$\textrm{mixture density network, MDN}$）[Bis94][^Bis94]; [ZS14][^ZS14] 或者 **深度混合专家**（$\textrm{deep mixture of experts}$）[CGG17][^CGG17]。图 $13.26$ 给出了一个模型的草图。

#### 13.6.2.3 层次 MOEs

如果每个专家本身就是一个 $\textrm{MoE}$ 模型，最终的模型将被称为 **层次混合专家**（$\textrm{hierarchical mixture of experts}$）[JJ94][^JJ94]。图 $13.27$ 给出了层次为 $2$ 的模型结构示意图。

具有 $L$ 级的 $\textrm{HME}$ 可以被认为是深度为L的“软”决策树，其中每个示例都通过树的每个分支，并且最终的预测是加权平均 （我们将在 $18.1$ 节讨论决策树）