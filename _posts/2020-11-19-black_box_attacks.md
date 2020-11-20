---
layout: post
title: 黑盒攻击的整理
categories: [深度学习, 对抗攻击]
description: 深度学习, 黑盒攻击，对抗攻击
keywords: 深度学习, 黑盒攻击，对抗攻击
---

## 梯度估计的方法
### 1. 零阶优化 ZOO （Zeroth Order Optimization）
零阶优化的方法可以用来估计梯度信息，从而产生对抗样本。假定攻击者知道模型的结果概率分布，
那么损失函数如下：

$$
ForTargetedAttack:\ f(x,t) = max\{max_{i\neq t}log[F(x)_i]-log[F(x)_t], -k\}
$$

$$
ForUntargetedAttack:\ f(x)=\{log[F(x)]_{t_0}-max_{i\neq t_0}log[F(x)]_i, -k\}
$$

作者然后使用对称差商（symmetric difference quotient）的方法来估计 $\frac{\partial f(x)}{\partial x_i}$:

$$
\hat{g}:= \frac{\partial f(x)}{\partial x_i} \approx \frac{f(x+he_i)-f(x-he_i)}{2h}
$$

这个简单的方法需要查询模型 $2p$ 次， 其中 $p$ 表示输入的维度。所以作者提出两种随机坐标方法（stochastic coordinate methods）：ZOO-Adam 和 ZOO-Newton，梯度是根据一个随机的坐标进行估计，并且更新公式使用 ADAM 的方法。作者还提出在低维产生噪声以提高效率。

### 2. Opt-Attack
Opt-Attack 的攻击场景更加困难，攻击者只能查询模型的预测标签（hard-label）。其目标函数如下：

$$
UntargetedAttack: \ g(\theta)=min_{\lambda>0} \ s.t\  f(x_0+\lambda*\frac{\theta}{||\theta||}) \neq y_0
$$

$$
TargetedAttack: \ g(\theta)=min_{\lambda>0}\ s.t \ f(x_0+\lambda*\frac{\theta}{||\theta||})=t
$$

这里 $\theta$ 是搜索的方向，$g(\theta)$ 表示输入图片$x_0$ 到最近的对抗样本的距离。通过求解 

$$
\theta^{*} = argmin_{\theta}g(\theta)
$$

 来获得对抗样本 
 
 $$
 x_0 + g(\theta^{*})*\frac{\theta^{*}}{||\theta^{*}||}
 $$
 
 。具体的方法是通过一个粗略的搜索找到一个决策边界，然后使用二叉搜索的方法进行方向更新。

### 有限差分
有限差分的损失函数主要是：

$$
L(x,y)=\phi(x+\delta)_y - max(\phi(x+\delta)_i:i\neq y)
$$

为了估计损失函数关于输入的梯度，简单的方法需要查询 $O(d)$ 次，其中 $d$ 是输入的维度。作者提出两种方法，第一种是直接随机选取一些特征，基于这些特征的组合进行梯度估计；另外一种方法是使用 PCA 对特征进行降维，然后对关键特征进行梯度估计。

### AutoZoom

AutoZoom 是基于自编码的零阶优化（Autoencoder based Zeroth Order Optimization）的简写。它的两个主要贡献点在于：1）提出自适应的随机梯度估计方法，可以使得查询次数和扰动更稳定；2）使用自编码器或者一个双线性插值的 resizer 来探索攻击过程。其攻击函数如下：

$$
min_{x\in[0,1]} Dist(x,x_0)+\lambda Loss(x, M(F(x)), t)
$$

作者提出的梯度估计方法如下：

$$
\textbf{g} = b. \frac{f(x+\beta u)-f(x)}{\beta}.u \\
\hat{\textbf{g}} = \frac{1}{q}\sum_{j=1}^q \textbf{g}_j
$$

他们证明 $\hat{\textbf{g}}$ 和 $\nabla f(x)$ 的 $L_2$ 距离具有上界：

$$
\mathcal{E} ||\hat{\textbf{g}}-\nabla f(x)|| \leq 4 (\frac{b^2}{d^2}+\frac{b^2}{dq}+\frac{(b-d)^2}{d^2}) ||\nabla f(x)||_2^2 + \frac{2q+1}{q} b^2\beta^2L^2
$$

当
$$
b\approx q
$$
时上界是最小值。

另外 AutoZoom 是在低维进行梯度估计的，
$$
d^{'}<d
$$
这样可以提高查询的效率。一个解码器 $D$ 将低维特征还原成原始特征：
$$
D: R^{d^{'}}\rightarrow R^d
$$
生成的对抗样本就是
$$
x=x_0+D(\delta^{'})
$$

### Bandit & Priors
Bandit 优化可以探索梯度的先验信息。其两个重要的观测是
- 对于梯度计算的输入数据点不是任意的，他的结构是可以预测的，并且反映梯度。
- 在 PGD 攻击方法中，不同迭代之间具有很高的相关性。

这两种观测分别表示时间依赖的先验和数据依赖的先验，可以提高黑盒对抗攻击的性能。作者使用 $t-1$ 时刻的梯度作为 $t$ 时刻梯度的先验。另一方面，像素点直接的相关性被用来理解相邻点梯度的特征。

## 迁移性方法



