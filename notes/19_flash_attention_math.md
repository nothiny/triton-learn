# 19 — Flash Attention 数学推导：从 Standard 到 Online Softmax 到 Flash

> 理解 Flash Attention 的数学是理解 IO-aware 算法设计的关键。这篇从头推导，每个公式都解释"为什么"。

---

## 0. 为什么需要这篇笔记？

Flash Attention 论文（Dao et al., 2022）的核心算法（Algorithm 1）只有 ~20 行，但每一行背后都有精妙的数学推导。如果你直接看代码，很容易"看得懂但不知道为什么对"。

这篇笔记分三步：
1. **Standard Attention** — 回顾基本公式和内存问题
2. **Online Softmax** — Flash Attention 的数学基础（Milakov & Gimelshein, 2018）
3. **Flash Attention** — 完整算法的逐行推导

---

## 1. Standard Attention — 起点

### 1.1 基本公式

给定: $$Q, K, V \in \mathbb{R}^{N \times d} \quad (N=\text{序列长度}, d=\text{head\_dim})$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

展开:
$$
\begin{aligned}
  S &= \frac{Q K^T}{\sqrt{d}}                &&\text{— attention scores (相似度矩阵)} \\
  P &= \text{softmax}(S, \dim=-1)            &&\text{— attention weights (概率分布)} \\
  O &= P V                                    &&\text{— output (values 的加权平均)}
\end{aligned}
$$

### 1.2 一步一步算

$$
\begin{aligned}
Q &= [q_1, q_2, \ldots, q_N]^T \quad (\text{每个 } q_i \text{ 是 } d \text{ 维向量}) \\
K &= [k_1, k_2, \ldots, k_N]^T \\
V &= [v_1, v_2, \ldots, v_N]^T
\end{aligned}
$$

$$S[i,j] = \frac{q_i \cdot k_j}{\sqrt{d}} \quad \text{— 第 } i \text{ 个 query 对第 } j \text{ 个 key 的相似度}$$

$$P[i,j] = \frac{\exp(S[i,j])}{\sum_k \exp(S[i,k])} \quad \text{— softmax，变成概率}$$

$$O[i,:] = \sum_j P[i,j] v_j \quad \text{— 对所有 value 的加权求和}$$

### 1.3 内存问题

Standard Attention 的内存占用:

S 矩阵: N × N 个元素
  对于 N=4096, fp16: 4096² × 2 bytes = 33.6 MB
  对于 N=8192: 8192² × 2 bytes = 134.2 MB
  对于 N=32768: 32768² × 2 bytes = 2.1 GB ← 一张 GPU 都放不下！

而且还有 P 矩阵（也是 N×N）→ 双倍内存。

Flash Attention 的洞察: S 和 P 不需要全部储存在 HBM 中。
可以通过分块计算——每次只加载一个 tile 到 SRAM，算出 O 的部分结果就扔掉 S tile。

---

## 2. Online Softmax — 关键的数学基础

Flash Attention 之前，需要先理解 **online softmax**——如何在不看到全部数据的情况下计算 softmax。

### 2.1 Naive Softmax（需要 3 次遍历）

标准 $\text{softmax}(x_1, \ldots, x_N)$:

单次遍历:
$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

数值稳定版（减去最大值）:
- $m = \max(x_1, \ldots, x_N)$ — 第 1 次遍历: 找最大值
- $y_i = \exp(x_i - m)$ — 第 2 次遍历: 计算 exp
- $s = \sum_i y_i$ — 累加
- $\text{softmax}(x_i) = y_i / s$ — 第 3 次遍历: 归一化

问题: 需要 3 次单独遍历数据 → 每遍历一次就是一次 HBM round-trip → 慢。

### 2.2 Online Softmax — 1 次遍历

Online softmax 的核心思想：**在遍历数据的同时，维护 running max 和 running sum，用 rescaling 修正之前的结果**。

**初始化:** $m = -\infty,\ s = 0$

**对于每个新元素 $x_i$:**
- $m_{\text{new}} = \max(m, x_i)$ — 更新 running max
- $s = s \cdot \exp(m - m_{\text{new}})$ ← 关键！旧的 $s$ 是基于旧 max 算的，需要缩放到新 max 的基准
- $s = s + \exp(x_i - m_{\text{new}})$ ← 新元素基于新 max
- $m = m_{\text{new}}$ — 更新 max

**所有元素处理完后:**
- $\text{softmax}(x_i) = \frac{\exp(x_i - m)}{s}$

### 2.3 为什么 rescaling 是对的？

这是推导的核心。我们一步步来验证。

关键在于理解这行:
$$s = s \cdot \exp(m - m_{\text{new}}) + \exp(x_i - m_{\text{new}})$$

假设之前处理了元素 $x_1, \ldots, x_{k-1}$:
$$s_{\text{old}} = \sum_{j=1}^{k-1} \exp(x_j - m_{\text{old}})$$

现在来了 $x_k$, 且 $m_{\text{new}} > m_{\text{old}}$。我们需要:
$$s_{\text{new}} = \sum_{j=1}^{k} \exp(x_j - m_{\text{new}})$$

其中旧元素的贡献:
$$
\begin{aligned}
\forall j < k: \quad \exp(x_j - m_{\text{new}}) &= \exp(x_j - m_{\text{old}} + m_{\text{old}} - m_{\text{new}}) \\
&= \exp(x_j - m_{\text{old}}) \cdot \exp(m_{\text{old}} - m_{\text{new}})
\end{aligned}
$$

所以旧元素在新 max 下的 sum $= s_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}})$
新元素的贡献 $= \exp(x_k - m_{\text{new}})$

所以:
$$s_{\text{new}} = s_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) + \exp(x_k - m_{\text{new}}) \quad \checkmark$$

这就是 rescale 公式的推导！它不是魔法——只是 exp 的乘法性质:
$$\exp(a - c) = \exp(a - b) \cdot \exp(b - c)$$

### 2.4 一个具体例子

数据: $[2, 1, 4, 3]$

标准 softmax:
$$
\begin{aligned}
m &= 4 \\
\exp(x-m) &: [\exp(-2), \exp(-3), \exp(0), \exp(-1)] \approx [0.135, 0.050, 1.0, 0.368] \\
s &= 1.553 \\
\text{softmax} &= [0.087, 0.032, 0.644, 0.237]
\end{aligned}
$$

Online softmax（逐步）:

**初始化:** $m = -\infty,\ s = 0$

- **$x_0 = 2$:**
  - $m_{\text{new}} = \max(-\infty, 2) = 2$
  - $s = 0 \cdot \exp(-\infty) + \exp(2-2) = 1.0$ (旧 $s=0$，直接加新值)
  - $m = 2$

- **$x_1 = 1$:**
  - $m_{\text{new}} = \max(2, 1) = 2$ (max 没变)
  - $s = 1.0 \cdot \exp(2-2) + \exp(1-2) = 1.0 + \exp(-1) = 1.0 + 0.368 = 1.368$
  - $m = 2$

- **$x_2 = 4$:**
  - $m_{\text{new}} = \max(2, 4) = 4$ (max 变了！)
  - $s = 1.368 \cdot \exp(2-4) + \exp(4-4) = 1.368 \cdot 0.135 + 1.0 = 0.185 + 1.0 = 1.185$
  - $m = 4$

- **$x_3 = 3$:**
  - $m_{\text{new}} = \max(4, 3) = 4$ (max 没变)
  - $s = 1.185 \cdot \exp(4-4) + \exp(3-4) = 1.185 + \exp(-1) = 1.185 + 0.368 = 1.553$
  - $m = 4$

最终: $m=4$, $s=1.553$

$$
\begin{aligned}
\text{softmax}(2) &= \exp(2-4)/1.553 = 0.135/1.553 = 0.087 \quad \checkmark \\
\text{softmax}(1) &= \exp(1-4)/1.553 = 0.050/1.553 = 0.032 \quad \checkmark \\
\text{softmax}(4) &= \exp(4-4)/1.553 = 1.0/1.553 = 0.644 \quad \checkmark \\
\text{softmax}(3) &= \exp(3-4)/1.553 = 0.368/1.553 = 0.237 \quad \checkmark
\end{aligned}
$$

验证通过！

---

## 3. Flash Attention — 完整推导

### 3.1 核心洞察：把 Online Softmax 应用到 Attention

Standard Attention (block-by-block):

$$\text{Attention}(Q, K, V) = \text{softmax}(Q K^T) V$$

如果直接分块计算 $Q K^T$ 会得到部分的 S，但 softmax 需要全局信息。
online softmax 正好解决了这个问题——可以在只知道部分列的情况下，
用 running max/sum 计算"部分的 softmax × V"，然后用 rescaling 修正。

算法结构:

**For each Q block** $(Q_i,\ \text{大小为 } B_r \times d)$:
- $O_i = 0,\ l_i = 0,\ m_i = -\infty$ ← 初始化 running 状态
- **For each KV block** $(K_j, V_j,\ \text{大小为 } B_c \times d)$:
  - $S_{ij} = Q_i K_j^T$ ← attention scores for this block
  - $m_{ij} = \text{rowmax}(S_{ij})$ ← local max
  - $m_{\text{new}} = \max(m_i, m_{ij})$ ← running max
  - $P_{ij} = \exp(S_{ij} - m_{\text{new}})$ ← stable exp (用新 max)
  - $l_{\text{new}} = \exp(m_i - m_{\text{new}}) \cdot l_i + \text{rowsum}(P_{ij})$ ← rescale + update sum
  - $O_i = \operatorname{diag}(\exp(m_i - m_{\text{new}})) \cdot O_i + P_{ij} V_j$ ← rescale + accumulate
  - $m_i = m_{\text{new}},\ l_i = l_{\text{new}}$ ← update running state
- $O_i = \operatorname{diag}(1/l_i) \cdot O_i$ ← final normalize
- Write $O_i$ to HBM

### 3.2 推导 O_i 的 rescaling

这是最核心但最不直观的部分：

为什么 $O_i = \operatorname{diag}(\exp(m_i - m_{\text{new}})) \cdot O_i + P_{ij} V_j$？

推导:

旧的 $O_i$ 是在旧 max $m_i$ 下计算的:
$$O_i = \sum_j P_j^{\text{old}} V_j$$
其中 $P_j^{\text{old}} = \exp(S_j - m_i)$ (每个 $S_j$ 是用旧 max 归一化的)

当 max 更新为 $m_{\text{new}} > m_i$，需要重新归一化 $P_j^{\text{old}}$:

$$
\begin{aligned}
P_j^{\text{new}} &= \exp(S_j - m_{\text{new}}) \\
&= \exp(S_j - m_i + m_i - m_{\text{new}}) \\
&= \exp(S_j - m_i) \cdot \exp(m_i - m_{\text{new}}) \\
&= P_j^{\text{old}} \cdot \exp(m_i - m_{\text{new}})
\end{aligned}
$$

所以 $O_i$ 的旧值需要乘以 $\exp(m_i - m_{\text{new}})$:
$$O_i^{\text{new}} = \exp(m_i - m_{\text{new}}) \cdot O_i^{\text{old}} + (\text{当前 block 的贡献})$$

注意 $\exp(m_i - m_{\text{new}}) < 1$ (因为 $m_{\text{new}} > m_i$),
所以旧 $O_i$ 被"缩小"了——新 max 让旧值的权重降低。

### 3.3 内存访问分析

标准 Attention:
$$
\begin{aligned}
&\text{HBM 读取: } Q(1\text{次}) + K(1\text{次}) + V(1\text{次}) = 3Nd \\
&\text{HBM 写入: } S(N \times N) + P(N \times N) + O(N \times d) \approx 2N^2 + Nd \\
&\text{— 关键问题: } N^2 \text{ 项不可忽略}
\end{aligned}
$$

Flash Attention:
$$
\begin{aligned}
&\text{HBM 读取:} \\
&\quad Q \text{ 的每个 block: } B_r \text{ 行 } \times d \\
&\quad \text{每 Q block 读一遍 K, V: } N \text{ 行 } \times d \\
&\quad \text{共 } \frac{B_r}{N} \times (B_r \times d + 2 \times N \times d) \approx 2 \times N \times d \times T_r \quad (T_r = \text{Q 的分块数}) \\
\\
&\text{HBM 写入: } O(N \times d) \text{ — 只有最终输出} \\
\\
&\text{关键: 没有 } N^2 \text{ 项！} \\
\\
&\text{Arithmetic intensity 大幅提升:} \\
&\quad \text{标准: } AI = \frac{4N^2 d}{2N^2 + 3Nd} \approx 2d \text{ FLOP/byte (对大 N)} \\
&\quad \text{Flash: } AI = \frac{4N^2 d}{4Nd} = d \text{ FLOP/byte → 对大 } d(64-128) \text{ 更接近 compute bound}
\end{aligned}
$$

### 3.4 直观理解

把 Flash Attention 想成"流水账":

你有一堆账单（K, V），你需要计算每个客户（Q）的总账（O）。

标准方法:
  把每个 (客户, 账单) 对的明细（S）全写在一本大账本上，然后统一计算。
  → 需要一本 N×N 的大账本（HBM）
  → 写一遍，读一遍

Flash Attention 的方法:
  一次只拿一个客户的资料 + 一批账单：
  1. 心里记住: "这个客户目前最大的一笔是多少"(m)
  2. 心里记住: "目前的总金额是多少"(l)
  3. 心里记住: "加权总额是多少"(O)
  4. 每看一批新账单，更新这三个数字
  5. 看完这个客户的所有账单后，最终算出总账

  → 不需要大账本（不写 N×N）
  → 只需要记住 3 个数字（O(B_r × d) SRAM）
  → 账单可以一批批地看（tiling）

---

## 4. Flash Attention v1 vs v2 — 数学层面的区别

### 4.1 相同的数学基础

v1 和 v2 的数学 formula 完全一致——它们基于同一个 online softmax。

### 4.2 算法层面的区别

**v1 (KV 外循环):**
- For each $K_j, V_j$:
  - Load $K_j, V_j$ (on chip)
  - For each $Q_i$:
    - Compute $S_{ij}$, update $O_i$

特点: KV block 被所有 Q block 复用 → 更好的 KV 数据复用
问题: 每个 Q block 需要维护自己的 running state $(m, l, O)$
- → 多个 Q block 之间没有交流
- → warp divergent

**v2 (Q 外循环):**
- For each $Q_i$:
  - Load $Q_i$ (on chip)
  - $m_i = -\infty,\ l_i = 0,\ O_i = 0$
  - For each $K_j, V_j$:
    - Compute $S_{ij}$, update $m_i, l_i, O_i$
  - $O_i = O_i / l_i$ ← 最终归一化只在 Q 循环末尾做一次
  - Write $O_i$

**改进:**
1. 最终归一化从内循环移到外循环 — 减少 non-matmul FLOPs
2. Q 外循环 — 同一 warp 内所有线程处理同一个 Q tile，更少的 divergence
3. 支持 causal masking（只需要遍历 $K_j \leq Q_i$ 的块）

### 4.3 为什么 v2 的归一化可以移到外循环？

v1: 内循环每次迭代都做一次 rescale
- **for** $K_j$:
  - $O_i = \alpha \cdot O_i + P_{ij} V_j$ ← 每次都要 rescale

v2: 延迟 rescale 到外循环
// 内循环不 rescale O_i，只更新 sum
- **for** $K_j$:
  - $P_{ij} = \exp(S_{ij})$
  - $O_i = O_i + P_{ij} V_j$ ← 不做 rescale！
  - $l_i = l_i + \text{sum}(P_{ij})$ ← 只累加 sum

// 外循环统一做 rescale + normalize
- $O_i = O_i / l_i$

等价性证明:
$$
\begin{aligned}
O_i^{\text{final}} &= \frac{\sum_j P_{ij} V_j}{\sum_j \text{sum}(P_{ij})} \\
&= \frac{\sum_j \exp(S_{ij}) V_j}{\sum_j \text{sum}(\exp(S_{ij}))} \\
&= \text{softmax}(S_i) V
\end{aligned}
$$

这个变形在数学上等价于 online softmax 的最后一步归一化。
v2 利用了这一点，把 rescaling 推迟到外循环，节省计算。

---

## 5. IO 复杂度分析

### 5.1 Formal 分析

标准 Attention:
$$\text{HBM 访问} = \Theta(Nd + N^2) \quad \text{← } N^2 \text{ 是毁灭性的}$$

Flash Attention:
每个 Q block 大小 $B_r$, KV block 大小 $B_c$

$$
\begin{aligned}
\text{Q blocks 数: } & T_r = N/B_r \\
\text{KV blocks 数: } & T_c = N/B_c \\
\\
\text{SRAM 要求: } & O(B_r \times d + B_c \times d) = O(\max(B_r, B_c) \times d) \\
\\
\text{HBM 访问: } & \\
& \text{每个 Q block: read } Q_i (B_r \times d) + \text{read all KV } (N \times d) + \text{write } O_i (B_r \times d) \\
& = T_r \times (B_r \times d + 2 \times N \times d + B_r \times d) \\
& = (N/B_r) \times (2 \times B_r \times d + 2 \times N \times d) \\
& = 2 \times N \times d + 2 \times N^2 \times d / B_r
\end{aligned}
$$

令 $B_r = \Theta(N)$:
$$= \Theta(Nd) \quad \text{← 没有 } N^2 \text{ 项！}$$

更精确地: $B_r = \Theta(\sqrt{N})$ 是最优的（SRAM 和 IO 的平衡）
$$\text{HBM 访问} = \Theta\left(\frac{N^2 d}{\sqrt{N}}\right) = \Theta(N^{1.5} d) \quad \text{← 仍然比 } \Theta(N^2) \text{ 好}$$

### 5.2 数值实例

$$N=4096,\ d=64,\ \text{fp16},\ B_r=B_c=128$$

标准 Attention:
$$
\begin{aligned}
S + P &: 2 \times 4096^2 \times 2 = 67.1\ \text{MB (写入)} \\
\text{读取 } Q,K,V &: 3 \times 4096 \times 64 \times 2 = 1.6\ \text{MB} \\
\text{Total } &\approx 68.7\ \text{MB per head}
\end{aligned}
$$

Flash Attention:
$$
\begin{aligned}
T_r &= 4096 / 128 = 32\ \text{Q blocks} \\
T_c &= 4096 / 128 = 32\ \text{KV blocks} \\
\\
\text{每个 Q block: } & \\
\quad \text{读 } Q_i &: 128 \times 64 \times 2 = 16\ \text{KB} \\
\quad \text{读 KV (全部)} &: 2 \times 4096 \times 64 \times 2 = 1.0\ \text{MB} \\
\quad \text{写 } O_i &: 128 \times 64 \times 2 = 16\ \text{KB} \\
\quad &\approx 1.03\ \text{MB per Q block} \\
\\
\text{总 HBM 访问 } &\approx 32 \times 1.03 = 33.0\ \text{MB (for all heads combined)}
\end{aligned}
$$

相比于标准的 67 MB per head, flash attention 用 ~1 MB per head
节省 ~67× (当 heads=32 时: 67×32 = 2.1 GB → 33 MB，~65× 减少)

---

## 6. 参考资料

- [FlashAttention (Dao et al., NeurIPS 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [Online Normalizer Calculation (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
- [Rabe & Staats (2021) — Self-Attention Does Not Need O(N²) Memory](https://arxiv.org/abs/2112.05682)
