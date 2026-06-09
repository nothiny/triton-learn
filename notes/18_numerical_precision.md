# 18 — 数值精度选择指南：fp32, fp16, bf16, tf32, fp8

> 选择正确的数据类型是 GPU kernel 性能优化的第一步。选错了 → 精度不够或性能浪费。这篇整理各种 dtype 的特性、适用场景和常见坑。

---

## 1. 各数据类型的精度对比

### 1.1 位宽和表示范围

```
fp32 (float32):  1 sign + 8 exponent + 23 mantissa = 32 bits
  范围: ±3.4×10³⁸, 精度: ~7 位十进制
  用于: 模型权重、优化器状态、需要高精度的场景

fp16 (float16):  1 sign + 5 exponent + 10 mantissa = 16 bits
  范围: ±65504, 精度: ~3.3 位十进制
  用于: GEMM 输入、activation（吞吐翻倍 vs fp32）

bf16 (bfloat16): 1 sign + 8 exponent + 7 mantissa = 16 bits
  范围: ±3.4×10³⁸ (同 fp32!), 精度: ~2 位十进制
  用于: 训练（范围大，不会溢出；精度低但训练可容忍）

tf32:            1 sign + 8 exponent(FROM fp32) + 10 mantissa(FROM fp16)
  范围: 同 fp32, 精度: ~3.3 位十进制
  用于: A100+ Tensor Core 的默认 fp32 模式
  无需代码改动 — 硬件自动转换
```

### 1.2 可视化

```
数值范围对比:

fp32:  [---可以表达极小的数 (2^-126) -----------可以表达极大的数 (3.4×10^38)---]
bf16:  [---同 fp32 的范围————————————————————————————精度低 2 位——————————————]
fp16:  [---范围小 (max 65504)---但精度比 bf16 高——————————  ]
tf32:  [---同 fp32 的范围 (8-bit exp)——精度同 fp16 (10-bit mant)————]

精度对比 (能区分的最小差值):
fp32:  1.0000000 和 1.0000001 → 能区分
fp16:  1.000 和 1.001 → 能区分
bf16:  1.00 和 1.01 → 能区分 (只有 2 位十进制精度！)
```

---

## 2. 什么时候用哪个？

### 2.1 快速决策表

| 场景 | 推荐 dtype | 原因 |
|------|-----------|------|
| GEMM 输入 (前向/反向) | fp16 / bf16 | Tensor Core 2× 吞吐 |
| GEMM 累加器 | fp32 | 避免累加精度损失 |
| Elementwise (激活函数) | fp16 / bf16 | 通常精度足够 |
| Softmax | fp32 中间, fp16 输入 | softmax 数值敏感 |
| LayerNorm / RMSNorm | fp32 中间 | 需要高精度统计量 |
| 模型权重 | fp32 (master copy) | 训练需要高精度 |
| 优化器状态 | fp32 | Adam 的 m, v 需要精度 |
| Loss | fp32 | 很小的值需要高精度 |
| LLM 推理 | fp8 (H100) 或 int8 | 精度损失可接受，速度优先 |

### 2.2 为什么 GEMM 用 fp16 输入但 fp32 累加？

```
GEMM: C = A @ B  (所有矩阵 fp16)

如果全程 fp16:
  C[0,0] = Σ A[0,k] * B[k,0]
  
  问题: 当 K 很大时（如 K=4096），累加 4096 个 fp16 值
  fp16 只有 ~3 位十进制精度
  → 大量小值的累加会被截断
  → 最终结果损失 1-2 位有效数字

解决: 输入/输出 fp16，累加器 fp32
  acc = Σ A[0,k] * B[k,0]  ← 每次乘加在 fp32 中进行
  C[0,0] = fp16(acc)       ← 只在最后截断一次

  fp32 有 ~7 位十进制精度
  → 累积 K=4096 次不会有可感知的精度损失
```

---

## 3. fp16 vs bf16 — 训练和推理的选择

### 3.1 比较

```
fp16:
  优点: 精度比 bf16 高（3.3 vs 2 位十进制）
  缺点: 范围小（max 65504）→ 容易溢出！
       需要 gradient scaling 来防止梯度下溢

bf16:
  优点: 范围同 fp32（不会溢出！）
        不需要 gradient scaling
  缺点: 精度低（2 位十进制）→ 对某些操作可能不够
       不是所有硬件都支持（V100 不支持）

实际使用:
  训练: bf16 越来越流行（A100+, H100 支持）
        省去了 gradient scaling 的复杂性和调参
  推理: fp16 通常足够（范围不会成为问题）
```

### 3.2 Gradient Scaling（fp16 训练必需）

```python
# 为什么 fp16 训练需要 gradient scaling？

# 问题: fp16 的最小正数是 ~6×10^-8
# 很多梯度比这小 → 变成 0 → 梯度下溢 → 训练停滞

# 解决: 先放大 loss → 计算梯度 → 缩小回来
loss_scale = 1024.0  # 典型的 scale factor
scaled_loss = loss * loss_scale
scaled_loss.backward()  # 梯度也放大了 1024×

# 更新权重前缩放回来:
for param in model.parameters():
    param.grad /= loss_scale

# PyTorch 自动做这个:
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()  # 动态调整 scale factor
```

---

## 4. tf32 — "默认可用的" 提速

### 4.1 什么是 tf32？

```
A100 引入的 TensorFloat-32:
  当你在 A100 上用 torch.matmul(fp32_a, fp32_b):
  A100 自动在内部用 tf32 做 MMA:

  1. 读取 fp32 输入
  2. 截断 mantissa 到 10 bits（丢掉 13 bits 精度）
  3. 用 Tensor Core 做 MMA（跟 fp16 一样快的硬件路径！）
  4. 结果累加到 fp32

  结果: 8× faster than fp32 CUDA Core, without code change
  精度: 介于 fp16 和 fp32 之间（约 3-4 位十进制）
```

### 4.2 控制和检查

```python
# 检查 tf32 是否启用
print(torch.backends.cuda.matmul.allow_tf32)  # True (default on A100+)

# 如果精度不够，可以关掉
torch.backends.cuda.matmul.allow_tf32 = False
# → 回到纯 fp32（慢 ~8×）

# 对某些需要高精度的操作关 tf32:
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, 
    enable_math=False,      
    enable_mem_efficient=True
):
    # SDPA 不用 tf32
    output = F.scaled_dot_product_attention(q, k, v)
```

---

## 5. FP8 — Hopper 的新武器

### 5.1 两种格式的用途

```
E4M3 (4 exponent, 3 mantissa):
  范围: ±448
  精度: ~1 位十进制
  用于: 前向（activation 和 weight 通常在 ±448 范围内）

E5M2 (5 exponent, 2 mantissa):
  范围: ±57344
  精度: ~0.6 位十进制
  用于: 反向（梯度范围更大但精度要求低）
```

### 5.2 Block-wise Scaling

```
直接用 fp8 精度不够？— 加 scaling:

原始 (fp16):
  [0.01, 0.02, 0.03, 100.0, 200.0, 300.0]  ← 动态范围大

直接量化到 fp8:
  [0.01, 0.02, 0.03, 100, 200, 300]  → 前面 3 个被截断为 0！

Block-wise scaling:
  分成两个 block:
  Block 0: [0.01, 0.02, 0.03] → scale=256 → [2.56, 5.12, 7.68]
  Block 1: [100, 200, 300]    → scale=1   → [100, 200, 300]
  
  每个 block 独立 scale → 小值不被截断！

  Triton 3.x 支持: tl.dot 接受 scale_a, scale_b 参数
```

---

## 6. 精度 vs 性能的 Tradeoff

### 6.1 不同精度在 H100 上的吞吐

```
数据搬运:
  fp16/bf16: 2 bytes → 带宽占用少
  fp32:      4 bytes → 2× 带宽占用

计算 (Tensor Core):
  fp16/bf16: m16n8k16 per cycle → 2048 ops/cycle
  tf32:      m16n8k8  per cycle → 1024 ops/cycle
  fp8:       m16n8k32 per cycle → 4096 ops/cycle

相对吞吐 (以 fp16=1.0 为基准):
  fp8:  2.0× (2× more ops per cycle)
  fp16: 1.0×
  bf16: 1.0×
  tf32: 0.5× (half the ops per cycle, but 8× faster than fp32 CUDA)
  fp32: 0.0625× (on CUDA Core — no Tensor Core for fp32)
```

### 6.2 精度损失的量化

```
以 GEMM (M=N=K=4096) 为例:

fp32 reference:      1.000000 (baseline)
tf32:                0.99995  (relative error ~5e-5)
fp16 accumulation:   0.9998   (relative error ~2e-4)
fp8 (w/ scaling):    0.997    (relative error ~3e-3)
int8:                0.995    (relative error ~5e-3)

对于训练: bf16 + fp32 accumulation → 通常和 fp32 的收敛行为相同
对于推理: fp16 → 通常足够
对于量化推理: fp8/int8 → 有轻微质量损失, 但 2-4× 速度提升
```

---

## 7. Triton 中的实际使用

### 7.1 声明和使用

```python
# Triton kernel 中输入类型由 torch tensor 的 dtype 决定

# fp16 kernel
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = matmul_kernel(a, b)  # 自动处理 fp16

# 累加器始终用 fp32
@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, ...):
    # 输入 a, b 是 fp16（由 tensor dtype 决定）
    a = tl.load(a_ptr + ...)  # fp16
    b = tl.load(b_ptr + ...)  # fp16
    
    # 累加器显式用 fp32
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    acc += tl.dot(a, b)  # tl.dot 自动在 fp32 中累加
    
    # 输出自动转为输入类型（fp16）
    tl.store(c_ptr + ..., acc)

# 如果输入是 fp32, tl.dot 会尝试用 tf32 (A100+) 或回退到 CUDA Core
```

### 7.2 混精度实践

```python
# 实战中常见的混精度模式:

# 模式 1: GEMM 用 fp16, 其他用 fp32
weights = torch.randn(..., dtype=torch.float32)  # 权重保持 fp32
x = x.half()  # 输入转 fp16

output = fused_linear(x, weights.half())  # GEMM 用 fp16
output = output.float()  # 结果转回 fp32

# 模式 2: 全部 fp16，关键路径用 fp32 累加
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = model(input)  # 自动转 fp16
    
# 模式 3: 推理时量化到 fp8
# 需要在量化框架如 TensorRT, vLLM 中处理
# Triton 目前对 fp8 支持仍在完善
```

---

## 8. 总结：选择 dtype 的决策树

```
你需要什么？

1. 纯计算（GEMM, conv）
   → fp16/bf16 (2× speed, Tensor Core)
   → 累加器始终 fp32

2. 需要大范围（训练）
   → bf16 (范围同 fp32, 无需 gradient scaling)
   → A100+ 推荐

3. 需要高精度（权重、优化器、loss）
   → fp32

4. 推理加速（H100）
   → fp8 (2× vs fp16, 需要 block-wise scaling)

5. 简单提速（A100+，不用改代码）
   → 确保 tf32 开启（默认就开着）

Golden rule:
  内存密集型 kernel: 优先用 fp16/bf16 (减少 2× 带宽)
  计算密集型 kernel: 优先用 fp16/bf16 (Tensor Core 2× 吞吐)
  精度敏感的 kernel: 用 fp32 (或 tf32)
```

---

## 参考资料

- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [TF32 Technical Overview](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
