# 15 — Multi-GPU 与分布式：NVLink、All-Reduce、Tensor Parallelism

> 单 GPU 性能优化到头后，下一个维度是 multi-GPU。这篇介绍 GPU 间通信的基础和你需要知道的 Triton 相关部分。

---

## 1. GPU 间通信的硬件层

### 1.1 通信方式对比

```
单 GPU 内 (SM ↔ SM):
  通过 L2 Cache / HBM
  带宽: ~10-50 TB/s
  无需显式通信

多 GPU 间:

PCIe (传统):
  GPU 0 ←→ CPU ←→ GPU 1
  带宽: ~32 GB/s (PCIe 4.0 ×16)
  延迟: ~10-20 μs
  
NVLink (NVIDIA 专有):
  GPU 0 ←→ GPU 1 (直连)
  带宽: 900 GB/s (A100 NVLink 3.0) / 450 GB/s (H100 NVLink 4.0) per direction
  延迟: ~1-3 μs
  
NVSwitch (更多 GPU):
  GPU 0 ↔ NVSwitch ↔ GPU 7
  8 个 A100/H100 全互联
  带宽: 同 NVLink
```

### 1.2 带宽数量级比较

```
HBM (GPU 内):         3.35 TB/s   ← baseline
NVLink 4.0 (GPU 间):  0.45 TB/s   ← ~7.4× 慢于 HBM
PCIe 4.0 (GPU 间):    0.032 TB/s  ← ~100× 慢于 HBM
InfiniBand (节点间):   0.025 TB/s  ← ~130× 慢于 HBM
```

**核心启示**: GPU 间通信比 GPU 内通信慢 10-100×。你的 kernel 设计需要考虑通信开销。

---

## 2. 常用通信原语

### 2.1 All-Reduce

```
All-Reduce: 所有 GPU 贡献数据，每个 GPU 得到相同的聚合结果

例: 3 个 GPU 各有一个 tensor:
  GPU 0: [1, 2, 3]
  GPU 1: [4, 5, 6]
  GPU 2: [7, 8, 9]

  All-Reduce(sum) → 每个 GPU 得到 [12, 15, 18]

用于: 数据并行训练中的梯度同步
```

### 2.2 All-Gather

```
All-Gather: 每个 GPU 把自己的数据发给所有 GPU

  GPU 0: [1, 2]  →  所有 GPU 得到 [[1,2], [3,4], [5,6]]
  GPU 1: [3, 4]
  GPU 2: [5, 6]

用于: 张量并行的前向 pass（收集各 GPU 的部分输出）
```

### 2.3 Reduce-Scatter

```
Reduce-Scatter: All-Reduce 的优化——先 reduce 再 scatter

  GPU 0: [1, 2, 3]     Reduce →  [12, 15, 18]
  GPU 1: [4, 5, 6]  ─────────→  Scatter → GPU 0: [12], GPU 1: [15], GPU 2: [18]
  GPU 2: [7, 8, 9]

用于: 反向传播的梯度同步（比 All-Reduce 更高效）
```

---

## 3. Tensor Parallelism（张量并行）

### 3.1 概念

```
问题: 一个 LLM 的权重矩阵太大，放不进一张 GPU。

解决方案: 把矩阵切成多份，放在不同的 GPU 上。

切分方式:

列并行 (Column Parallel):
  原始: Y = X @ W                          (X: [B, D_in], W: [D_in, D_out])
  分割: W = [W₁ | W₂]                       (W₁: [D_in, D_out/2], W₂: [D_in, D_out/2])
  GPU 0: Y₁ = X @ W₁                        (Y₁: [B, D_out/2])
  GPU 1: Y₂ = X @ W₂                        (Y₂: [B, D_out/2])
  最后: All-Gather 拼回 Y = [Y₁ | Y₂]

行并行 (Row Parallel):
  原始: Z = Y @ V                          (Y: [B, D_in], V: [D_in, D_out])
  分割: Y = [Y₁ | Y₂]                       (列并行的输出)
  GPU 0: Z₁ = Y₁ @ V₁                       (V₁: [D_in/2, D_out])
  GPU 1: Z₂ = Y₂ @ V₂                       (V₂: [D_in/2, D_out])
  最后: All-Reduce Z = Z₁ + Z₂
```

### 3.2 在 Triton 中写 Tensor Parallel 的 kernel

```python
# Triton kernel 本身不关心数据在哪个 GPU 上
# Tensor parallelism 由 PyTorch 的分布式框架管理

# 你的 Triton kernel 还是处理本 GPU 上的数据:
@triton.jit
def fused_linear_activation(x_ptr, w_ptr, out_ptr, ...):
    # w_ptr 指向本 GPU 上的 W 的一部分 (列并行)
    # 这与单 GPU 完全相同 — Triton 不需要知道分布式
    ...

# 分布式部分由 PyTorch Distributed 处理:
from torch.distributed import all_gather, all_reduce

def column_parallel_linear(x, w_shard):
    # Step 1: Triton kernel on local shard
    local_out = fused_linear(x, w_shard)
    
    # Step 2: All-Gather to collect other GPUs' outputs
    full_out = all_gather(local_out)  # PyTorch distributed
    
    return full_out
```

---

## 4. 通信-计算重叠

### 4.1 问题

```
串行执行:
  GPU 0: [Compute]------[Send]------[Compute]------[Send]------
  GPU 1: [Compute]------[Recv]------[Compute]------[Recv]------
  
  问题: 通信时计算单元空闲（反之亦然）

重叠执行:
  GPU 0: [Compute][Send(prev)][Compute][Send(prev)]
  GPU 1: [Compute]...[Recv]......[Compute]...[Recv]
  
  让通信和计算尽可能同时发生
```

### 4.2 在 Triton 中的实现

```python
# Triton kernel 本身不处理通信重叠
# 但可以通过 CUDA streams 手动管理:

comm_stream = torch.cuda.Stream()
comp_stream = torch.cuda.Stream()

with torch.cuda.stream(comp_stream):
    # 启动 Triton kernel 在当前 shard 上
    output = triton_kernel(x, w_local)
    
with torch.cuda.stream(comm_stream):
    # 同时异步发送上一轮的结果
    dist.all_reduce(prev_output, async_op=True)

# 两个 stream 重叠执行
torch.cuda.synchronize()
```

---

## 5. 实践建议

### 5.1 优化优先级

```
1. 先优化单 GPU kernel（最重要！）
   一个慢的 kernel × 8 GPU = 8 个慢的 kernel
   先做到 70%+ cuBLAS 再考虑多 GPU

2. 减少通信量
   - 用更小的 dtype（fp16 → 通信量减半）
   - 用 gradient accumulation 减少同步频率

3. 重叠通信和计算
   - 用 CUDA stream 让发送/接收和计算重叠

4. 选择正确的并行策略
   - 小模型 → Data Parallel（DP）
   - 大模型（单 GPU 放不下）→ Tensor Parallel（TP）
   - 超大模型 → TP + PP（Pipeline Parallel）+ DP
```

### 5.2 常见模式

```
模式 1: Data Parallel + All-Reduce
  每个 GPU 有完整模型副本
  前向 pass: 独立计算
  反向 pass: 独立计算梯度 → All-Reduce 平均梯度
  适用: 模型能放入单 GPU

模式 2: Tensor Parallel + All-Gather/Reduce-Scatter
  模型权重分布在多个 GPU 上
  前向: 各 GPU 计算自己的 shard → All-Gather
  反向: 各 GPU 计算梯度 → Reduce-Scatter
  适用: 模型太大放不进单 GPU

模式 3: Pipeline Parallel
  模型按层拆分到不同 GPU
  GPU 0 负责 layers 0-7, GPU 1 负责 layers 8-15, ...
  适用: 层数很多的超大型模型
```

---

## 6. 与 Triton 的关系

```
Triton 本身是 single-GPU kernel 语言。
Multi-GPU 部分由 PyTorch Distributed / NCCL 管理。

Triton 的作用:
  ✅ 优化单 GPU 上的计算（GEMM, attention, fusion）
  ✅ 减少通信前的计算时间（让通信开始得更早）
  ✅ 融合通信后的操作

你需要知道:
  当设计 multi-GPU 的训练/推理时:
  1. Triton kernel 负责"单 GPU 内的计算"
  2. NCCL 负责"GPU 间的通信"
  3. PyTorch Distributed 负责"前端接口"
```

---

## 参考资料

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
