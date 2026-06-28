# 22 — 生产级 Triton Kernel 优化清单

> **目标**: 一份系统化的 checklist，覆盖从"能跑"到"生产可用"的每个维度——代码、内存、计算、调度、数值、调优、测试、profiling。
> **前置**: 笔记 01-02（编程模型、内存层级）、笔记 19（Block Pointer API）、至少写过 3 个 kernel

---

## 0. 总览: 生产级 Kernel 的 8 个维度

```
                    生产级 Triton Kernel 检查清单

  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ 1. 代码   │  │ 2. 内存   │  │ 3. 计算   │  │ 4. 调度   │
  │ 模式      │  │ 层级      │  │ 单元      │  │ 并发      │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ 5. 数值   │  │ 6. 调优   │  │ 7. 测试   │  │ 8. Profile│
  │ 精度      │  │ 策略      │  │ 验证      │  │ 剖析      │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## 1. 代码模式 — 用对 API

### ✅ Checklist

- [ ] **用 `tl.make_block_ptr` 替代手工指针拼接**（所有规整的 tile 访问）
- [ ] **用 `boundary_check` 替代手工 `mask=..., other=0.0`**
- [ ] **用 `tl.advance` 替代手工 K 偏移更新**
- [ ] **`order` 参数匹配内存布局**（row-major → `order=(1,0)` 或 `(ndim-1, ..., 0)`）
- [ ] **strides 用元素数而非 bytes 数**（Triton 的 strides 是元素数）
- [ ] **`@triton.autotune` 的 `key` 用 shape 参数，不用 block size**
- [ ] **grid 用 `lambda meta: (...)` 传 `meta["BLOCK_SIZE"]`**
- [ ] **kernel 签名清晰**: 先 raw pointers → shape ints → strides → block sizes (constexpr)

### ❌ 常见反模式

```python
# ❌ 还在用手工指针拼接（新代码不应出现）
offs_m = pid_m * BM + tl.arange(0, BM)
a_ptrs = a_ptr + offs_m[:, None] * stride_am + ...
a = tl.load(a_ptrs, mask=..., other=0.0)

# ✅ 正确
p_a = tl.make_block_ptr(...)
a = tl.load(p_a, boundary_check=(0, 1))

# ❌ autotune key 包含 BLOCK_SIZE（cache miss，重复编译）
@triton.autotune(configs=[...], key=["M", "N", "K", "BLOCK_SIZE"])

# ✅ 正确
@triton.autotune(configs=[...], key=["M", "N", "K"])

# ❌ 硬编码 grid（autotune 无法调 BLOCK_SIZE）
grid = (triton.cdiv(M, 128), triton.cdiv(N, 64))

# ✅ 正确
grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
```

---

## 2. 内存层级 — 最大化带宽利用

### 2.1 Global Memory (HBM)

- [ ] **Coalesced access**: 同一 warp 内相邻线程访问相邻地址 → `order` 正确
- [ ] **128-byte alignment**: `base` 指针对齐到 128 bytes（PyTorch tensor 默认满足）
- [ ] **Minimize HBM round-trips**: 能 fuse 就 fuse（如 RMSNorm + Residual）
- [ ] **Dtype 选择**: 能 fp16/bf16 就不用 fp32（减半 HBM 流量）

```python
# Coalescing 自查:
# Q: 相邻 thread 的地址差是多少？
# A: 应该是 1 个元素（即 thread 沿 stride=1 的维度分布）
# → order=(ndim-1, ..., 0) for row-major PyTorch tensors
```

### 2.2 Shared Memory

- [ ] **`num_stages >= 2`**: 启用 double buffering / software pipelining
- [ ] **Shared memory 用量 < 228 KB/SM**（否则 occupancy 暴跌）
- [ ] **Bank conflict awareness**: 虽然 Triton 编译器自动 swizzle，但了解访问模式

```python
# 计算 shared memory 用量
# 例: 2-stage double buffering, 2 个 input tile + 1 个 acc
# A tile: BM×BK×sizeof(fp16) = 128×64×2 = 16 KB
# B tile: BK×BN×sizeof(fp16) = 64×128×2 = 16 KB
# 双缓冲 ×2 = 64 KB
# Acc: BM×BN×sizeof(fp32) = 128×128×4 = 64 KB
# 总计: 128 KB → 在 H100 的 228 KB 之内 ✅
```

### 2.3 TMA (Hopper)

- [ ] **`order` 按 stride 递增排序**: TMA 映射要求
- [ ] **block_shape 各维是 16 bytes 的倍数**: fp16→8 的倍数，fp32→4 的倍数
- [ ] **考虑启用 `TRITON_ENABLE_TMA=1`**: 在 H100 上试

---

## 3. 计算单元 — 榨干 Tensor Core

### ✅ Checklist

- [ ] **用 `tl.dot` 而不是手写乘法累加**: `tl.dot` 映射到 MMA 指令（Tensor Core）
- [ ] **accumulator 用 `tl.float32`**: 即使输入是 fp16，acc 用 fp32 防精度损失
- [ ] **MMA tile 大小匹配 Tensor Core 规格**: M16N16K16 (fp16) 或 M16N8K32 (fp8)
- [ ] **`block_shape` 是 MMA tile 的整数倍**: 如 fp16 → BM/BN/BK 都是 16 的倍数

```python
# ❌ 手写 reduce（用不到 Tensor Core）
acc = 0
for k in range(K):
    acc += a[k] * b[k]

# ✅ tl.dot（映射到 MMA m16n16k16）
acc = tl.zeros([BM, BN], dtype=tl.float32)  # acc 用 fp32
for k in range(0, K, BK):
    a = tl.load(p_a, boundary_check=(0, 1))
    b = tl.load(p_b, boundary_check=(0, 1))
    acc += tl.dot(a, b)  # → PTX: mma.sync.aligned.m16n16k16...
```

### 3.1 Tensor Core 利用率自查

```
利用率 = 实际 TFLOPS / GPU Peak TFLOPS

H100 (fp16): peak ~990 TFLOPS
  利用率 > 80%: ✅ 优秀
  利用率 60-80%: ⚠️ 还行，还有优化空间
  利用率 < 60%: ❌ 需要检查: 是否 memory-bound？block size 太小？

检查方法:
  python -c "
  from utils.roofline import roofline_analysis
  roofline_analysis(flops=2*M*N*K, bytes=..., time_ms=...)
  "
```

---

## 4. 调度并发 — 喂饱 SM

### 4.1 Occupancy 基础

```
Occupancy = 每个 SM 上活跃的 warp 数 / 理论最大 warp 数

影响因素:
  1. 每个 thread 的寄存器数（register pressure）
  2. 每个 block 的 shared memory 用量
  3. block size（threads per CTA）

寄存器压力是最常见的 occupancy 杀手:
  - 过多局部变量 → 寄存器溢出 (spill to HBM) → 性能暴跌
  - 每个 SM 的寄存器池有限 (H100: 65536 寄存器/SM)
  - 寄存器/thread × threads/block × blocks/SM ≤ 65536
```

### 4.2 ✅ Checklist

- [ ] **Grid 充分利用 SM**: `grid 的 CTA 数 > 2 × num_sms`
- [ ] **Block size 合理**: 128-512 threads/block（太小浪费 SM，太大 occupancy 差）
- [ ] **每个 SM 至少 2 个 block**: 一个 block 等 memory 时另一个可以执行
- [ ] **Persistent kernel 场景**: 小 grid → 考虑用 persistent kernel pattern
- [ ] **3D grid 扩展 V 维**: FlashAttention 用 3D grid `(Q_tiles, B×HQ, V_tiles)`

### 4.3 Register Pressure 诊断

```bash
# 用 ncu 看寄存器用量
ncu --set-registers python my_kernel.py

# 看 "Registers Per Thread" metric
# > 255: ❌ 严重溢出（H100 max 255/thread before spilling）
# 200-255: ⚠️ 边界，优化局部变量
# < 200: ✅ 健康
```

### 4.4 减少寄存器压力的技巧

```python
# ❌ 寄存器压力大: 太多临时变量
tmp1 = tl.load(p1); tmp2 = tl.load(p2); tmp3 = tl.load(p3)
result1 = tmp1 * scale1; result2 = tmp2 * scale2
result3 = tmp3 * scale3; ...

# ✅ 减少同时 live 的变量: pipeline 化
acc = 0
for k in range(0, K, BK):
    a = tl.load(p_a, boundary_check=(0, 1))   # 用了就消费
    b = tl.load(p_b, boundary_check=(0, 1))
    acc += tl.dot(a, b)                        # a, b 的生命周期结束
    p_a = tl.advance(p_a, (0, BK))

# ✅ 使用 tl.multiple_of 提示: 帮助编译器优化地址计算
a_ptrs = a_ptr + offs_m[:, None] * stride_am
a_ptrs = tl.multiple_of(a_ptrs, (16,))  # 编译器知道是 16 的倍数 → 更少指令
```

---

## 5. 数值精度 — 做对不只做快

### 5.1 ✅ Checklist

- [ ] **Accumulator 用 fp32**: `acc = tl.zeros([...], dtype=tl.float32)`
- [ ] **Reduction 用 fp32**: 求和/求 max 的中间值用 fp32
- [ ] **Online softmax 的正确实现**: rescale 顺序和公式正确
- [ ] **除法用 `+ eps` 防除零**: `rms = tl.sqrt(sum_sq / N + 1e-6)`
- [ ] **fp8 用缩放因子**: `x_scaled = x * scale`, `y = result / scale`
- [ ] **bf16 优于 fp16 的场景**: 大 reduction（bf16 浮点范围更大，不易溢出）

### 5.2 常见数值陷阱

```python
# ❌ fp16 accumulation → 精度损失
acc = tl.zeros([BM, BN], dtype=tl.float16)  # 每个 dot 的结果截断到 fp16
acc += tl.dot(a, b)

# ✅ fp32 accumulation
acc = tl.zeros([BM, BN], dtype=tl.float32)
acc += tl.dot(a, b)
acc = acc.to(tl.float16)  # 只在最后 store 前转 fp16

# ❌ Online softmax 的 rescale 错误
# 错误: 先 exp 再加
l_i = l_i + tl.sum(tl.exp(b_s - m_new))  # 忘了 rescale 旧值

# ✅ 正确: 先 rescale 旧值
alpha = tl.exp(m_i - m_new)
l_i = l_i * alpha + tl.sum(tl.exp(b_s - m_new))
```

### 5.3 容忍度选择

```python
# 不同场景的推荐容忍度
TOLERANCES = {
    "fp32 elementwise": 1e-5,
    "fp16 elementwise": 1e-3,
    "bf16 GEMM": 1e-2,
    "fp16 GEMM": 5e-2,
    "fp8 GEMM": 5e-1,           # fp8 量化噪声大
    "attention fp16": 5e-2,
    "attention fp32": 1e-4,
    "LayerNorm fp16": 1e-2,     # reduction 累积误差
    "RMSNorm fp16": 1e-2,
    "softmax fp16": 1e-3,
    "atomic_add GEMM": 0.1,     # atomic 的累积误差按 sqrt(K) 增长
}
```

---

## 6. 调优策略 — 让 Autotune 真正有效

### 6.1 ✅ Checklist

- [ ] **Config space 合理**: 6-24 个 config，覆盖关键组合
- [ ] **`num_warps` 在 4, 8 之间扫**: 影响 occupancy 和寄存器分配
- [ ] **`num_stages` 在 2, 3, 4 之间扫**: 更多 stage = 更深的 pipeline = 更多 shared memory
- [ ] **`key` 参数正确**: 只包含决定哪组 config 的最少参数
- [ ] **Prune configs 用 `pre_hook` 或条件 config**: 排除不合理的组合

```python
# ✅ 好的 autotune 设置
@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],  # ← 只包含 shape 参数！
)
# Triton 缓存: 对每组 (M,N,K) 编译所有 config，bench 后选最优
# 下一次遇到同样的 (M,N,K) 直接使用缓存
```

### 6.2 Config Pruning 策略

```python
# 策略 1: pre_hook 排除
def prune_configs(configs, named_args):
    """排除 shared memory 超限的 config"""
    pruned = []
    for cfg in configs:
        sm_required = (
            cfg.kwargs["BM"] * cfg.kwargs["BK"] * 2 +  # A tile
            cfg.kwargs["BK"] * cfg.kwargs["BN"] * 2 +  # B tile
            cfg.kwargs["BM"] * cfg.kwargs["BN"] * 4    # accumulator
        ) * cfg.num_stages
        if sm_required < 228 * 1024:  # H100: 228 KB/SM
            pruned.append(cfg)
    return pruned

# 策略 2: 条件排除
# 小矩阵 → 大 BLOCK 没意义
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=2),
        # ...
    ],
    key=["M", "N", "K"],
    pre_hook=prune_configs,
)
```

---

## 7. 测试验证 — 覆盖边界

### 7.1 ✅ Checklist

- [ ] **正确性: 至少 3 组 shape 测试**（小→中→大）
- [ ] **边界: 不整除的 shape**（如 127, 257, 509 等质数/奇数）
- [ ] **边界: 极值**（M=1, N=1, K=1 或超大）
- [ ] **Dtype: 每种 dtype 都测**（fp16, bf16, fp32）
- [ ] **Strided: 非连续内存**（如 transpose 后的 tensor）
- [ ] **Batch: 至少测 batch=1 和 batch>1**
- [ ] **GQA: 测 G=1 (MHA), G>1 (GQA), G=HQ (MQA)**
- [ ] **Causal: 测 causal=True 和 False**
- [ ] **与至少 2 个 reference 对比**（PyTorch + 另一个框架或手写 ref）

### 7.2 测试模式

```python
# 好的测试: 覆盖 shape 的多种组合
SHAPES = [
    # (M, N, K) 或 (B, H, N, D)
    (1, 1, 64),       # 极小 → 边界测试
    (7, 13, 17),      # 质数 → 不整除任何 tile
    (128, 64, 256),   # 小矩阵
    (256, 128, 512),  # 中矩阵
    (1024, 1024, 1024),  # 大矩阵
    (4096, 4096, 4096),  # 极大
]

# 好的测试: strided input
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
a_strided = a.T.contiguous().T  # 改变 stride

# 好的测试: 与 SotA 对比
# GEMM → cuBLAS (torch.mm)
# Attention → SDPA / flash-attn
# LayerNorm → PyTorch / Liger Kernel
```

---

## 8. Profiling 剖析 — 找到真正的瓶颈

### 8.1 分层 profiling

```
Level 1: 快速诊断 (30s)
  → python -c "from utils.roofline import roofline_analysis; ..."
  → 回答: memory-bound 还是 compute-bound？

Level 2: Triton 层面 (2 min)
  → TRITON_KERNEL_DUMP=1 python my_kernel.py | grep -E "cp.async|mma|tma"
  → 回答: cp.async 是否在用？MMA 指令对不对？

Level 3: ncu 详细分析 (10 min)
  → ncu --set full python my_kernel.py
  → 回答: occupancy, register pressure, memory throughput,
           Tensor Core utilization, L1/L2 hit rate

Level 4: Nsight Systems 时间线 (10 min)
  → nsys profile python my_kernel.py
  → 回答: 哪个 kernel 是瓶颈？launch overhead 多少？
           CPU/GPU 同步点多吗？
```

### 8.2 关键 ncu metrics

```bash
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,        # SM 利用率
  dram__throughput.avg.pct_of_peak_sustained_elapsed,      # HBM 带宽利用率
  l1tex__throughput.avg.pct_of_peak_sustained_elapsed,     # L1/Tex 利用率
  lts__throughput.avg.pct_of_peak_sustained_elapsed,       # L2 利用率
  sm__warps_active.avg.pct_of_peak_sustained_elapsed,      # Occupancy
  sm__inst_executed.avg.pct_of_peak_sustained_elapsed,     # Issue slot 利用率
  smsp__average_warps_issue_stalled_barrier_per_cycle,     # Barrier wait
  smsp__average_warps_issue_stalled_long_scoreboard_per_cycle, # Memory wait
  launc__kernel_duration,                                   # Kernel duration
  python my_kernel.py
```

### 8.3 常见瓶颈 → 解决方案速查

| 症状 | 指标 | 根因 | 解决 |
|------|------|------|------|
| SM 利用率低 (<60%) | `sm__throughput` | Memory stall | 加 `num_stages`; 用 TMA |
| HBM 利用率高 (>80%) | `dram__throughput` | Memory-bound | Fusion; 减少 HBM round-trip; fp16 |
| Occupancy 低 (<50%) | `warps_active` | Register pressure | 减少局部变量; 降低 `num_warps` |
| Barrier stall 高 | `stalled_barrier` | 同步等待多 | 调整 `num_stages`; 调整 block size |
| Memory stall 高 | `stalled_scoreboard` | 等 HBM 数据 | 增加 prefetch 距离; `num_stages=3,4` |
| MMA 利用率低 | `sm__throughput` | Compute under-utilized | 增大 `block_shape`; 确保 tile 是 16 倍数 |

---

## 9. 生产部署前的最终检查

### 9.1 性能

- [ ] 与 cuBLAS/SDPA 的性能差距 < 15%（或说明原因）
- [ ] 小 batch (B=1) 和大 batch (B=64) 都测过
- [ ] 不同序列长度都测过（64, 128, 256, 512, 1024, 2048, 4096）
- [ ] TFLOPS / bandwidth utilization 报告

### 9.2 正确性

- [ ] 边界 shape（质数、1、大数）全部通过
- [ ] Strided input 通过
- [ ] 与至少 2 个 reference 对比通过
- [ ] GQA/MQA 变体通过

### 9.3 工程

- [ ] 有 `# PERFORMANCE NOTES` 文档注释
- [ ] 有 autotune configs（或说明为什么不）
- [ ] 环境检测: `is_cuda()` guard
- [ ] 单元测试在 `tests/` 中

### 9.4 可维护性

- [ ] kernel 签名清晰（参数名有意义，constexpr 标注了 `# [COMPILER]`）
- [ ] grid lambda 用了 `meta`
- [ ] 没有 magic number（block size 都是 constexpr 或 autotune 参数）
- [ ] import 干净（没有 `import *`）

---

## 10. 总结: 优化顺序（先做什么、后做什么）

```
Step 1: 写对（验证正确性）
  → 手工指针 or block_ptr 都可以
  → 测试多种 shape

Step 2: 搬数据（内存优化）
  → 换 block_ptr + boundary_check
  → num_stages >= 2
  → 检查 coalescing

Step 3: 算得快（计算优化）
  → tl.dot → Tensor Core
  → fp32 accumulator
  → block_shape 是 MMA tile 的倍数

Step 4: 喂得饱（调度优化）
  → autotune grid/block sizes
  → 检查 occupancy
  → 必要时 persistent kernel

Step 5: 压得紧（微调）
  → 用 ncu 找瓶颈
  → TRITON_ENABLE_TMA=1
  → 减少寄存器压力
  → autotune 扩大 config space

Step 6: 生产化（工程）
  → 单元测试
  → perf model / roofline
  → 文档
```

**每个 step 之间验证性能——不要跳过 Step 1！**

---

## 参考资料

- 本笔记系列的所有前置笔记（00-18）
- `utils/roofline.py` — Roofline 分析工具
- `utils/profiler.py` — `KernelProfiler` 和 `GPUInfo`
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Triton Performance Tips](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
