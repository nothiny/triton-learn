# 16 — Autotuning 策略指南：从"随便试"到"科学搜索"

> Autotune 不是"多加点 config 就更好"。搜索空间太大 → 编译时间长、cache 爆炸；太小 → 找不到最优。这篇讲如何设计高效的 autotune 策略。

---

## 1. Autotune 的工作原理

### 1.1 一个完整的 autotune 周期

```
@triton.autotune(configs=[...], key=['M', 'N', 'K'])
@triton.jit
def my_kernel(...):
    ...

第一次调用 (M=1024, N=1024, K=1024):
  1. 检查 cache: ~/.triton/cache/ 中是否有 M=1024,N=1024,K=1024 的结果？
     → 没有 → 需要 autotune
  
  2. 对每个 config 生成编译版本（JIT compile）
     → 20 个 configs × ~200ms 编译时间 = ~4 秒
  
  3. 对每个 config 在 GPU 上实际运行 1-2 次
     → 20 个 configs × ~5ms = ~100ms
  
  4. 选最快的 config → 缓存到 ~/.triton/cache/
  
  5. 用最优 config 运行真正的 kernel

后续调用 (M=1024, N=1024, K=1024):
  1. 检查 cache → hit → 直接用缓存的 config → 0 开销！
```

### 1.2 Autotune 的开销

```
首次运行开销:
  JIT 编译: 每个 config ~100-500ms (取决于 kernel 复杂度)
  Benchmark: 每个 config ~2-10ms
  总: config 数 × (编译+bench)

Cache 存储:
  每个 config × 每个 key 组合 = 一个编译版本
  20 configs × 3 key values = 最多 60 个编译文件
  → 可能占用几十到几百 MB

建议:
  - 开发时用少量 config (4-6 个) 快速迭代
  - 最终优化时用更多 config (15-30 个)
  - 定期清理 cache: rm -rf ~/.triton/cache/
```

---

## 2. 搜索空间设计

### 2.1 BLOCK 尺寸的选择原则

```python
# ❌ 太随意的搜索
configs = [
    triton.Config({'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k}, ...)
    for m in [32, 64, 96, 128, 160, 192, 224, 256]  # 非 2 的幂浪费了
    for n in [32, 64, 96, 128, 160, 192, 224, 256]
    for k in [16, 32, 48, 64, 80, 96, 112, 128]
]
# 8×8×8=512 个 configs → 太多！编译几十分钟

# ✅ 基于硬件约束的 smart search
configs = [
    # BLOCK_M: 16 的倍数（MMA M=16）
    # BLOCK_N: 8 的倍数（MMA N=8）
    # BLOCK_K: 16 的倍数（MMA K=16），通常 32 或 64 就够了
    triton.Config({'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k}, ...)
    for m in [64, 128, 256]              # 3 options
    for n in [64, 128, 256]              # 3 options
    for k in [32, 64]                    # 2 options
]
# 3×3×2=18 个 configs ← 合理
```

### 2.2 硬件约束速查

```
Block 大小约束 (Ampere/Hopper):
  - BLOCK_M % 16 == 0 (MMA M dimension)
  - BLOCK_N % 8 == 0 (MMA N dimension)
  - BLOCK_K % 16 == 0 (MMA K dimension)
  - BLOCK_M × BLOCK_N × thread_size ≤ max threads per block (2048 in Triton)
  - BLOCK_M × BLOCK_K × dtype_size × num_stages ≤ shared memory (164-228 KB)
  
num_warps 约束:
  - num_warps 必须是 2 的幂: 4, 8, 16, 32
  - num_warps × 32 × registers_per_thread ≤ 65536 (register file)
  - 一般 4-8 就够，超过 8 很少有用

num_stages 约束:
  - num_stages × tile_size × dtype_size ≤ shared memory
  - 通常 2-3 最优，很少有理由用 4+
```

### 2.3 Pruning 策略

```python
# 生成 configs 时过滤掉不合法的组合
configs = []

for m in [64, 128, 256]:
    for n in [64, 128, 256]:
        for k in [32, 64]:
            for num_warps in [4, 8]:
                for num_stages in [2, 3, 4]:
                    # Prune 规则:
                    
                    # 1. num_warps=4 时不要太多 stages
                    if num_warps == 4 and num_stages > 2:
                        continue
                    
                    # 2. 大 tile + 少 warps → 每个 warp 做太多 → 慢
                    if num_warps == 4 and m * n > 128 * 128:
                        continue
                    
                    # 3. Shared memory 检查
                    # 2 buffers (num_stages) × tile_size × 2 bytes (fp16)
                    # A: m×k, B: k×n → total = num_stages × (m×k + k×n) × 2
                    sm_usage = num_stages * (m * k + k * n) * 2
                    if sm_usage > 180 * 1024:  # 180 KB (conservative)
                        continue
                    
                    # 4. 总线程数检查
                    if num_warps * 32 > 1024:  # 通常用不到 1024
                        continue
                    
                    configs.append(triton.Config(
                        {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    ))
```

---

## 3. Key 的选择

### 3.1 Key 是什么？

```python
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K'],  # ← 按什么特征选择配置
)

# 含义:
# - 对于不同的 (M,N,K) 组合，Triton 分别找最优 config
# - cache 按 (M,N,K) 的值分组存储
# - 新的大小 → 可能触发新的 autotune

# 常见 key 选择:
# elementwise kernel (只有大小重要):
#   key=['n_elements']
# GEMM (矩阵形状重要):
#   key=['M', 'N', 'K']
# Attention (序列长度重要):
#   key=['N_CTX']
```

### 3.2 Key 的粒度

```python
# 太粗: 所有大小用同一个 config
key = []  # 不推荐 — 大矩阵和小矩阵的最优 config 不同

# 适中: 按主要维度分组
key = ['M']  # OK — 如果 N 和 K 通常和 M 正相关

# 太细: 每个 (M,N,K) 都独立 autotune
key = ['M', 'N', 'K']  # 可能会 trigger 很多次 autotune
# 对于 elementwise: 过分了 — 有很多不同的 M 值
# 对于 GEMM: 合理 — (M,N,K) 的组合有限
```

---

## 4. 性能评估指标选择

### 4.1 默认 vs 自定义

```python
# Triton 默认: 最小化执行时间
# 但你可以提供自定义的 prune 函数:

@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': my_prune_fn,  # 预过滤
    }
)

# prune 函数:
def my_prune_fn(configs, named_args):
    """在编译前就过滤掉明显不好的 config"""
    M = named_args['M']
    pruned = []
    for config in configs:
        # 对于小 M，大 BLOCK_M 不合适（block 太少，GPU 空闲）
        if M < 512 and config.kwargs['BLOCK_M'] > 128:
            continue
        pruned.append(config)
    return pruned
```

### 4.2 自定义性能指标

```python
# 有时你关心的不全是时间 — Triton 允许用 result 的字段过滤

# 默认: 选最快的（time_ms 最小）
# 也可以基于 TFLOPS, bandwidth, 或自定义 metric

# Triton 目前不直接支持自定义 metric 选择
# 但可以用 perf_model 给出初始排序:
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K'],
    # 可以用 use_global_memory=false 强制检查 register/spared memory
)
```

---

## 5. 实战策略

### 5.1 开发阶段（快速迭代）

```python
# 只用 2-4 个"安全"的 config
FAST_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, 
                   num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, 
                   num_warps=8, num_stages=2),
]

@triton.autotune(configs=FAST_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def kernel_dev(...):
    ...
```

### 5.2 优化阶段（找最优）

```python
# 15-30 个 configs，覆盖更多可能
FULL_CONFIGS = []

for m in [64, 128, 256]:
    for n in [64, 128, 256]:
        for k in [32, 64]:
            for w in [4, 8]:
                for s in [2, 3]:
                    if is_valid_config(m, n, k, w, s):
                        FULL_CONFIGS.append(
                            triton.Config(
                                {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k},
                                num_warps=w, num_stages=s,
                            )
                        )

# 同时加 prune 函数减少无意义的测试
```

### 5.3 读取 Autotune 输出

```python
# 运行时设置环境变量看 autotune 过程:
# TRITON_PRINT_AUTOTUNING=1 python my_kernel.py

# 输出:
# autotune: kernel_name, config 1/20: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, 
#           num_warps=4, num_stages=2 → 0.234ms
# autotune: kernel_name, config 2/20: BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, 
#           num_warps=4, num_stages=2 → 0.198ms
# ...
# autotune: kernel_name selected config #7: 0.156ms

# 选中的 config 信息缓存在:
# ~/.triton/cache/<hash>/__triton_kernel_config.json
```

---

## 6. Autotune 的局限性

```
1. 不跨 GPU 型号
   A100 上 autotune 出来的 config 不一定在 H100 上最优
   每个 GPU 型号需要独立的 autotune

2. 输入相关的性能
   config 对于 M=1024 是最优的，但在 M=4096 上可能不是
   这就是为什么 key 很重要

3. 不感知"相邻 kernel"的 cache 影响
   autotune 测试时 kernel 独立运行
   但在实际 pipeline 中，前后的 kernel 会影响 L2 cache 状态

4. 编译时间长
   30+ configs 的第一次运行可能需要 30 秒到 1 分钟编译时间
   对于开发阶段，这是很高的 overhead
```

---

## 7. 决策速查

```
问自己 3 个问题:

1. 我的 kernel 是 memory-bound 还是 compute-bound?
   → memory-bound: 优先调 num_stages（pipeline 隐藏延迟）
   → compute-bound: 优先调 BLOCK 大小（Tensor Core 利用率）

2. 输入大小变化大吗？
   → 大: 用多个 key，每个大小范围独立 autotune
   → 小: 一个 key 就够

3. 开发速度 vs 性能哪个重要？
   → 开发: 4 个 configs → 秒级反馈
   → 优化: 20 个 configs → 分钟级编译
   → 极致: 50+ configs + 手动微调 → 小时级
```

---

## 参考资料

- [Triton Autotuning Guide](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)
- [Triton Config API](https://triton-lang.org/main/python-api/triton.html#triton.Config)
