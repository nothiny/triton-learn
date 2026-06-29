# 09 — Benchmarking Methodology: 如何正确测量 GPU Kernel 性能

> ⚠️ GPU kernel 的 benchmark 比 CPU 复杂得多，有很多容易踩的坑。这篇笔记整理了正确做法和常见误判。

## TL;DR Cheat Sheet

    新手最常见的 5 个错误:
    
    ❌ 用 time.time() 计时
    ✅ 用 torch.cuda.Event
    
    ❌ 不 warmup 直接测
    ✅ 至少 10-25 次 warmup 后再测
    
    ❌ 只测一个尺寸就下结论
    ✅ 扫描多个规模，看趋势
    
    ❌ 不看数值正确性就 bench
    ✅ 先 assert max_diff < 1e-3，再 bench
    
    ❌ 只报 TFLOPS，不报 % of ceiling
    ✅ 两个都报：TFLOPS + "% of ceiling" 或 "% of cuBLAS"
    
    正确的 benchmark 流程:
      make check-gpu → warmup → 测正确性 → CUDA Event 计时 → 报告 median + % ceiling

---

## 1. 为什么不能用 `time.time()`

```python
# ❌ 错误做法
import time
t0 = time.time()
my_kernel(a, b)
t1 = time.time()
print(f"{t1 - t0:.4f} seconds")
```

GPU kernel launch 是**异步**的。`my_kernel(a, b)` 只是把任务提交到 GPU 的 command queue，不等 GPU 执行完就返回了。`time.time()` 测到的是 CPU 侧的 submit 时间（通常 <1μs），不是 GPU 执行时间。

```python
# ✅ 正确做法: 用 torch.cuda.Event 做 GPU 侧计时
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
my_kernel(a, b)
end.record()
torch.cuda.synchronize()  # 等 GPU 完成

elapsed_ms = start.elapsed_time(end)
```

`torch.cuda.Event` 在 GPU 的 command stream 中插入时间戳，`elapsed_time()` 返回两个时间戳之间的 GPU 时钟周期换算结果。这才是 GPU 执行时间。

> 🔧 **编译器类比**: `torch.cuda.Event` 类似在 IR 中插入 timing intrinsic（如 x86 的 `RDTSC`），而不是在 host 侧计时。

---

## 2. Warmup 的必要性

GPU kernel 第一次运行时会触发：

1. **JIT 编译**：Triton 需要在第一次调用时编译 kernel（Python AST → TTIR → TTGIR → LLVM IR → PTX → CUBIN），这需要几十到几百毫秒
2. **CUDA context 初始化**：第一次 CUDA 调用初始化 driver context
3. **Kernel cache 写入**：编译结果写入 `~/.triton/cache/`
4. **GPU clock boost**：GPU 从 idle clock 升到 boost clock 需要 ~100μs

因此至少需要 **10-25 次 warmup** 迭代（不纳入测量）。

```python
# Warmup
for _ in range(25):
    my_kernel(a, b)
torch.cuda.synchronize()

# Now benchmark
times = []
for _ in range(100):
    start.record()
    my_kernel(a, b)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
```

> 🔧 **编译器类比**: warmup 类似 JIT warmup（JVM、V8）。第一次运行触发 compilation pipeline，后续运行命中 cache。

---

## 3. Median vs Mean

GPU kernel 的执行时间不是正态分布的，通常有右偏长尾，原因：

- **Thermal throttling**：温度升高后 GPU 降频
- **Scheduling jitter**：OS / driver 层面的短暂干扰
- **DRAM refresh**：HBM 的周期性刷新
- **Power management**：GPU boost clock 的动态调整

因此 **median 比 mean 更可靠**。用 mean 会被偶发的长尾拖高，高估真实的稳态性能。

```python
import statistics
median_ms = statistics.median(times)   # 更鲁棒
mean_ms = statistics.mean(times)       # 会被 outlier 影响
std_ms = statistics.stdev(times)       # 看变异程度

# 好的 benchmark: std/median < 2%
# 差的 benchmark: std/median > 10% → 可能有 clock 波动或 throttling
```

常见做法：报告 median，附带 std 作为稳定性指标。

---

## 4. Roofline 模型：判断你的 kernel 是 compute-bound 还是 memory-bound

### 4.1 核心概念

Roofline 模型用一个二维图告诉你 kernel 的性能瓶颈在哪里：

- **X 轴**：Arithmetic Intensity（算术强度）= FLOPs / Bytes（每个字节数据上做多少次浮点运算）
- **Y 轴**：Achievable TFLOPS（实际达到的算力）
- **Ceiling**（天花板）：一条水平线 = GPU 峰值 TFLOPS，一条斜线 = 峰值带宽 × 算术强度
- **Ridge Point**（转折点）= 峰值 TFLOPS / 峰值带宽：算术强度超过这个值 → compute-bound，否则 memory-bound

### 4.2 具体计算

```python
# 以 GEMM (M=4096, N=4096, K=4096, fp16) 为例
M, N, K = 4096, 4096, 4096
dtype_bytes = 2  # fp16

# 理论 FLOPs: 每个 C 元素需要 K 次乘加 = 2K FLOPs
flops = 2 * M * N * K  # = 137,438,953,472 ≈ 137.4 GFLOPs

# 算法理论 HBM 流量（不含 L2 命中，最坏情况）
bytes_read = (M*K + K*N) * dtype_bytes   # A + B
bytes_written = (M*N) * dtype_bytes      # C
bytes_accessed = bytes_read + bytes_written

# 算术强度
ai = flops / bytes_accessed  # ≈ 137.4G / 201.3MB ≈ 682 FLOP/byte
```

对于 H100 SXM5：
- 峰值 FP16 = 989.4 TFLOPS
- 峰值 HBM BW = 3350 GB/s
- Ridge Point = 989.4 / 3.350 ≈ 295 FLOP/byte

682 > 295 → **compute-bound**

### 4.3 不同 kernel 类型的典型瓶颈

| Kernel 类型 | 算术强度 | 典型瓶颈 |
|-------------|---------|---------|
| Elementwise (ReLU, add) | ~0.1-0.5 FLOP/byte | **Memory-bound** |
| Reduction (softmax, norm) | ~1-5 FLOP/byte | **Memory-bound** |
| Small GEMM (M,N < 512) | ~50-200 FLOP/byte | 取决于 GPU，可能在 ridge 附近 |
| Large GEMM (M,N ≥ 4096) | ~500+ FLOP/byte | **Compute-bound** |
| Flash Attention (long seq) | ~50-200 FLOP/byte | 通过 tiling 从 memory-bound 拉到 compute-bound |
| Convolution | 在 GEMM 和 elementwise 之间 | 取决于 kernel size |

### 4.4 Roofline 的正确使用

- **Arithmetic intensity 是 kernel 的内在属性**，由算法决定，不随实现改变
- **Achieved TFLOPS 取决于实现质量**：好的实现（shared memory、register blocking）接近 roofline，差的实现被带宽或延迟限制
- **优化方向由 bottleneck 决定**：
  - Memory-bound → 减少 HBM 访问（fusion、tiling、better cache use）
  - Compute-bound → 减少 FLOPs 或更好利用 Tensor Core（warp tiling、MMA layout）

---

## 5. cuBLAS 为什么难超越

很多初学者 benchmark 自己的 Triton GEMM 后发现远不如 `torch.matmul`，这是正常的：

1. **手写汇编级优化**：cuBLAS 为每个 GPU 架构（Volta、Ampere、Hopper）手写 SASS/PTX，利用了所有硬件特性
2. **Triton 无法表达的指令**：
   - Hopper 的 `wgmma`（warp group matrix multiply-accumulate）— 异步 Tensor Core
   - TMA（Tensor Memory Accelerator）— 硬件加速的地址计算 + 数据搬运
   - 这些指令在 Triton 3.x 中没有对等的抽象
3. **Tile 尺寸穷举**：cuBLAS 在安装时对所有可能的 tile 尺寸做 exhaustive search，Triton 的 autotune 通常只扫几十个配置

**但这也是 Triton 的价值所在**：
- 用 20% 的精力达到 cuBLAS 的 70-80%，然后做 cuBLAS 做不到的 **operator fusion**
- Fused softmax + matmul 在 Triton 里 30 行代码，在 CUDA 里要 500+ 行

---

## 6. "% of cuBLAS" vs "% of Ceiling" — 哪个更有参考价值

### % of Roofline Ceiling
- 最诚实的数字：你的实现离物理极限还有多远
- **问题**：cuBLAS 自己也只能达到 ceiling 的 80-85%（Tensor Core 利用率达不到 100%），所以 ceiling 不是一个可实现的目标

### % of cuBLAS
- 更实际的对标：业界最优实现是多少
- **业界标准**：Triton 实现达到 cuBLAS 的 **85%+** 就算优秀
- 对于做 fusion 的 kernel（如 Flash Attention），对标对象不是 cuBLAS 而是 `flash-attn` 库

**建议两个都报告**。`% of ceiling` 告诉你还有多少理论空间，`% of cuBLAS` 告诉你离最优实践还有多远。

---

## 7. 不同问题规模的性能特征

### 小规模（M, N < 256）
- **Launch overhead 主导**：kernel launch 的固定开销（~5-10μs）可能比执行时间还长
- **不要优化这里**：小矩阵用 PyTorch eager 就行，Triton kernel 的 launch overhead 可能倒挂
- 如果你必须优化，用 **persistent kernel**（一个 grid 持续运行，避免多次 launch）

### 中等规模（1024-4096）
- **优化的主战场**：这是大多数 LLM 推理的实际尺寸
- Shared memory、block tiling、autotune 的效果在这里最明显

### 超大规模（>8192）
- 通常受 HBM 带宽限制（除非算术强度极高）
- 即使 compute-bound 的大 GEMM，接近 ceiling 后也很难继续优化
- 关注点转向 multi-GPU / tensor parallelism

---

## 8. 常见 benchmark 误区

### 误区 1: 只测一个尺寸
```python
# ❌ 只测 (1024, 1024, 1024) 就说"我的 kernel 达到 X TFLOPS"
# ✅ 扫描多个尺寸，看趋势。很多 kernel 在小尺寸好、大尺寸差（或反过来）
```

### 误区 2: 不检查数值正确性就 bench
```python
# ❌ 直接 benchmark，发现很快，但结果是错的
max_diff = (my_output - reference).abs().max()
assert max_diff < 0.01, f"Wrong! max_diff={max_diff}"
# 再去 bench
```

### 误区 3: 用 PyTorch eager 当 baseline
- `torch.matmul` 底层是 cuBLAS，不是"naive 实现"
- 真正的 baseline 应该是一个**朴素的 Triton 实现**（如 01_matmul_naive.py），才能看 tiling/shared memory 带来了多少提升

### 误区 4: 比较不同 dtype 的 TFLOPS
- FP32 的峰值 TFLOPS 远低于 FP16（Tensor Core 的 FP16 吞吐是 FP32 的 ~15x）
- 比较 TFLOPS 时必须确保所有实现用同一 dtype，或在报告中注明不同 dtype 的峰值

### 误区 5: 忽略 torch.profiler 和 ncu
- `torch.profiler` 能看到 kernel 的时间线、CPU/GPU 重叠、memcpy
- `ncu`（NVIDIA Nsight Compute）能看到 achieved occupancy、memory throughput、compute throughput、register spills
- 只用 wall-clock 时间做优化 = 盲人摸象

---

## 9. 推荐的 Benchmark 流程

    1. make check-gpu          → 确认 GPU 型号和峰值参数
    2. 跑 correctness test     → 确保 kernel 数值正确
    3. torch.profiler          → 看 GPU kernel 的时间线，确认没有异常
    4. ncu --set full          → 看 roofline、occupancy、memory/compute throughput
    5. 扫描多个规模             → 画 TFLOPS vs size 的折线图
    6. 报告 % of ceiling + % of cuBLAS（或 % of flash-attn）

---

---

## 10. 如何解读 `ncu`（NVIDIA Nsight Compute）输出

对于初学者，`ncu --set full` 的输出可能很吓人。这里只关注最关键的几个指标：

### 最重要的 5 个指标

| 指标 | 在哪看 | 什么算好 | 什么算差 |
|------|--------|---------|---------|
| **Achieved Occupancy** | `Launch Statistics` | >60% | <30%（warp 太少，延迟隐藏不足） |
| **Compute Throughput** | `GPU Speed Of Light` | >60% | <30%（SM 计算单元空闲太多） |
| **Memory Throughput** | `GPU Speed Of Light` | >60%（memory-bound时） | 对于 compute-bound kernel 低是好事 |
| **Register Per Thread** | `Launch Statistics` | 64-128 | >200（寄存器压力，导致 occupancy 下降） |
| **Shared Memory Bank Conflict** | `Memory Workload Analysis` | 0% | >10%（有 bank conflict） |

### 如何判断 kernel 瓶颈（3 步法）

    1. 看 "GPU Speed of Light" 的 Compute Throughput
       → Compute > 60%: kernel 在努力计算，compute bound
       → Compute < 30%: SM 计算单元空闲，继续看 Memory
    
    2. 看 "GPU Speed of Light" 的 Memory Throughput
       → Memory > 60%: kernel 内存受限，memory bound
       → Memory < 30%: 既不用计算也不等内存 → 可能是 launch overhead 或同步问题
    
    3. 看 Achieved Occupancy
       → occupancy < 30%: 检查寄存器使用量，可能 num_warps 太大
       → occupancy > 60%: 资源利用正常，问题在别处

### 快速诊断命令

```bash
# 只看最重要的几个指标
ncu --set full --section SpeedOfLight \
    --section LaunchStatistics \
    --section MemoryWorkloadAnalysis \
    python my_kernel.py
```

---

## 参考资料

- [Roofline Model (Williams, Waterman, Patterson; 2009)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Triton Autotuning Guide](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [CUDA C++ Best Practices Guide — Timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing)
