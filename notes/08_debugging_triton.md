# 08 — 调试 Triton Kernel：从"跑不通"到"跑得对"

> ⚠️ 最容易被忽视但最重要的笔记。GPU kernel debug 比 CPU 难 10 倍——没有 printf、没有断点、错误信息晦涩。这篇整理了所有可用的调试工具和常见 Bug 清单。

---

## 1. 为什么 GPU Kernel Debug 这么难？

| CPU 调试 | GPU 调试 |
|---------|---------|
| `printf` 随便用 | `tl.device_print` 受限、输出截断 |
| gdb/lldb 断点 | 无标准 debugger |
| 崩溃有 stack trace | CUDA error 通常只有 `misaligned address` |
| 可以 `assert` | `tl.device_assert` 在 release 模式下被跳过 |
| 逻辑错误容易复现 | 并行执行 → 不确定性 → 难以复现 |

**核心心态**: 在 GPU 上，你无法"看看变量是什么"。必须用结构性方法排查。

---

## 2. 调试工具链（从简单到复杂）

### 2.1 最简单的检查：正确性验证

```python
# 第一步永远是：你的 kernel 输出对不对？
def check_kernel(triton_fn, ref_fn, *args, rtol=1e-3, atol=1e-3):
    actual = triton_fn(*args)
    expected = ref_fn(*args)
    
    abs_diff = (actual - expected).abs()
    max_diff = abs_diff.max().item()
    exceed = (abs_diff > atol + rtol * expected.abs()).float().mean().item()
    
    print(f"Max diff: {max_diff:.6e}")
    print(f"Exceed fraction: {exceed:.6e}")
    
    if max_diff > atol * 100:
        # 找到差异最大的位置
        worst_idx = abs_diff.argmax().item()
        print(f"Worst @ {worst_idx}: actual={actual.flatten()[worst_idx]:.6f}, "
              f"expected={expected.flatten()[worst_idx]:.6f}")
    
    return max_diff < atol * 10
```

### 2.2 缩小问题规模

```python
# ❌ 直接在 4096×4096 上 debug — 太慢了，看不出规律
# ✅ 先用极小尺寸验证逻辑

# 最小可复现测试
M, N, K = 4, 4, 4  # 可以手算验证
a = torch.arange(M*K, device='cuda', dtype=torch.float32).reshape(M, K)
b = torch.arange(K*N, device='cuda', dtype=torch.float32).reshape(K, N)
# 用已知的输入，对比已知的输出
```

### 2.3 `tl.device_print` — GPU 上的 printf

```python
@triton.jit
def debug_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 🔍 打印中间值（只在第一个 program 的前几个元素上）
    if pid == 0:
        tl.device_print("x values:", x)        # 整个向量
        tl.device_print("offsets:", offsets)    # 偏移量
    
    # 计算
    result = x * 2 + 1
    
    if pid == 0:
        tl.device_print("result:", result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

# ⚠️ device_print 限制:
# 1. 只在第一个 program/thread 打印（避免刷屏）
# 2. 输出被截断（只显示前几个元素）
# 3. 有性能开销——不要在生产代码里留
```

### 2.4 `tl.device_assert` — GPU 上的 assert

```python
@triton.jit
def kernel_with_checks(ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # ✅ 检查指针不为空（Triton 3.2+）
    # tl.device_assert(ptr != 0, "Null pointer!")
    
    # ✅ 检查偏移在范围内
    # 注意: assert 会严重影响性能，仅用于 debug
    mask = offsets < N
    
    x = tl.load(ptr + offsets, mask=mask)
    tl.store(ptr + offsets, x, mask=mask)

# 启用 assert（默认在 non-release 构建中启用）
# TRITON_DEBUG=1 python my_kernel.py
```

### 2.5 `TRITON_INTERPRET` — 在 CPU 上解释执行

```bash
# 最强大的 debug 模式: 在 CPU 上逐指令解释 Triton kernel
TRITON_INTERPRET=1 python my_kernel.py

# 好处:
# 1. 可以加 Python 断点（import pdb; pdb.set_trace()）
# 2. 错误信息包含 Python traceback
# 3. 支持逐操作检查
# 4. 在 kernel 内部使用 print() 会正常输出

# 限制:
# 1. 很慢（解释执行，不做 JIT）
# 2. 行为有细微差异（没有真正的 GPU 并发问题）
# 3. tl.device_print 不可用
```

### 2.6 用已知内核对标

```python
# 方法: 从工作的 kernel 开始，逐步修改
# 
# Step 1: vector_add.py（肯定能跑）— 验证环境
# Step 2: 修改成你的新逻辑，一次只改一个东西
# Step 3: 每次修改后验证正确性
# Step 4: 如果某次修改后坏了 → 就是这个修改的问题

# 例: 开发 tiled matmul
# 先从 naive matmul 开始 → 加 shared memory → 加 tiling → 加 autotune
# 每步验证，而不是一步写完再测
```

---

## 3. 常见 Bug 清单

### Bug 1: Mask 错误 — 边界处理不正确

```python
# ❌ 常见错误: mask 只检查一维
# 2D tensor: M=100, N=100, BLOCK_SIZE=32
offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

# 错误: 只 mask 了 M 维，N 维没检查
mask = offsets_m < M  # ❌

# 正确: 两维都要 mask
mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)  # ✅
```

### Bug 2: Stride 错误 — 访问了错误的内存位置

```python
# ❌ 常见错误: stride 顺序搞反
# PyTorch tensor (M, N) 的 stride 是 (N, 1) 对于 row-major
ptr = x_ptr + row * stride_0 + col * stride_1

# 如果你的 row/col 混淆了:
# ptr = x_ptr + row * stride_1 + col * stride_0  # ❌ 行列颠倒了
```

### Bug 3: dtype 不匹配

```python
# ❌ fp16 输入 + fp32 累加器 + 忘记转换
@triton.jit
def kernel(a_ptr, b_ptr, out_ptr, ...):
    a = tl.load(a_ptr + offsets)  # fp16
    b = tl.load(b_ptr + offsets)  # fp16
    # 累加器用 fp32（正确做法）
    acc = tl.zeros([...], dtype=tl.float32)
    
    acc += tl.dot(a, b)  # tl.dot 自动处理
    # 但如果手动计算:
    # result = a * b  # fp16 × fp16 = fp16 → 精度损失！
    result = a.to(tl.float32) * b.to(tl.float32)  # ✅
    
    tl.store(out_ptr + offsets, result)  # 写回时自动转换
```

### Bug 4: Grid 大小不对

```python
# ❌ 常见错误: grid 是 tuple，不是整数
grid = triton.cdiv(N, BLOCK_SIZE)  # ❌ 这是 int，不是 tuple
kernel[grid](...)  # RuntimeError: grid must be a tuple

# ✅
grid = (triton.cdiv(N, BLOCK_SIZE),)  # 或者 lambda meta: (...)
# 注意逗号: 单元素 tuple 需要尾部逗号
```

### Bug 5: tl.constexpr 参数忘了传或传错

```python
# ❌ 用 autotune 时: 参数名不匹配
@triton.autotune(configs=[...], key=['N'])
@triton.jit
def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):  # autotune 参数名为 BLOCK_SIZE
    ...

# config 中写错了:
# triton.Config({'BLOCK': 256}, ...)  # ❌ 应该是 'BLOCK_SIZE'

# ✅
# triton.Config({'BLOCK_SIZE': 256}, ...)
```

### Bug 6: Reduction 的 axis 参数

```python
# ❌ tl.sum(x)  vs tl.sum(x, axis=0) — 效果完全不同
# tl.sum(x)  → 对所有元素求和，返回标量
# tl.sum(x, axis=0) → 沿第 0 维求和，返回向量

# softmax 中:
row_max = tl.max(x, axis=1)  # 沿列方向求 max → 结果是每行的 max
# 如果你写了 axis=0 → 结果的形状完全不对
```

### Bug 7: 忘记 torch.cuda.synchronize()

```python
# ❌ kernel 还没执行完就读结果
result = my_kernel(a, b)
print(result[0])  # 可能读到未写完的数据！

# ✅
result = my_kernel(a, b)
torch.cuda.synchronize()
print(result[0])
```

### Bug 8: 用 Python int 代替 tl.constexpr

```python
# ❌ Python 变量不是编译时常量
BLOCK_SIZE = 256  # Python int
@triton.jit
def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):  # ❌ 不会工作
    ...

# ✅ 直接传字面量或用 autotune
@triton.jit
def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
# 调用时: kernel[grid](ptr, N, BLOCK_SIZE=256)
```

---

## 4. 系统性 Debug 工作流

    遇到 bug 时，按顺序尝试:
    
    □ 1. 缩小问题规模 (M=N=K=4 或 8)
         → 使用已知的输入（arange, ones, zeros）
         → 手算预期输出，对比
    
    □ 2. TRITON_INTERPRET=1
         → 在 CPU 上运行，获得 Python traceback
         → 可以加 pdb 断点
    
    □ 3. tl.device_print
         → 打印中间值（第一个 program 的前几个元素）
    
    □ 4. 从工作 kernel 逐步修改
         → vector_add → 一点一点改成你的目标 kernel
         → 每步验证
    
    □ 5. 检查 IR dump
         → TRITON_KERNEL_DUMP=1
         → 看 TTGIR 中 layout 是否符合预期
         → 检查是否有意外的 ConvertLayout
    
    □ 6. 检查 PTX
         → 看生成的 PTX 中 ld/st 指令地址计算是否正确
         → 确认使用了 mma.sync（如果有 tl.dot）
    
    □ 7. ncu 检查
         → 看 achieved occupancy、memory throughput
         → 排除硬件层面的异常（stall、bank conflict）

---

## 5. 常见 CUDA Error 解码

| 错误信息 | 可能原因 | 解决 |
|---------|---------|------|
| `misaligned address` | 指针访问了未对齐的地址 | 检查 offset 计算，确保 dtype 匹配 |
| `an illegal memory access was encountered` | 数组越界 | 检查 mask、检查 grid 大小 |
| `out of memory` | GPU 内存不足 | 减小 batch/tensor 大小 |
| `too many resources requested` | shared memory 或寄存器超限 | 减小 BLOCK_SIZE 或 num_warps |
| `driver shutting down` | 上一个 kernel 崩溃后没恢复 | 重启 Python（`import torch; torch.cuda.init()`） |

---

## 6. 调试辅助工具

```python
# utils/debug.py — 一个简单的 debug 辅助函数

import torch

def describe_tensor(name: str, t: torch.Tensor) -> None:
    """Print tensor metadata for debugging."""
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
          f"device={t.device}, stride={tuple(t.stride())}, "
          f"contiguous={t.is_contiguous()}, "
          f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
          f"mean={t.float().mean().item():.4f}")

def check_output(actual, expected, name="output", rtol=1e-3, atol=1e-3):
    """Validate triton kernel output."""
    abs_diff = (actual.float() - expected.float()).abs()
    max_diff = abs_diff.max().item()
    exceed_frac = (abs_diff > atol + rtol * expected.float().abs()).float().mean().item()
    
    print(f"[{name}] max_diff={max_diff:.2e}, exceed_frac={exceed_frac:.2e}")
    
    if max_diff > atol * 10:
        worst_idx = abs_diff.argmax().item()
        multi_idx = unravel_index(worst_idx, actual.shape)
        print(f"  Worst @ {multi_idx}: actual={actual.flatten()[worst_idx]:.6f}, "
              f"expected={expected.flatten()[worst_idx]:.6f}")
        return False
    return True

def unravel_index(idx, shape):
    """Convert flat index to multi-dimensional index."""
    result = []
    for dim in reversed(shape):
        result.append(idx % dim)
        idx //= dim
    return tuple(reversed(result))
```
