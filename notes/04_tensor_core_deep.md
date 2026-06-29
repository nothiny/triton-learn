# 04 — Tensor Core 深入：从 PTX 指令到性能优化

> Tensor Core 是 NVIDIA GPU 上矩阵乘法的硬件加速器。理解它的工作原理，是达到 >70% peak TFLOPS 的关键。

---

## 1. Tensor Core 做了什么？— 一个直观的类比

```
普通 CUDA Core 做乘法:
  1 个 FMA (Fused Multiply-Add) = c += a × b
  每 cycle: 1 个结果
  4 bytes 的 FMA

Tensor Core 做乘法:
  1 个 MMA (Matrix Multiply-Accumulate) = D = A @ B + C
  每 cycle: 16×8×16 = 2048 个结果（Ampere fp16）
  2048 个 FMA 的等价计算

  类比: CUDA Core = 一支笔（一次写一个字）
       Tensor Core = 一个印刷机（同时印一面纸）
```

---

## 2. MMA 指令详解

### 2.1 指令格式

$$
\text{mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32}
$$

解读:

- **m16n8k16**: 输出 tile 大小: $M = 16,\ N = 8,\ K = 16$
- **row.col**: A 矩阵是 row-major, B 是 col-major（相对 PTX 来说）
- **f32** (第1个): D（输出/累加）的类型
- **f16** (第1个): A 的类型
- **f16** (第2个): B 的类型
- **f32** (第2个): C（累加器输入）的类型

### 2.2 不同架构的 MMA 规格

| GPU 架构 | 指令 | M×N×K | 数据类型 |
|---------|------|-------|---------|
| Volta (V100) | m8n8k4 | 8×8×4 | fp16→fp16/fp32 |
| Ampere (A100) | m16n8k16 | 16×8×16 | fp16/bf16→fp32 |
| Ampere (A100) | m16n8k8 | 16×8×8 | tf32→fp32 |
| Ampere (A100) | m16n8k16 | 16×8×16 | int8→int32 |
| Hopper (H100) | m16n8k16 | 16×8×16 | fp16/bf16→fp32 |
| Hopper (H100) | m16n8k32 | 16×8×32 | fp8→fp32 |

### 2.3 Fragment 布局

```
一个 warp（32 线程）协作计算 m16n8k16 MMA:

A fragment (16×16 fp16 = 256 elements, 512 bytes):
  线程 0:  持有 A[0:2, 0:4]    (8 个 fp16)
  线程 1:  持有 A[2:4, 0:4]    (8 个 fp16)
  ...(按特定模式分配)
  线程 31: 持有 A[14:16, 4:8]  (8 个 fp16)
  
B fragment (16×8 fp16 = 128 elements, 256 bytes):
  类似地分配到 32 个线程

C/D fragment (16×8 fp32 = 128 elements, 512 bytes):
  每个线程持有 4 个 fp32 元素（128/32 = 4）

[COMPILER] 这是 Triton 的 MmaEncodingAttr 描述的内容。
Triton 编译器自动决定 fragment 分配方案。
```

---

## 3. Tensor Core 的约束 — 什么是 Triton 不能做的

### 3.1 MMA 的硬性要求

1. **Tile 大小固定**: 不能做 $15 \times 7 \times 15$ 的 MMA — 必须是硬件支持的尺寸。Triton 处理: `tl.dot` 自动选择合适的 MMA 指令。

2. **数据类型固定**: Ampere 支持 `fp16`, `bf16`, `tf32`, `int8`, `int4`, `int1`。`fp32` 不能直接用 Tensor Core（只能用 CUDA Core）。

3. **Fragment 布局固定**: 输入张量必须以特定的布局排列（`DotOperandEncodingAttr`），不同于常规的 `BlockedEncoding`。

4. **需要 warp 内同步**: 32 个线程必须同时执行 MMA（锁步执行）。硬件隐式同步。

### 3.2 Triton 如何映射 tl.dot

```python
# 你写的:
acc += tl.dot(a, b)  # a: [128, 32] fp16, b: [32, 256] fp16

# Triton 编译器做的:
# 1. 分析 a, b 的 layout encoding
# 2. 如有必要，插入 ConvertLayout 到 DotOperandEncodingAttr
# 3. 将 128×32×256 的 dot 分解为多个 m16n8k16 的 MMA
#    128/16=8, 256/8=32 → 8×32=256 次 MMA 调用
# 4. 生成对应的 mma.sync 指令序列
# 5. 管理 fragment 分配（哪个线程持有结果的哪部分）
```

---

## 4. FP8 — H100 的新能力

### 4.1 为什么需要 FP8？

**FP16** (16 bits, 5 exponent + 10 mantissa + 1 sign): 范围 $\pm 65504$, 精度: 3.3 位十进制

**FP8 E4M3** (8 bits, 4 exponent + 3 mantissa + 1 sign): 范围 $\pm 448$, 精度: 1 位十进制

**FP8 E5M2** (8 bits, 5 exponent + 2 mantissa + 1 sign): 范围 $\pm 57344$, 精度: 0.6 位十进制

关键: FP8 的精度不够直接做整个 GEMM。但通过 block-wise scaling（每 block 独立 scale），可以保持精度
$\rightarrow$ H100 上用 FP8 GEMM $\approx 2\times$ FP16 的吞吐量
$\rightarrow$ 对 LLM inference 特别有用（少量精度损失，换来 $2\times$ 速度）

### 4.2 Triton 中的 FP8

```python
# Triton 3.x 开始支持 FP8
# 需要用 tl.load 的 special dtype

@triton.jit
def fp8_matmul(a_ptr, b_ptr, c_ptr, ...):
    # FP8 输入
    a = tl.load(a_ptr + offsets, ...)  # 需要数据已经是 fp8 格式
    b = tl.load(b_ptr + offsets, ...)
    
    # FP32 累加器（仍需高精度累加）
    acc = tl.zeros([...], dtype=tl.float32)
    acc += tl.dot(a, b)  # Triton 自动用 FP8 MMA
    
    tl.store(c_ptr + offsets, acc)
```

---

## 5. Sparsity — 2:4 结构化稀疏

### 5.1 什么能被加速？

Ampere+ 的硬件 sparsity 支持: 只有在"每组 4 个连续元素中恰好有 2 个为 0"时才能加速
$\rightarrow$ 2:4 structured sparsity

例: `[a, 0, c, 0, e, 0, g, 0]` $\leftarrow$ 每 4 个中有 2 个零 $\rightarrow$ 可以 $2\times$
`[0, 0, c, d, 0, 0, g, h]` $\leftarrow$ 不行（前面 3 个零）

Tensor Core 会自动:
- 跳过 0 元素的计算
- 将有效元素打包为原来的 $1/2$ 的存储
- 吞吐翻倍（$2\times$ peak TFLOPS）

### 5.2 Triton 中的 Sparsity

Triton 目前不直接支持 2:4 sparsity。但通过:
1. 在数据准备阶段做 2:4 pruning
2. 用 `torch.sparse` 格式传递
3. Triton kernel 中绕过零元素

实际上这个流程很复杂，Triton 的 sparsity 支持仍在发展中。

---

## 6. Tensor Core 性能 Checklist

□ 1. 使用了 `fp16`/`bf16`/`tf32` 输入？$\rightarrow$ `fp32` 不能用 Tensor Core（除非用 `tf32`）

□ 2. `BLOCK_K` 是 16 的倍数？$\rightarrow$ MMA 的 K 维是 16，确保 `BLOCK_K % 16 == 0`

□ 3. `BLOCK_M` 是 16 的倍数，`BLOCK_N` 是 8 的倍数？$\rightarrow$ 确保 `tl.dot` 能被整数次 MMA 覆盖

□ 4. 数据在 GPU 上是 row-major 的？$\rightarrow$ PyTorch 默认 row-major，Triton 默认处理 row-major

□ 5. 累加器用 `fp32`？$\rightarrow$ `acc = tl.zeros([...], dtype=tl.float32)`

□ 6. 检查 PTX 中有 `mma.sync` 指令？$\rightarrow$ `TRITON_KERNEL_DUMP=1` $\rightarrow$ `grep mma.sync` $\rightarrow$ 确认用了 Tensor Core

---

## 7. 参考资料

- [NVIDIA Tensor Core Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [PTX ISA — MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)
- [CUTLASS — Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
