# Benchmarks — 性能对比系统

## 对比矩阵

| 对比对象 | Kernel 来源 | 说明 |
|----------|-----------|------|
| **Your Triton** | `phase1_fundamentals/`, `phase2_compute/` | 你手写的 Triton kernel |
| **PyTorch** | `torch.nn.functional`, `torch.mm` | PyTorch 内置 → cuBLAS/cuDNN |
| **Liger Kernel** | `liger_kernel.transformers.functional` | LinkedIn 的开源 Triton kernel 库 |
| **PyTorch SDPA** | `F.scaled_dot_product_attention` | PyTorch 内置 FlashAttention-2 |

## 运行方式

```bash
# 快速对比（所有 kernel）
make bench

# 只看 normalization（含 Liger LayerNorm/RMSNorm）
python benchmarks/bench_runner.py --category normalization

# 只看 elementwise（含 Liger SwiGLU/GeGLU）
python benchmarks/bench_runner.py --category elementwise

# 带 profiling + chrome traces
python benchmarks/bench_runner.py --profile --trace-out traces/

# 导出 JSON
python benchmarks/bench_runner.py -j reports/results.json
```

## 输出解读

```
Kernel                          Size    Triton(ms)  Ref(ms)   Speedup   Triton TFLOPS  Ref TFLOPS
MatMul Tiled (fp16, autotuned)   4K²      0.404      0.299    ✅ 1.35x       340.3         460.2
```

- **Speedup**: Triton 时间 / Ref 时间
  - 🔥 <1.0x: Triton 更快
  - ✅ 1.0-1.5x: 差距不大
  - ⚠️ 1.5-5.0x: 有差距
  - ❌ >5.0x: 差距很大
- **TFLOPS**: 计算吞吐量（越高越好）
- **Roofline**: 显示 H100 理论峰值，帮助你判断瓶颈

## 添加新 Benchmark

编辑 `benchmarks/bench_cases.py`，在 `build_cases()` 中添加：

```python
cases.append(BenchCase(
    name="My New Kernel",
    category="gemm",                  # elementwise/reduction/gemm/attention/normalization
    triton_fn=my_triton_fn,           # 你的 Triton 包装函数
    ref_fn=lambda a, b: torch.mm(a, b),  # PyTorch reference
    input_gen=gen_my_inputs,          # (size_idx) → (args, kwargs)
    flops_calc=flops_my_kernel,       # (inputs) → FLOP count
    bytes_calc=bytes_my_kernel,       # (inputs) → total bytes r/w
    sizes=[256, 512, 1024],           # problem sizes
    size_labels=["S", "M", "L"],
))
```

## Chrome Trace 可视化

```bash
# 1. 生成 trace
python benchmarks/bench_runner.py --profile --trace-out traces/

# 2. 打开 Chrome
# 地址栏输入: chrome://tracing
# 拖入 traces/ 目录下的 .json 文件

# 可以看到：
# - GPU kernel 的时间线
# - CPU kernel launch 时间
# - 数据传输 (memcpy)
# - CUDA sync 点
```

## Profile 输出示例

```
Name                          Self CUDA   Self CUDA%
matmul_tiled_kernel             28.063us      100.00%   ← 你的 kernel
nvjet_hsh_128x80_64x8_4x1_v_bz  22.528us      100.00%   ← cuBLAS kernel
```

cuBLAS 的 kernel 名可以看出它用的 tile 尺寸 (`128x80x64`)，对比你自己的 tile 尺寸，就能找到优化方向。
