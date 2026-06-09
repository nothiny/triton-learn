# Utils — 工具模块

## 文件列表

| 文件 | 内容 | 用法 |
|------|------|------|
| `profiler.py` | GPU kernel 性能测量 + roofline 分析 | benchmark 系统使用 |
| `checker.py` | 数值正确性验证（vs PyTorch reference） | 测试和调试使用 |
| `ir_dump.py` | Triton IR dump 工具 + PTX 注释 | Phase 3 使用 |

## 各模块说明

### `profiler.py`

```python
from utils.profiler import KernelProfiler, GPUInfo, quick_bench

# GPU 信息
info = GPUInfo.detect()
print(info.name)           # "NVIDIA H100 PCIe"
print(info.peak_fp16_tflops)  # 989.0

# 单次 benchmark
result = quick_bench(
    lambda: my_kernel(a, b),
    name="my_kernel",
    flops=2 * M * N * K,       # 总 FLOPs
    bytes_read=M * K * 2,      # 读取字节数
    bytes_written=M * N * 2,    # 写入字节数
)
# 输出: TFLOPS, bandwidth, bottleneck (compute/memory bound)
```

### `checker.py`

```python
from utils.checker import check_allclose, check_max_diff

# 对比 Triton 输出 vs PyTorch reference
ok = check_allclose("my_kernel", actual=out_triton, expected=out_torch)

# 快速 check
max_diff = check_max_diff("my_kernel", out_triton, out_torch)
```

### `ir_dump.py`

```python
from utils.ir_dump import enable_ir_dump, annotate_ptx

# 启用 IR dump
enable_ir_dump()
# ... 运行 kernel ...
# → 检查 ~/.triton/cache/ 下的 .ttir/.ttgir/.ll/.ptx

# 注释 PTX
annotated = annotate_ptx(ptx_source)
# 标注: [REG], [SHARED], [MMA], [BARRIER], [LD.GLOBAL], ...
```
