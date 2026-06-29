# 26 — IR 级性能调试方法论

> 从"kernel 慢"到"找到根因"的系统化流程——4 种常见问题的 IR 诊断方法。
> 配合 `phase4_compiler/11_debugging_with_ir.py` 和 `19_env_vars.py`。

---

## 0. "我的 kernel 很慢" → 从哪里开始？

**核心原则：不要猜，看 IR。**

```
怀疑 → Dump IR → 读 IR → 定位问题 → 修复 → 重新 Dump 验证
  ↑___________________________________________________________↓
```

这个笔记教你：**每种性能症状对应 IR 的哪个文件的哪一行**。

---

## 1. 诊断工作流

### 1.1 第一步：建立 Baseline

```bash
TRITON_KERNEL_DUMP=1 python my_kernel.py
# → 生成 .ttir, .ttgir, .ptx 到 ~/.triton/cache/
```

### 1.2 第二步：按症状检查

| 症状 | 首先检查 | 看什么 |
|------|---------|--------|
| TFLOPS 远低于峰值 | `.ptx` | 有没有 `mma.sync`？ |
| 带宽利用率低 | `.ttgir` | 有多少 `convert_layout`？ |
| 性能随 BLOCK_SIZE 剧烈波动 | `.ptx` | 寄存器数量变了吗？ |
| 不同 GPU 上性能差异巨大 | `.ptx` + `.sass` | MMA 形状对吗？spill 了吗？ |
| 编译慢、第一次运行卡 | `.ttgir` / cache | 是不是在重复编译？ |

### 1.3 第三步：验证修复

```bash
# 改代码后强制重编译
TRITON_KERNEL_DUMP=1 TRITON_KERNEL_OVERRIDE=1 python my_kernel.py

# 对比旧 IR 和新 IR
diff <(cat old_ir.ptx) <(cat ~/.triton/cache/*.ptx)
```

---

## 2. 案例 1：tl.dot 没有触发 Tensor Core

### 症状

GEMM kernel TFLOPS 只有峰值的 15-20%。

### 诊断

**Step 1：检查 TTIR**
```bash
grep "tt.dot" ~/.triton/cache/*.ttir
```
✅ 有 `tt.dot` → tl.dot 被正确识别为矩阵乘。

**Step 2：检查 PTX**
```bash
grep "mma.sync" ~/.triton/cache/*.ptx
```
❌ 没有 `mma.sync` → MMA lowering 没有触发！

### 根因分析

| 可能原因 | 确认方法 | 修复 |
|---------|---------|------|
| fp32 输入（H100 不支持 fp32 MMA） | 看 dtype | 换 fp16/bf16 |
| K 维度不是 16 的倍数 | 看 BLOCK_K | 调整为 16 的倍数 |
| BLOCK 太小（< MMA tile） | 看 config | 增大 BLOCK_M/BLOCK_N |
| Triton 版本不支持 | `triton.__version__` | 升级 Triton |

### 验证修复

```bash
TRITON_KERNEL_DUMP=1 python my_kernel.py
grep "mma.sync" ~/.triton/cache/*.ptx
# → 现在应该看到 mma.sync.aligned.m16n8k16...
```

### 预期结果

修复后 TFLOPS 应该达到峰值的 60-80%（剩余 20-40% 是内存延迟）。

---

## 3. 案例 2：convert_layout 过多

### 症状

Bandwidth utilization 高但 TFLOPS 低，或者 shared memory 用量异常高。

### 诊断

**Step 1：统计 convert_layout**
```bash
grep -c "convert_layout" ~/.triton/cache/*.ttgir
```
输出：8 → 有 8 次 layout 转换，可能太多。

**Step 2：定位转换位置**

在 TTGIR 中搜索：
```mlir
ttg.convert_layout %x : tensor<..., #blocked> → tensor<..., #slice>
```
这说明 `blocked → slice` 的转换发生在 `%x` 上 → 往前追溯到 `%x` 的定义 → 找到触发它的 op。

### 根因分析

常见模式：
```python
x = tl.load(...)                         // blocked
s = tl.sum(x, axis=1)                    // slice（convert 1）
y = x - s[:, None]                       // blocked ← slice（convert 2）
v = tl.sum(y * y, axis=1)               // slice（convert 3）
z = y / v[:, None]                       // blocked ← slice（convert 4）
```

每次 `blocked ↔ slice` 都是 shared memory round-trip + barrier。

### 修复

1. **合并 elementwise**：把 `x_centered = x - mean[:, None]` 和 `x_norm = x_centered / rstd[:, None]` 合并？
2. **重新组织计算顺序**：先做完所有 reduce，再做所有 elementwise
3. **使用更高效的算法**：生产级 LayerNorm 用 welford 或 parallel reduction 减少转换

### 验证

```bash
# 修复前后对比
grep -c "convert_layout" old.ttgir  # 8
grep -c "convert_layout" new.ttgir  # 2（期望减少到 2-3 个）
```

---

## 4. 案例 3：寄存器 Spill

### 症状

性能突然大幅下降（2-5x），ncu 显示大量 `l1tex__data_bank_conflict` 或 local memory 访问。

### 诊断

**Step 1：检查 PTX**
```bash
grep "st.local\|ld.local" ~/.triton/cache/*.ptx
```
输出：`st.local.f32 [%rd+...], %f42;` → **LLVM 层面已经 spill！**

**Step 2：统计寄存器**
```bash
grep -c "\.reg \.f32" ~/.triton/cache/*.ptx
```
输出：210 → > 200，非常接近 255 上限。

**Step 3（进阶）：检查 SASS**

如果 PTX 中没有 `st.local` 但性能仍然异常：
```bash
cuobjdump -sass ~/.triton/cache/*.cubin | grep "STL\|LDL"
```
如果有输出 → ptxas 发现物理寄存器不够，即使 PTX 声明了足够的虚拟寄存器。

### 修复

```python
# 尝试 1：减小 BLOCK
@triton.autotune(configs=[
    triton.Config({'BLOCK': 64}, num_warps=4),   # 原来 128
    triton.Config({'BLOCK': 128}, num_warps=4),
])

# 尝试 2：用 fp16
x = tl.load(ptr, ...)  # dtype=fp16 代替 fp32

# 尝试 3：减少 num_warps（给每 warp 更多寄存器）
# 原来 num_warps=8 → 试试 num_warps=4
```

### 验证

```bash
grep -c "\.reg \.f32" new.ptx   # 期望 < 150
grep "st.local" new.ptx          # 期望无输出
```

---

## 5. 案例 4：未合并的内存访问（Uncoalesced Access）

### 症状

带宽利用率只有峰值的 30-50%，尽管 kernel 是明显的 memory-bound。

### 诊断

**Step 1：检查地址模式**

在 PTX 中看 `ld.global` 的地址模式：
```asm
// Coalesced（好）：相邻线程访问相邻地址
thread 0: ld.global.f32 %f1, [%rd + 0]
thread 1: ld.global.f32 %f1, [%rd + 4]

// Strided（坏）：相邻线程跨大步
thread 0: ld.global.f32 %f1, [%rd + 0]
thread 1: ld.global.f32 %f1, [%rd + 512]
```

**Step 2：检查 TTGIR 的 layout order**

```bash
grep "order" ~/.triton/cache/*.ttgir
```

`order=[0, 1]` → dim 0 是 innermost → 线程沿 dim 0 连续 → 如果内存是 dim 1 连续的（如 B 矩阵），则**访问不连续** → uncoalesced。

### 修复

```python
# 如果数据是 column-major（dim 1 连续），用 order=[1, 0]
# 或调换 load 时的索引顺序：
x = tl.load(ptr + offs_n[:, None] * stride + offs_m[None, :])
# 而不是 x = tl.load(ptr + offs_m[:, None] * N + offs_n[None, :])
```

### 验证

修复后带宽利用率应接近峰值（80-90%）。

---

## 6. 通用诊断速查表

| 检查什么 | 在哪层 IR | 命令 |
|---------|---------|------|
| `tt.dot` 被识别 | `.ttir` | `grep "tt.dot" *.ttir` |
| MMA 触发 | `.ptx` | `grep "mma.sync" *.ptx` |
| convert_layout 数量 | `.ttgir` | `grep -c "convert_layout" *.ttgir` |
| Tensor shapes | `.ttir` | `grep "tensor<" *.ttir` |
| Layout 参数 | `.ttgir` | `grep "#blocked\|#mma\|#slice" *.ttgir` |
| 寄存器数量 | `.ptx` | `grep -c "\.reg \.f32" *.ptx` |
| PTX 级 spill | `.ptx` | `grep "st.local\|ld.local" *.ptx` |
| SASS 级 spill | `.sass` | `cuobjdump -sass *.cubin \| grep "STL\|LDL"` |
| Shared mem 用量 | `.ptx` | `grep "\.shared" *.ptx` |
| GPU 架构 | `.ttgir` | `grep "ttg.target" *.ttgir` |
| Coalesced? | `.ptx` | 看地址模式（手动） |

---

## 7. 关键环境变量

| 变量 | 用途 |
|------|------|
| `TRITON_KERNEL_DUMP=1` | 生成所有 IR 到 cache |
| `TRITON_KERNEL_OVERRIDE=1` | 强制重编译（不用 cache） |
| `MLIR_PRINT_IR_AFTER_ALL=1` | 每个 pass 后 dump IR（输出量巨大） |
| `TRITON_PRINT_AUTOTUNING=1` | 查看 autotune 测试了哪些 config |
| `TRITON_INTERPRET=1` | CPU 解释执行（支持 Python 断点调试！） |

---

## 8. 参考文件

| 主题 | 教程文件 |
|------|---------|
| IR 调试实战 | `11_debugging_with_ir.py` |
| 环境变量速查 | `19_env_vars.py` |
| IR 分析工具 | `27_ir_analysis_tools.py` |

---

## 9. 总结

性能诊断的黄金法则: 看 IR，不要猜。

4 个最常见问题:
  1. MMA 没触发 → 检查 .ptx 有没有 mma.sync
  2. convert_layout 太多 → 检查 .ttgir，目标 <3
  3. 寄存器 spill → 检查 .ptx + .sass 的 st.local/STL
  4. Uncoalesced → 检查 address pattern + layout order

每次性能异常:
  → Dump IR → 按症状对照上表 → 找到问题 → 修复 → Dump 验证
