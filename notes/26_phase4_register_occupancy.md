# 26 — 寄存器分配与 Occupancy 优化

> GPU SM 的三资源约束模型、spill 检测、num_warps/BLOCK_SIZE/num_stages 的权衡——Triton 性能调优的核心知识。
> 配合 `phase4_compiler/10_register_pressure.py`。

---

## 0. 问题：为什么改了 num_warps 性能就变了？

你通过 autotune 搜到最优的 `num_warps=8`。但为什么是 8 而不是 4？为什么不是 16？

答案在 GPU SM（Streaming Multiprocessor）的物理限制中。

---

## 1. GPU SM 的三资源约束模型

### 1.1 每个 SM 的固定资源

```
┌──────────────────────┬─────────────────────┐
│ 资源                  │ H100 每 SM          │
├──────────────────────┼─────────────────────┤
│ 32-bit 寄存器         │ 65536              │
│ Shared Memory         │ 228 KB（可配置）     │
│ 最大 Warps            │ 64                 │
│ 最大 Thread Blocks    │ 32                 │
└──────────────────────┴─────────────────────┘
```

### 1.2 你的 Kernel 每 CTA（block）消耗的资源

```
每 CTA 寄存器 = num_warps × 32 threads/warp × registers_per_thread
每 CTA Shared Memory = 所有 .shared 声明的总和
每 CTA Warps = num_warps
```

### 1.3 Occupancy 的计算

```
Occupancy = 同时驻留在 SM 上的 CTA 数

受限于:
  max_ctas_by_regs   = floor(65536 / regs_per_cta)
  max_ctas_by_shared = floor(228KB / shared_per_cta)
  max_ctas_by_warps  = floor(64 / num_warps)
  max_ctas_by_blocks = 32

  occupancy = min(max_ctas_by_regs, max_ctas_by_shared,
                  max_ctas_by_warps, max_ctas_by_blocks)
```

**举例**：

```
num_warps=4, regs_per_thread=128
  → regs_per_cta = 4×32×128 = 16384
  → max_ctas_by_regs = 65536/16384 = 4 个 CTA 可同时驻留
  → max_ctas_by_warps = 64/4 = 16
  → occupancy = min(4, 16, 32, ...) = 4 ✓

num_warps=8, regs_per_thread=128
  → regs_per_cta = 8×32×128 = 32768
  → max_ctas_by_regs = 65536/32768 = 2 个 CTA
  → max_ctas_by_warps = 64/8 = 8
  → occupancy = min(2, 8, 32, ...) = 2

num_warps=16, regs_per_thread=128
  → regs_per_cta = 16×32×128 = 65536
  → max_ctas_by_regs = 65536/65536 = 1 个 CTA
  → occupancy = 1  ← 寄存器成了瓶颈！
```

**增加 num_warps 不一定提高 occupancy**——注册器用光时反而降低。

---

## 2. 寄存器压力的"甜蜜点"

### 2.1 每线程寄存器数的含义

| regs/thread | 状态 | 行动 |
|------------|------|------|
| < 64 | 很轻松 | 可以考虑增加 num_warps 或 BLOCK_SIZE |
| 64-128 | 正常 | 保持，大多数 kernel 在这个范围 |
| 128-200 | 较高 | 注意 occupancy，大 kernel 要关注 |
| > 200 | 很高 | 很可能 spill，考虑减小 BLOCK_SIZE |
| > 255 | H100 物理上限 | **一定会 spill**，必须优化 |

### 2.2 从 PTX 中读寄存器使用量

```bash
# 统计 PTX 中的寄存器声明
grep "\.reg" ~/.triton/cache/*.ptx | wc -l

# 详细分类
grep -c "\.reg \.f32" ~/.triton/cache/*.ptx   # float 寄存器
grep -c "\.reg \.b32" ~/.triton/cache/*.ptx   # 32-bit 通用
grep -c "\.reg \.b64" ~/.triton/cache/*.ptx   # 64-bit（计为 2 个 32-bit）
```

或使用分析工具：

```python
from phase4_compiler.27_ir_analysis_tools import IROpStats
stats = IROpStats.from_ir_text(ptx_text)
print(f"Estimated registers: ~{stats.register_estimate}")
```

### 2.3 什么增加了寄存器需求？

| 因素 | 影响 |
|------|------|
| `BLOCK_SIZE ↑` → `sizePerThread ↑` | 直接增加（每线程持有更多元素） |
| 更多活跃的中间变量 | Triton 编译器管理，你控制不了太多 |
| fp32 代替 fp16 | 寄存器宽度翻倍 |
| 复杂的计算图（多个 load→多种计算→store） | 中间结果需要寄存器保存 |

---

## 3. Register Spilling：性能的隐形杀手

### 3.1 什么是 Spilling？

当 LLVM（或 ptxas）发现物理寄存器不够时，将一些值"溢出"到**local memory**（线程私有的，存在 L1 cache 中）：

```
寄存器 → local memory（STL/store local）
需要时 → 从 local memory 加载回来（LDL/load local）
```

local memory 的延迟 ~20-100 cycles（取决于 L1 hit），远高于寄存器的 0 cycles。

### 3.2 两级 Spill 检测

**PTX 级别**（LLVM 做的 spill）：
```bash
grep "st.local\|ld.local" ~/.triton/cache/*.ptx
# 有输出 → LLVM 已经 spill 了
```

**SASS 级别**（ptxas 做的 spill——更隐蔽）：
```bash
cuobjdump -sass *.cubin | grep "STL\|LDL"
# PTX 中没有 st.local，但 SASS 中有 STL → ptxas 发现物理寄存器不够
# 这是看 SASS 的核心价值
```

### 3.3 Spill 的连锁反应

```
每线程需要的寄存器太多
  → LLVM spill 一些到 local memory
  → PTX 中出现 st.local/ld.local
  → 即使 PTX 中声明了 255 个虚拟寄存器，ptxas 也可能 spill 更多
  → 额外的 load/store 指令
  → 2-5x 性能下降
```

---

## 4. Shared Memory 约束

### 4.1 什么消耗 Shared Memory？

- **`num_stages` 的 pipeline buffer**：`num_stages × tile_size × sizeof(dtype) × 2`（A + B 各一份）
- **`convert_layout` 的 staging buffer**：需要 rearranged 的数据暂存
- **手动分配的 shared memory**（较少见）

### 4.2 num_stages 的权衡

| num_stages | Shared Memory | 内存延迟隐藏 | Occupancy 影响 |
|-----------|-------------|------------|---------------|
| 1 | 1× tile（最小） | 无 | 最小（好） |
| 2 | 2× tile | 中等 | 小 |
| 3-4 | 3-4× tile | 高 | 可能显著 |
| 5+ | 5+× tile | 很高 | 很可能降低 |

```
如果 shared_per_cta > 228KB:
  → 无法在 H100 上启动 kernel（编译失败或运行时错误）
如果 shared_per_cta 接近 228KB:
  → occupancy = 1（每个 SM 只能驻留 1 个 CTA）
  → GPU 利用率低
```

---

## 5. Triton 参数 ↔ 资源消耗映射

```
参数                影响              资源
─────────────────────────────────────────────
num_warps ↑         每 CTA warp 多      寄存器需求 ↑（总池固定，每 warp 分得少）
                                        占用更多 warp 槽位
BLOCK_SIZE ↑        sizePerThread ↑    寄存器需求 ↑（每线程管更多元素）
num_stages ↑        buffer 多           Shared Memory ↑↑
dtype=fp32          fp32 代替 fp16     寄存器宽度 ↑↑（翻倍）
```

**最优配置是三维空间中最靠近"刚好不溢出"的点**——autotune 做的就是在这个空间里搜。

---

## 6. 实战：分析你的 Kernel 的资源使用

### 6.1 快速检查清单

```bash
# 1. 生成 PTX
TRITON_KERNEL_DUMP=1 python my_kernel.py

# 2. 寄存器
echo "=== Registers ==="
grep -c "\.reg \.f32" ~/.triton/cache/*.ptx
grep -c "\.reg \.b32" ~/.triton/cache/*.ptx
grep -c "\.reg \.b64" ~/.triton/cache/*.ptx

# 3. Shared Memory
echo "=== Shared Memory ==="
grep "\.shared" ~/.triton/cache/*.ptx

# 4. Spill 检测
echo "=== Spill Check ==="
grep "st.local\|ld.local" ~/.triton/cache/*.ptx && echo "SPILL DETECTED!"
```

### 6.2 如果发现 Spill

1. **减小 BLOCK_SIZE** → 减少 sizePerThread → 减少寄存器需求
2. **用 fp16/bf16 代替 fp32** → 寄存器宽度减半
3. **减小 num_warps** → 每 warp 分到更多寄存器 → 减少 LLVM spill（但这可能降低 occupancy）
4. **简化 kernel 逻辑** → 减少中间变量
5. **检查是否真的需要额外的 load** → 复用已有数据

### 6.3 如果 Occupancy 太低

1. **减少 num_warps** → 更少的 warp 槽位占用（但更多的寄存器/warp）
2. **减少 num_stages** → 减少 shared memory
3. **减小 BLOCK_SIZE** → 减少寄存器需求 → 可以增加 num_warps

> 💡 **没有万能解**——这三个参数相互制衡。这就是为什么需要 autotune。

---

## 7. 参考文件

| 主题 | 教程文件 |
|------|---------|
| 寄存器压力分析 | `10_register_pressure.py` |
| PTX 注释分析 | `07_ptx_assembly.py` |
| PTX → SASS | `21_ptx_to_sass.py` |
| autotuner 原理 | `18_autotuner.py` |

---

## 8. 总结

```
GPU SM 的三资源约束:

  寄存器 (65536/SM)  ─┐
  Shared Mem (228KB)  ─┼── 三者最小值 → Occupancy
  Warp Slots (64)     ─┘

Triton 的三个控制参数:
  num_warps    → 寄存器 + Warp Slots
  BLOCK_SIZE   → 寄存器（sizePerThread）
  num_stages   → Shared Memory

调优策略:
  寄存器 spill? → 减小 BLOCK_SIZE / 用 fp16 / 减少 num_warps
  Occupancy 低? → 减少 num_warps / 减少 num_stages
  一切都好但仍慢? → 检查是否是 memory-bound 而非 compute-bound
```
