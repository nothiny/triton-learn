# 04 — CuTe 预览（CuTE: CUTLASS Templates）

> CuTe 是 CUTLASS 3.x 的底层 C++ 模板库。如果把 Triton 比作 Python（高层抽象、快速开发），CuTe 就是 C++ 模板元编程（底层控制、极致性能）。
> **建议**: 先学完 Phase 1-3（Triton），再来读这篇，建立 CuTe 概念即可。

---

## 0. 为什么需要关心 CuTe？— Triton 做不到的事

### 0.1 Triton 的天花板

Triton 的设计哲学是"编译器帮你做决策"。大多数时候这很好——你写 30 行代码，编译器自动生成合并访问、shared memory staging、software pipelining。

但有些优化 Triton 做不到：

```
H100 上 Triton 无法使用的硬件特性:
  ✗ wgmma (warp group MMA) — 4 个 warp 协作做更大的矩阵乘
  ✗ TMA (Tensor Memory Accelerator) — 硬件数据搬运单元
  ✗ Warp specialization — producer warp 做加载、consumer warp 做计算
  ✗ 精确的 shared memory bank 控制 — Triton 自动 swizzle，但不一定最优

这就是为什么 H100 上 Triton GEMM 只能达到 cuBLAS 的 70-85%。
要追求那最后 15-30%，你需要像 CuTe 这样能精确控制每个 bit 的工具。
```

### 0.2 定位：什么时候用哪个

| 场景 | Triton | CuTe |
|------|--------|------|
| 快速原型开发 | ✅ 首选 | ❌ 太重 |
| 常见 kernel（GEMM, softmax, layernorm） | ✅ 够用 | ❌ 过度 |
| Operator fusion | ✅ 最好 | ❌ 繁琐 |
| 极致性能优化（H100） | ⚠️ 接近但不是最快 | ✅ 能榨干硬件 |
| Warp specialization | ❌ 不支持 | ✅ 核心功能 |
| 学习成本 | 中等（Python） | 高（C++17 模板） |
| 代码量 | 30-200 行 | 200-1000+ 行 |

> 💡 **经验法则**: 一个 kernel，先写 Triton 版本（1-2 天）。如果性能不够（差 cuBLAS 20%+），再考虑用 CuTe 重写（1-2 周）。

---

## 1. CuTe 的核心抽象

### 1.1 Layout = Shape × Stride

CuTe 中最核心的概念——**Layout 就是"从逻辑坐标到内存偏移的映射"**：

```cpp
// 概念: 对于维度为 (M, N) 的矩阵，
//       元素 (i, j) 在内存中的位置 = i × stride[0] + j × stride[1]

// Row-major 矩阵: (M, N) with stride (N, 1)
Layout shape  = make_shape(M, N);
Layout stride = make_stride(N, Int<1>{});
// 元素 (i, j) 的偏移 = i*N + j*1

// Column-major 矩阵: stride (1, M)
Layout stride = make_stride(Int<1>{}, M);
// 元素 (i, j) 的偏移 = i*1 + j*M
```

**为什么这很重要？**

Triton 的 layout encoding 是编译器帮你选的——你不知道也不需要知道具体映射。CuTe 的 layout 是你自己构建的——你知道每个元素在内存中的精确位置，所以能做到 Triton 做不到的优化。

> 🔧 **Compiler Perspective**: Layout = 仿射映射 f: ℤⁿ → ℤ（N 维坐标 → 1 维线性地址）。这正是 polyhedral 编译器中 iteration domain → data space 的映射函数。Layout 的组合（`composition`、`complement`）对应仿射变换的复合和求逆。

### 1.2 Tensor = Iterator + Layout

```cpp
// 一个 CuTe Tensor 由一个数据指针和一个 Layout 组成
Tensor A = make_tensor(ptr, layout);  // ptr 是原始 HBM 指针

// 访问元素
auto val = A(i, j);  // 等价于 ptr[layout(i, j)] = ptr[i*stride0 + j*stride1]
```

这比 Triton 的 `tl.load(ptr + offsets)` 更底层——你控制的是"坐标→地址"的映射，而不仅仅是偏移。

### 1.3 核心操作：Layout 代数

```cpp
// Layout 的组合运算
auto C = composition(A, B);      // 复合: C(i) = A(B(i))
auto D = complement(A, B);       // 求补: A ∘ D = B (用于推导需要的数据布局)
auto E = right_inverse(A);       // 右逆: 用于反推坐标
auto F = product(A, B);          // 笛卡尔积: 组合两个 layout

// 这些操作让你可以在编译时推导出：
//   "如果我想这样访问数据，需要 shared memory 怎么布局？"
//   "TMA 需要什么 stride 才能正确拷贝这个 tile？"
```

如果你不熟悉这些代数操作——没关系。CuTe 的初学曲线很陡，**这是 Phase 4 的内容，先建立概念即可**。

---

## 2. MMA Atom — Tensor Core 的 C++ 接口

### 2.1 一个 MMA Atom 是什么？

```cpp
// Ampere fp16 MMA: 一个 warp 在 1 个 cycle 内计算 16×8×16 的矩阵乘
using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
//             ↑     ↑  ↑  ↑   ↑   ↑   ↑   ↑
//             |     |  |  |   |   |   |   └── A transposed, B not
//             |     |  |  |   |   |   └── C/D type (fp32)
//             |     |  |  |   |   └── B type (fp16)
//             |     |  |  |   └── A type (fp16)
//             |     |  |  └── K dim size (16)
//             |     |  └── N dim size (8)
//             |     └── M dim size (16)
//             └── SM architecture (Ampere)

// 把这个 MMA op × layout tiling → 覆盖更大的矩阵
auto tiled_mma = make_tiled_mma(MMA_Op{}, ...);

// 每个线程持有 fragment（矩阵 tile 的碎片）
auto C_frag = tiled_mma.make_fragment_C(...);
// 32 个线程协作完成一个 16×8×16 MMA
```

### 2.2 与 Triton 的对比

```
Triton 中:
  acc += tl.dot(a, b)
  // 你不需要知道这个 dot 映射到哪个 MMA 指令
  // 编译器自动选择: m16n8k16 (A100) 或 m16n8k32 (H100)
  // 编译器自动处理 fragment 分配

CuTe 中:
  // 你显式指定 MMA 指令类型
  using MMA = SM80_16x8x16_F32F16F16F32_TN;
  auto tiled_mma = make_tiled_mma(MMA{}, ...);
  // 你显式管理 fragment
  auto tCrA = tiled_mma.partition_fragment_A(A);
  // 你显式调用 MMA
  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
```

---

## 3. 进阶概念速览（建立词汇表，后续深挖）

| 概念 | 一句话解释 | 类比 |
|------|----------|------|
| **Layout** | Shape × Stride，描述数据在内存中的排列 | Triton 的 `BlockedEncodingAttr` 但由你构建 |
| **TiledCopy** | 描述 global→shared 数据拷贝的 layout 分解 | Triton 编译器自动插入的 cp.async |
| **TiledMMA** | MMA 指令 × layout tiling，覆盖更大的矩阵 | Triton 的 `tl.dot` 但更显式 |
| **Warp Specialization** | Producer warp 做 TMA copy，Consumer warp 做 MMA | 硬件流水线——Hopper 的关键优化 |
| **Copy_Atom** | 一次硬件拷贝操作的抽象（如 `ld.global`, `cp.async`, TMA） | Triton IR 中的 `tt.load` → PTX ld.global |

---

## 4. Warp Specialization（Hopper）— CuTe 的杀手锏

这是目前 Triton（3.x）完全无法表达、但 H100 上提升最大的特性：

```
传统 kernel（Triton 的做法）:
  所有 warp 做同样的事: load → compute → store
  Warp 0: [load]--[compute]--[store]--[load]--[compute]--[store]
  Warp 1: [load]--[compute]--[store]--[load]--[compute]--[store]
  ...

Warp Specialization（CuTe 在 H100 上的做法）:
  两个 warp group 分工:
  
  Producer Warps (warp 0-3):
    [TMA load tile 0] [TMA load tile 1] [TMA load tile 2] ...
    专门负责用 TMA 从 HBM 搬到 shared memory
  
  Consumer Warps (warp 4-7):
    [MMA tile 0] [MMA tile 1] [MMA tile 2] ...
    专门负责从 shared memory 读数据做矩阵乘

  好处:
  - Producer 只做搬运 → TMA 利用率 100%
  - Consumer 只做计算 → Tensor Core 利用率接近 100%
  - 两者完全并行 → shared memory 作为 pipeline buffer

  这是为什么 H100 上 cuBLAS 能达到 >80% peak TFLOPS，
  而 Triton kernel 通常只有 60-70%。
```

> 🔧 **Compiler Perspective**: Warp specialization 本质上就是硬件的 functional unit pipelining——类似 CPU 超标量处理器中"load/store unit"和"ALU"的分离并行，但在 warp 粒度上实现。

---

## 5. 与 Triton 编译器的概念映射

| CuTe 概念 | Triton 编译器等价操作 | 控制权 |
|-----------|-------------------|--------|
| Layout | Layout encoding (编译器自动推导) | CuTe: 手动构建 |
| TiledCopy | `ConvertTritonToTritonGPU` + `TritonGPUPipeline` | CuTe: 手动构建 |
| TiledMMA | `TritonGPUAccelerateMatmul` | CuTe: 手动构建 |
| Pipeline state | `num_stages` 参数 | CuTe: 手动管理状态机 |
| ConvertLayout | `ttg.convert_layout` op | CuTe: 手动 layout 转换 |

> 🔧 **Compiler Perspective**: CuTe 可以理解为"嵌入在 C++ 模板中的编译器"——Layout 代数做的是编译器 IR 层的仿射分析，但由程序员手动完成而非编译器自动推导。这给了程序员最大控制权，但也带来了最高复杂性。

---

## 6. 学习路径

```
Phase 1-3 (Triton) ──→ 掌握 GPU kernel 思维
        │
        ▼
Phase 4 预备 ──→ 环境检测 (phase4_cute/placeholder.py)
        │        │ 读这篇笔记
        │        └ 读 00_layout_algebra.md（CuTe layout 代数入门）
        │
        ▼
CUTLASS 安装 ──→ git clone https://github.com/NVIDIA/cutlass.git
        │
        ▼
CuTe 快速入门 ──→ cutlass/media/docs/cute/00_quickstart.md
        │        │ 理解 Layout, Tensor, MMA Atom 的 C++ API
        │
        ▼
CUTLASS examples ──→ cutlass/examples/ 中的 CuTe 示例
        │        │ 从 simple_gemm 开始
        │
        ▼
进阶 ──→ include/cute/layout.hpp（Layout 代数实现）
        │ include/cute/atom/mma_atom.hpp（MMA Atom 定义）
        │ 实现 warp specialization（Hopper only）
```

---

## 参考资料

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CuTe 快速入门](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CUTLASS 3.x 编程指南](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/programming_guidelines.md)
- [CuTe Layout 代数详解](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout_algebra.md)
