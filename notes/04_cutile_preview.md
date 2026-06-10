# 04 — cuTile Python 预览

> cuTile Python (`cuda-tile`) 是 NVIDIA 官方的 GPU kernel 编程语言，基于 Tile IR。可以理解为"如果 NVIDIA 来设计 Triton，会是什么样"。
> **建议**: 学完 Phase 1-3（Triton）后再读这篇，建立 cuTile 概念即可。

---

## 0. 背景：为什么 NVIDIA 要做 cuTile？

### 0.1 Triton 的成功与局限

Triton (OpenAI, 2019) 证明了"Python DSL → GPU kernel"这条路的可行性。它的 block-level 抽象让写 GEMM、Flash Attention 变得简单，编译器自动处理 coalescing、shared memory staging、software pipelining。

但 Triton 有两个 NVIDIA 无法忽视的问题：

1. **Triton 不归 NVIDIA 控制**。Triton 是 OpenAI 的开源项目，编译器优化方向、新硬件支持优先级都由 OpenAI 决定。如果 NVIDIA 想在 H200/B200 上暴露某个新特性（如 TMA 的某种用法），Triton 不一定会及时支持。

2. **Triton 的编译器抽象有上限**。Triton IR 的 block-level 语义没法表达 warp specialization（producer/consumer warp 分工），这是 H100+ 上榨干硬件性能的关键技术。

所以 NVIDIA 做了 cuTile Python——**由 NVIDIA 官方维护、基于 Tile IR、能访问全部硬件特性**的 Python GPU kernel DSL。

### 0.2 cuTile vs Triton：设计哲学

```
Triton 的设计:                           cuTile 的设计:
  "编译器帮你做决策"                       "你告诉编译器精确要什么"
  
  你写: tl.dot(a, b)                     你写: 显式 tile 形状, 显式 pipeline stage
  编译器: 自动选 MMA 指令,                编译器: 严格按你说的做, 不做"聪明"变换
         自动分配 fragment
         
  优点: 快速开发, 30 行写 GEMM             优点: 精确控制, 接近手写 CUDA 的性能
  缺点: 无法使用 wgmma/TMA/warp spec      缺点: 代码更长, 需要更深硬件理解
```

| 维度 | Triton (OpenAI) | cuTile Python (NVIDIA) |
|------|:---:|:---:|
| 编译器 IR | Triton IR → TTGIR → LLVM → PTX | Tile IR → PTX |
| 抽象层级 | Block-level (program) | Tile-level (更细粒度) |
| MMA 指令选择 | 编译器自动 | 程序员显式指定 |
| Warp specialization | ❌ 不支持 | ✅ 支持 |
| TMA | ❌ 不支持 | ✅ 支持 |
| 硬件要求 | CUDA ≥ 12.0, Ampere+ | CUDA ≥ 13.1, Blackwell/Ampere/Ada |
| 维护方 | OpenAI (社区) | NVIDIA (官方) |
| 学习曲线 | 中等 | 较高 |

---

## 1. cuTile 编程模型速览

### 1.1 第一个 cuTile kernel：Vector Add

```python
import cuda.tile as ct
import cupy

TILE_SIZE = 16

@ct.kernel
def vector_add_kernel(a, b, result):
    # 获取当前 block 的索引（类似 tl.program_id）
    block_id = ct.bid(0)
    
    # 从 global memory 加载一个 tile（类似 tl.load）
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    
    # 逐元素计算（类似 Triton 的向量化操作）
    result_tile = a_tile + b_tile
    
    # 写回 global memory（类似 tl.store）
    ct.store(result, index=(block_id,), tile=result_tile)

# 启动 kernel（类似 Triton 的 grid + kernel[grid](...)）
grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
```

### 1.2 和 Triton 语法的逐行对比

```
Triton:                               cuTile:
─────────────────────────────────     ─────────────────────────────────
@triton.jit                            @ct.kernel
def kernel(x_ptr, ...):                def kernel(x, ...):

pid = tl.program_id(0)                 block_id = ct.bid(0)

offsets = pid*BLOCK + tl.arange(0,B)   tile = ct.load(x, index=(block_id,),
x = tl.load(x_ptr + offsets)                         shape=(TILE_SIZE,))

output = x + y                         result_tile = a_tile + b_tile

tl.store(out_ptr + offsets, output)    ct.store(result, index=(block_id,),
                                                   tile=result_tile)

grid = lambda meta: (...)              grid = (n_blocks, 1, 1)
kernel[grid](x, y, out, N, B=1024)     ct.launch(stream, grid, kernel, (a, b, result))
```

核心差异：
- **Triton**: 你操作的是指针 + 偏移（`x_ptr + offsets`），更接近 C 语义
- **cuTile**: 你操作的是 tile 对象（`ct.load` 返回一个 tile），更接近数组语义
- **Triton**: `BLOCK_SIZE` 是编译时常量，通过 autotune 搜索
- **cuTile**: tile 形状在 `ct.load` 时指定，更灵活

---

## 2. cuTile 的核心抽象：Tile

### 2.1 什么是 Tile？

cuTile 的核心概念是 **Tile**——一个多维数据块。`ct.load` 从 global memory 加载一个 tile，返回一个 tile 对象。tile 对象支持逐元素算术运算，`ct.store` 把它写回 memory。

```python
# 1D tile: 长度为 256 的向量段
tile_1d = ct.load(x, index=(block_id,), shape=(256,))

# 2D tile: 128×64 的矩阵块（用于 GEMM）
tile_2d = ct.load(A, index=(bm, bn), shape=(128, 64))

# 3D tile: 用于卷积
tile_3d = ct.load(X, index=(b, c, h, w), shape=(1, 32, 8, 8))
```

### 2.2 Tile IR：cuTile 的中间表示

cuTile 的编译器管线比 Triton 更短：

```
Triton:  Python AST → Triton IR → TTGIR → LLVM IR → PTX → SASS
cuTile:  Python AST → Tile IR → PTX → SASS

cuTile 跳过了 LLVM IR 这一步！Tile IR 直接编译到 PTX。
这意味着 cuTile 可以生成 Triton（通过 LLVM）无法表达的 PTX 指令，
比如 wgmma、TMA 指令等。
```

Tile IR 是 NVIDIA 为 GPU kernel 专门设计的 IR，不经过 LLVM 中转。这是一个关键架构差异。

---

## 3. cuTile 能做到而 Triton 做不到的事

### 3.1 TMA (Tensor Memory Accelerator)

H100 的 TMA 是硬件数据搬运单元，可以异步地在 global memory 和 shared memory 之间搬运 5D tensor tile。Triton 3.x 不支持 TMA（只能通过 `num_stages` 参数间接影响 cp.async 的使用）。

cuTile 中可以直接使用 TMA：

```python
# 概念代码（cuTile 支持 TMA descriptor）
# TMA 可以异步搬运多维 tensor tile，
# 支持边界自动处理、转置、reduction 等
tma_copy = ct.tma_load(src, dst_shared, descriptor)
```

### 3.2 Warp Specialization

H100 上最大的性能提升来自 warp specialization——把 warp 分成 producer 组和 consumer 组：

```
传统 kernel (Triton):
  所有 warp 轮流: load → compute → store → load → ...

Warp Specialization (cuTile):
  Producer Warps (warp 0-3):
    [TMA load tile 0] [TMA load tile 1] [TMA load tile 2] ...
    只做数据搬运 → TMA 利用率 100%

  Consumer Warps (warp 4-7):
    [wgmma tile 0] [wgmma tile 1] [wgmma tile 2] ...
    只做矩阵乘 → Tensor Core 利用率接近 100%

  两者通过 shared memory pipeline 异步协作
```

这是 H100 上 cuBLAS 能做到 >80% peak TFLOPS 的原因。Triton GEMM 通常只有 60-70%。

### 3.3 wgmma (Warp Group MMA)

H100 的 wgmma 指令让 4 个 warp 协作做更大的矩阵乘（比传统 mma 指令吞吐量高 2x）。Triton 无法直接使用 wgmma，cuTile 可以。

---

## 4. 什么时候用 Triton，什么时候用 cuTile？

```
快速原型 ──→ Triton（1-2 天出结果）
    │
    ▼
性能评估 ──→ 对比 cuBLAS/Liger
    │
    ├── 达到 80%+ peak → ✅ 不需要 cuTile
    │
    └── 差 20%+ → 考虑 cuTile 重写
                    │
                    ├── 需要 TMA/warp spec → cuTile
                    ├── 需要 wgmma → cuTile
                    └── 只是 tuning 不够 → 多试几个 Triton autotune config
```

| 场景 | Triton | cuTile |
|------|:---:|:---:|
| 快速原型开发 | ✅ 首选 | ❌ 太重 |
| Elementwise / reduction / norm | ✅ 够用 | ❌ 过度 |
| GEMM (A100) | ✅ ~85% cuBLAS | ⚠️ 提升不大 |
| GEMM (H100) | ⚠️ ~70% cuBLAS | ✅ >90% cuBLAS |
| Flash Attention | ✅ 适合 | ⚠️ 可做但 Triton 生态更好 |
| Operator fusion (自定义) | ✅ 最灵活 | ✅ 也可做 |
| 需要精确控制 shared memory | ❌ 编译器决定 | ✅ 显式控制 |

---

## 5. 学习路径

```
Phase 1-3 (Triton) ──→ 掌握 GPU kernel 思维 + 编译管线
        │
        ▼
环境准备 ──→ CUDA Toolkit ≥ 13.1
        │    pip install cuda-tile[tileiras]
        │    python phase4_cutile/placeholder.py
        │
        ▼
cuTile 入门 ──→ https://docs.nvidia.com/cuda/cutile-python
        │    │ 读 quickstart + 跑 TileGym 示例
        │    │ (https://github.com/NVIDIA/TileGym)
        │
        ▼
对比练习 ──→ 用 cuTile 重写 Phase 1 的 vector_add
        │    │ 理解 tile-level vs block-level 的差异
        │
        ▼
  进阶 ──→ GEMM with TMA + warp specialization (Hopper)
           │ 这是 cuTile 相比 Triton 优势最大的场景
```

---

## 6. 和 Triton 编译器的概念映射

| cuTile 概念 | Triton 等价 | 谁做决策 |
|-----------|-----------|---------|
| `ct.load` shape | `BLOCK_SIZE` + masking | Triton: autotune 搜索；cuTile: 你指定 |
| Tile | `tl.arange` + `tl.load` 的向量化 | Triton: 编译器隐式 tile；cuTile: 显式 tile 对象 |
| `ct.store` | `tl.store` | 基本相同 |
| Pipeline stage | `num_stages` (config 参数) | Triton: config 控制；cuTile: 更细粒度 |
| MMA selection | 编译器自动选择 | cuTile: 可在 Tile IR level 指定 |
| 内存层级 | 编译器管理 shared memory | cuTile: 可显式分配和布局 |

---

## 参考资料

- [cuTile Python 官方文档](https://docs.nvidia.com/cuda/cutile-python)
- [GitHub: NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)
- [TileGym — cuTile 示例集](https://github.com/NVIDIA/TileGym)
- [Tile IR 文档](https://docs.nvidia.com/cuda/tile-ir/)
- [cuTile Python PyPI](https://pypi.org/project/cuda-tile/)
