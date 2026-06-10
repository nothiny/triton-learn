# cuTile Tile 模型 — Notes

> cuTile 的核心抽象: **Tile** —— 一个多维数据块，是 compute 和 memory 操作的基本单元。

## Tile 基础

```python
import cuda.tile as ct

# ct.load 从 global memory 加载一个 tile
# index: tile 在 global tensor 中的起始位置（逻辑坐标）
# shape: tile 的形状
tile = ct.load(src, index=(block_id,), shape=(TILE_SIZE,))

# tile 支持逐元素算术
result_tile = a_tile + b_tile

# ct.store 把 tile 写回 global memory
ct.store(dst, index=(block_id,), tile=result_tile)
```

## Tile 的维度

cuTile 支持 1D 到 5D 的 tile：

```python
# 1D: 向量段
tile_1d = ct.load(x, index=(i,), shape=(256,))

# 2D: 矩阵块（GEMM）
tile_2d = ct.load(A, index=(bm, bn), shape=(128, 64))

# 3D: 立方体块（3D 卷积）
tile_3d = ct.load(X, index=(b, c, d), shape=(1, 32, 16))

# 5D: Tensor（TMA 搬运单元的标准维度）
tile_5d = ct.load(T, index=(n, c, d, h, w), shape=(1, 32, 16, 16, 16))
```

## 与 Triton 的对比

| 概念 | Triton | cuTile |
|------|--------|--------|
| 数据单元 | `tl.arange(B) + tl.load(ptr+offsets)` (指针+偏移) | `ct.load(src, index, shape)` (tile 对象) |
| 形状指定 | `BLOCK_SIZE: tl.constexpr` (编译时常量) | `shape=(...)` (运行时也可变) |
| 指针操作 | 显式 (`x_ptr + offsets`) | 隐式 (tile 对象封装) |
| Masking | 手动 `mask=offsets < N` | `ct.load` 自动处理边界 |

## 后续填充

- Tile IR 编译管线详解
- TMA descriptor 构建
- Pipeline stage 配置
- 与 CUTLASS C++ Layout 代数的映射

请参考 `notes/04_cutile_preview.md` 了解完整预览。
