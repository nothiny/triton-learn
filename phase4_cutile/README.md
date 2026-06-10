# Phase 4 — cuTile Python 探索

## 当前状态

📝 **骨架阶段** — cuTile Python (`cuda-tile`) 是 NVIDIA 官方的 GPU kernel 编程语言，基于 Tile IR。可以理解为 NVIDIA 版的 Triton。本阶段目前只有环境检测，等你学完 Triton 后再深入对比。

## 文件列表

| # | 文件 | 内容 |
|---|------|------|
| — | `README.md` | 本文件 |
| 1 | `placeholder.py` | 环境检测：检查 cuTile Python 是否可用 |
| 2 | `00_layout_algebra.md` | Tile IR Layout 代数笔记（骨架） |

## 运行方式

```bash
python phase4_cutile/placeholder.py
```

## 前置条件

```bash
# cuTile Python 需要 CUDA Toolkit ≥ 13.1
# 安装 cuTile Python
pip install cuda-tile[tileiras]

# 验证
python -c "import cuda.tile; print('cuTile OK')"
```

## 学习路径

1. ✅ 学完 Phase 1-3（Triton）
2. 📖 读 `notes/04_cutile_preview.md`
3. 💻 安装 cuTile Python: `pip install cuda-tile[tileiras]`
4. 📖 阅读 [cuTile Python 官方文档](https://docs.nvidia.com/cuda/cutile-python)
5. 💻 从 [TileGym](https://github.com/NVIDIA/TileGym) 示例入手
6. 🔬 用 cuTile 重写 Phase 1 的 vector_add，对比两种 DSL 的设计

## Triton vs cuTile 互补

| 场景 | Triton | cuTile |
|------|:---:|:---:|
| 快速原型 | ✅ | ⚠️ (仍快速, 但更底层) |
| 常见 kernel 优化 | ✅ | ✅ |
| 极致性能控制 | ⚠️ | ✅ (Tile-level 控制) |
| 学习成本 | 中 | 较高 |
| 文档 + 社区 | 成熟 | 较新 (2025) |
| NVIDIA 官方支持 | ❌ (第三方) | ✅ (一等公民) |
