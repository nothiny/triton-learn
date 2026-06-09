# Phase 4 — CuTe 预备

## 当前状态

📝 **骨架阶段** — CuTe 是 CUTLASS 3.x 的底层 C++ 模板库。本阶段目前只有环境检测和概念笔记，等你学完 Triton 后再深入。

## 文件列表

| # | 文件 | 内容 |
|---|------|------|
| — | `README.md` | 本文件 |
| 1 | `placeholder.py` | 环境检测：检查 CUTLASS 是否可用 |
| 2 | `00_layout_algebra.md` | CuTe Layout 代数笔记（骨架） |

## 运行方式

```bash
# 环境检测
python phase4_cute/placeholder.py

# 输出示例：
#   CUTLASS not found.
#   Install: git clone https://github.com/NVIDIA/cutlass.git
```

## 前置条件

```bash
# 安装 CUTLASS（header-only）
git clone https://github.com/NVIDIA/cutlass.git
export CUTLASS_PATH=/path/to/cutlass

# 需要有 nvcc（CUDA C++ compiler）
nvcc --version
```

## 学习路径

1. ✅ 学完 Phase 1-3（Triton）
2. 📖 读 `notes/04_cute_preview.md`
3. 💻 安装 CUTLASS
4. 📖 阅读 [CuTe 官方文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
5. 💻 从 CUTLASS examples 入手

## 与 Triton 的互补

| 场景 | Triton | CuTe |
|------|--------|------|
| 快速原型 | ✅ | ❌ |
| 常见 kernel 优化 | ✅ | ❌ |
| 极致性能 (warp specialization) | ❌ | ✅ |
| 学习成本 | 中 | 高 |
