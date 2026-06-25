# Phase 2 — Production-Grade Kernels

使用 Triton 现代 API（`tl.make_block_ptr`, `tl.advance`, `triton.heuristics` 等）
的生产级 kernel 实现。与 `phase2_compute/` 中的教学版本形成对比学习。

## 核心 API 升级

| 老写法 (phase2_compute) | 新写法 (phase2_production) | 说明 |
|---|---|---|
| 手工指针拼接 + mask | `tl.make_block_ptr` + `boundary_check` | 编译器可推理访问模式 |
| `for k in range(0, K, BK)` 手工偏移 | `tl.advance(p, (BK, 0))` | 语义化的指针推进 |
| 固定 block size | `triton.heuristics` 动态选择 | 根据输入 shape 选择最优配置 |
| `tl.load(ptr, mask=..., other=0.0)` | `tl.load(p, boundary_check=(0,1))` | 自动边界处理 |

## Kernel 列表

| # | 文件 | 对标 phase2_compute | 学什么 |
|---|------|-------------------|--------|
| 01 | `matmul_block_ptr.py` | `02_matmul_tiled.py` | block_ptr GEMM, TMA-ready |
| 02 | `attention_decoding.py` | `07_flash_attention_v1.py` | 推理 attention, GQA |

## 与 phase2_compute 的关系

`phase2_compute/` 是"先理解原理"：
- 手工指针拼接让你看清地址计算
- 显式 mask 让你理解边界处理
- 固定配置让你理解 block size 的影响

`phase2_production/` 是"再学最佳实践"：
- 块指针让编译器做优化
- 自动边界处理减少 boilerplate
- heuristics 做动态 dispatch
