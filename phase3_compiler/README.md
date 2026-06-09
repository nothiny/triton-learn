# Phase 3 — 编译器内部

## 学习目标

理解 Triton 的编译管线，能 dump 并阅读各阶段 IR，掌握 layout encoding 系统。

## 文件列表

| # | 文件 | 内容 | 难度 |
|---|------|------|------|
| 1 | `01_dump_ir.py` | Dump TTIR/TTGIR/LLVM/PTX | ⭐⭐ |
| 2 | `02_layout_analysis.py` | 解析 BlockedEncoding/MmaEncoding/SliceEncoding | ⭐⭐⭐ |
| 3 | `custom_pass/` | 自定义 MLIR Pass 框架 + Python AST 分析 | ⭐⭐⭐⭐ |
| 4 | `04_ptx_analysis.py` | PTX 指令识别、寄存器压力分析 | ⭐⭐⭐ |

## 运行方式

```bash
# 1. Dump 各阶段 IR
python phase3_compiler/01_dump_ir.py
# → 输出在 ~/.triton/cache/ 下：.ttir, .ttgir, .ll, .ptx 文件

# 2. 分析 layout encoding
python phase3_compiler/02_layout_analysis.py
# → 解析 BlockedEncodingAttr 的参数含义

# 3. PTX 分析
python phase3_compiler/04_ptx_analysis.py
# → 带注释的 PTX，标注每类指令的作用

# 使用 Makefile
make dump-ir
make layout-analysis
make ptx-analysis
```

## 关键输出

### `01_dump_ir.py` 运行后

```
~/.triton/cache/
├── XXXXX.ttir    ← Triton IR (tt dialect)，无 GPU 信息
├── XXXXX.ttgir   ← Triton GPU IR (ttg dialect)，含 #blocked<> layout
├── XXXXX.ll      ← LLVM IR (NVPTX target)
└── XXXXX.ptx     ← PTX 汇编，最终执行的指令
```

### `02_layout_analysis.py` 会解释

```
BlockedEncodingAttr{
  sizePerThread  = (1, 4)      ← 每线程持有 1×4 个元素
  threadsPerWarp = (2, 16)     ← warp 布局 2×16=32 threads
  warpsPerCTA    = (4, 1)      ← 4 warps 沿 dim 0
  order          = (0, 1)      ← dim 0 为 innermost
}

→ 每个 CTA 处理 128×64 = 8192 个元素
→ Thread (t0, t1) 持有元素 [t0, t1*4 : (t1+1)*4]
```

## 前置阅读

必须读完 `notes/03_triton_compiler_pipeline.md`，这是整个 Phase 3 的理论基础。

## 进阶

- 用 `MLIR_PRINT_IR_AFTER_ALL=1` dump 每个 pass 前后的 IR
- 用 `TRITON_KERNEL_DUMP=1` dump 所有阶段的 IR
- 阅读 Triton 源码：`triton/lib/Conversion/`, `triton/lib/Dialect/`
