# 13 — Triton 内部机制：JIT、Cache、Driver

> 了解 Triton 如何编译和管理 kernel，帮助你 debug 编译问题、管理 cache、理解性能。

---

## 1. Triton 的 JIT 编译流程

### 1.1 从调用到执行

```
你的代码:
  kernel[grid](a, b, c, N, BLOCK_SIZE=1024)

Triton 内部:
  ┌──────────────────────────────────────────┐
  │ 1. 检查 cache                           │
  │    ~/.triton/cache/ 中有这个 kernel +    │
  │    这组参数组合的编译结果吗？             │
  └──────────┬───────────────────────────────┘
             │
       ┌─────┴─────┐
       │ cache hit? │
       └─────┬─────┘
        Yes  │  No
         │   │
         │   ├──→ 2. 提取 kernel 的 AST
         │   │       从 @triton.jit 装饰的函数中解析 Python AST
         │   │
         │   ├──→ 3. AST → TTIR (MLIR)
         │   │       用 Triton 的 AST builder 生成 MLIR
         │   │
         │   ├──→ 4. TTIR → TTGIR
         │   │       运行 compiler passes (ConvertTritonToTritonGPU, ...)
         │   │
         │   ├──→ 5. TTGIR → LLVM IR
         │   │       运行 ConvertTritonGPUToLLVM
         │   │
         │   ├──→ 6. LLVM IR → PTX
         │   │       调用 LLVM NVPTX backend
         │   │
         │   ├──→ 7. PTX → CUBIN
         │   │       调用 ptxas (NVIDIA 汇编器，需要 CUDA toolkit)
         │   │
         │   └──→ 8. 缓存结果
         │           写入 ~/.triton/cache/<hash>/ 目录
         │
         └──→ 9. 加载 CUBIN 到 GPU driver
              → 10. 启动 kernel
```

### 1.2 Cache 哈希

```python
# Triton 的 cache key 由以下因素决定:
# 1. kernel 源代码的 hash
# 2. 所有 tl.constexpr 参数的值
# 3. autotune config (num_warps, num_stages, ...)
# 4. GPU 架构 (SM version)
# 5. Triton 版本
# 
# 任何一项变化 → 不同的 cache key → 重新编译

# 查看 cache:
ls ~/.triton/cache/
# 输出: 一堆 hash 目录

# 清理 cache:
rm -rf ~/.triton/cache/
# 下次运行会重新编译所有 kernel
```

### 1.3 强制重编译

```bash
# 方法 1: 环境变量
TRITON_KERNEL_OVERRIDE=1 python my_kernel.py
# 跳过 cache，强制重新编译

# 方法 2: 删除 cache
rm -rf ~/.triton/cache/

# 方法 3: 改 Triton 版本或升级 CUDA
# 这会改变 cache key → 自动重编译
```

---

## 2. Triton 的 Python AST 分析

### 2.1 `@triton.jit` 做了什么？

```python
@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...

# @triton.jit 装饰器:
# 1. 解析 my_kernel 的 Python 源码
# 2. 识别 tl.constexpr 参数（必须编译时已知）
# 3. 识别 triton.language 的调用（tl.load, tl.store, tl.dot, ...）
# 4. 构建 TTIR generation 函数
# 5. 包装为一个 JITFunction 对象
# 
# 当 kernel[grid](args...) 被调用时:
# 1. 检查是否需要用当前参数编译
# 2. 编译（或从 cache 加载）
# 3. 将编译结果加载到 GPU
# 4. 启动 kernel
```

### 2.2 AST 限制

```
Triton 的 Python AST 分析有限制:

✅ 支持:
  - for 循环（固定范围）
  - if/else（tl.constexpr 条件）
  - 算术运算
  - tl.* 函数调用
  - 局部变量

❌ 不支持:
  - Python 标准库调用（len(), range() 可以用 tl.arange 代替）
  - 动态内存分配
  - 递归
  - try/except
  - 闭包/高阶函数
  - generator/yield
```

---

## 3. Triton 与 CUDA Driver 的交互

### 3.1 Driver API 调用流程

```
Triton 调用序列（简化）:

1. cuModuleLoad(compiled_cubin)
   → 将编译好的 kernel 二进制加载到 CUDA context

2. cuModuleGetFunction(module, "kernel_name")
   → 获取 kernel 入口函数

3. cuLaunchKernel(kernel, grid, block, shared_mem, stream, args)
   → 启动 kernel

4. (kernel 在 GPU 上执行)

5. cuStreamSynchronize(stream)
   → 如果 torch.cuda.synchronize() 被调用
```

### 3.2 Stream 管理

```python
# Triton 使用 CUDA stream 管理 kernel 执行

# 默认: 使用当前 torch 的 CUDA stream
kernel[grid](args...)  # 在 current stream 上 launch

# 可以用 torch.cuda.stream 创建新 stream:
with torch.cuda.stream(torch.cuda.Stream()):
    kernel[grid](args...)  # 在新 stream 上 launch
```

---

## 4. Triton 编译器性能调试

### 4.1 编译耗时分析

```python
# 如果第一次运行很慢（几秒到几十秒），那是正常的 — 这是 JIT 编译

# 查看编译缓存:
ls ~/.triton/cache/
# 每个 hash 目录对应一个 kernel 的一组参数

# 如果每次运行都重新编译:
# 1. 检查是不是每次都改了 autotune config
# 2. 检查 cache 目录是否可写
# 3. 检查 disk 空间
```

### 4.2 Autotune Cache

```python
# Autotune 的结果也缓存在 ~/.triton/cache/ 中
# 格式: cache/<hash>/autotune_config.json

# 第一次运行: 测试所有 config → 缓存最优
# 后续运行: 直接从 cache 读取最优 config → 快速启动

# 如果改了 kernel 代码 → cache 失效 → 重新 autotune
# 如果只改了输入大小 → 不同 key → 不同 cache entry
```

---

## 5. Triton 环境变量速查

| 变量 | 作用 | 何时使用 |
|------|------|---------|
| `TRITON_KERNEL_DUMP=1` | Dump TTIR/TTGIR/LLVM/PTX | Debug IR |
| `TRITON_KERNEL_OVERRIDE=1` | 强制重编译 | 改代码后想看新 PTX |
| `MLIR_PRINT_IR_AFTER_ALL=1` | 每个 pass 后打印 IR | 深入 debug pass pipeline |
| `TRITON_INTERPRET=1` | CPU 解释执行 | Debug（支持 Python 断点） |
| `TRITON_ALWAYS_COMPILE=1` | 总是重新编译 | 开发时确保用最新代码 |
| `TRITON_PRINT_AUTOTUNING=1` | 打印 autotune 过程 | 查看 autotune 测试了哪些 config |

---

## 6. 参考资料

- [Triton GitHub — lib/](https://github.com/triton-lang/triton/tree/main/lib) — 编译器源码
- [Triton GitHub — python/triton/compiler/](https://github.com/triton-lang/triton/tree/main/python/triton/compiler)
- [Triton JIT 论文 (Tillet et al., 2023)](https://www.eecs.harvard.edu/~htk/publication/2023-pact-tillet-kung-cox.pdf)
