# 28 — 为什么 Triton 不支持 Lambda（以及如何自己加上）

> **背景**: `tl.associative_scan(val, axis=0, combine_fn=lambda a, b: a + b)` 报错 `unsupported AST node type: Lambda`
>
> **核心结论**: 不是设计选择，而是 JIT 编译器的 AST 遍历器没实现 `Lambda` 这一种 AST 节点。加上它需要改两处代码。

---

## 1. 错误的完整链路

### 1.1 入口

```python
@triton.jit
def my_kernel(...):
    result = tl.associative_scan(val, axis=0, combine_fn=lambda a, b: a + b)
    #                                                         ^^^^^^^^^^^^^^^^
    #                                                         这个 lambda 触发了报错
```

### 1.2 发生了什么

```
1. @triton.jit 编译 my_kernel
   → triton/compiler/compiler.py:80  ast_to_ttir()
   → 遍历 my_kernel 的 Python AST
   → 遇到 tl.associative_scan(...) 调用
   → 遍历其参数, 包括 combine_fn=lambda a, b: a + b
   → 遇到 ast.Lambda 节点
   → code_generator.py:1552  generic_visit()
   → 抛出 "unsupported AST node type: Lambda"
```

### 1.3 即使过了 AST 遍历，还有第二关

`combine_fn` 最终被 `associative_scan` 这样使用（`core.py:2757`）：

```python
results = _generator.call_JitFunction(combine_fn, args, kwargs={})
```

`call_JitFunction` 要求 `combine_fn` 是 `@triton.jit` 生成的 `JitFunction` 对象——它有自己的 compiled kernel。lambda 是普通 Python 函数对象，类型不匹配。

---

## 2. 要改的两处代码

### 位置 1: AST 遍历器 — `code_generator.py`

文件: `triton/compiler/code_generator.py`

**问题行** (1552):
```python
def generic_visit(self, node):
    raise self._unsupported(node, "unsupported AST node type: {}".format(type(node).__name__))
```

**需要添加 `visit_Lambda`**，参考已有的 `visit_FunctionDef`:

```python
def visit_Lambda(self, node):
    # Lambda 本质上是一个匿名函数:
    #   lambda a, b: a + b
    # AST 结构:
    #   Lambda(args=arguments(args=[arg('a'), arg('b')]), body=BinOp(a, Add, b))
    #
    # 思路: 把 lambda body 当作一个内联的 triton function 来处理
    # 1. 创建一个匿名函数名 (e.g., "_lambda_0")
    # 2. 把 lambda args → 函数参数
    # 3. 把 lambda body → 函数体
    # 4. 注册到 code generator 的 function registry
    # 5. 返回对这个匿名函数的引用

    # 核心难点: Triton 的 compiler 假设所有函数都是顶层 @triton.jit 函数，
    # lambda 是嵌套在另一个函数内的，需要处理闭包/作用域。
    ...
```

**更简单的方案**：不去改通用的 AST 遍历器，而是在 `visit_Call` 里检测 lambda 参数并提前转换：

```python
def visit_Call(self, node):
    # 在 visit 之前, 检查参数里有没有 lambda
    for kw in node.keywords:
        if isinstance(kw.value, ast.Lambda):
            # 提前把 lambda 转成命名函数
            lambda_fn_name = self._register_lambda(kw.value)
            kw.value = ast.Name(id=lambda_fn_name)
    ...
```

### 位置 2: `associative_scan` / `reduce` — `core.py`

文件: `triton/language/core.py`

**问题行** (2757):
```python
results = _generator.call_JitFunction(combine_fn, args, kwargs={})
```

`combine_fn` 必须是 `JitFunction`。如果 lambda 已经被编译为匿名的 JitFunction，这里就能正常工作。否则需要加类型检查：

```python
if isinstance(combine_fn, types.LambdaType):
    combine_fn = triton.jit(combine_fn)  # 动态 jit
results = _generator.call_JitFunction(combine_fn, args, kwargs={})
```

---

## 3. 更深层的问题

即使加了 `visit_Lambda`，还有三个硬问题：

### 3.1 闭包捕获

Lambda 可能引用外层变量：

```python
scale = 2.0
result = tl.associative_scan(val, combine_fn=lambda a, b: a + b * scale)
#                                                              ^^^^^
#                                                              这个 scale 从哪里来？
```

Triton 的 JIT 编译发生在**函数级别**——每个 `@triton.jit` 函数被独立编译成独立的 kernel。lambda 如果引用外层变量，这些变量需要在编译期可解析为 `constexpr`。

### 3.2 类型推断

Triton 是强类型的（在 MLIR 层面）。lambda 的输入类型 `a, b` 需要在编译期确定。在 `associative_scan` 的上下文中，类型可以从 `val` 推导出来——但编译器需要知道这个上下文。

### 3.3 为什么 Triton 团队没做

- **投入产出比低**：lambda 的唯一使用场景是 `reduce`/`associative_scan` 的 `combine_fn`，写成 `@triton.jit` 具名函数只需要多 2 行代码
- **增加编译器复杂度**：闭包捕获、作用域分析、类型推导都是不小的工程
- **可替代方案**：内置 `tl.sum`、`tl.max` 等已经覆盖了 95% 的使用场景

---

## 4. 实用建议

### 不要等 Triton 支持，现在就这样写：

```python
# 替代方案 1: 定义具名 combine 函数
@triton.jit
def _add(a, b):
    return a + b

result = tl.associative_scan(val, axis=0, combine_fn=_add)

# 替代方案 2: 用内置操作
result = tl.associative_scan(val, axis=0, combine_fn=tl.add)  # 如果支持的话

# 替代方案 3: 用 PyTorch
result = torch.cumsum(x, dim=0)  # 3x faster, 已在 cuBLAS 优化
```

### 如果你真想改 Triton 源码练手

1. Fork `triton-lang/triton`
2. 从 `code_generator.py` 的 `visit_Call` 入手，检测 lambda 参数
3. 把 lambda 包装成匿名 `JitFunction`
4. 在 `core.py` 的 `associative_scan`/`reduce` 中接受非 JitFunction 的 callable
5. 写一个最简单的测试: `tl.associative_scan(v, combine_fn=lambda a,b: a+b)`

**预计工作量**: 熟悉 Triton 编译器架构 1 周 + 实现 2-3 天。这个过程中你会深入理解 Triton 的 AST→TTIR→TritonGPU IR 全链路——比写 10 个 kernel 都值。

---

## 5. 相关文件索引

| 文件 | 作用 |
|------|------|
| `triton/compiler/code_generator.py:1552` | `generic_visit` — 报错的地方 |
| `triton/compiler/code_generator.py:visit_FunctionDef` | 参考实现: 如何处理函数定义 |
| `triton/compiler/code_generator.py:visit_Call` | 参考实现: 如何处理函数调用 |
| `triton/language/core.py:2757` | `call_JitFunction` — 要求 JitFunction |
| `triton/language/core.py:2733` | `associative_scan` 入口 |
| `triton/compiler/compiler.py:80` | `ast_to_ttir` — AST→IR 入口 |
