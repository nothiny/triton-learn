# Custom MLIR Pass — 参考

## 当前状态

自定义 pass 的教学内容已整合到 `13_custom_pass.py` 中。
Python pass API 在 Triton 3.x 中仍在发展，最稳定的分析方法是 Python AST 分析。

## 学习路径

1. 运行 `python phase4_compiler/13_custom_pass.py` 学习 AST 分析工具
2. 阅读 Triton 源码中的 pass 定义: `triton/lib/Conversion/`, `triton/lib/Dialect/`
3. 深入了解: [MLIR Pass 文档](https://mlir.llvm.org/docs/PassManagement/)

## References

- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [Triton MLIR Dialect](https://github.com/triton-lang/triton/tree/main/lib/Dialect)
- [Writing a New MLIR Pass](https://mlir.llvm.org/docs/Tutorials/QuickStartRewrites/)
