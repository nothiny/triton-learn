# Tests — 测试

## 运行方式

```bash
# 全部测试
pytest tests/ -v

# 只跑 GPU 测试
pytest tests/ -v -m gpu

# 跳过慢测试
pytest tests/ -v -m "not slow"

# 使用 Makefile
make test        # GPU 测试
make test-all    # 全部测试
```

## 文件

| 文件 | 测试内容 | 需要 GPU |
|------|----------|----------|
| `conftest.py` | Pytest 配置、fixtures、skip 逻辑 | — |
| `test_phase1.py` | vector_add, softmax, relu, layernorm | ✅ |
| `test_phase2.py` | matmul_naive, matmul_tiled, flash_attn_v1 | ✅ |

## 添加新测试

```python
# 在 test_phase1.py 或 test_phase2.py 中添加
import importlib

def _load(module_path, func_name):
    spec = importlib.util.spec_from_file_location(...)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)

class TestMyNewKernel:
    def test_correctness(self):
        my_fn = _load("phase2_compute/my_kernel", "my_fn")
        # ... test ...

    @pytest.mark.slow
    def test_large(self):
        # 大尺寸测试（慢）
        ...
```

## 注意事项

- 模块名以数字开头（如 `01_vector_add.py`），不能直接 `import`，需要用 `importlib`
- GPU 不可用时，所有测试自动 skip
- LayerNorm 测试当前 `xfail`（简化版已知数值误差）
