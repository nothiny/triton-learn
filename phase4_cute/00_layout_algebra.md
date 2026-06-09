# CuTe Layout Algebra — Notes

> CuTe 的核心抽象: Layout = Shape × Stride
> 这是 CuTe 的"代数"——所有操作都可以用 Layout 操作来描述。

## 基础定义

```cpp
// Layout: 从 N-D 逻辑坐标 → 1-D 物理偏移的映射
struct Layout {
    Shape  shape;   // 逻辑形状，如 (M, N, K)
    Stride stride;  // 步长，如 (1, M, M*N) for column-major
};

// 坐标 (i,j,k) 映射到的内存偏移:
// offset = i*stride[0] + j*stride[1] + k*stride[2]
```

## Layout 操作

- `composition(layout_a, layout_b)`: 嵌套索引
- `complement(layout, target_shape)`: 填充到目标形状
- `right_inverse(layout)`: 从偏移反查坐标
- `product(layout_a, layout_b)`: 笛卡尔积

> 🔧 **Compiler Perspective**: Layout 代数本质上是仿射映射的组合和求逆，
> 类似 polyhedral 编译器中使用的 Presburger 算术。

## 后续填充

请参考 `notes/04_cute_preview.md` 了解完整预览。
