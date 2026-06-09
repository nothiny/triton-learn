# 17 — Triton 反向传播：写 Backward Pass 和 Autograd 集成

> 写 forward kernel 只完成了一半。真正训练需要 backward pass。这篇讲如何写 Triton kernel 的反向传播，以及如何和 PyTorch autograd 集成。

---

## 1. 基础：Autograd 的工作原理

### 1.1 Forward 和 Backward 的关系

```
Forward:  给定输入 x, w，计算输出 y
Backward: 给定 ∂L/∂y（损失对输出的梯度），计算 ∂L/∂x 和 ∂L/∂w

链式法则:
  ∂L/∂x = ∂L/∂y · ∂y/∂x
  ∂L/∂w = ∂L/∂y · ∂y/∂w

你不需要知道 L 是什么 — 只需要知道 ∂L/∂y（由下游传过来）
```

### 1.2 PyTorch 的 `torch.autograd.Function`

```python
import torch

class MyLinearFunction(torch.autograd.Function):
    """
    自定义 autograd 函数。
    
    三个必须实现的方法:
    - forward: 前向计算
    - setup_context: 保存反向传播需要的信息
    - backward: 反向计算
    """
    
    @staticmethod
    def forward(ctx, x, w, bias):
        # ctx: context，保存 backward 需要的信息
        output = x @ w.T + bias
        ctx.save_for_backward(x, w, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: ∂L/∂y，下游传过来的梯度
        x, w, bias = ctx.saved_tensors
        
        grad_x = grad_output @ w        # ∂L/∂x = ∂L/∂y @ W
        grad_w = grad_output.T @ x      # ∂L/∂w = (∂L/∂y)^T @ X
        grad_bias = grad_output.sum(0)  # ∂L/∂b = sum(∂L/∂y, dim=0)
        
        return grad_x, grad_w, grad_bias
```

---

## 2. Triton + Autograd 集成

### 2.1 模式：Triton 写计算，PyTorch Autograd 管框架

```python
class TritonMatmulFunction(torch.autograd.Function):
    """
    用 Triton kernel 做 forward 和 backward 的 matmul
    """
    
    @staticmethod
    def forward(ctx, a, b):
        # Triton forward
        c = matmul_tiled(a, b)
        ctx.save_for_backward(a, b)
        return c
    
    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        
        # grad_a = grad_c @ b.T
        grad_a = matmul_tiled(grad_c, b.T)
        
        # grad_b = a.T @ grad_c
        grad_b = matmul_tiled(a.T, grad_c)
        
        return grad_a, grad_b
```

### 2.2 在 Triton kernel 中实现 backward

```python
# 完整示例: Fused Linear + ReLU 的 forward + backward

@triton.jit
def linear_relu_fwd_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    # ... strides, BLOCK sizes
):
    """
    Forward: out = ReLU(x @ w.T + bias)
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    
    # GEMM
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + 
                     (k + offs_k)[None, :] * stride_xk, ...)
        b = tl.load(w_ptr + offs_n[None, :] * stride_wn +
                     (k + offs_k)[:, None] * stride_wk, ...)
        acc += tl.dot(a, b)
    
    # + bias + ReLU
    result = tl.maximum(acc + bias[None, :], 0.0)
    
    tl.store(out_ptr + ..., result, ...)


@triton.jit
def linear_relu_bwd_kernel(
    x_ptr, w_ptr, grad_out_ptr,  # forward inputs + upstream gradient
    grad_x_ptr, grad_w_ptr,      # output gradients
    M, N, K,
    # strides, BLOCK sizes
):
    """
    Backward: 
      grad_x = (grad_out ⊙ (x @ w.T + b > 0)) @ w
      grad_w = (grad_out ⊙ (x @ w.T + b > 0)).T @ x
    
    具体推导:
      y = ReLU(x @ w.T + b)      前向
      ∂L/∂y = grad_out           上游梯度
      
      设 z = x @ w.T + b
      则 ∂y/∂z = (z > 0)         ReLU 的导数
      
      ∂L/∂z = grad_out ⊙ (z > 0)   (elementwise)
      ∂L/∂x = ∂L/∂z @ w           (matmul)
      ∂L/∂w = ∂L/∂z.T @ x         (matmul)
    """
    pid = tl.program_id(0)
    # ... (实现 matmul 的 backward)
    # 关键: grad_out ⊙ (z > 0) 和 forward 的 z 依赖
```

---

## 3. 常见 Kernel 的 Backward 推导

### 3.1 Elementwise

```python
# Forward: y = op(x)
# Backward: grad_x = grad_y * op'(x)

# 常见 elementwise op 的导数:

# ReLU: y = max(0, x)
#   dy/dx = 1 if x > 0 else 0

# GELU: 太复杂，通常用近似
#   dy/dx = Φ(x) + x * φ(x)  (Φ=CDF, φ=PDF of standard normal)

# SiLU: y = x * sigmoid(x)
#   dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

### 3.2 Softmax

```python
# Forward: y_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
# Backward (标准版本):
#   dy_i/dx_j = y_i * (δ_ij - y_j)
#   其中 δ_ij = 1 if i==j else 0
#
# 向量形式:
#   grad_x = y * (grad_y - sum(y * grad_y))
#   其中 * 是 elementwise multiply

# Flash Attention 的 backward:
#   见 Flash Attention 论文 Algorithm 2
#   比 forward 复杂得多（需要 recompute softmax statistics）
```

### 3.3 LayerNorm

```python
# Forward: y = γ * (x - μ) / √(σ² + ε) + β
# 其中 μ = mean(x), σ² = var(x)
#
# Backward:
#   推导较复杂，通常用 PyTorch 的公式
#   或者直接用 torch.autograd.gradcheck 验证你手写的 backward

# 核心: LayerNorm 的 backward 涉及两类参数:
#   grad_x: 对输入的梯度（传给上一层）
#   grad_γ, grad_β: 对权重和偏置的梯度（用于更新参数）
```

### 3.4 MatMul

```python
# Forward: C = A @ B
# Backward:
#   grad_A = grad_C @ B.T     (grad_C: [M,N], B.T: [N,K] → grad_A: [M,K])
#   grad_B = A.T @ grad_C     (A.T: [K,M], grad_C: [M,N] → grad_B: [K,N])
#
# 注意: grad_A 和 grad_B 的类型与 A, B 相同
#   如果 forward 是 fp16, backward 的 grad 也是 fp16
#   但中间的 matmul 应该用 fp32 累加

# 在 Triton 中，通常用已有的 GEMM kernel 实现 backward:
# def backward(ctx, grad_C):
#     A, B = ctx.saved_tensors
#     grad_A = matmul_tiled(grad_C, B.T)
#     grad_B = matmul_tiled(A.T, grad_C)
#     return grad_A, grad_B
```

---

## 4. 数值验证：`torch.autograd.gradcheck`

### 4.1 用法

```python
import torch
from torch.autograd import gradcheck

# 验证你的 autograd Function 是否正确
def test_my_function():
    x = torch.randn(4, 8, device='cuda', dtype=torch.float64, requires_grad=True)
    w = torch.randn(16, 8, device='cuda', dtype=torch.float64, requires_grad=True)
    
    # gradcheck 用有限差分法验证你的 analytic backward
    # 对比: 数值梯度 (finite diff) ≈ 解析梯度 (你的 backward)
    test = gradcheck(MyLinearFunction.apply, (x, w), eps=1e-6, atol=1e-4)
    print(f"Gradient check passed: {test}")

# ⚠️ 注意:
# - 必须用 float64 (double) — 有限差分法对数值精度要求高
# - 你的 kernel 需要支持 float64 输入（或至少在 backward 中支持）
# - gradcheck 很慢（O(N) 次 backward 调用），只用于测试
```

### 4.2 常见的 gradcheck 失败原因

```
1. 忘记 bias 的梯度:
   forward: y = x @ w.T + bias
   backward 只返回了 grad_x, grad_w，忘了 grad_bias

2. ReLU/Dropout 的 mask 不一致:
   forward 用的 mask 和 backward 用的 mask 必须完全相同
   → 使用 ctx.save_for_backward 保存 mask

3. 数值稳定性:
   softmax 在 float64 下行，但在 float32 下行为不同
   → 检查 gradcheck 的 atol

4. Stride 错误:
   forward 中 stride 是 (N, 1)，backward 中 T 之后的 stride 变了
   → 仔细检查 .T 之后的 tensor 的 stride
```

---

## 5. 完整示例：Fused Linear + ReLU

```python
class FusedLinearReLU(torch.autograd.Function):
    """
    Forward: y = ReLU(x @ w.T + b)
    Backward: grad_x = (grad_y ⊙ mask) @ w
              grad_w = (grad_y ⊙ mask).T @ x
              grad_b = sum(grad_y ⊙ mask, dim=0)
    
    其中 mask = (x @ w.T + b > 0)
    """
    
    @staticmethod
    def forward(ctx, x, w, b):
        # x: [M, K], w: [N, K], b: [N]
        M, K = x.shape
        N, _ = w.shape
        
        # Forward: z = x @ w.T + b
        z = torch.empty(M, N, device=x.device, dtype=x.dtype)
        linear_fwd_kernel[grid](x, w, b, z, M, N, K, ...)
        
        # ReLU
        mask = z > 0
        y = torch.where(mask, z, torch.zeros_like(z))
        
        # 保存 backward 需要的信息
        ctx.save_for_backward(x, w, mask)  # mask 是关键！
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        x, w, mask = ctx.saved_tensors
        M, K = x.shape
        N = w.shape[0]
        
        # grad_z = grad_y ⊙ mask (ReLU 的导数)
        grad_z = grad_y * mask
        
        # grad_x = grad_z @ w
        grad_x = matmul_tiled(grad_z, w)  # [M, N] @ [N, K] → [M, K]
        
        # grad_w = grad_z.T @ x
        grad_w = matmul_tiled(grad_z.T, x)  # [N, M] @ [M, K] → [N, K]
        
        # grad_b = sum(grad_z, dim=0)
        grad_b = grad_z.sum(0)
        
        return grad_x, grad_w, grad_b
```

---

## 6. 性能考量

### 6.1 Backward 的 FLOPs

```
Backward 的计算量 ≈ Forward 的 2× (对于 GEMM 占主导的 kernel)

以 Linear 为例:
  Forward: y = x @ w.T         (2×M×N×K FLOPs)
  Backward: grad_x = grad_y @ w (2×M×N×K FLOPs)
            grad_w = grad_y.T @ x (2×M×N×K FLOPs)
  总共: Backward ≈ 2× Forward

以 Softmax 为例:
  Forward: O(N) FLOPs
  Backward: O(N²) FLOPs (需要完整的 attention matrix)
  → 这也是 Flash Attention 需要专门 backward 算法的原因
```

### 6.2 内存：Save for Backward

```
ctx.save_for_backward 保存的内容影响显存使用:

以 Linear + ReLU 为例:
  - x: [M, K] fp16 = M×K×2 bytes
  - w: [N, K] fp16 = N×K×2 bytes (如果在 backward 中还需要的話)
  - mask: [M, N] bool = M×N bytes

  对于 M=4096, N=4096, K=1024:
    x: 4096×1024×2 = 8.4 MB
    w: 4096×1024×2 = 8.4 MB
    mask: 4096×4096 = 16.8 MB
    总共: ~33.6 MB per layer

  对于 32 层: 33.6 × 32 = 1.1 GB — 不小但通常够

  Flash Attention 的 memory saving:
    标准 attention 需要保存 attention matrix (N²)
    Flash Attention 只保存 softmax statistics (O(N))
    → 显著减少 save_for_backward 的内存
```

---

## 7. 参考资料

- [PyTorch Custom Autograd Function](https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd)
- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [Flash Attention Backward (Dao et al.)](https://arxiv.org/abs/2205.14135) — Algorithm 2
