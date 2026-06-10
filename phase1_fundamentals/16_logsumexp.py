"""
16_logsumexp.py — Row-Wise LogSumExp kernel

学习目标:
  - 掌握 LogSumExp 的数值稳定实现 (max-subtraction trick)
  - 理解 LogSumExp 和 Softmax 的关系: softmax = exp(x - LSE(x))
  - 学习行级 reduction + elementwise 的组合模式

数学公式:
  LogSumExp(x_row) = log(sum(exp(x_i)))

  数值稳定版:
    m = max(x_row)
    LSE = m + log(sum(exp(x_i - m)))

为什么重要:
  - Softmax 分母的对数 = LogSumExp (log_softmax = x - LSE)
  - Cross Entropy = -x_target + LSE (见 18_cross_entropy.py)
  - 比直接 log(sum(exp(x))) 数值更稳定

运行: python phase1_fundamentals/16_logsumexp.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def logsumexp_kernel(x_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    逐行 LogSumExp: output[row] = log(sum(exp(x[row, :]))).
    每个 program 处理一行.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Pass 1: 找 max
    row_max = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        row_max = tl.maximum(row_max, x)
    global_max = tl.max(row_max, axis=0)

    # Pass 2: sum(exp(x - max))
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        sum_exp += tl.exp(x - global_max)
    global_sum = tl.sum(sum_exp, axis=0)

    # LogSumExp = max + log(sum(exp(x - max)))
    result = global_max + tl.log(global_sum)
    tl.store(output_ptr + row_idx, result)


def logsumexp_row(x: torch.Tensor) -> torch.Tensor:
    """逐行 LogSumExp, 返回 (N_ROWS,)."""
    n_rows, n_cols = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    grid = (n_rows,)
    logsumexp_kernel[grid](x, output, n_cols, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("16_logsumexp — Row-Wise LogSumExp")
    print("=" * 60)
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)
    for name, shape in [("small", (64, 128)), ("medium", (256, 1024)), ("large", (1024, 4096))]:
        x = torch.randn(*shape, device="cuda")
        out_triton = logsumexp_row(x)
        out_torch = torch.logsumexp(x, dim=-1)
        max_diff = (out_triton - out_torch).abs().max().item()
        print(f"  [{name}] shape={shape} max_diff={max_diff:.2e} {'✅' if max_diff<1e-4 else '❌'}")
    print("\n--- Performance ---")
    x = torch.randn(1024, 8192, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({"Triton (ours)": lambda: logsumexp_row(x), "PyTorch (ref)": lambda: torch.logsumexp(x, dim=-1)}, flops=n*5, bytes_accessed=n*4, dtype="fp32")
    print_compare_report(result)

# PERFORMANCE NOTES
# =================
# - LogSumExp 是 Softmax 的"对数值版本":
#   softmax(x) = exp(x - LSE(x)), log_softmax(x) = x - LSE(x)
# - max-subtraction trick 防止 exp 溢出 (exp(100) 会爆 fp32)
# - 2-pass: max → sum_exp (和 softmax 相同), 最后加一个 log 操作

if __name__ == "__main__": main()
