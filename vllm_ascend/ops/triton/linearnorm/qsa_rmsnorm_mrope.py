import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def _validate_rope_positions(
    positions: torch.Tensor,
    cache: torch.Tensor,
    tensor: torch.Tensor,
    rope_dim: int,
) -> None:
    """Check the direct-gather ABI using metadata only (safe during capture)."""
    if positions.ndim not in (1, 2) or (positions.ndim == 2 and positions.shape[0] != 3):
        raise ValueError("QSA positions must have one or three axes")
    if positions.shape[-1] != tensor.shape[0] or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA positions must be integer positions for every token")
    if cache.ndim != 2 or cache.shape[0] == 0 or cache.shape[1] != rope_dim or cache.stride(1) != 1:
        raise ValueError("QSA rotary cache must have nonempty, contiguous rotary rows")
    if tensor.dtype != torch.bfloat16 or cache.dtype != tensor.dtype:
        raise ValueError("QSA direct rotary gathering requires a BF16 cache and input")
    if positions.device != tensor.device or cache.device != tensor.device:
        raise ValueError("QSA positions, rotary cache and input must share a device")


@triton.jit(do_not_specialize=["position_stride_axis", "position_stride_token", "cache_rows"])
def _qsa_rmsnorm_mrope_kernel(
    input_ptr,
    weight_ptr,
    cos_sin_ptr,
    output_ptr,
    input_stride_token,
    input_stride_head,
    cos_sin_stride_axis,
    cos_sin_stride_token,
    output_stride_token,
    output_stride_head,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EPS: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    positions_ptr,
    position_stride_axis,
    position_stride_token,
    cache_rows,
    GATHER_CACHE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_DIM)
    mask = offsets < HEAD_DIM
    input_row = input_ptr + token_id * input_stride_token + head_id * input_stride_head
    values = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32) + 1.0
    variance = tl.sum(values * values, axis=0) / HEAD_DIM
    reciprocal_std = tl.rsqrt(variance + EPS)
    normalized = (values * reciprocal_std * weight).to(tl.bfloat16)

    HALF_ROPE_DIM: tl.constexpr = ROPE_DIM // 2
    rotary_offsets = tl.arange(0, HALF_ROPE_DIM)
    if GATHER_CACHE:
        if HAS_MROPE:
            # Pair indices cycle through t/h/w (0/1/2); each section count
            # becomes an interleaved index cutoff after multiplying by 3.
            h_mask = (rotary_offsets % 3 == 1) & (rotary_offsets < MROPE_SECTION_H * 3)
            w_mask = (rotary_offsets % 3 == 2) & (rotary_offsets < MROPE_SECTION_W * 3)
            axis = tl.where(h_mask, 1, tl.where(w_mask, 2, 0))
            position = tl.load(positions_ptr + axis * position_stride_axis + token_id * position_stride_token)
        else:
            # Keep the shared position scalar: an all-zero axis vector creates
            # a repeated-address gather unsupported by Triton Ascend lowering.
            position = tl.load(positions_ptr + token_id * position_stride_token)
        position = tl.minimum(tl.maximum(position, 0), cache_rows - 1)
        cache_row = cos_sin_ptr + position.to(tl.int64) * cos_sin_stride_token
        cos = tl.load(cache_row + rotary_offsets).to(tl.float32)
        sin = tl.load(cache_row + rotary_offsets + HALF_ROPE_DIM).to(tl.float32)
    elif HAS_MROPE:
        h_mask = (rotary_offsets % 3 == 1) & (rotary_offsets < MROPE_SECTION_H * 3)
        w_mask = (rotary_offsets % 3 == 2) & (rotary_offsets < MROPE_SECTION_W * 3)
        cache_row = cos_sin_ptr + token_id * cos_sin_stride_token
        cos_t = tl.load(cache_row + rotary_offsets).to(tl.float32)
        cos_h = tl.load(cache_row + cos_sin_stride_axis + rotary_offsets).to(tl.float32)
        cos_w = tl.load(cache_row + 2 * cos_sin_stride_axis + rotary_offsets).to(tl.float32)
        sin_t = tl.load(cache_row + rotary_offsets + HALF_ROPE_DIM).to(tl.float32)
        sin_h = tl.load(cache_row + cos_sin_stride_axis + rotary_offsets + HALF_ROPE_DIM).to(tl.float32)
        sin_w = tl.load(cache_row + 2 * cos_sin_stride_axis + rotary_offsets + HALF_ROPE_DIM).to(tl.float32)
        cos = tl.where(h_mask, cos_h, tl.where(w_mask, cos_w, cos_t))
        sin = tl.where(h_mask, sin_h, tl.where(w_mask, sin_w, sin_t))
    else:
        cache_row = cos_sin_ptr + token_id * cos_sin_stride_token
        cos = tl.load(cache_row + rotary_offsets).to(tl.float32)
        sin = tl.load(cache_row + rotary_offsets + HALF_ROPE_DIM).to(tl.float32)
    first = tl.load(input_row + rotary_offsets).to(tl.float32)
    first_weight = tl.load(weight_ptr + rotary_offsets).to(tl.float32) + 1.0
    first = (first * reciprocal_std * first_weight).to(tl.bfloat16)
    second_offsets = rotary_offsets + HALF_ROPE_DIM
    second = tl.load(input_row + second_offsets).to(tl.float32)
    second_weight = tl.load(weight_ptr + second_offsets).to(tl.float32) + 1.0
    second = (second * reciprocal_std * second_weight).to(tl.bfloat16)
    first_cos = (first * cos).to(tl.bfloat16)
    second_sin = (second * sin).to(tl.bfloat16)
    second_cos = (second * cos).to(tl.bfloat16)
    first_sin = (first * sin).to(tl.bfloat16)
    rotated_first = (first_cos - second_sin).to(tl.bfloat16)
    rotated_second = (second_cos + first_sin).to(tl.bfloat16)

    output_row = output_ptr + token_id * output_stride_token + head_id * output_stride_head
    tl.store(output_row + offsets, normalized, mask=mask)
    tl.store(output_row + rotary_offsets, rotated_first)
    tl.store(output_row + rotary_offsets + HALF_ROPE_DIM, rotated_second)


def triton_qsa_rmsnorm_mrope(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    rope_dim: int,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize and rotate, using a full cache when positions are supplied."""
    if tensor.ndim != 3:
        raise ValueError("QSA norm+RoPE input must be [tokens, heads, head_dim]")
    if tensor.dtype != torch.bfloat16 or weight.dtype != tensor.dtype:
        raise ValueError("QSA norm+RoPE requires BF16 input and weight")
    if tensor.stride(-1) != 1 or weight.ndim != 1 or weight.shape[0] != tensor.shape[-1]:
        raise ValueError("QSA norm+RoPE requires contiguous head rows and matching weight")
    if not weight.is_contiguous():
        raise ValueError("QSA norm+RoPE requires contiguous weight")
    if cos_sin.ndim not in (2, 3) or cos_sin.shape[-1] != rope_dim:
        raise ValueError("QSA norm+RoPE received an invalid rotary cache view")
    if rope_dim <= 0 or rope_dim > tensor.shape[-1] or rope_dim % 2:
        raise ValueError("QSA norm+RoPE requires an even rotary width")
    if positions is None and cos_sin.shape[-2] != tensor.shape[0]:
        raise ValueError("QSA norm+RoPE token and rotary rows must match")

    if positions is not None:
        _validate_rope_positions(positions, cos_sin, tensor, rope_dim)
    has_mrope = cos_sin.ndim == 3 if positions is None else positions.ndim == 2
    if has_mrope:
        if (positions is None and cos_sin.shape[0] != 3) or len(mrope_section) != 3:
            raise ValueError("QSA MRoPE requires three position axes and sections")
        cos_sin_stride_axis = cos_sin.stride(0)
    else:
        cos_sin_stride_axis = 0

    output = torch.empty_like(tensor)
    num_tokens, num_heads, head_dim = tensor.shape
    if not num_tokens:
        return output
    _qsa_rmsnorm_mrope_kernel[(num_tokens, num_heads)](
        tensor,
        weight,
        cos_sin,
        output,
        tensor.stride(0),
        tensor.stride(1),
        cos_sin_stride_axis,
        cos_sin.stride(-2),
        output.stride(0),
        output.stride(1),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        EPS=eps,
        MROPE_SECTION_H=mrope_section[1] if has_mrope else 0,
        MROPE_SECTION_W=mrope_section[2] if has_mrope else 0,
        ROPE_DIM=rope_dim,
        HAS_MROPE=has_mrope,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        positions_ptr=positions,
        position_stride_axis=positions.stride(0) if positions is not None and positions.ndim == 2 else 0,
        position_stride_token=positions.stride(-1) if positions is not None else 0,
        cache_rows=cos_sin.shape[0],
        GATHER_CACHE=positions is not None,
        num_warps=1,
        num_stages=1,
    )
    return output


def triton_qsa_rmsnorm_mrope_fake(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    rope_dim: int,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(tensor)


direct_register_custom_op(
    op_name="triton_qsa_rmsnorm_mrope",
    op_func=triton_qsa_rmsnorm_mrope,
    fake_impl=triton_qsa_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
