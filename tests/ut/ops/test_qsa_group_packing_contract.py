# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the pack wrapper contract, without executing device kernels."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def packing():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/qwen4_exp/qsa.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pack_qsa_group_indices_npu")
    launch = MagicMock()
    namespace = {"torch": torch, "HAS_TRITON": True, "_pack_qsa_group_indices_kernel": launch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name], launch


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_sequence_length_layout_contract(packing, dtype, strided):
    pack, launch = packing
    storage = torch.tensor([4, 0, 8, 0], dtype=dtype)
    lengths = storage[::2] if strided else storage[1:3]
    # Only the blocks' device metadata is mocked; lengths are real tensor views.
    blocks = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(2, 2), stride=lambda dim: (2, 1)[dim])
    positions = torch.tensor([3, 7], dtype=torch.int32)
    owners = torch.tensor([0, 1], dtype=torch.int32)
    output = torch.empty(2, 5, dtype=torch.int32)
    if strided:
        with pytest.raises(ValueError, match="requires contiguous sequence lengths"):
            pack(blocks, positions, lengths, owners, 4, 8, output)
        launch.__getitem__.assert_not_called()
    else:
        assert lengths.is_contiguous() and lengths.storage_offset() == 1
        assert pack(blocks, positions, lengths, owners, 4, 8, output) is output
        launch.__getitem__.assert_called_once_with((2,))
        launch.__getitem__.return_value.assert_called_once()
        assert launch.__getitem__.return_value.call_args.args[2] is lengths
