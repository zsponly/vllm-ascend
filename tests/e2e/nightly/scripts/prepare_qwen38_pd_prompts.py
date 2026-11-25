"""Prepare full-length GSM8K splits for the A3 prefill and decode nightlies."""

import argparse
from itertools import cycle, repeat
from pathlib import Path

from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from tools.aisbench import maybe_download_from_modelscope


def prepare_prompts(mode: str) -> None:
    for name, dataset, prefill_count, decode_count in (
        ("128k", "vllm-ascend/GSM8K-in131072-bs100-qwen3", 352, 512),
        ("16k", "vllm-ascend/GSM8K_prefix0_in16384_bs200_qwen", 300, 176),
    ):
        source = Path(maybe_download_from_modelscope(dataset, repo_type="dataset")) / "test.jsonl"
        output = Path(get_data_path(".")) / f"qwen38_a3_{mode}" / name
        output.mkdir(parents=True, exist_ok=True)
        (output / "train.jsonl").write_bytes(b"")
        with source.open("rb") as src, (output / "test.jsonl").open("wb") as dst:
            # Decode reuses a single warmed prefix. Prefill cycles through the source
            # split with prefix caching disabled to reach the measured request count.
            rows = (row.rstrip(b"\r\n") + b"\n" for row in src)
            if mode == "decode":
                prompts, count = repeat(next(rows)), decode_count
            else:
                prompts, count = cycle(rows), prefill_count
            for _ in range(count):
                dst.write(next(prompts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prefill", "decode"))
    prepare_prompts(parser.parse_args().mode)
