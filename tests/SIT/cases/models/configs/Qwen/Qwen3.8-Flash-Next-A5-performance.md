# Qwen3.8 Flash Next A5 performance measurements

These AISBench measurements were collected on November 26–27, 2026, using the
Qwen3.8 Flash Next W4A8 MXFP checkpoint on one eight-device Ascend A5 server.
They provide measured operating points for the
[A5 prefill](Qwen3.8-Flash-Next-w4a8-mxfp-PD-Prefill-A5.yaml) and
[A5 decode](Qwen3.8-Flash-Next-w4a8-mxfp-PD-Decode-A5.yaml) nightly cases.

## Test environment

### Runtime

The November 26–27 performance runs used basic 1P1D, with four A5 devices per
role and DP2/TP2/EP4/PP1 on each side. Tiered KV pooling was disabled.

| Component | Version or configuration |
| --- | --- |
| Container base | A5 openEuler r177, CANN 9.3.0, shared by P, D, and proxy |
| vLLM | v0.34.0, commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` |
| vLLM-Ascend | Wheel built from `4f49ae332de1b1e572587bb9162cfb991eadf2ed`, with no separately applied PRs |
| Python and NPU stack | Python 3.12.13, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post7, Triton Ascend 3.6.0.post2 |
| Other packages | Fastokens 0.3.5, MemCache Hybrid and MemFabric Hybrid 1.5.0.dev20261123 |
| Benchmark client | AISBench 3.1.20261013 with fixed-interval pacing when `burstiness=0` |

The runtime wheel installation had two local adjustments. The Static Kernel
blacklist excluded Slice, and the PyTorch profiler retained CANN analysis output
for separate profiling runs. AISBench used its native metric calculations.
Warmup and profiling requests were excluded from the scored rounds.

### Prefill and decode launch settings

Both roles set `--max-model-len 143360`, `--max-num-seqs 64`,
`--max-num-batched-tokens 8192`, and `--gpu-memory-utilization 0.9`.
They enabled Ascend quantization, async scheduling, expert parallelism, and
`enable_flashcomm1=true`. Mamba cache used `align` mode with `bfloat16` SSM
state. Both roles used `qwen3_5_mtp` with three speculative tokens.

| Setting | Prefill (P) | Decode (D) |
| --- | --- | --- |
| Visible device IDs | 0–3 | 4–7 |
| HTTP port | 18347 | 18348 |
| DP RPC port | 19345 | 19346 |
| Distributed master port | 29450 | 29460 |
| Local prefix cache | Off for prefill-only without a shared prefix. On for the 90% single-prefix case, end-to-end, and decode-only. | Off |
| Fastokens | Enabled | Disabled |
| Graph mode | `NONE` | `FULL_DECODE_ONLY` |
| Ascend compilation settings | `enable_npugraph_ex=true`, Static Kernel off, SuperKernel off | `enable_npugraph_ex=true`, Static Kernel on with Slice excluded, SuperKernel off |
| API server count | 1 | 1 |
| Renderer workers | Default | `--renderer-num-workers 4` |
| MooncakeConnectorV2 | `kv_producer`, port 33470 | `kv_consumer`, port 33480 |
| `HCCL_NPU_SOCKET_PORT_RANGE` | 61400–61499 | 61500–61599 |
| `HCCL_IF_BASE_PORT` | 62000 | 63000 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 2400 | 4800 |

The launchers sourced the Ascend toolkit, Ascend NPU IR from the active toolkit
tree, and NNAL ATB with C++ ABI 1. Python and its package directories were
resolved from the interpreter in the container. The library path included the
installed PyTorch, torch-npu, MemCache, MemFabric, and Mooncake libraries. The
launchers also sourced vLLM-Ascend's installed custom-op environment when present.

The communication environment used `HCCL_OP_EXPANSION_MODE=AIV`,
`HCCL_BUFFSIZE=8192`, `HCCL_EXEC_TIMEOUT=204`, and `HCCL_CONNECT_TIMEOUT=120`.
Set the service IP and NIC for the target host using the existing
`VLLM_HOST_IP`, `HCCL_IF_IP`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and
`TP_SOCKET_IFNAME` settings. Both roles used the v2 model runner,
`TASK_QUEUE_ENABLE=1`, and `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.

### Proxy and benchmark routing

The 1P1D proxy used the same vLLM-Ascend source revision as the wheel. With
`ascend_repo` set to its checkout and `service_ip` set to the P/D service address,
the launcher used this port mapping:

```bash
python3 "$ascend_repo/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
    --host 0.0.0.0 --port 18349 \
    --prefiller-hosts "$service_ip" --prefiller-ports 18347 \
    --decoder-hosts "$service_ip" --decoder-ports 18348
```

Prefill-only requests went directly to P. End-to-end and decode-only requests
went through the proxy. Decode-only runs used warmed prompts with P-side prefix
caching. The 90% single-prefix cases used one shared prefix covering 90% of each
input. With a positive configured request rate, AISBench sent requests at fixed
intervals. A configured rate of zero means no pacing limit.

## Workloads and selected rounds

Each case ran three measured rounds with zero failed requests. The tables use one
representative round per case. For prefill and end-to-end, it is the middle round
when sorted by full-run mean TTFT. For decode, it is the middle round when sorted
by steady-state mean TPOT, with steady-state QPS ascending as the tie-breaker.

| ID | Workload | P prefix cache | Selected round | Mean input tokens | Mean output tokens | Requests issued | Max concurrency | Measured concurrency | Configured req/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 128K/1K end-to-end, 90% single prefix | On | 3 | 131072 | 1024 | 160 | 40 | 34.0422 | 4.966642 |
| B | 128K/1 prefill, no shared prefix | Off | 1 | 131084.1750 | 1 | 40 | 10 | 4.3138 | 0.341371 |
| C | 128K/1 prefill, 90% single prefix | On | 3 | 131072 | 1 | 128 | 32 | 17.9828 | 3.498213 |
| D | 128K/1K decode, 150-concurrency retest | On | 1 | 131072 | 1024 | 600 | 150 | 150 | 0 |
| E | 16K/1K end-to-end, no shared prefix | On | 2 | 16396.1558 | 1024 | 1252 | 313 | 26.1168 | 2.018750 |
| F | 16K/1 prefill, no shared prefix | Off | 2 | 16396.0898 | 1 | 256 | 64 | 4.6038 | 2.669779 |
| G | 16K/1K decode | On | 2 | 16396 | 1024 | 1000 | 250 | 250 | 0 |

The decode rows report the AISBench steady-state slices of 450 requests for D and
750 for G. Requests issued are full-run counts. The other rows use full-run metrics.

## Latency

| ID | Mean TTFT (ms) | TTFT P90 (ms) | Mean TPOT (ms) | TPOT P90 (ms) | Mean ITL (ms) | Mean E2E latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 6655.9 | 8105.4 | 6.8 | 7.8 | 27.7 | 13.6309 |
| B | 14718.0 | 21312.0 | — | — | 0.1 | 14.7181 |
| C | 6644.4 | 10617.9 | — | — | 0.0 | 6.6445 |
| D | 23783.6 | 25856.5 | 8.5 | 9.1 | 39.5 | 32.4380 |
| E | 1970.2 | 2071.6 | 11.0 | 17.0 | 22.9 | 13.1933 |
| F | 1742.7 | 1842.8 | — | — | 0.0 | 1.7428 |
| G | 13132.9 | 14046.2 | 13.0 | 13.3 | 47.4 | 26.3934 |

TPOT is undefined for the single-output-token cases. Their ITL values are included
as AISBench reports them.

## Throughput

Token throughputs are in tokens/s. Per-device throughput divides by eight for
end-to-end and by four for prefill-only and decode-only.

| ID | QPS | QPM | Input | Input/device | Output | Output/device | Total | Total/device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 2.4974 | 149.84 | 327343.7231 | 40917.9654 | 2557.3728 | 319.6716 | 329901.0960 | 41237.6370 |
| B | 0.2931 | 17.59 | 38420.1621 | 9605.0405 | 0.2931 | 0.0733 | 38420.4552 | 9605.1138 |
| C | 2.7064 | 162.38 | 354737.6095 | 88684.4024 | 2.7064 | 0.6766 | 354740.3159 | 88685.0790 |
| D | 4.6484 | 278.90 | 609271.0429 | 152317.7607 | 4759.9300 | 1189.9825 | 614030.9730 | 153507.7432 |
| E | 1.9796 | 118.78 | 32457.1314 | 4057.1414 | 2027.0668 | 253.3834 | 34484.1982 | 4310.5248 |
| F | 2.6416 | 158.50 | 43311.7351 | 10827.9338 | 2.6416 | 0.6604 | 43314.3767 | 10828.5942 |
| G | 10.0046 | 600.28 | 164035.1354 | 41008.7838 | 10244.6925 | 2561.1731 | 174279.8279 | 43569.9570 |

## Speculative acceptance

Acceptance uses the D-side Prometheus counter increases across the full
measurement round, summed over both DP ranks. The rate is accepted draft tokens
divided by proposed draft tokens.

| ID | Accepted draft tokens | Proposed draft tokens | Acceptance |
| --- | ---: | ---: | ---: |
| A | 117752 | 138939 | 84.75% |
| D | 446622 | 505746 | 88.31% |
| E | 666920 | 1849473 | 36.06% |
| G | 743697 | 844968 | 88.01% |

The A5 decode nightly YAML uses a fully repeated prompt. Benchmark comparisons
should match its workload, serving configuration, and measurement window.

## Historical GPQA Diamond accuracy

A separate A5 run on November 21, 2026, used 1P1D with tiered KV pooling.
The third round scored **91.41% (181/198)** on the full GPQA Diamond set, the
highest of three completed rounds. The other two rounds each scored 89.90%
(178/198). Sampling used `do_sample=true`, temperature 1.0, top-k 20, and top-p 0.95.

This measurement applies to the pooled deployment used in that run.
