# Qwen3.8 Flash Next W4A8 A3 performance and accuracy measurements

These reference measurements were collected on Atlas A3 on 2026-11-26 with AISBench. They accompany [the Qwen3.8 Flash Next nightly proposal](https://github.com/vllm-project/vllm-ascend/pull/22291). The manual runs used their own request counts, rates, cache settings, and routing. CI regression thresholds require hardware runs with the nightly configuration.

## Test environment

- One prefill instance and one decode instance. Each role used four physical A3 cards (eight logical NPU devices). P used DP1/TP2/EP2/PP4 and D used DP4/TP2/EP8/PP1.
- A3 openEuler r136 image with CANN 9.3.0, vLLM `0.34.0+gc410543d9f`, and torch-npu `2.10.0.post7`.
- vLLM-Ascend build `4267a2e18` used main at `d86f630b2` with overlays from #21851, #20311, #21373, #22723, #22641, and #21627. The A3 Union GBSA operator package was loaded.
- Both roles used MTP3, a maximum model length of 143,360, 32 sequences per DP rank, and an 8,192-token batch budget. P used Fastokens 0.3.5. D used Static Kernel, SuperKernel, and `npugraph_ex`. KV cache pooling was off.
- Pure prefill went directly to P and generated one token per request. End-to-end and pure decode went through the PD proxy. Pure decode warmed the shared prompt with 16 requests before each measured round. D local prefix caching was off. The P cache setting appears in the workload table.

## Workloads and selected rounds

Each workload had three measured rounds. The tables show the middle round by mean TTFT for pure prefill and end-to-end, and by mean steady-stage TPOT for pure decode. Steady-stage QPS in ascending order breaks TPOT ties. The 128K end-to-end and 16K pure decode rows use the follow-up runs at RR 2.08 and concurrency 110, respectively. All selected rounds completed without request failures.

| Workload | Shared input prefix | P local cache | Round | Input tokens | Mean output tokens | Requests sent | Max concurrency | AISBench concurrency | Offered req/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K end-to-end | 90% | On | 2 | 131072 | 1024 | 396 | 99 | 43.1114 | 2.080000 |
| 128K pure P | 0% | Off | 3 | 131072 | 1 | 440 | 110 | 2.7977 | 0.239975 |
| 128K pure P, shared prefix | 90% | On | 3 | 131072 | 1 | 500 | 125 | 9.0678 | 2.064097 |
| 128K pure D | 100% | On | 3 | 131072 | 1019.4141 | 512 | 128 | 128.0000 | 0 |
| 16K end-to-end | 0% | On | 1 | 16384 | 1024 | 348 | 87 | 23.0179 | 1.534056 |
| 16K pure P | 0% | Off | 1 | 16384 | 1 | 376 | 94 | 3.8361 | 2.142476 |
| 16K pure D | 100% | On | 1 | 16384 | 1024 | 440 | 110 | 110.0000 | 0 |

An offered rate of zero means unpaced sending. The pure D input sharing ratio describes the dataset. The steady-stage concurrency column reports the AISBench steady-stage value. The full-run concurrency for the selected 128K pure D round was 114.3634.

## Latency and MTP acceptance

The pure D rows use AISBench `stable_stage --reuse`. The other rows use the AISBench full-run table. The TPOT percentile is the native AISBench P90. MTP acceptance is calculated from D-side counter deltas over the full round, including for pure D rows. Accepted tokens per step is the mean number of accepted draft tokens per speculative step.

| Workload | Mean TTFT (ms) | TTFT P90 (ms) | Mean TPOT (ms) | TPOT P90 (ms) | Mean ITL (ms) | Mean E2E latency (s) | MTP token acceptance | Accepted tokens per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K end-to-end | 5431.2 | 7120.2 | 18.4 | 25.1 | 50.7 | 24.2575 | 58.82% | 1.76 |
| 128K pure P | 11746.0 | 16786.1 | — | — | 0.1 | 11.7461 | — | — |
| 128K pure P, shared prefix | 4496.1 | 6049.7 | — | — | 0.1 | 4.4962 | — | — |
| 128K pure D | 17984.7 | 20124.1 | 18.6 | 18.3 | 67.0 | 36.8120 | 89.72% | 2.69 |
| 16K end-to-end | 1972.0 | 2051.6 | 14.0 | 18.8 | 42.9 | 16.2726 | 69.49% | 2.08 |
| 16K pure P | 1807.9 | 2082.8 | — | — | 0.1 | 1.8080 | — | — |
| 16K pure D | 1403.5 | 1704.2 | 19.5 | 20.9 | 68.1 | 21.3696 | 83.76% | 2.51 |

## Throughput

Token throughput is in tokens per second. The per-card columns use a four-physical-card divisor for all rows, matching the per-role normalization. The end-to-end totals cover the full eight-physical-card deployment. QPM is QPS multiplied by 60.

| Workload | Input tokens/s | Input per card | Output tokens/s | Output per card | Total tokens/s | Total per card | QPS | QPM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K end-to-end | 232946.11 | 58236.53 | 1819.89 | 454.97 | 234766.00 | 58691.50 | 1.7772 | 106.632 |
| 128K pure P | 31219.01 | 7804.75 | 0.2382 | 0.0595 | 31219.25 | 7804.81 | 0.2382 | 14.292 |
| 128K pure P, shared prefix | 264342.83 | 66085.71 | 2.0168 | 0.5042 | 264344.85 | 66086.21 | 2.0168 | 121.008 |
| 128K pure D | 474161.18 | 118540.29 | 3687.79 | 921.95 | 477848.97 | 119462.24 | 3.6176 | 217.056 |
| 16K end-to-end | 23175.47 | 5793.87 | 1448.47 | 362.12 | 24623.94 | 6155.98 | 1.4145 | 84.870 |
| 16K pure P | 34762.34 | 8690.59 | 2.1217 | 0.5304 | 34764.46 | 8691.12 | 2.1217 | 127.302 |
| 16K pure D | 103382.12 | 25845.53 | 6461.38 | 1615.35 | 109843.51 | 27460.88 | 6.3099 | 378.594 |

For reference, the full-run AISBench table for the selected 128K pure D round reported 512 requests, mean TTFT of 16,810.1 ms, mean TPOT of 18.6 ms, 3,276.9468 output tokens/s, and 3.2144 QPS. The steady-stage row above uses the 384-request analysis of that same run.

## GPQA Diamond accuracy

AISBench evaluated all 198 GPQA Diamond questions in each of three rounds through the A3 1P1D proxy. The evaluation used concurrency 32, P local prefix caching enabled, D local prefix caching disabled, and KV cache pooling disabled. Sampling used temperature 1.0, top-k 20, and top-p 0.95 with normal EOS handling.

| Evaluation | Selected round | Correct / total | Accuracy |
| --- | ---: | ---: | ---: |
| GPQA Diamond, full split | 3 (median of three) | 181 / 198 | 91.41% |

This measurement uses the disaggregated service. The nightly proposal specifies a colocated GPQA job for its own validation.
