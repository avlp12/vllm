# Spark kpool prefill candidate

I keep runtime compatibility with 487ecf187 and retain the existing tail-pool reservation correction. VLLM_GLM_KPOOL_ROW_SPLIT=1 is an experimental, default-off query-row split for a single eager FP8 kpool prefill. All ranks vote on the setting. Mixed/decode batches, multiple prefills, DCP, FP4, short or uneven row partitions use the original path.

The mechanism is informed by [RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE) and [issue53691](https://github.com/vllm-project/vllm/issues/53691). I implemented a narrower admission rule after that issue reported concurrent correctness failures.

Validation: 25 host tests pass. The first four-GPU test failed exact selected-index equality. The candidate is NOT admitted for model serving and remains off. A second four-GPU diagnostic found random score tensors bit-exact and sorted selected sets equal, but native full-vs-full ordering was already nondeterministic. Tied-score inputs produced different selected sets on every rank. I do not treat ordering noise as an excuse to ignore those set differences. No serving speedup is claimed.
