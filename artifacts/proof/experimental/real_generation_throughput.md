# Real Generation Throughput Report

**Model:** Qwen/Qwen2.5-0.5B-Instruct  
**Seed:** 42  
**Configs:** baseline_fp16, k8_v5_gs64, k8_v5_gs32, turbo_polar, adaptive, experimental_hybrid

## baseline_fp16 @ 128 tokens

- **Tokens/sec:** 46.30
- **Total E2E ms:** 2764.35
- **Compression ratio:** 3145728.00x
- **Logit cosine vs FP16:** 1.0000
- **Top-5 overlap vs FP16:** 1.0000
- **KL vs FP16:** 0.000000

## baseline_fp16 @ 512 tokens

- **Tokens/sec:** 42.43
- **Total E2E ms:** 3016.73
- **Compression ratio:** 7864320.00x
- **Logit cosine vs FP16:** 1.0000
- **Top-5 overlap vs FP16:** 1.0000
- **KL vs FP16:** 0.000000

## baseline_fp16 @ 1024 tokens

- **Tokens/sec:** 8.22
- **Total E2E ms:** 15581.21
- **Compression ratio:** 14155776.00x
- **Logit cosine vs FP16:** 1.0000
- **Top-5 overlap vs FP16:** 1.0000
- **KL vs FP16:** 0.000000

## k8_v5_gs64 @ 128 tokens

- **Tokens/sec:** 40.78
- **Total E2E ms:** 3139.00
- **Compression ratio:** 4.46x
- **Logit cosine vs FP16:** 0.7938
- **Top-5 overlap vs FP16:** 0.3094
- **KL vs FP16:** 9.837869

## k8_v5_gs64 @ 512 tokens

- **Tokens/sec:** 32.25
- **Total E2E ms:** 3968.86
- **Compression ratio:** 2.79x
- **Logit cosine vs FP16:** 0.9998
- **Top-5 overlap vs FP16:** 0.9797
- **KL vs FP16:** 0.000034

## k8_v5_gs64 @ 1024 tokens

- **Tokens/sec:** 10.50
- **Total E2E ms:** 12194.52
- **Compression ratio:** 2.51x
- **Logit cosine vs FP16:** 0.9999
- **Top-5 overlap vs FP16:** 0.9812
- **KL vs FP16:** 0.000014

## k8_v5_gs32 @ 128 tokens

- **Tokens/sec:** 36.18
- **Total E2E ms:** 3537.58
- **Compression ratio:** 4.17x
- **Logit cosine vs FP16:** 0.7943
- **Top-5 overlap vs FP16:** 0.3109
- **KL vs FP16:** 9.821530

## k8_v5_gs32 @ 512 tokens

- **Tokens/sec:** 32.91
- **Total E2E ms:** 3888.92
- **Compression ratio:** 2.61x
- **Logit cosine vs FP16:** 0.9998
- **Top-5 overlap vs FP16:** 0.9844
- **KL vs FP16:** 0.000031

## k8_v5_gs32 @ 1024 tokens

- **Tokens/sec:** 8.51
- **Total E2E ms:** 15041.71
- **Compression ratio:** 2.35x
- **Logit cosine vs FP16:** 0.9999
- **Top-5 overlap vs FP16:** 0.9797
- **KL vs FP16:** 0.000013

## turbo_polar @ 128 tokens

- **Tokens/sec:** 34.22
- **Total E2E ms:** 3740.47
- **Compression ratio:** 4.63x
- **Logit cosine vs FP16:** 0.7372
- **Top-5 overlap vs FP16:** 0.1641
- **KL vs FP16:** 7.170052

## turbo_polar @ 512 tokens

- **Tokens/sec:** 33.27
- **Total E2E ms:** 3846.75
- **Compression ratio:** 2.89x
- **Logit cosine vs FP16:** 0.6933
- **Top-5 overlap vs FP16:** 0.1438
- **KL vs FP16:** 9.549608

## turbo_polar @ 1024 tokens

- **Tokens/sec:** 11.16
- **Total E2E ms:** 11469.91
- **Compression ratio:** 2.61x
- **Logit cosine vs FP16:** 0.9233
- **Top-5 overlap vs FP16:** 0.6391
- **KL vs FP16:** 0.021768

## adaptive @ 128 tokens

- **Tokens/sec:** 35.12
- **Total E2E ms:** 3644.71
- **Compression ratio:** 4.63x
- **Logit cosine vs FP16:** 0.6932
- **Top-5 overlap vs FP16:** 0.1078
- **KL vs FP16:** 10.799811

## adaptive @ 512 tokens

- **Tokens/sec:** 34.32
- **Total E2E ms:** 3729.41
- **Compression ratio:** 2.89x
- **Logit cosine vs FP16:** 0.8700
- **Top-5 overlap vs FP16:** 0.6141
- **KL vs FP16:** 0.038245

## adaptive @ 1024 tokens

- **Tokens/sec:** 11.51
- **Total E2E ms:** 11120.14
- **Compression ratio:** 2.61x
- **Logit cosine vs FP16:** 0.9241
- **Top-5 overlap vs FP16:** 0.6375
- **KL vs FP16:** 0.021542

## experimental_hybrid @ 128 tokens

- **Tokens/sec:** 42.69
- **Total E2E ms:** 2998.52
- **Compression ratio:** 5.91x
- **Logit cosine vs FP16:** 0.9455
- **Top-5 overlap vs FP16:** 0.6766
- **KL vs FP16:** 0.029672

## experimental_hybrid @ 512 tokens

- **Tokens/sec:** 35.23
- **Total E2E ms:** 3633.40
- **Compression ratio:** 3.72x
- **Logit cosine vs FP16:** 0.7518
- **Top-5 overlap vs FP16:** 0.2953
- **KL vs FP16:** 8.429742

## experimental_hybrid @ 1024 tokens

- **Tokens/sec:** 8.72
- **Total E2E ms:** 14679.08
- **Compression ratio:** 3.35x
- **Logit cosine vs FP16:** 0.9657
- **Top-5 overlap vs FP16:** 0.7438
- **KL vs FP16:** 0.011899

