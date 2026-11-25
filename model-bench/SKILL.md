---
name: model-bench
description: 对已运行的 SGLang 文本生成服务执行端到端性能压测。用于固定输入输出 token 长度的并发扫描、单波或稳态吞吐对比、TTFT/TPOT/ITL/E2E 延迟和 SLO 判定，输出逐档原始数据、日志和汇总 CSV。
---

# SGLang 模型服务压测

1. 确认目标服务地址、模型与 tokenizer、并发范围及对比目标。先检查服务健康、模型身份、启动参数和真实流式短请求。HCU 上记录容器、卡号及占用、SGLang/插件提交、TP/DP/PP/EP/ACP、模型精度与 KV dtype、投机解码参数、上下文上限、`max_running_requests` 和 `max_total_num_tokens`。服务配置以运行参数和对应版本的 server info 为准。测试范围仅覆盖已授权的目标服务。
2. 在负责发请求的机器上运行官方 benchmark 客户端。优先用 `python -m sglang.benchmark.serving --help` 检查该安装；老版本可用 `python -m sglang.bench_serving --help`。记录客户端版本、所在节点、到服务的网络路径、随机种子、流式模式、缓存口径和 EOS 策略。默认沿用当前服务配置和缓存状态；仅在明确要求、服务归属已确认时重启或清缓存。共享服务上避免并发运行其他压测。
3. 先用小输入/输出和并发 1 验证 CLI、token 计数、日志及结果。按目标选择下述模式，再逐档执行。遇到请求失败、结果缺失、超时或实际 token 预算不符，保留现场并停止扩大负载。已有占卡进程交接和服务部署属于对应部署流程。
4. 比较同模式、同配置和同负载。报告服务吞吐、首 token 延迟（TTFT）、首 token 后每 token 延迟（TPOT）、流式 chunk 间延迟（ITL）与端到端延迟（E2E）。均值与 P95/P99 分开；单波的小样本尾分位仅作描述。若用户给了 SLO，逐档核对对应分位阈值及完整成功数，报告符合 SLO 的已测最高并发和对应吞吐。保留全部失败档、可用样本数和缺失字段；结果默认留在测试机器。

## 固定长度并发扫描

固定长度预设 `4k_1k`：4096 输入、1024 输出、`random-ids`、`random_range_ratio=1`、`request_rate=inf`。单波预设使用 `num_prompts=max_concurrency`，并发为 1–8、10–32 偶数档和 40。参考服务是 Qwen3.8-27B-FP8、TP2/DP1、FP8 KV、投机解码 3/1/4。服务配置是历史复现条件，实际服务按任务选择。

脚本通过 `--tokenize-prompt` 直接发送 token ID，保留流式和官方客户端的默认忽略 EOS 行为。每档用基础种子加并发值生成数据，减少跨档复用同批提示词。参考表未记录原始种子及请求编码方式，比较时以重跑的数据为准。

- **单波（`burst`）**：每档一次性发出一波请求，`num_prompts=max_concurrency`。用于复现参考表，不把整轮吞吐或 1 秒窗口峰值当成稳态吞吐。
- **稳态（`steady`）**：默认每档请求数是并发上限的 5 倍，及时补充完成的请求。延长轮次、重复测量，检查启动和排空对整轮指标的影响。若需要生产到达速率、真实文本、多轮或多模态，按安装版本用官方 CLI 选择对应 dataset 和到达模型，另记工作负载及参数。

在验证过的客户端环境中运行（示例先测 1、2、4 档）：

```bash
python scripts/model_bench.py --base-url http://127.0.0.1:30000 \
  --model /path/to/model --tokenizer /path/to/tokenizer \
  --mode burst --concurrency 1,2,4 --output-dir ./model-bench-run
```

指定 `--input-len` / `--output-len` 做冒烟或其他固定长度场景；省略 `--concurrency` 时使用参考档位。输出目录必须是新路径，每档保存客户端命令、日志和官方 JSONL，`summary.csv` 在各档结束后更新。稳定态使用 `--mode steady`，可用 `--prompt-multiplier` 增加每档请求数。旧版本用 `--module sglang.bench_serving`。认证由当前客户端环境提供；命令和报告里不保存凭据。

## 表格口径

- `max_concurrency` 是客户端上限，`concurrency` 是各请求延迟之和除以整轮耗时，`peak_concurrency` 使用客户端实测峰值。三者与服务端实际 running batch 分开。
- `duration_s` 对应客户端 `duration`，`rps` 对应请求吞吐；`generate_throughput_tok_s` 对应官方 `output_throughput`。保留服务端输出 token 计数与客户端 retokenized 计数。峰值从官方字段取得，并记录所用客户端版本的窗口口径。
- 使用客户端输出的 latency 指标。TPOT 对单个请求近似 `(E2E - TTFT)/(输出 token 数 - 1)`；ITL 是客户端收到流式增量的间隔，MTP 时一个 chunk 可含多个 token。不可用的 accept length 留空，并核实客户端是本轮采样还是服务端累计值。
- 可随汇总附上容量字段：`tokens_per_request=ISL+OSL`。容量估算使用该次运行的 token 池和运行请求上限。只有明确 token 池和请求上限的全局/每 DP 口径后才对 DP>1 计算或拆分；实际 per-DP batch 需要服务端证据。

参考：[SGLang Bench Serving Guide](https://docs.sglang.io/docs/developer_guide/bench_serving)
