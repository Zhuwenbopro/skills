---
name: hcu-sglang-trace
description: 在海光 HCU/DTK 上为 SGLang 抓取和下载逐 rank 的 torch.profiler trace，覆盖 CUDA Graph 完整采集开关、覆盖检查、SHA256 校验及 prefill/decode 计时口径。用户说抓 trace、抓 prof、Graph 算子漏抓或比较流分布时使用；不替代模型部署或插件实现开发。
---

# HCU SGLang Trace

交付原始逐 rank trace、相同请求和运行配置、覆盖验收以及下载校验。沿用用户已给的节点、容器、设备、服务和工作目录，不把历史实验参数视为新任务的固定要求。

## 关键配置

在当前已验证的 DTK Graph 环境，**服务启动、HIP 初始化及 Graph capture 前**设置：

```bash
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export ACCUMULATE_DISPATCH_PROFILE_LEVEL=FULL_PROFILE
```

缺失时可能只有 hipGraphLaunch 或部分 kernel，形成假的可见空白。若运行中的服务启动时未设置，核实实际进程环境后重启并重新 capture Graph；HTTP /start_profile 不能替代启动环境。保留 Graph 和累积派发，不用 eager trace 拼补时间线。

版本与可迁移边界见 [references/qwen38-graph-example.md](references/qwen38-graph-example.md)。不同 DTK/模型先核对实际版本和 kernel 预期；FULL_PROFILE 的效果不自动推广到所有 ROCm/DTK。

## 工作流程

1. 核对当前 PID/完整命令、实际进程环境、三个仓库 commit/dirty、Torch/HIP/DTK、模型、TP、设备和 Graph 配置。单靠新 shell 中 export 不能证明老进程已生效。已有健康服务且环境满足时直接复用。需要部署或重启时可用已有 `hcu-sglang-deploy`；只停止经身份核对的自有进程组，保留其他设备和自有占卡程序，不复用历史 PID。
2. 准备独立实验目录。固定 input_ids、采样、输出长度、并发、缓存政策、KV/GDN 流策略；A/B 只改变目标变量。默认单请求 1024 输入/32 输出、预热两次、抓取前 flush prefix cache，用户另有要求则遵从。共享服务 flush 会影响其他请求，不能未经任务范围核实照搬。
3. 在容器可见的路径运行 `scripts/trace_tool.py capture`：health→预热→flush→server_info→start_profile→generate→finally stop_profile→health→验收。详见 [references/runbook.md](references/runbook.md)。接口或响应与脚本不匹配时先读实际 SGLang profiler 入口，保留原尝试；不升级依赖来迁就工具。
4. 检查所有 TP rank、Graph replay、每步 kernel 及模型层数预期。kernel 非零或计数稳定都不能单独证明完整。当前实验的 1015、31、stream72 只适用于该固定配置。主流优先从 RCCL kernel 定位；跨进程编号变化正常。默认允许多流，仅在单流实验显式要求单流。
5. 保存 manifest 后用 `download` 将整个实验目录拉回工作目录，核对原始 trace 的长度与 SHA256。优先逐 rank 文件；保留 gzip 原件、原始响应和诊断日志，分析另写文件。不要覆盖旧实验。
6. 交付节点/容器/配置、trace 链接、覆盖证据、已知限制；完成后保留服务，除非用户要求停止。

## 可复用工具

`scripts/trace_tool.py` 有三个子命令，先看对应 `--help`：

- `capture`：在目标容器调用已有 SGLang HTTP profiler。可直接重用 `--request-json` 的请求或 workload.json；仅生成 input_ids 时需要目标服务环境的 transformers/tokenizer，使用本地模型且不下载。其他操作只需 Python 标准库。
- `inspect`：对原始 `.trace.json(.gz)` 按 Graph correlation 统计；支持明确指定 TP、replay、每步 kernel 数和单流预期。输出结构检查与覆盖预期的区别，不把“有事件”写成“已完整”。未知版本的 rank 命名或 correlation 格式需适配后重验。
- `download`：本机通过既有 SSH 别名、Docker tar 流下载；无 shell 插值拼接用户参数；安全解包并独立验证 manifest。推荐本机 Python 3.12+，要求 tarfile.data_filter。拒绝覆盖已下载实验。下载成功不代表 manifest 的覆盖验收已经通过。

抓取或网络失败时先核实 profiler 状态和原进程；同一原因无新证据不反复抓取、重启或重载模型。超时的 /start_profile 可能已生效，不盲目重复。stop_profile 失败、trace 未 flush、验收失败都保留日志，不产生 capture.done。

## 覆盖和性能解释

漏抓诊断、已验证的无效替代办法与时间戳限制见 [references/coverage-and-timing.md](references/coverage-and-timing.md)。

- GPU kernel 覆盖、CPU→GPU flow、Python stack 分开验收。Graph replay 不重跑各层 Python/aten，不能把缺少 Python scope 当成 GPU 没执行。
- 可见 kernel gap 不等于物理 GPU idle；kernel 首尾区间也不是完整服务 step。Graph 内、Graph 间与 prefill 分别观察；保留 duplicate flow 等警告。
- 带 profiler 的每步 decode、无 profiler 的 TPOT/ITL、TTFT/prefill 和整请求 E2E 是不同指标。不能把 E2E 中位数叫 decode 延迟，也不能将中位单步差乘步数当精确请求差值。
- 性能结论使用无 profiler 的同条件 A/B；固定缓存政策、预热、shape、并发和采集开关，交替/重复测量并保存分布。原记录只有 E2E 时直接说明无法定位是哪一阶段变化，不把 prefill 变慢当作已证实原因。分阶段或长 decode 压力测试仅在用户要求或因问题确有必要时进行，不把基准测试加入每次普通抓 trace 的固定动作。
