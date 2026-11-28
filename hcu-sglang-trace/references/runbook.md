# 采集与下载操作

## 启动条件

先核对真实服务 `/proc/<已核对PID>/environ` 中的两个 Graph profiling 变量和命令。新启动脚本在 exec 服务解释器前设置：

```bash
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export ACCUMULATE_DISPATCH_PROFILE_LEVEL=FULL_PROFILE
```

保留目标服务原有 PYTHONPATH、adapter/plugin 入口、设备、TP 和 Graph 配置。已有服务若满足条件直接采集。若需要重启，先准备完整启动配置、日志与真实 API 验收，再核对 PID/命令/进程组，停止自有服务并启动替代进程。不要用 pkill python。重连后先确认原操作结果，不重复启动模型。当前配置样例见 qwen38-graph-example.md；部署细节由已有部署 Skill 处理。

SSH 脚本传输推荐标准输入，避免 Windows 长命令/base64 命令行超长和 PowerShell 的美元变量展开。例如本机 Python：

```python
import shlex
import subprocess
from pathlib import Path
host = "hcu-node"
container = "sglang-container"
script = Path("prepared-remote.sh").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
command = shlex.join(["docker", "exec", "-i", "-w", "/home", container, "bash", "-s"])
subprocess.run(["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", host, command], input=script.encode("utf-8"), check=True)
```

按当前目标替换节点和容器。复制工具可通过 stdin/base64 文件内容、scp 或已有部署 helper，不复制凭据，不依赖新的公网下载。脚本按当前工具的权限流程执行；不要为此改全局 SSH 设置。

## 单请求 capture

将 `trace_tool.py` 放入容器可读的工具目录。输出根可以预先保存 start.sh/profile/版本信息，但不得含旧 workload、manifest、capture 标记或旧 trace。以下命令在容器内执行；路径为示例，使用本次新目录：

```bash
HCU_ADAPTER_HOST=none \
  /workspace/sglang-deploy/venv/bin/python /workspace/sglang-trace/trace_tool.py capture \
  --base-url http://127.0.0.1:30000 \
  --output-dir /workspace/sglang-trace/runs/new-run \
  --model-path /models/Qwen3.8-27B \
  --input-tokens 1024 --output-tokens 32 --expected-output-tokens 32 \
  --expect-tp 2 --expect-replays 31
```

当前单请求、非 MTP 配方的第一枚输出由 prefill 产生，其余 31 次 decode replay。其他输出长度、speculative/MTP、chunked prefill、并发等不能沿用 31 的预期。

使用已保存请求做精确 A/B 时，将 `--model-path/--input-tokens/--output-tokens` 替换为 `--request-json /path/to/workload.json`。接受原始请求 JSON 或含 request 键的 workload.json，不改写请求。不要用不同 tokenization 或提示词比较代码改动。

`--with-stack` 可帮助 CPU scope 映射，不保证 Graph 内层级栈；映射需求可另抓 eager，不能拿 eager GPU 时间线拼装 Graph trace。默认 profiler 配置为 CPU+GPU、record_shapes=true、with_stack=false、profile_by_stage=false、merge_profiles=false，不设置 num_steps，避免旧版本停止计数状态机差异。

该 helper 是已有服务的 HTTP 客户端，不设置/验证服务器进程启动环境，不负责启动/停止模型或修改源码。HCU_ADAPTER_HOST=none 只适用于客户端，不可照搬为服务入口变量。服务解释器、来源和实际 profiler schema 以当前部署为准。

## 覆盖验收

采集 helper 自动保存 trace_manifest.json。结构检查后还需模型预期。例如仅对当前已验证的 Qwen3.8-27B BF16/TP2/两种辅助流关闭配方，可增加：

```bash
python /workspace/sglang-trace/trace_tool.py inspect /workspace/sglang-trace/runs/new-run \
  --output /workspace/sglang-trace/runs/new-run/trace_manifest.json \
  --expect-tp 2 --expect-replays 31 \
  --expect-kernels-per-replay 1015 --single-stream
```

1015 不是其他模型/版本的通用常数。用模型层数、后端、融合/量化和当前代码确定预期，核对至少 GEMM、SiLU、GDN norm、KV store/RCCL 等适用类别。单流断言仅对既有单流改动生效的实验使用。

对 eager 控制 trace 使用 `--mode eager`，不要求 Graph replay。普通未给明确每步预期的检查即使通过，也只说明结构可用；报告 `coverage=not_proven_by_event_presence`，不能声称已完整。

## 下载

在本机运行，无需本机安装 Torch：

```bash
python /path/to/skill/scripts/trace_tool.py download \
  --host hcu-node --container sglang-container \
  --remote-dir /workspace/sglang-trace/runs/new-run \
  --output-dir /path/to/workspace/traces/new-run
```

目录与主机来自当前任务。Windows 解包使用扩展长度路径，并在安全过滤后规范化 tar 的 ./ 与分隔符，兼容实验源码副本中的长文件名；不修改系统全局长路径设置。下载保留 `artifacts.tar.gz`，安全解包到 artifacts，并输出 download-check.json。支持本 Skill 的对象式 manifest 和早期实验列表式 manifest；逐个校验原始文件长度与 SHA256。下载允许保留验收失败的原始 trace，不能将 checksum verified 解释为覆盖通过。包含多份 capture 子目录时，每个 manifest 相对其所在目录验证。

## 交付

保留完整模型参数、源 SHA/dirty、Torch/HIP/DTK、两个启动变量、实际启动/Graph 配置、设备、原始请求和响应、profiler 配置、所有 rank trace、manifest、服务/采集日志及 SHA256 下载检查。另存简短调查/比较文档，明确覆盖预期的来源和未验证条件。含错误的首次尝试不要覆盖；后续用新目录记录修复。
