---
name: hcu-sglang-deploy
description: 在海光 HCU/DTK 节点通过 SSH 和 Docker 部署官方 SGLang、adapter-das 与 sglang-plugin-das，覆盖容器准备、GitHub 授权、版本与依赖、模型定位、显存交接、后台启动及真实 API 验收。用于从新容器到模型服务就绪的完整部署，也适用于恢复、重启和更换模型；不用于 CUDA/NVIDIA 通用部署或修改插件实现。
---

# HCU SGLang 部署

把指定模型部署到用户选择的节点和容器，交付可重复启动的服务、确切版本、日志和真实生成证据。以实际节点、模型配置及仓库源码为准。

## 入口与工具

已有会话给过节点、容器、模型、卡或版本时直接沿用，不重复确认。节点不明确且影响资源写入时才询问。已有健康服务若正是目标部署，先核实后复用。

对于 Qwen3.8-27B，可选用已验证的 [TP2 MTP 配方](references/qwen38-mtp-tp2-example.json)，参数为 NEXTN、steps=3、topk=1、draft tokens=4。用户明确要求 MTP 时，需要验证该路径实际执行。如果实际版本不支持，说明阻塞。其他模型先核对其 MTP 支持。沿用已确认的 Graph、流和设备配置。

- 完整操作与依赖安装：[references/runbook.md](references/runbook.md)。开始部署时阅读。
- 已验证的版本和模型配方：[references/known-recipe.md](references/known-recipe.md)。需要参照历史版本时阅读。
- 编译、加载、KV 或预热错误：[references/troubleshooting.md](references/troubleshooting.md)。按实际首条原因选择条目。
- `scripts/remote_exec.py`：UTF-8/base64 SSH 传输，避免 PowerShell 改写 `$变量`、嵌套引号和 CRLF。
- `scripts/render_launch.py`：从 JSON profile 生成启动脚本，不连接节点。
- `scripts/reserve_vram.py`：在 Linux 容器保留已分配卡的显存，支持信号释放，不持续做计算。
- `scripts/verify_service.py`：等待服务、核对模型身份、请求真实短答案并检查正常停止，保存完整响应。

脚本仅依赖标准库。占显存另需容器中的 HIP Torch。用 `--help` 查看参数。Qwen3.8 可参考 [TP2 MTP 配方](references/qwen38-mtp-tp2-example.json) 或 [TP4 非 MTP 配方](references/qwen38-tp4-example.json)。复制后替换节点、容器、路径、设备和端口，再核对源码版本。

## 运行边界

- 只使用用户指定或已分配的节点、容器和卡。`hy-smi` 同时检查显存与 HCU 利用率；利用率为零但显存常驻的卡仍可能属于他人。
- 用户已分配测试卡且要求占卡时，保留自有占卡进程，配置和预检查完成后才交接给模型。核对 PID、完整命令及进程组后再停止自有进程。
- 不复用历史 PID、授权验证码、空闲读数或镜像“最新”结论；运行前重新检查。
- GitHub 认证在目标环境独立完成，保留直接 fetch/切分支能力。复制另一节点的仓库不能代替目标节点认证；不复制 token、私钥或 credential 文件。
- 使用模型只读挂载，不复制几十 GB 权重到用户主目录。路径略有差异时先枚举相邻目录、读取 config/index 并说明修正。
- 不从公共 PyPI 覆盖匹配 DTK 的 Torch/Triton/AITER/LightOp。历史 Python 3.10 示例不能照搬到 Python 3.12 镜像。
- 未要求修改源码时优先通过匹配版本、公开配置和后端选择部署。需要修改实现则遵守目标仓 AGENTS.md/相关 Skill、保留差异和回归证据；不隐藏 soft error。

## 分阶段执行

### 1. 连接、容器、资源与模型

使用已有 SSH 别名和默认认证。若执行工具阻断网络，按工具权限流程处理。

读取 hostname、whoami、hy-smi 和 docker ps -a。容器存在则复用，停止则按部署需要启动。仅在容器不存在时检查并执行用户提供的创建脚本。脚本末尾若因无 TTY 退出，仍须 inspect 容器并执行简单命令确认容器是否已经创建。

检查 `/models` 挂载、模型实际配置、全部权重分片及 tokenizer/chat template。确认上下文、架构、dtype、量化、TP 可分性和卡容量。只在无法辨认目标时询问。

### 2. 仓库、身份与版本

标准位置为 `/workspace/sglang-deploy/src/adapter-das`、`/workspace/sglang-deploy/src/sglang-plugin-das`、`/workspace/sglang-deploy/src/sglang`。origin 分别是 `https://github.com/HYGON-AI/adapter-das.git`、`https://github.com/HYGON-AI/sglang-plugin-das.git`、`https://github.com/sgl-project/sglang.git`。

不存在才 clone；已有目录先检查 origin、HEAD 和 dirty，不覆盖用户修改。私仓认证失败时用 GitHub CLI 浏览器设备授权，展示本次网址/验证码并等待用户完成，不索取聊天中的 token。授权后在同一容器执行 `gh auth setup-git`，实际 `git fetch origin`。

读取相关仓库的 AGENTS.md、当前 pyproject、安装文档和入口。用 SGLang v0.5.22 基线时，按已验证配方检查两个插件的远程 ref 和兼容性。官方按兼容要求固定并记录解析后的 commit。

### 3. 安装与加载预检查

按 runbook 建立独立 venv，可继承 system-site-packages 复用 DTK 栈。以 `--no-deps --no-build-isolation` 安装匹配三仓，再针对性补齐 Python 依赖；`--no-deps` 不代表依赖已齐全。

安装和元数据检查可临时使用 `HCU_ADAPTER_HOST=none`，服务时恢复 sglang。核对版本、模块来源和运行相关验证工具时，使用服务相同的解释器与 PYTHONPATH。容器默认路径可能遮蔽 venv 中的包。

保证新进程启动时能发现 adapter 的 sitecustomize。Plugin 要有安装元数据中的 `sglang.srt.plugins` entry point：`hcu = hcu_sglang:register`。推荐 `SGLANG_PLUGINS=hcu` 和 `python -m hcu_sglang`，让包装入口先初始化 AITER 兼容。已验证 loader 中未设置/空 SGLANG_PLUGINS 表示加载全部，不能用于停用。

完成 tvm-ffi 单进程预热和 `--help` 入口检查。记录相关 Hook 来源与 soft errors；入口检查和模型验收分开。

### 4. 配置、显存交接与后台启动

按实际 config 和已验证配方选择后端、KV 页、TP、上下文、并发和显存预算。不能对所有模型固定 TP4 或 fa3。示例 profile 是 Qwen3.8 BF16/gfx936 配方。

生成独立部署目录中的 profile、start.sh、日志和 PID 文件。用 render_launch.py 生成脚本；同一进程只用一种设备可见性变量。已验证配方使用 HIP_VISIBLE_DEVICES 并清除继承的 ROCR/CUDA/ordinal 过滤。

确认端口仍空闲、安装和配置已就绪后，经身份核对释放自有占卡程序，随即以 `nohup setsid bash start.sh` 启动并保存真实 PID。重启只清理该进程组和遗留 worker。SSH 中断先检查原进程与日志，不重复加载。

### 5. 观察、诊断与验收

区分导入、注册、collective、权重、KV、冷 JIT 和预热阶段。HTTP 监听但 health 503 仍未 ready。按具体原因调整配置并保留每次日志；同一失败无新证据不盲目重复。两次相同无进展失败后补充诊断；依赖或权限确实不可得时说明解除条件并保留/恢复自有显存占用，不无限重试。

执行 verify_service.py，验证 health 200、`/v1/models` 身份、真实回答和 `finish_reason=stop`。不能只检查 HTTP 200 或长输出中出现目标数字。通过后只因新变更或疑点才重复/扩大测试。

MTP 部署还需核对实际 PID 的完整命令、`/get_server_info` 中生效的 speculative 参数，并用真实生成的 draft/verify 计数或服务运行统计确认该路径已执行；保存原始证据。当前 SGLang 可能将 NEXTN 归一化为 EAGLE，应记录请求配置与有效配置，不能仅凭显示名判断未开启。

保存三仓 commit/dirty、解释器及模块来源、镜像/设备、完整参数、当前 PID、日志、原始响应和已测范围。重新查看 hy-smi，确认资源由模型接管。仅文本测试不能声明图片、视频、Graph、吞吐或完整插件合同通过。

简洁交付节点/容器、实际模型路径、API base/model、卡、启动脚本/日志和验收结果；保留服务运行，除非用户要求停止。
