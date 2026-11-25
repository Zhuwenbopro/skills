---
name: hcu-bench
description: 在不同型号的海光 HCU/DTK 节点上识别硬件、选择可用设备，测量显存带宽、卡间 P2P 单向/双向通信和 BF16/INT8 GEMM，保存可复用硬件基线。用于硬件性能基准和矩阵规模对照；不用于模型部署、抓 trace 或自定义 kernel 开发。
---

# HCU 硬件基线

适用于 Torch CUDA 兼容接口下的 HCU/DTK 环境，包含 BW1000、BW151 等型号。型号以现场 `hy-smi` 和 Torch 设备属性为准。

使用 [scripts/hcu_bench.py](scripts/hcu_bench.py) 测量官方 Torch 算子；[scripts/bw1000_bench.py](scripts/bw1000_bench.py) 是旧名称入口。旧报告 schema 不变，新运行输出 schema v2。

## 识别环境和选择设备

- 使用用户给出的节点、账号和连接路径，重新验证连通性。VPN 内网直连与 ProxyJump 分开处理；遇到本机全局 `SSH_ASKPASS_REQUIRE=force` 与本次密码冲突时，仅对本次 SSH 子进程移除 `SSH_ASKPASS` 并设 `SSH_ASKPASS_REQUIRE=never`，通过终端输入用户提供的密码，不改全局配置，不将凭据存入脚本或报告。
- 在宿主机检查 `hy-smi`、`--showproductname`、`--showmeminfo vram`、`--showpids`、`--showbus`、`--showdriverversion`、`--showtopo`。不同版本按 `hy-smi --help` 调整选项。记录型号、显存、PCI BDF、驱动、拓扑与 NUMA；需要时从 PID 的 `/proc/<pid>/cgroup` 核对容器归属。
- 卡号由当前占用和用户约束决定，不固定为 0、1。显存占用和利用率都低还不够，需检查对应 KFD 进程。用户要求使用可用卡时，可直接选择经检查无任务占用的卡运行；无需先建立占卡服务。该检查不构成排他锁，运行前再确认，发现目标卡出现其他任务则停止扩展测试。
- 优先选同 NUMA、HSL 直连的可用卡，保存选择依据。跨 NUMA/PCIe 的结果单独标注；两卡测试不能代表整机所有链路。
- 不停止、释放或修改其他任务。已有用户占卡策略继续遵守；只有用户要求预留卡时才创建或恢复预留程序，并确认所有权。
- 使用已验证的 Torch/HIP 解释器；可用现有容器启动独立测试进程，不修改容器服务或软件环境。记录容器名/镜像、解释器、Torch/HIP/DTK、可见设备变量及警告。不要把驱动、DTK 与 Torch 的 HIP 版本混为一谈。
- 将宿主卡号与 Torch 逻辑编号通过 PCI BDF/UUID 对齐。容器和 `HIP_VISIBLE_DEVICES` 等可能重新编号；脚本 `--devices` 始终指 Torch 逻辑编号，`--physical-devices` 是已核对的宿主卡号标签，脚本不会自动证明映射。

## 测量与保存

1. 保存测试前的型号、拓扑、KFD PID、显存和服务状态。目标卡有用户服务常驻时，记录 `service_resident_no_request`；目标卡空闲而其他卡有任务时，记录 `selected_devices_idle_shared_host`，不要称整机独占峰值。只对当前任务的服务检查已知 `/health`，不默认请求他人的 `/flush_cache`。
2. 先用小 buffer/矩阵检查算子可用性、映射和正确性。然后运行正式参数，使用独立输出路径保存每次结果，保留失败日志。
3. 默认显存 copy buffer 512 MiB，P2P 每方向 256 MiB；warmup 5、迭代 20、重复 3 轮。GEMM 默认 `10240³`，可增加 `4096³` 作规模对照。它们是统一参考参数，不保证每个型号达到饱和或峰值；显存不足时显式缩小并记录原因。
4. 保存原始每轮值及中位数、环境、参数、正确性检查和前后显存。脚本在各阶段增量保存 JSON；单卡跳过 P2P，多卡测指定卡之间所有无序对。P2P 不支持或 INT8 后端明确不支持时记录 `unsupported`，不以 CPU/host staging 或不同精度冒充结果。
5. OOM、设备错误、VMFault、正确性失败等保留失败状态并停止扩展，不盲目重试。测试后检查 `hy-smi`、KFD PID、内核错误和相关服务健康，确认本次进程退出、目标卡恢复；不要清理其他任务。
6. 下载脚本、JSON、前后环境、命令与日志到按日期和型号命名的新目录，校验 SHA256。对实际在远端运行的脚本归档，避免更新后的本地版本冒充原运行版本。

示例（容器内；这里假设宿主 6、7 已映射为 Torch 0、1）：

```bash
HIP_VISIBLE_DEVICES=6,7 HCU_ADAPTER_HOST=none SGLANG_PLUGINS=none \
  /path/to/verified/python /tmp/hcu_bench.py \
  --output /tmp/hcu-baseline-new-run.json \
  --devices 0,1 --physical-devices 6,7 \
  --gemm-sizes 4096,10240 --scope selected_devices_idle_shared_host
```

先检查并处理已有可见设备变量，避免叠加掩码产生错误映射。`--hardware-label` 可附加操作人员标签；`--environment-json` 可合入外部预检 JSON。脚本不部署服务、不更改频率/功耗、不占满显存，不设置预留进程。

## 结果口径

- 显存：同卡 BF16 `copy_`，事件计时，吞吐按读+写 `2 × buffer_bytes / 秒`。这是固定访问模式的参考值，不是所有模型的实际带宽。
- P2P：BF16 `copy_`，分别单向和双向并发；墙钟包含主机提交与两卡同步开销。单向按一个 buffer 计，双向按两个 buffer 合计。不能将双向合计当作单向，也不能把所有链路称为 HSL。
- GEMM：BF16 `torch.mm`，INT8 `aten._int_mm`（INT32 输出），输入 A、B 均为连续行主序。吞吐 `2 × N³ / 秒 / 10¹²`，分别为 TFLOPS、TOPS；事件计时，包含设备时间范围内的算子开销。正确性用少量输出位置与 CPU float64 点积对照，不等同完整数值验证。
- 同型号比较也需对齐矩阵规模、layout、dtype、软件版本、计时方式、重复次数及共享/服务占用状态。不同型号不套用 BW1000 理论上限；没有可靠规格来源时只报告实测。
- 不把自定义 HIP/Triton kernel 混入官方算子基线，不从 GEMM 吞吐推断模型端到端性能，也不把一次大矩阵测试宣称为硬件理论峰值。