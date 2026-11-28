# 覆盖诊断与计时解释

## 当前 DTK 漏抓反例

2026-11-04 同环境执行 GEMM、SiLU、RMSNorm、index_put、Triton marker 五种 kernel，重复三次，预期15个。结果：

| 条件 | replay kernel 数 |
|---|---:|
| eager | 15 |
| 默认，先建 Graph 后开 profiler | 0（仍有3次 hipGraphLaunch，数值正确） |
| 默认，profiler 开启期间建 Graph | 9（另有 capture 准备事件，不可混计） |
| GPU_FORCE_QUEUE_PROFILING=1 | 11 |
| HIP_GRAPH_ACCUMULATE_DISPATCH=0 | 15 |
| 累积派发=1，ACCUMULATE_DISPATCH_PROFILE_LEVEL=FULL_PROFILE | 15 |

默认案例原始 kineto_results.events() 已无 GPU kernel；原生 hipprof 默认也漏 Graph 中的 Triton marker。问题定位到当前 DTK Graph 累积派发路径的 activity 上报，不能单靠换 Perfetto、record_shapes、with_stack、扩大 buffer 或提前 /start_profile 来解决。没有厂商源码，不声称已定位具体 C++ 错误行。

首选 FULL_PROFILE，保持 Graph 与累积派发。关闭 HIP_GRAPH_ACCUMULATE_DISPATCH 是改变派发策略的备用诊断控制，不作为不加说明的性能基线。其他 Torch/DTK 版本先做小复现确认。

## 判定顺序

先核对 actual worker 的环境、运行时库版本和 capture 时机，再看每 rank Graph correlation、每步计数和各模型层数。需要时重用历史 coverage_repro.py；目录不可用时构造同类可验证的小例子即可，不必为了普通抓取编译额外探针。

非零或稳定的少量 kernel 仍可能是部分覆盖。相同总数也不自动证明具体算子都出现。merged trace 的多个进程 correlation 可能冲突，优先独立 rank 文件；遇到不同格式，应保留原始数据并调整解析，不能把无法关联的 kernel 丢掉后宣称完整。

`ROCTracer produced duplicate flow start: 11` 在补全 kernel 后仍出现。记录警告；不能宣称 CPU→GPU flow 已修复。GPU kernel 与 Python/aten stack 是两层：Graph replay 不重跑 Python 层，eager 可用于源码映射但不能替代真实 Graph timing。

## 可见 gap 与真实等待

补全后的图内相邻事件约有 1ns 边界，不能据此宣布硬件利用率≈100%。原生 HIP Graph 插入外部 event 等待的控制实验：side stream host callback 延迟50ms，Graph 同步耗时约51.56ms，trace 的两个 marker kernel 之间仍显示约43.29ms空白。等待先于 Graph launch 开始，因此不要求 kernel 间空白精确等于50ms。此验证支持显式等待可保留，不证明所有累积节点内的纳秒级时间戳精度。

Torch external=True Event 在该 ROCm 构建上报 External events are disallowed in rocm；不要无限重复该接口。若时间戳精度对问题必要，再用可用的原生 HIP API/厂商工具做有界验证。

## decode 变快但 E2E 变慢的已知案例

GDN 辅助流禁用前后，各抓一次相同 1024→32 请求：

| 带 profiler 的观察 | KV单流/GDN双流 | KV+GDN单流 |
|---|---:|---:|
| CPU Graph launch 间隔中位数 | 37.18ms | 34.68ms |
| 可见 GPU Graph 首尾区间中位数（旧 trace 有漏抓） | 36.88ms | 35.22ms |
| step[EXTEND] CPU scope | 228.05ms | 345.83ms |
| 同轮5次无 profiler 整请求中位数 | 1.334809s | 1.409198s |

这是两份 trace 的实际观察，不证明无 profiler 的那五次也发生相同阶段变化。原5次记录只有 E2E 和完整响应，没有 TTFT/逐 token 时间；prefill 增加是带 profiler 观察到的抵消项，无 profiler 的因果尚未验证。客户端与服务端 E2E 都增加，不能简单归因于 SSH 或下载耗时；两组测量相隔约40分钟，也不是交替、同时间窗口的 A/B。五次之间较稳定仍不能消除跨轮的频率/温度、runtime或状态漂移。

已核对该版本官方 `_forward_input_proj`：辅助流分支要求 `get_is_capture_mode()` 且 token 数<1024。本次 prefill Graph disabled，1024-token prefill 原本走顺序分支，所以没有直接代码证据表明 GDN 流开关使 prefill 额外串行化；prefill scope 变长不能直接归因于该改动。

精确术语：TTFT 含首 token 前的排队/调度/prefill/传输；ITL 是相邻 token 间隔；TPOT 的定义需明示；CPU step scope、hipGraphLaunch CPU耗时、launch间隔、可见GPU kernel区间各有不同边界。非流式 E2E 包含首 token 与其余 token，不能倒推出单步 decode。

用户要求解释或量化优化时，优先核对已存计时口径；缺少阶段数据就说明缺失。需要新的 A/B 时固定 FULL_PROFILE 环境、缓存、预热、模型、请求、设备及其他变量；在无 profiler 下同时收集 TTFT/ITL或明确服务端 decode 计时和 E2E，并做合理重复/交替。长输出、短输入可以放大 decode 占比，但须标为新增 workload，不能拿其结果冒充原1024→32请求效果。
