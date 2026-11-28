# Codex 技能集

这里收录了海光 HCU 算子开发、SGLang 推理性能分析和飞书工作汇报用到的 Codex 技能。点击技能名可查看使用说明。安装后，在 Codex 请求中写入 `$技能名` 即可调用对应技能。

## HCU 硬件与算子

| 技能 | 适用场景 |
| --- | --- |
| [hcu-bench](hcu-bench/SKILL.md) | 为 HCU 节点建立性能基线，测量显存带宽、卡间 P2P 单向与双向带宽，以及 BF16、INT8 GEMM 吞吐。 |
| [hcu-kernel](hcu-kernel/SKILL.md) | 在 gfx936、gfx938 等 HCU 上开发和调优 GEMM、规约及逐元素算子。以访存延迟地板为参照，在固定频率下用纯访存探针和反汇编检查布局与预取，加回计算后逐点验证，绘制延迟和加速比图。 |

## SGLang 部署与性能分析

| 技能 | 适用场景 |
| --- | --- |
| [hcu-sglang-deploy](hcu-sglang-deploy/SKILL.md) | 在 HCU/DTK 节点部署或恢复 SGLang 推理服务，配置容器、模型与 DAS 插件，并用实际生成请求验收。 |
| [model-bench](model-bench/SKILL.md) | 对已运行的 SGLang 服务做端到端压测。固定输入输出长度，扫描并发，比较吞吐、首 token 延迟（TTFT）和每个输出 token 的耗时（TPOT）。 |
| [hcu-sglang-trace](hcu-sglang-trace/SKILL.md) | 在 HCU 上采集 SGLang 各 rank 的 `torch.profiler` trace，检查 Graph 算子覆盖，核对 prefill 和 decode 的计时范围。 |
| [hcu-performance-report](hcu-performance-report/SKILL.md) | 用已有 trace、运行配置和硬件基线整理性能报告，按实际调用顺序解释推测解码 DRAFT 与 VERIFY 阶段的算子耗时和利用率。 |

## 飞书工作汇报

| 技能 | 适用场景 |
| --- | --- |
| [feishu-session-report](feishu-session-report/SKILL.md) | 按需把当前 Codex 会话整理成飞书工作简报，在正文列出关键数据，并归档支撑结论的测试结果。 |
| [feishu-weekly-report](feishu-weekly-report/SKILL.md) | 从已配置的飞书索引读取最近七天或指定日期范围的工作汇报，按项目生成中文周报并附原文链接。 |

## 调用示例

> 用 `$hcu-kernel` 调优 gfx936 上的 INT8 GEMM。固定核心和 HBM 频率，从 M=1 到 256 逐点测量，画出延迟和相对 LightOP 的加速比。

> 用 `$feishu-weekly-report` 把最近七天的工作按项目整理成周报，附上每项工作的原文链接。
