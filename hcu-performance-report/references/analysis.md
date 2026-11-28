# 分析与校验口径

分析、绘图和校验全部在目标服务器完成。本文件规定内部工作口径，不是报告正文中的“模块映射与计算口径”章节。

## 输入与溯源

先建立一份服务器内部清单，关联以下已有资料：

| 资料 | 需要确认的内容 |
| --- | --- |
| 原始 trace | 文件、rank、时间、采集窗口、阶段标记、Graph/eager、过滤条件 |
| 捕获时配置 | 模型、dtype、TP、卡号、版本、MTP、Graph、流设置、环境变量 |
| 请求及采集记录 | 实际输入/输出 token、并发、采样、预热、缓存、采集范围 |
| 硬件基线 | 卡、dtype、矩阵规模、计时范围、单位、P2P 方向及聚合方式 |
| 模型与运行时源码 | 对应版本的真实模块、函数、shape、张量并行分片和阶段边界 |
| 已有报告 | 当前内容、用户指定的更新范围、可沿用的图和有效证据 |

相同节点也可能有不同版本或启动配置。不要把启动前的环境快照误用为之后的 Graph 捕获配置，也不要用当前服务状态倒推过去。配置只能由已有命令、日志、server_info 记录或其他直接证据确认；本 skill 不发送生成请求来补验。

资料缺失时，先完成不依赖缺项的分析。无法确定的 shape 或函数留空缺标记，明确内部证据边界。利用率优先从服务器 checkpoint tensor headers、模型配置和实际访问公式建立逻辑参考；只有在这些资料也缺失时才留空，不要因为没有硬件计数器就把整列都写成“—”。不要为补齐报告而自动采集新数据。

## 从 trace 到调用序列

1. 按实际阶段标记划分 DRAFT / VERIFY；没有出现的阶段不能补造。
2. 以设备、流和时间信息建立 GPU 执行序列，再通过 runtime correlation / External id 关联 CPU op、模型注解及源码。CPU 提交顺序不等于跨流 GPU 的全局依赖顺序；并行分支保留流和依赖关系，不伪造严格串行关系。
3. Graph trace 的流、关联和注解可能不完整。优先区分 `user_annotation`（CPU 提交/调度）与 `gpu_user_annotation`（device lane GPU scope）；DRAFT/VERIFY 的 Graph 时间用后者包围的 kernel 事件，CPU 注解只做 host 侧补充。已有 eager 注解 trace 可提供语义及 shape，干净 Graph trace 可提供时间；分别保存来源，不能把较短 CPU 注解当成 GPU 阶段耗时，也不能据此宣称 Graph 加速。
4. Graph/eager 采用有约束的顺序匹配。重复通用 kernel 会产生多解；检查最早/最晚匹配和上下文。无法唯一确认时标记映射不确定。
5. GEMM main kernel 与同一 External id 的 PostGSU 可以合为一个逻辑运算：时间求和，FLOPs 与逻辑字节只计一次。保留原始 kernel 归属。
6. 使用每个 rank / replay 对应出现位置的 shape，尤其是动态 KV 长度；不能把首个 replay 的 shape 复制给所有 replay。报告 TP 分片后的真实输入、权重和输出。
7. 延迟执行的 FFN allreduce 可能位于下一层的边界注解内。根据源码关联其生产者，但保持实际 kernel 位置，不能为了模块分组移动通信。
8. 父 GPU annotation 可能包含子 GPU annotation。建立唯一 kernel 集合时，先按时间和 kernel identity 扣除子 scope，再分别记录父 scope 的 wall span 和子 scope 的 kernel duration；否则会把 `draft_extend` 与 `DRAFT_EXTEND_V2` 重复相加。
8. Graph 独有的收尾操作可能没有 eager External id。仅在源码结构足够支持时命名，并记录这是源码证据；不能伪造关联 id。

只把 draft、TARGET_VERIFY、DRAFT_EXTEND_V2 这类外层 user annotation 当作阶段边界，不当作最终算子表。最终表应继续向下关联 CPU op、correlation、kernel 名、模型源码和 shape；如果仍无法唯一映射，保留粗行但明确它是未展开的阶段摘要，并在报告中写出未覆盖的调用链。

缺失 CPU flow 或 Graph kernel 覆盖时，已见事件不能证明完整调用链。保留 kernel 集合、未映射项和覆盖缺口；无法确认顺序的部分不能包装成完整顺序表。

## 折叠重复调用

- 从原始有序逻辑调用序列识别重复片段，优先按真实模型层次折叠，而不是按函数名聚合。
- 嵌套结构须保留。例如实际为“3 个 GDN 层 → 1 个全注意力层”重复 16 次，就表示为这个嵌套重复体，不能先列全部 GDN 再列全部全注意力。
- 相邻相同操作可以折叠并注明先后，例如 K 写入后 V 写入。非相邻同名操作保留原始位置。
- 每个片段写明层范围或重复次数；片段内部顺序完整。第一层残差、RMSNorm 等特殊变体保留在行内说明，差异影响顺序或指标时拆行。
- 片段序号可用 2.1、2.2 等表示层次。“次数/replay”表示该行展开后的实际总次数，已包含外层重复，读者不能再乘一次。
- 每行保留覆盖的原始事件 id、rank / replay、形状变体、时间列表及展开规则，供内部验证。
- 不能唯一重建顺序时少折叠，不能靠猜测减少行数。

## 时间、计数与分类

- 单项时间使用对应 GPU kernel 的实测 duration；合并项只相加其拥有的 kernel。
- 各行“总耗时”默认表示一个 decode step 的平均 GPU kernel 耗时；“次数/replay（单 step）”只显示一个 step 的展开调用次数，“单次耗时”由单 step 总耗时除以该次数得到。selected replay 数、累计调用次数和原始时间列表保存在审计数据中，不能把 9 个 replay 直接显示成正文主表次数。
- rank / replay 汇总规则写入测试方法表。同一请求的两个 rank 是并行观测，不是两次独立请求；不能把它们相加冒充请求延迟。
- GPU kernel 时间之和、阶段首尾跨度、请求延迟和每 token 延迟是不同量。多流重叠时 kernel 时间之和可以大于阶段跨度。
- Perfetto 的 `Wall Duration` 是 slice 的首尾跨度，`Self Duration` 是去除子 slice 后的自身时间；两者都不能直接替代唯一 GPU kernel duration。报告必须在数据文件中保留事件类别和去重后的 kernel id。
- 对重复的 DRAFT / VERIFY replay，正文主指标默认取每个周期的平均 GPU kernel duration；多次 replay 的合计仅作为服务器内部审计值，除非用户明确要求累计量。报告必须同时保留周期起始间隔、scheduler.run_batch 或其他 wall/CPU annotation 的独立统计，不能把它们称为同一个 step 时间。
- 饼图以同一选定阶段的 GPU kernel 耗时之和为分母。每个 kernel 唯一归属通信、GEMM、逐元素、其他四类之一；不确定项归“其他”并保留原因。
- 双饼图使用相同的选取、rank / replay 汇总和四舍五入口径。不能将不同配置或窗口拼接在一张对比图中。

## 利用率的内部依据

| 项目 | 可使用的计算及限制 |
| --- | --- |
| GEMM 算力 | 对标准矩阵乘，FLOPs = 2×M×K×N；除以对应实测时间，再除以相同卡、dtype 和适当规模的已测 GEMM 参考。融合项或低精度变体按实际语义计算 |
| GEMM 带宽 | 依据实际 A、W、Y 形状和 dtype 计算逻辑读写字节；需要重复读写、额外融合输入时据实处理，不能把权重分片还原成全局权重 |
| 逐元素/嵌入/KV | 按实际访问及读写语义计逻辑字节；embedding 计被选中的行，KV 写入计选中位置的数据及索引，不计整个已分配池 |
| 通信 | 使用与通信方向、卡数、拓扑及聚合方法匹配的 P2P/集合通信基线。TP2 的两 rank 有效负载求和可对应双向总带宽；不能推广为任意 TP 或任意集合通信的统一公式 |
| 缺少可信口径 | 细小辅助项可用“—”；DRAFT/VERIFY 阶段和主要 GEMM/MoE/KV 行必须优先给出基于 shape、dtype 和访问公式的逻辑参考 |

W4A8 模型的权重精度、激活精度和 kernel 算术路径分别记录。硬件没有 INT4 峰值时，W4A8 的算力利用率统一使用同型号 INT8 GEMM 实测上限作为参考分母；结果必须标记为 `INT8 reference`，不能称为 INT4 利用率。W4 权重的逻辑字节来自 checkpoint 的 packed INT8 tensor 与 scale，MTP/BF16 权重按实际 BF16 字节数，FP8 KV 按实际 K/V head、head_dim、context 和 token 数计算。带宽利用率是逻辑字节/实测 HBM copy 上限，不是 DRAM 计数器；正文表头或脚注必须标记“逻辑/参考”。“decode 更接近带宽受限”可以作为基于这些计算、batch、KV/状态访问和 kernel 形态的工程判断。

阶段利用率至少按下式计算并在 `analysis-data.json` 留存：

```text
compute_util = estimated_flops / (gpu_kernel_time × measured_INT8_peak)
bandwidth_util = logical_bytes / (gpu_kernel_time × measured_HBM_copy_peak)
```

`measured_INT8_peak` 使用同卡、同型号 10240³ INT8 GEMM 实测中位数；`measured_HBM_copy_peak` 使用同卡 BF16 read+write copy 实测中位数。逻辑带宽没有扣除 cache reuse 时，可以超过 100%，必须明确这是访问上界而非物理 HBM 利用率。

有效算力、字节数、FLOPs、分母出处和置信范围全部内部保存。逻辑带宽与流式 HBM 基线的比值可能因缓存或访问方式而超过 100%；它不是 DRAM 计数器测得的物理利用率。不要裁剪为 100%，也不要隐瞒口径差异。存在这种情况时，在表头或相关单元格用最短标记说明“逻辑带宽/参考”，不追加方法论章节。

硬件参考用于对照，不是理论峰值。逐卡原始结果保留；只有相同方法和单位且差异不影响参考意义时才合并，保留代表值的取法，不编造固定的相近阈值。

## 发布前检查

至少在服务器核对以下实质不变量：

- 每个选中 kernel 恰好属于一个逻辑操作和一个类别，遗漏项有明确记录。
- 折叠结构展开后的事件序列与原始选中序列一致；并行分支的依赖和归属没有改变。
- 行时间之和、分类时间之和与源数据相等，允许的误差只来自明确的单位换算和显示舍入。
- 对所有纳入的 rank / replay 核对次数、时间和形状变体，不只验证一个样例。
- 对每个折叠行核对展开后的次数、层范围、shape 变体和顺序；特别检查 MTP draft/verify 的 forward 次数、GDN 与 full-attention 层组、TP collective 和 logits/sampling 收尾。
- 饼图比例由发布数据产生；类别、标签、颜色及合计一致。正文数值与同一份数据一致。
- 绘图脚本必须在生成前删除或覆盖旧图，生成后核对文件修改时间、图内阶段数值与 `analysis-data.json`。绘图库不可用时使用可验证的备用绘图实现，不能让旧图留在交付目录。
- 图表未引用旧节点或其他配置；未知字段保留“—”，未用示例值填充。
- 只有在 GPU lane 的 kernel 排列、Graph 周期起始间隔和 scheduler wall span 同时核对后，才可判断空泡；CPU annotation 之间的空白不能单独证明 GPU 空泡。

保留足以重现分析的服务器产物和校验结论。校验通过表示此次已有数据的分析一致，不代表重新测试过模型或硬件。
