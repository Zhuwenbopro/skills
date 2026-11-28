# 可交付报告契约

调用本 skill 的结果分为 `complete` 和 `incomplete`。只有满足本契约并通过 `scripts/validate_report_bundle.py` 的结果，才能称为完整报告。

## 产物目录

服务器上的报告目录至少包含：

```text
report/
  README.md
  draft-verify-kernel-time.png
  analysis-data.json
  source-map.json
  coverage.json
  validation.json
```

`analysis-data.json` 保存原始时间、分类、折叠行、次数和时间口径；`source-map.json` 保存 CPU op、correlation、kernel、源码函数和 shape 的映射；`coverage.json` 保存已覆盖、未映射和无法证明的事件。正文只呈现可读结果，不把这些审计文件整段复制进去。

## 正文结构

正文按以下顺序生成：

1. 标题：模型、节点和运行环境。
2. `运行环境与有效配置`。
3. `硬件实测参考`。
4. `测试方法`。
5. 一张 DRAFT / VERIFY 双饼图。
6. `DRAFT 调用顺序` 或 `DRAFT 算子调用顺序`。
7. `VERIFY 调用顺序` 或 `VERIFY 算子调用顺序`。

DRAFT 和 VERIFY 表都使用以下列：

```text
顺序 | 模块/函数 | 归属类别 | 简短作用 | 输入/权重/输出 shape | 次数/replay |
总耗时 | 单次耗时 | 算力利用率 | 带宽利用率
```

表中应是展开后的模型调用链。`draft`、`DRAFT_EXTEND_V2`、`TARGET_VERIFY`、`step[...]` 等外层 annotation 只能作为阶段边界或未展开摘要，不能构成最终表的全部内容。对于 Qwen MTP 类报告，通常应能看到 embedding、norm、QKV、attention/KV、GDN 或 full-attention、MoE/gate、TP collective、logits 和 sampling 等可由证据支持的连续片段；不能识别的片段保留为“未映射”并在覆盖文件中说明。

## 时间和计数

“归属类别”逐行记录 `GEMM`、`Element-wise` 或 `Other`，与饼图图例及发布数据使用完全相同的名称；实际存在通信时才增加 `Communication`。按该列汇总的时间必须与饼图分类时间一致，不能为分类重新排序调用表。

每一行保留展开后的 `expanded_calls`。正文默认显示每个 decode step 的平均 GPU kernel duration：

```text
row_total_ms = mean(该行在选定 rank/replay 中拥有的 kernel duration 之和)
single_us = row_total_ms * 1000 / expanded_calls
```

正文“次数/replay（单 step）”只显示一个 step 的展开调用次数；多次 replay 的总次数、selected cycle 数和累计值只放在审计数据中。阶段首尾跨度、`scheduler.run_batch` 和 decode 周期起始间隔单独命名，不能加到 DRAFT/VERIFY kernel 合计中，也不能把多次合计当作一个 step。

Graph trace 必须记录事件类别。`gpu_user_annotation` 表示 device lane 上的阶段范围，`user_annotation` 表示 CPU 提交/调度范围；DRAFT/VERIFY 主表使用前者关联的 kernel。父 GPU scope 包含子 scope 时，父项的 kernel 合计扣除子项，父项 wall span 与子项 kernel duration 分开呈现。

折叠行必须能按层范围、嵌套次数、shape 变体和事件 id 展开回原始序列。次数已经包含外层重复，读者不能再次相乘。饼图和表格必须来自同一份发布数据，使用相同的 rank/replay 选择和时间口径。

## W4 / W4A8 规则

环境表同时写出权重格式、激活/计算路径和硬件参考的适用范围。硬件没有 INT4 峰值时，W4A8 的算力利用率统一以同型号 INT8 GEMM 实测上限作参考，并明确标记 `INT8 reference`，不能声称为 INT4 峰值。W4 权重、MTP/BF16 权重和 FP8 KV 的逻辑字节必须来自 checkpoint tensor headers、模型配置和访问公式。带宽利用率使用逻辑字节除以已有 HBM copy 实测上限，标记为“逻辑/参考”；它不是 HBM 物理计数器，未扣 cache reuse 时可以超过 100%。

正文必须有阶段资源利用率表或等价图，至少包含 DRAFT、VERIFY 的 GPU kernel 时间、估算 FLOPs、逻辑字节、INT8 算力利用率和 HBM 参考带宽利用率。算子表对主要 GEMM、MoE、KV 行填入对应值，只有无法建立可信公式的细小辅助项可以保留 `—`。

## 完成门槛

- DRAFT、VERIFY 均有展开后的调用顺序表；每个有数据的阶段至少包含 8 行可辨识的模块或函数。只有 3–4 个外层 annotation 行时，状态必须为 `incomplete`。
- 每一行能追溯到 `source-map.json` 或明确标记为未映射；shape、次数和时间口径不能用示例值填充。
- 展开序列、行时间、分类时间、饼图和 `analysis-data.json` 通过时间守恒与次数检查。
- 图表无零时间空扇区；报告中的数字与发布数据一致。
- 图必须包含本次发布数据的类别时间、Graph wall/cadence 参考和阶段利用率；绘图失败不能继续使用旧 PNG，必须重新生成或显式失败。
- `validation.json.status` 为 `complete`。任何缺口、覆盖未证明或 annotation-only 结果都必须在正文或 `coverage.json` 中保留，并将状态设为 `incomplete`。
