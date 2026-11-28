---
name: hcu-performance-report
description: "读取服务器已有的 SGLang trace、运行配置和硬件基线，分析 DRAFT / VERIFY 的真实调用顺序、算子耗时及利用率，生成或更新以表格和双饼图为主的简洁性能报告。用户要求整理已有 profiling 结果、重写性能报告或更新飞书分析文档时使用；不负责发起测试、采集 trace、运行硬件基准或部署和重启服务。"
---

# HCU 性能分析报告

将已有测量数据整理成可核查、简洁的报告。分析证据保存在服务器，正文只呈现核心数据。完整报告必须遵守 [可交付报告契约](references/report-contract.md)，并通过 `scripts/validate_report_bundle.py`；只有外层 annotation 摘要的结果只能标为 `incomplete`。

## 范围与执行位置

- 只读取已有 trace、基线、请求记录、日志、模型源码和配置；可以在服务器编写并执行分析、绘图与校验脚本。
- 不发起模型请求或性能测试，不抓新 trace，不跑硬件基准，不部署、重启或调整模型服务。缺少测量资料时列明缺项，由用户另行安排采集。
- 项目脚本的编写、修改、运行、trace 分析、绘图和中间产物默认全部在目标服务器。只有用户明确要求传输脚本时才回传本地，并限定在其授权范围。
- 不为编辑方便先下载脚本再上传，不以本地副本替代服务器工作目录。发布时可以使用已认证客户端，在内存中桥接必要的文档内容；这不授权下载脚本或原始 trace。
- 这个 skill 的本地安装文件是用户明确要求的交付物。它不改变上述服务器执行纪律。

## 工作流程

1. **定位已有资料。** 优先从当前项目和用户给定路径确认服务器、容器、结果目录及目标文档。先检查可达性和文件存在性；节点、卡号、进程和服务状态不能直接沿用历史记录。不要扫描无关目录或输出凭据。
2. **核对测量身份。** 确认 trace 的时间、节点、卡、模型、TP、dtype、代码版本、Graph、MTP、流设置，以及真实输入/输出长度、并发和采样条件。报告捕获时的配置；当前进程配置只能作为补充核验，不能覆盖历史捕获配置。证据冲突保存在内部，缺证据写“—”。
3. **分析调用与时间。** 阅读 [分析与校验口径](references/analysis.md)，建立逐 rank、逐 replay 的实际执行序列，关联模块、函数和 shape，计算实测耗时及有依据的利用率。使用已有硬件基线，不补跑基准。
4. **按契约折叠与排版。** 先阅读 [可交付报告契约](references/report-contract.md)，再阅读 [报告版式与发布](references/report-format.md)。重复片段按真实层次折叠；DRAFT、VERIFY 各一张展开后的顺序表，新增“归属类别”列，逐行使用与饼图一致的类别名称；保持实际调用顺序。
5. **生成固定报告包。** 在服务器报告目录生成 `README.md`、`draft-verify-kernel-time.png`、`analysis-data.json`、`source-map.json` 和 `coverage.json`。`source-map.json` 至少记录模块/函数、CPU op、correlation、kernel、shape、rank/replay 和证据状态；`coverage.json` 至少记录已映射、未映射、未证明覆盖和 annotation-only 缺口。不要只生成一张图或一张阶段摘要表。
6. **运行自动校验。** 执行 `python scripts/validate_report_bundle.py <report-dir> --write-validation`，并检查生成的 `validation.json`。校验未通过时，交付状态必须为 `incomplete`，列出缺失映射或覆盖证据；不能用“报告已生成”掩盖失败。
7. **生成或更新文档。** 用户要求写入文档即授权对应编辑范围，无须重复询问。只在明确要求整篇重写时整篇替换；否则原位更新相应内容。飞书操作使用当前可用的 lark-doc skill / 已认证工具，按其文档格式和版本约束执行。先在服务器留存原文与待发内容，写后回读核验。
8. **简短交付。** 返回文档或服务器产物位置、`complete/incomplete` 状态、校验结果和实质缺项。不要把内部审计清单、路径清单和计算推导再追加到报告正文。

## 必须保留的判断

- 时间来自实际 kernel 测量，不能按 FLOPs 比例分摊某个总耗时。
- 用源码、调用注解和关联信息确认模块名；只有 kernel 名时，不冒充已识别模型函数。
- 折叠表示展开后必须仍是实际调用顺序；不能把非相邻同名函数汇总成伪调用链。
- 参数量、FLOPs、映射证据、字节口径和校验产物留在服务器；正文不展示这些展开细节。
- 未知、不适用或缺少可信分母的指标用“—”；不填零、不造数。
- 单次案例中的模型、层数、MTP 参数、卡号、文档 token、测试长度和时间数值都不是模板默认值。

## 本项目的时间与量化约定

- DRAFT / VERIFY 正文默认以每个 decode step 的平均 GPU kernel duration 为主口径。调用顺序表中的“总耗时”对应单 step 平均周期耗时，“单次耗时”按单 step 展开调用次数计算；“次数/replay”列只显示一个 step 的调用次数，不显示 9 个 replay。多次 replay 的原始合计和 selected cycle 数保存在服务器审计数据中，只有用户明确要求累计量时才放入正文主表。
- 必须分开标注 GPU kernel duration、阶段首尾跨度、scheduler.run_batch annotation、相邻 decode 周期起始间隔和请求 E2E。它们不能互相替代，也不能把多次合计误写成一个 step 的耗时。
- Graph trace 中同名的 `user_annotation` 与 `gpu_user_annotation` 不是同一口径：`user_annotation` 主要反映 CPU 提交/调度范围，`gpu_user_annotation` 才是 Perfetto device lane 上的 GPU scope。DRAFT/VERIFY 的 Graph 阶段时间必须优先从 `gpu_user_annotation` 关联 kernel；CPU annotation 只能作为 host 调度补充，不能用较短的 CPU `TARGET_VERIFY` duration 代替 GPU `TARGET_VERIFY`。
- GPU scope 可能嵌套。父 scope（例如 `draft_extend`）的 wall duration 可以包含 `DRAFT_EXTEND_V2`；计算唯一 kernel duration 时必须扣除子 scope 的 kernel，再与子 scope 分开列出。Perfetto 的 annotation wall duration、kernel self duration 和 scheduler wall span 必须分别命名。
- W4A8 / W4 权重与 HCU 的 INT8 计算路径分开记录。已有 INT8 GEMM 基线只能作为同型号计算参考，不能直接换算成 W4A8 MoE 小 batch 的有效算力。
- 对 decode 是否带宽受限必须给出基于 batch、权重格式、KV/状态访问和 kernel 名的工程判断，并把模型逻辑流量除以已有 HBM copy 实测上限，填入“带宽利用率（逻辑/参考）”。这不是 HBM 物理计数器读数；如果没有逐 kernel 字节量，使用服务器 checkpoint tensor headers、模型配置和可复核的访问公式估算，并在表头或脚注标明“逻辑/参考”。

## 参考报告的最低交付形态

- 正文顺序固定为：运行环境与有效配置、硬件实测参考、测试方法、DRAFT / VERIFY 类别饼图、DRAFT 算子调用顺序、VERIFY 算子调用顺序。分析不能用一张粗粒度摘要表替代后两张调用顺序表。
- DRAFT / VERIFY 表格必须尽量列出真实模块或函数、短作用、输入/权重/输出 shape、次数/replay、每 rank/replay 平均总耗时、单次微秒耗时、算力利用率和带宽利用率。只看到外层 annotation 时，先继续用 correlation、源码、模型配置和 kernel 名做映射；无法唯一映射的行标明缺口，不能把 draft、TARGET_VERIFY 等外层 annotation 冒充完整模型调用链。
- 重复结构按真实层级折叠：保留 MTP forward、GDN 层组、全注意力层组、残差边界、TP collective、logits 和采样等相邻顺序。折叠行必须能展开回原始调用顺序，次数已经包含外层重复，不能让读者再乘一次。
- 饼图和表格的“归属类别”列从同一份发布数据读取，名称完全一致。本次无通信的报告使用 `GEMM`、`Element-wise`、`Other` 三类；有通信时按证据保留 `Communication`。分类用于逐行标注及饼图汇总，不改变算子顺序。零时间类别不绘制空扇区或空图例。
- W4A8 / W4 报告必须同时写清权重精度、激活/计算 kernel 路径和硬件参考的适用范围。硬件没有 INT4 峰值时，算力利用率统一用同型号、同卡的 INT8 GEMM 实测上限作参考，并明确这是 INT8 reference，不得声称为 INT4 峰值。W4 权重按服务器实际 packed tensor 与 scale 字节数计算逻辑流量；MTP、KV 和状态访问按其实际 dtype/shape 单独计算。
- 正文必须增加阶段资源利用率表或等价图：至少给出 DRAFT、VERIFY 的 GPU kernel 时间、估算 FLOPs、逻辑字节、INT8 算力参考利用率和 HBM copy 参考带宽利用率。算子表对可由 checkpoint shape 与 kernel 语义计算的 GEMM/MoE/KV 行填数；无法建立可信公式的细小辅助项可保留“—”，但不能让整张表的利用率列为空。
- 图必须从本次 `analysis-data.json` 重新绘制，并同时标明 GPU kernel duration、Graph wall span/cadence 和利用率参考；不能只保留旧的类别饼图。绘图失败时必须使用可验证的备用绘图路径或报错退出，禁止保留旧 PNG 冒充新图；发布前核对 PNG 修改时间和图中关键数字。

## 完成定义

- `README.md` 按固定顺序包含环境、硬件、方法、双饼图以及 DRAFT/VERIFY 调用顺序。
- DRAFT、VERIFY 每个有数据的阶段至少有 8 行可追溯的展开调用；只有 `draft`、`DRAFT_EXTEND_V2`、`TARGET_VERIFY` 等外层 annotation 的 3–4 行摘要属于中间结果。
- Graph 阶段表必须记录事件类别选择（GPU user annotation / CPU user annotation）和嵌套去重规则；不能根据 CPU annotation 比 GPU device scope 更短就宣称存在空泡。只有 lane 级 GPU 时间线、周期起始间隔和调度范围共同支持时，才可下空泡结论。
- 每一行的次数已含折叠重复，shape、时间和分类来自同一份发布数据；`source-map.json` 和 `coverage.json` 明确证据边界。
- `validation.json.status` 为 `complete`。未通过校验、只能保留粗粒度映射或 Graph 覆盖未证明时，必须保留 `incomplete`。
