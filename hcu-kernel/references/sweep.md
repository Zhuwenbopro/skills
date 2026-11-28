# 逐点数据与交付

命令从本技能根目录运行，路径按实际实验目录填写。地板和诊断只用 Python 标准库；绘图使用 Python 3.10+ 和 Matplotlib。

```bash
python scripts/floor.py gemm --arch gfx936 --m 63 --n 8192 --k 5120 --tile-m 32 --tile-n 128 --x-item-bytes 1 --w-item-bytes 1 --y-item-bytes 2
python scripts/floor.py stream --arch gfx938 --read-bytes 8192 --write-bytes 256
```

第一例有 2 个 M 块、64 个 N 块，HBM 同时计入 BF16 输出，地板约 39.243 µs。第二例的 8448 B 读写地板为 0.004224 µs，调用启动等固定耗时会在这种小工作量下格外明显。编写扫描驱动时可直接导入 `floor.py` 的 `gemm_floor` 或 `stream_floor`，在每个 M 选定路由后填写对应地板。

按硬件、时钟、dtype、形状族、缓存和计时口径分组。每次扫描提供 CSV 或 CSV.gz，每个算子一行一个 M，覆盖声明区间的每个整数。保留每轮原始时序，单独保存绘图用的中位数表。表头如下。

```csv
op,m,floor_us,optimized_us,baseline_us,ops,peak_tops,mem_us
```

- `op` 是本组算子名，`m` 是 batch 或每 rank 的 M。三个延迟列均以 µs 为单位，`optimized_us` 取已通过正确性检查的实际选路复测中位数。
- `baseline_us` 取 LightOP、原版或用户指定基线。使用相同输入、频率、缓存和计时范围逐点采集，整列逐点配对后才画加速比。对于刚编写的算子，尽早留存一版可正确运行的初始实现。如果完全没有可比实现，留空并说明加速比缺少分子。
- `ops` 为**有效**运算次数，GEMM 为 `2 × M × N × K`。`peak_tops` 用该设备、dtype、指令类型和时钟下的实测持续峰值。gfx936 INT8 在本项目对应条件下用 784 TOPS。算力利用率为 `ops / (optimized_us × peak_tops × 10⁶)`。展示百分比并保留超出 100% 的点供复核。对逐元素和规约，若有匹配的运算计数与峰值则同法计算，两者确实无法定义时留空，在图旁解释并报告带宽利用率。
- `mem_us` 为同布局、同访存范围的去计算探针中位数，供小 batch 诊断使用。在逐点结果旁记录 `tile_m/tile_n`、块数、路由、是否拆分、HBM/L2 模型字节、带宽来源，以及实测核心/HBM MHz，便于定位跳变。

```bash
python scripts/diagnose_sweep.py sweep.csv --start 1 --end 256 --out candidates.csv --split-independent
python scripts/plot_sweep.py sweep.csv --start 1 --end 256 --out-dir figures --title "gfx936 INT8 TP2" --baseline-label LightOP
```

诊断表内，M=1–64 的纯访存项以 `floor_us / mem_us` 为 ratio。90%–100%（含两端）达标，低于 90% 列为调优候选，高于 100% 保留原值并列为带宽与模型复核项。

另外三种信号的 ratio 为当前优化延迟除以相应参照延迟。倍数项的参照为 k 次小调用时间之和的估计，最终胜负要实测整个拆分路径。完整区间同时有基线时输出 `latency.png` 和 `speedup.png`。多个算子各占一幅子图。图中保留测得的台阶、回退和毛刺，优化结论以独立复测为准。
