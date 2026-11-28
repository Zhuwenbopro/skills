# 带宽校准与反汇编

在目标节点的 DTK 环境、已锁频的预留卡上执行。先将 GPU 物理编号与进程可见卡对应，从 `hcu_kernel` 仓库根目录执行。下例的 `GPU` 是本次选定的卡。

```bash
source /opt/dtk/env.sh
bash hbm_read_gfx936/build.sh
HIP_VISIBLE_DEVICES="$GPU" python3 hbm_read_gfx936/benchmark.py
```

gfx936 的参考点来自流式纯读，约 1.50 TB/s。[gfx936 HBM 读带宽程序](https://github.com/Zhuwenbopro/hcu_kernel/tree/main/hbm_read_gfx936)

```bash
python3 hbm_read_gfx938/build.py
HIP_VISIBLE_DEVICES="$GPU" python3 hbm_read_gfx938/benchmark.py --blocks 64 --depth 12 --cu-count 64 --output /tmp/hbm64-q12.json
```

gfx938 在 64 个开放 CU、64 CTA、12 组滚动加载下记录了 2.066 TB/s 有效读带宽。本技能以 2.0 TB/s 建模。复测时记录目标机实际开放 CU、频率、访问宽度及工作集。对大量写入或读写混合算子，同时用匹配流量比例的探针量化纯读参考与实际吞吐的差距。[gfx938 HBM 读带宽程序及结果](https://github.com/Zhuwenbopro/hcu_kernel/tree/main/hbm_read_gfx938)

**L2 的 6.0 TB/s 是项目给定的建模参考值**。以上两个程序测量 HBM 读取。校准 L2 时将工作集控制在有效 L2 容量内，同时大于 CU 私有缓存有效容量，先预热再从全局地址反复读取，保留访问并检查 ISA。扫描容量和并发度，找稳定吞吐平台；可用计数器时核对 L2 命中及同次 dispatch 的 HBM 流量。测 HBM 则使用明显大于 L2 的工作集或轮换缓冲。分别记录逻辑字节和硬件计数字节、读写方向、计数层级、同次 dispatch 的时间。新增机型应填写实测率，`floor.py` 可用 `--hbm-tbps`、`--l2-tbps` 覆盖参考值。

**ISA 检查**使用该架构随 DTK 发布的工具，对实际启动的内核符号归档 hsaco、编译参数和反汇编。若为 JIT 实现，取该 shape/特化与当前启动配置的缓存产物。例：

```bash
arch=gfx936
"${DTK:-/opt/dtk}/aillvm/bin/llvm-objdump" -d --mcpu="$arch" kernel.hsaco > kernel.isa
rg -n 'global_load|buffer_load|global_store|buffer_store|ds_read|ds_write|s_waitcnt|s_barrier|scratch|mfma' kernel.isa
```

定位热循环的 prologue、steady state 和 drain。对同一组 VGPR 或同一 LDS 缓冲追踪“发起 → 第一次消费 → 可安全覆写”。核对每轮的未完成读数量、第一条消费者之前的 wait 和循环携带的依赖。太早的全等待会缩短可重叠区间，过深的预取可能导致 VGPR spill。纯访存探针应保有生产路径的地址序列和数据宽度；加回计算后再对照两份 ISA，并同时报告吞吐、寄存器和占用变化。
