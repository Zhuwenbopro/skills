# Qwen3.8-27B Graph 已验证配方（2026-11-04）

本页记录历史验证条件。表中的节点、容器和路径是示例，运行前按目标环境替换，并重新检查设备、进程和软件版本。

| 项目 | 已验证值 |
|---|---|
| SSH / 容器 | hcu-node / sglang-container（示例） |
| 服务设备 | HIP_VISIBLE_DEVICES=0,1；清除冲突的 ROCR_VISIBLE_DEVICES、CUDA_VISIBLE_DEVICES、GPU_DEVICE_ORDINAL |
| 其他设备 | 本次 2、3 原占卡进程保留；不触碰 4–7。新任务按实际分配，不假定这些卡属于自己 |
| 模型 / served name | /models/Qwen3.8-27B / Qwen3.8-27B |
| 架构 | Qwen3_5ForConditionalGeneration，64 层，其中 48 GDN、16 full attention |
| Python | /workspace/sglang-deploy/venv/bin/python |
| PYTHONPATH | /workspace/sglang-deploy/src/adapter-das:/workspace/sglang-deploy/src/sglang/python:/workspace/sglang-deploy/src/sglang-plugin-das |
| 服务入口 | HCU_ADAPTER_HOST=sglang，SGLANG_PLUGINS=hcu，python -u -m hcu_sglang |
| attention / linear | fa3 / triton |
| TP / dtype / KV page | 2 / BF16 / 64 |
| 其他参数 | mem-fraction-static=0.85，context-length=8192，max-running-requests=8，disable-custom-all-reduce，watchdog-timeout=600 |
| Graph | decode full bs=[1], max_bs=1；prefill disabled；普通生成，MTP 关闭 |
| 流策略 | 插件已有 KV pool enable_alt_stream=False；GDN 构造后 alt_stream=None。抓 trace 本身不授权改这些实现 |
| Torch / HIP | 2.11.0+dtk2610.2611031312.g529def / 7.2.26365 |

当前版本 Graph JSON 参数为：

```bash
--cuda-graph-config '{"decode":{"backend":"full","bs":[1],"max_bs":1},"prefill":{"backend":"disabled"}}'
```

已验证源 SHA（插件含前述未提交改动）：

- sglang：baa9befa381511cd4d56f83980951e5a45fffc6d
- adapter-das：c43693f3621c98a14c6ad65601400bc3b2b7d1f1
- sglang-plugin-das：0aa4114d9f7a170b658e87310190a2f824c8b6e8

加两个 FULL_PROFILE 启动变量、重新启动和 capture 后，相同单请求 1024→32，两个 rank 各 31 次 replay，每步 1015 个 kernel：64 SiLU、48 GDN layer norm、128 fused add RMSNorm、16 QK/RoPE、32 KV index_put、417 GEMM（含 PostGSU）、130 RCCL；相关 kernel 均位于当时的主流72。stream 编号不可跨进程固定。

原始证据：

- 归档包含逐 rank trace 和 coverage-comparison.json。
- 等待结论另使用最小复现及等待探针验证。


上述证据目录不可用时，Skill 的普通采集流程仍可独立执行；不要宣称历史证据已重新验证。
