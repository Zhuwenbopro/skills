# Copilot Skills

| Skill | 功能 |
|---|---|
| `eval` | 自动调度 GPU、启动 SGLang，并使用 EvalScope 运行指定数据集评测。 |
| `compile-sglang` | 在 HCU/ROCm 环境中编译并安装 SGLang、`sgl-kernel` 及相关依赖。 |
| `experimental-bug-investigation` | 通过真实复现和可证伪实验调查 Bug，持续记录证据、根因与验证结果。 |
| `bug-experience-writing` | 将已解决且已验证的 Bug 提炼为简洁经验条目；由 Bug 调查流程内部调用。 |
| `hello-skill` | 生成一句问候，用于验证 Skill 的发现和调用。 |

## 调用案例

在 Copilot Chat 中输入 `/eval`，指定数据集并粘贴完整的 SGLang 启动命令：

```text
/eval 使用下面的服务命令测试 humaneval

export SGLANG_ENABLE_SPEC_V2=1
unset NCCL_TOPO_FILE
sglang serve --model-path /models/Qwen3.6-35B-A3B \
  --tp-size 2 \
  --mem-fraction-static 0.8
```

`eval` 会等待可用 GPU、选择端口、启动并检查服务，然后运行 HumanEval 评测。
