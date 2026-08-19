# 自动等待 GPU 并运行 SGLang + EvalScope

用户日常只需要重点修改两个命令文件：

- `server_command.sh`：直接粘贴 SGLang 环境变量和启动命令。
- `eval_command.sh`：直接修改 EvalScope 数据集、batch 和生成参数。

`config.env` 只保存等卡、端口范围、超时、重试和结果目录等调度参数；
`auto_eval.sh` 负责完整生命周期。`lib/server_command_parser.py` 是内部解析器，
通常不需要修改。

## 服务命令的自动解析

`server_command.sh` 支持以下两种启动形式：

```bash
sglang serve --model-path /models/Qwen3-8B
--tp-size 2
--pp-size 1
```

```bash
python -m sglang.launch_server \
  --model-path /models/Qwen3-8B \
  --tp-size 2 \
  --pp-size 1
```

参数可以正常使用反斜杠，也可以像第一种形式一样逐行粘贴。解析器会得到：

| 参数 | 用途 | 缺省值 |
| --- | --- | --- |
| `--model-path` | 模型路径，并自动推导模型名 | 必填 |
| `--served-model-name` | EvalScope 使用的模型名 | 模型目录名 |
| `--tp-size` | TP 大小 | `1` |
| `--pp-size` | PP 大小 | `1` |

需要的 GPU 数量自动计算为：

```text
REQUIRED_GPUS = TP_SIZE × PP_SIZE
```

下面两个粘贴值不会直接使用：

- `--port`：由 `auto_eval.sh` 在 `config.env` 指定的范围内自动寻找并覆盖。
- `HIP_VISIBLE_DEVICES`：由空闲 GPU 检测结果覆盖。

因此可以直接复制原有启动命令，不需要先删除固定卡号和固定端口。

服务文件中只能放简单的 `export NAME=value`、变量赋值和一条 SGLang 命令；
不要添加 `nohup`、`&`、管道、日志重定向或命令替换，因为后台运行和日志由
`auto_eval.sh` 统一管理。

## EvalScope 命令

评测 batch、数据集和 generation config 直接放在 `eval_command.sh`：

```bash
BATCH=64
DATASETS=(math_500)
```

它可以使用等待器提供的变量：

| 变量 | 含义 |
| --- | --- |
| `MODEL_PATH` / `MODEL_NAME` | 从服务命令解析出的模型路径和模型名 |
| `TP_SIZE` / `PP_SIZE` | 从服务命令解析出的并行大小 |
| `HEALTH_HOST` / `PORT` | 本轮实际服务地址和自动选择的端口 |
| `HIP_VISIBLE_DEVICES` | 本轮自动选中的 GPU |
| `RUN_DIR` | 本轮结果目录 |

## 启动

修改命令后，在当前 automation 目录执行：

```bash
mkdir -p /home/eval_results
cd /path/to/eval/automation
nohup bash ./auto_eval.sh > /home/eval_results/auto_eval.log 2>&1 &
echo $!
```

查看日志：

```bash
tail -f /home/eval_results/auto_eval.log
```

停止：

```bash
pgrep -af '[a]uto_eval\.sh'
kill -TERM <启动时输出的 PID>
```

收到 `TERM` 或 `INT` 后，脚本会停止本轮 EvalScope 和整个 SGLang 进程组，
然后释放 GPU 锁。

## 失败处理

以下情况都会清理资源并重新等卡：

- 服务启动失败或超过启动超时。
- EvalScope 返回非零退出状态。
- 评测期间 SGLang Server 意外退出。

模型路径不存在、命令无法解析等静态错误会在进入循环前直接退出。

当前 `rocm-smi` 解析方式沿用原环境格式：GPU 编号、显存占用和 HCU 利用率
分别位于第 1、6、7 列。如果机器输出格式不同，需要调整 `auto_eval.sh` 中的
`query_free_gpus`。
