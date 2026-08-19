# Copilot Skills

这个仓库保存可复用的 VS Code Copilot Agent Skills。

## 包含的 Skills

### `eval`

用于自动运行 SGLang + EvalScope 评测，支持：

- 自动等待空闲 GPU
- GPU 文件锁，避免多个任务抢占同一张卡
- 自动选择服务端口
- SGLang 健康检查
- EvalScope 评测
- 失败清理和自动重试
- 多数据集和数据集专属 generation config

### `hello-skill`

用于验证 Skill 是否能被 Copilot 发现和调用的简单示例。

## 使用方法

### 1. 将仓库放到工作区

克隆仓库：

```bash
git clone https://github.com/Zhuwenbopro/skills.git
```

如果希望在当前 `/home` 工作区中使用，可以确保目录结构如下：

```text
/home/
└── .github/
    └── skills/
        ├── eval/
        │   ├── SKILL.md
        │   └── automation/
        └── hello-skill/
            └── SKILL.md
```

在 VS Code 中打开包含 `.github/skills` 的工作区，然后重新加载窗口。

### 2. 调用 `eval`

在 Copilot Chat 中输入：

```text
/eval 使用下面命令测一下 math500
```

然后粘贴完整的 SGLang 启动命令，例如：

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HIP_VISIBLE_DEVICES=0,1
unset NCCL_TOPO_FILE
sglang serve --model-path /models/Qwen3.6-35B-A3B \
    --tp-size 2 \
    --pp-size 1
```

Skill 会自动完成：

1. 清理 `nohup`、固定端口、固定 GPU、日志重定向和后台符号。
2. 将命令写入 `eval/automation/server_command.sh`。
3. 解析模型路径、TP、PP，并计算所需 GPU 数量。
4. 等待符合条件的空闲 GPU。
5. 自动选择端口并启动 SGLang。
6. 等待 `/health` 检查通过。
7. 使用 EvalScope 执行指定数据集。
8. 失败时清理进程和 GPU 锁并重试。

标准数据集由 EvalScope 根据数据集名称自行解析，不需要手动创建本地测试集目录。
例如：

```text
/eval 测试 humaneval
/eval 测试 math500,gsm8k
/eval 开启 thinking，测试 humaneval
```

默认配置：

```text
thinking=false
GPU_VRAM_MAX_PERCENT=6
```

### 3. 查看日志

默认调度日志：

```bash
tail -f /home/eval_results/auto_eval.log
```

每轮运行结果位于：

```text
/home/eval_results/<model-name>/<timestamp>_attempt<N>/
```

其中通常包括：

```text
server.log
eval.log
```

### 4. 停止评测

使用启动 Skill 后返回的 scheduler PID：

```bash
kill -TERM <scheduler-pid>
```

调度器会停止 EvalScope、SGLang 进程组并释放 GPU 锁。

## 配置文件

`eval/automation/config.env` 保存调度参数，例如：

```bash
EVAL_DATASETS="math_500"
EVAL_ENABLE_THINKING="false"
GPU_VRAM_MAX_PERCENT=6
GPU_HCU_MAX_PERCENT=0
GPU_POLL_INTERVAL=30
MAX_ATTEMPTS=0
```

`eval/automation/server_command.sh` 保存 SGLang 环境变量和启动命令。

`eval/automation/eval_command.sh` 保存 EvalScope 执行逻辑。

不要在 `server_command.sh` 中加入 `nohup`、`&`、管道或日志重定向，这些由调度器统一管理。

## 新容器使用

新容器中只要将仓库挂载到工作区并让 VS Code 打开该工作区即可：

```bash
docker run --rm -it \
  -v "$PWD/skills:/workspace/.github/skills" \
  your-image
```

或者在容器中重新克隆：

```bash
git clone https://github.com/Zhuwenbopro/skills.git /workspace/.github/skills
```

注意：Skill 文件可以跨容器复用，但 SGLang、EvalScope、模型文件、ROCm 工具和评测运行环境仍需要在新容器中单独准备。

## 本地验证

运行 `eval` Skill 自带的调度器回归测试：

```bash
bash eval/automation/tests/test_retry.sh
```

检查脚本语法：

```bash
bash -n eval/automation/auto_eval.sh \
  eval/automation/config.env \
  eval/automation/eval_command.sh \
  eval/automation/server_command.sh

python3 -m py_compile eval/automation/lib/server_command_parser.py
```
