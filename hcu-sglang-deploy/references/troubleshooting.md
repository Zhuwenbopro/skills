# 按首条原因处理

先保存当前日志、参数、SHA、环境和进程状态；不要在旧进程仍编译/占卡时并发启动第二份。每次只根据新证据调整适用配置。需要修改 plugin 实现时读目标仓 hcu-plugin-triage / hcu-plugin-dev；常规部署修正参数不必改源码。

## SSH 与脚本传输

- 沙盒内跳板机 Permission denied、沙盒外成功：按执行工具权限机制处理网络，不改密钥或关主机校验。
- BatchMode 下认证失败，但默认流程成功：保留默认认证及现有 askpass/用户机制。
- Banner exchange timeout：先区分未执行与 SSH 中断。确认原动作状态后有限重试；同时建立两条跳板 SSH 在本次曾失败，串行操作可降低干扰。
- 远程 `$repo` 消失、PowerShell ParserError、bash unexpected EOF：使用 remote_exec.py。其 UTF-8 编码与换行规范化避免双层 shell 解释，不用 JSON.stringify 代替 shell 转义。
- script 最后 docker exec -it 报 input device not a TTY：检查容器和初始化结果；创建成功无需重跑整套脚本。

## tvm-ffi 冷缓存递归

症状：只停在 one-time build，父子 Python 等锁，编译器没在工作。Editable 安装的 sitecustomize 仍会自动加载，剥除 PYTHONPATH 不足以断开递归。先停止/等待本次失败的自有预检查进程，确认没有并发编译；随后在目标解释器下以 HCU_ADAPTER_HOST=none 手工构建真实 HIP 扩展。

将下面作为容器 Python 脚本，通过设置 HCU_ADAPTER_HOST=none 的 shell 调用（P 为目标 venv Python）：

```python
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import torch

assert torch.version.hip, "Must use the actual HIP Torch environment"
spec = importlib.util.find_spec("tvm_ffi")
helper = Path(spec.origin).parent / "utils/_build_optional_torch_c_dlpack.py"
tag = "torch" + "".join(torch.__version__.split(".")[:2])
cache = Path.home() / ".cache/tvm-ffi"
cache.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.pop("PYTHONPATH", None)
env["HCU_ADAPTER_HOST"] = "none"
subprocess.run([sys.executable, str(helper), "--output-dir", str(cache),
                "--libname", f"libtorch_c_dlpack_addon_{tag}-rocm.so",
                "--build-with-rocm"], env=env, check=True, timeout=300)
assert (cache / f"libtorch_c_dlpack_addon_{tag}-rocm.so").is_file()
```

成功后恢复 HCU_ADAPTER_HOST=sglang 再检查入口。不得为了跳过编译伪造空 .so。跨容器拷编译结果必须另证 Torch/ABI 一致，不作为默认步骤。

## 导入链与版本

- 首条 ModuleNotFoundError 后大量 KeyError/AttributeError：优先补真实缺失包，再在新进程复验；不要修补被早期失败污染的导入缓存。
- IPython、blobfile、outlines 或 kernels 的传递依赖缺失：检查目标环境 metadata 的 Requires-Dist，按实际 marker 补齐；已验证缺口见 known-recipe.md。
- `--no-deps` 安装成功但启动失败：它跳过全部依赖安装，不能等同于就绪。
- metadata 与服务版本不一致时，核对 sys.executable、sys.prefix、PYTHONPATH、distribution location 和 module __file__。容器默认 PYTHONPATH 可能遮蔽 venv 中的包。显式使用服务的路径配置再次检查。
- 可选 audio/diffusion/import soft errors：区分当前模型相关与未要求能力。当前任务相关失败必须处理；无关缺口保留限制，不通过删日志、mock 或 broad fallback 宣称全通过。

## HCU KV pool 与注意力

### 四维缓存送入官方 Triton

本次 plugin 的 `hcu_sglang/mem_cache/hcu_mha_pool.py` 默认 SGLANG_KV_LAYOUT_HCU_FA=true，生成四维 K[pages,H,page,D]、V[pages,H,D,page]。官方 v0.5.22 `_extract_kv_strides` 要求三维，因而报 unexpected KV buffer ndim=4。

已验证修正是选择 plugin 配套 `--attention-backend fa3`，保留 `--linear-attn-backend triton`。不要未经完整契约验证就把 `.view()` 加进官方源码。也不能仅关一个布局变量而不检查融合 RoPE/KV-store 等其他使用方。

### gfx936 paged KV 页大小

原生调用报 `Paged KV cache block size must be divisible by 64 at gfx936 platform` 时，当前配方显式 `--page-size 64`。核对新 server_args 的实际页大小，并重新跑服务预热和真实 decode，不能只看参数解析成功。

### Graph 与资源

先用任务接受的简单路径部署并说明是否关闭 Graph；不能用关闭 Graph 的结果证明 Graph lane 正确。Qwen3.8-27B 本次基线使用 eager。

健康监听后 503 + HCU JIT compiling 可能只是冷预热。观察真实 compiler/日志进展，不立即重启。同一编译一直无进展时检查子进程和锁。已发生错误时框架可能等待 coredump 一分钟；先明确实际失败，不当作正常加载。

服务失败后的清理只针对本次完整进程组。检查显存确已释放，再按用户占卡要求保留仍归本任务的卡；不清理其他用户进程。

## 验收不能降格

检查原始 message、模型名和 finish_reason。只看到数字、health 200 或响应长度很短均不充分。max_tokens 截断、输出错误、模型不符、server crash 都记失败。仅三道文本题通过不等于全插件目标、精度基准、多模态或性能通过。
