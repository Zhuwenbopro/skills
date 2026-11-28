# Qwen3.8-27B 已验证部署配方

验证日期为 2026-11-04。下文的节点、容器、目录、设备编号和端口都需要按实际环境替换。历史配置只适用于对应版本、镜像和模型。

## MTP 配方（2026-11-05）

兼容的 Qwen3.8-27B 部署可参考 [TP2 MTP profile](qwen38-mtp-tp2-example.json)，参数为 `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`。执行前按当前资源调整卡和端口，重新核对源码版本。启用 MTP 时沿用任务确认的 Graph 和流设置。下文的 TP4 参数与验收对应 2026-11-04 的非 MTP 基线。

2026-11-05 已在 hcu-node、Qwen3.8-27B BF16 TP2、卡 2/5 验证该 MTP profile（Graph 关闭）。三道文本题均正确且正常停止；64-token 原生生成请求记录 22 次 verify、66 个 proposed drafts、43 个 correct drafts。`/get_server_info` 的配置在该版本直接位于 JSON 根层，NEXTN 显示为 EAGLE。该短请求仅证明 MTP 实际执行，不能用其接受率或耗时当作性能基准。当前 adapter/plugin 为 `release/sglang_v0.5.22` 的 `3ad198a3eefcf858ffa425ed1ff5f03543ab4391` / `40dee477cc3e63e038f420e2365916a4921624a1`；官方仍为下表的 v0.5.22 commit。

## 节点与容器惯例

- 使用自己环境中的 SSH 别名和现有认证配置。需要跳板时沿用已有 SSH config。
- 每次连接前核对用户本次指定的节点以及已有容器归属。
- 示例容器名为 `sglang-container`。容器缺失时使用用户提供的创建流程。
- 历史环境的镜像含 Ubuntu 22.04、DTK 26.10 与 Python 3.12。以用户批准的镜像、实际安装的组件和启动入口为准。
- 宿主 hy-smi 编号与占卡结果已对照；目标设备 gfx936/BW200、每卡约 63.98 GiB。

## 模型与代码

历史模型为 Qwen3.8-27B。示例路径为 `/models/Qwen3.8-27B`，需按实际挂载修改。18 个权重分片，约 55.6 GB。config 架构是 Qwen3_5ForConditionalGeneration，BF16、64 层 full/linear 混合注意力，attn output gate 打开。模型来源保持只读挂载。

| 仓库 | 分支/版本 | 代码 commit |
| --- | --- | --- |
| 官方 SGLang | v0.5.22，detached | baa9befa381511cd4d56f83980951e5a45fffc6d |
| adapter-das | sglang_v0.5.22 | c43693f3621c98a14c6ad65601400bc3b2b7d1f1 |
| sglang-plugin-das | sglang_v0.5.22 | 0aa4114d9f7a170b658e87310190a2f824c8b6e8 |

包：sglang 0.5.22、sgl-kernel 0.1.0、hcu-sglang-plugin 0.3.2.dev77。要精确复现用这些 commit；要跟进更新则 fetch 当前目标分支并重新验证。

## 环境与参数

独立 venv `/workspace/sglang-deploy/venv` 继承镜像依赖：Torch 2.11.0+dtk2610、Triton 3.6.0+dtk2610、AITER 0.1.8+dtk2610、LightOp 0.6.0+dtk2610、TileLang 0.1.15+dtk2610、apache-tvm-ffi 0.1.11。

本次在 venv 额外安装/固定：Transformers 5.19.0、tokenizers 0.23.2、compressed-tensors 0.18.0、outlines 0.1.11、outlines-core 0.1.26、blobfile 3.0.0、kernels 0.14.1、soundfile 0.13.1；以及 distro、easydict、IPython、zstandard、nest_asyncio、diskcache、airportsdata、pycryptodomex、kernels-data、tomlkit 和构建工具。按实际依赖补齐，不重复覆盖已匹配包，也不将此清单视为通用锁文件。

启动参数由 [qwen38-tp4-example.json](qwen38-tp4-example.json) 提供：卡 0–3、TP4、fa3 attention、triton linear attention、page-size 64、mem-fraction-static 0.85、context 8192、max-running-requests 8、CUDA Graph 关闭、custom all-reduce 关闭、watchdog 600。

默认示例省略 TOKENIZERS_PARALLELISM：本配方模型走 SGLang v0.5.22 的多模态 tokenizer 初始化分支，该分支在 `sglang/srt/managers/tokenizer_manager.py` 设置 false。原始成功脚本曾显式设 false，但它不是本配方必须再加的环境变量；这是通过当前分支源码确认的省略，未为此重启健康服务。其他模型/启动入口应查看有效配置，不泛化为 Tokenizers 库本身默认关闭。

## 结果与范围

服务健康 200，模型列表身份 Qwen3.8-27B。三次文本生成均返回预期答案及 finish_reason=stop。最终测试卡显存占用约 86–87%，占卡程序已交接给模型。

仅验证文本服务、无 Graph；未验证图片、音频、视频、长文、并发或吞吐。可选图像生成 Hook 有加载错误，不能宣称全量插件合同通过。三仓跟踪源码均干净。

部署目录 `/workspace/sglang-deploy/runs/model-example` 内有 start.sh、server.pid、server.log、deployment.json、validation.json、README.md 和前两轮失败日志。历史 PID 已过期风险高，任何操作先读当前 PID 文件并检查归属。
