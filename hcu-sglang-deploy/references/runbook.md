# 从节点到模型服务

## 1. 执行方式与连接

在本机调用 Skill 脚本时使用本机 Python；容器内命令使用容器解释器。不要把 Windows 路径传给 Linux Python。

把一阶段的 Linux 命令写成 UTF-8 文件，再通过：

```text
python <skill>/scripts/remote_exec.py --host hcu-node --file inspect-host.sh
python <skill>/scripts/remote_exec.py --host hcu-node --container sglang-container --file inspect-container.sh
```

脚本通过 subprocess 参数列表调用 SSH，不使用 PowerShell 对远程脚本做字符串展开。保留 SSH 已有 ProxyJump 和自动认证。工具中的本地超时或断线不代表远程进程已退出，重试前先检查。

容器不存在才执行用户提供的创建脚本，例如宿主的 `bash /path/to/run_docker.sh`。先确认脚本存在、基本语法和镜像及挂载参数。结束时检查 `docker inspect` 的状态与 Mounts，并验证 `docker exec ... pwd/whoami`。读取脚本时避免将认证设置带入日志或 Skill。

按实际 `docker inspect` 核对宿主模型目录到容器的只读挂载。确认任务工作目录可写。现有停止容器可启动。重建和删除已有容器需要用户明确要求。

## 2. 模型和资源

在改环境前运行 hy-smi。若用户已要求占卡，使用既有程序保持资源；尚无程序时，在确认分配和空闲后将 reserve_vram.py 放到任务部署目录，并用容器当前 HIP Torch 启动：

```bash
nohup python -u /home/<部署目录>/reserve_vram.py \
  --devices 0,1,2,3 --fraction 0.90 \
  --pid-file /home/<部署目录>/reservation.pid \
  </dev/null >/home/<部署目录>/reservation.log 2>&1 &
```

设备编号示意，按实际分配替换。查看 READY、PID 和 hy-smi；检查到 0% 利用率但显存占用高时，不将卡认作空闲。Torch 的空闲读数可能与 hy-smi 不一致，以两者共同检查及已知归属为依据。占卡不是调度器层面的独占保证。

模型读取只需 config.json、tokenizer 配置、分片索引、文件存在性与大小，不读出所有权重。解析 architectures、text_config、量化、attention heads/KV heads、层类型和最大上下文。确认模型完整性及所选 TP 支持；上下文和 max-running-requests 根据任务与显存选择。

## 3. 仓库认证和固定基线

每个仓库先检查存在性、origin、工作区、分支、tag 和 HEAD。克隆或 fetch 后再切换用户版本。已有修改不覆盖、不自动 reset/stash；必要时为本次部署建立新干净 checkout 并记录新路径。

两个 HYGON-AI 仓库在本次操作中需要 GitHub 身份。目标容器已有 gh 且已登录时直接 setup-git/fetch；没有 gh 时优先用允许的软件源或 GitHub CLI 官方 release 的匹配 Linux 架构二进制。软件源失联时不要无限 apt update；核对错误后选择官方分发方式。

在目标容器发起：

```bash
GH_BROWSER=true gh auth login --hostname github.com --git-protocol https --web
```

展示命令实际返回的网址和本次验证码，等待用户在其浏览器确认。保持该登录进程，用户回复后读取完成结果，再执行：

```bash
gh auth setup-git
git -C /workspace/sglang-deploy/src/adapter-das fetch origin
git -C /workspace/sglang-deploy/src/sglang-plugin-das fetch origin
```

不将网页登录当成所有节点共享认证，不转移另一个容器的 token 文件。不要在 Git origin 中放凭据。私仓远程更新成功后再声明该节点具备直连仓库能力。

本项目插件分支为 sglang_v0.5.22，官方 v0.5.22 是 annotated tag，应记录 `git rev-parse v0.5.22^{commit}`，不能把 tag object SHA 当成代码 commit。新版本始终阅读实际 pyproject 和入口，不沿用旧文档 dev34 的数字来替代当前版本验证。

## 4. 隔离环境和依赖

先记录镜像、系统包版本及三仓状态，然后在任务目录选择独立 venv。下面使用示例路径，运行前按当前容器修改。

```bash
python -m venv --system-site-packages /workspace/sglang-deploy/venv
P=/workspace/sglang-deploy/venv/bin/python
HCU_ADAPTER_HOST=none "$P" -m pip install setuptools-rust setuptools-scm wheel build
HCU_ADAPTER_HOST=none SGLANG_BUILD_RUST_EXTS=none "$P" -m pip install \
  --no-deps --no-build-isolation \
  -e /workspace/sglang-deploy/src/sglang/python -e /workspace/sglang-deploy/src/adapter-das -e /workspace/sglang-deploy/src/sglang-plugin-das
```

`SGLANG_BUILD_RUST_EXTS=none` 是该基线支持的构建选择：本次只用 REST，未编译可选 Rust 扩展；不是伪造 native 检测。需要依赖这些扩展的能力时重新核对构建要求，不把 REST 通过当作 Rust 能力通过。

读取项目 dependencies 和已安装 metadata，按实际 import 错误/模型需求补齐通用 Python 包及其传递依赖。不要运行一个会自动拉入 NVIDIA Torch、CUDA 库和全套 GPU wheel 的不受控安装。先补真实依赖，再判断其他 AttributeError 是否是早期导入失败的连锁结果。

已有 Python 包与模型配方存在明确版本要求时在 venv 内固定；不要改共享 site-packages。本次用到的具体包见 known-recipe.md，不能将其当作所有未来镜像的统一 requirements。

检查 metadata 和 import 来源时显式设置与服务一致的 PYTHONPATH。Wheel 模式不要加入 plugin 源码造成遮蔽；源码 editable 模式保留源码和安装 metadata。Adapter 的 sitecustomize 需要在 Python 启动时可发现，不能只在应用导入完成后补 sys.path。

运行前确认 tvm-ffi 已预热；按 troubleshooting.md 处理 editable 安装的自递归陷阱。先 `python -m hcu_sglang --help`，保留软错误；对本次模型相关的失败必须处理。若仅缺少与任务无关的可选能力，明确记录未覆盖范围，不能声明全合同通过。

## 5. 生成与启动

复制合适 profile 到本次任务目录，按当前模型改绝对路径、卡、端口及参数，然后生成：

```text
python <skill>/scripts/render_launch.py --profile <本次profile.json> --output <本地start.sh>
```

通过 remote_exec.py 的容器脚本模式写入 start.sh 和所需 helper；可将文件 UTF-8 bytes 编成 base64，在容器内解码到明确目标。避免本机 shell 转义或二进制管道损坏。

启动前再次检查端口和进程，核对占卡 PID 的完整 `/proc/<pid>/cmdline` 是本次程序，发 SIGTERM 并等它清理。使用本次 reserve_vram.py 配置的 PID 路径，并核实进程归属。

```bash
nohup setsid bash /home/<部署目录>/start.sh \
  </dev/null >/home/<部署目录>/server.log 2>&1 &
echo $! > /home/<部署目录>/server.pid
```

此阶段脚本中使用的 `$!`、`$P` 必须由 Linux shell 展开；不要先被 PowerShell 替换。正常 SSH 执行返回后模型继续在后台运行。

重启时核对 PID、cmdline、`os.getpgid(pid)==pid`，只向该自有 session/group 发信号，等待并检查非 zombie worker。只有同一自有组仍不退出时才对该组升级信号，不能扫描所有 python 进程杀掉。保留首轮日志，使用唯一 attempt 名称。

## 6. 验收和记录

复制 verify_service.py 到任务部署目录，使用不触发 adapter 的系统 Python 访问 HTTP 即可：

```bash
python -u /home/<部署目录>/verify_service.py \
  --base-url http://127.0.0.1:30000 --model Qwen3.8-27B \
  --pid-file /home/<部署目录>/server.pid \
  --output /home/<部署目录>/validation.json
```

默认三题仅做模型部署冒烟，支持自定义 cases 文件。不能以此替代精度评测、长文、Graph、TP 吞吐或多模态测试。思考模型如不支持 enable_thinking=false，应按其协议提供合适预算和答案校验，不以截断思考当作正常答案。

保存 profile、start.sh、安装日志、三仓完整 commit/dirty、实际解释器与模块来源、镜像/挂载和设备、服务 PID、完整请求响应、失败轮次及已测/未测项目。最后确认 hy-smi 和当前进程。失败且暂时无法继续时，在确认卡仍归本任务且空闲后恢复自有保留程序；说明真实阻塞，不宣布部署成功。
