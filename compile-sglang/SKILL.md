---
name: compile-sglang
description: "Compile HYGON-AI sglang-das for HCU/ROCm environments, including dependencies, sgl-kernel, and the editable SGLang package. Use when the user asks to compile, build, install, or rebuild SGLang, sglang-kernel, sgl-kernel, HIP, or HCU support."
argument-hint: "Optionally provide the source directory; defaults to /home/sglang-das"
user-invocable: true
---

# Compile SGLang for HCU

Use this Skill to compile the HYGON-AI SGLang fork and its AOT kernel package in the current Python environment.

## Input

The user may provide an optional source directory. Use `/home/sglang-das` by default. Do not silently use a different Python environment or source tree.

Before changing anything, check:

```bash
python3 --version
pip3 --version
git --version
```

If any command is unavailable, stop and report the missing prerequisite. If the source directory already exists, inspect it briefly and reuse it; do not delete it or clone over it. Clone only when the directory does not exist:

```bash
git clone https://github.com/HYGON-AI/sglang-das.git /home/sglang-das
```

## Build Procedure

Run each step from the source directory and stop immediately when a command fails. Preserve the user's active environment and do not add `sudo`.

1. Install the fork's dependencies:

   ```bash
   cd /home/sglang-das
   pip3 install -r requirements_hcu.txt \
      -i https://mirrors.ustc.edu.cn/pypi/simple \
      --trusted-host mirrors.ustc.edu.cn
   ```

2. Remove the previously installed kernel package, then build and install the AOT kernel:

   ```bash
   cd /home/sglang-das
   pip3 uninstall -y sglang-kernel
   cd python/sglang/kernels/aot
   python3 setup_hip.py install
   ```

3. Install SGLang in editable mode using the HCU/ROCm extras. Keep the supplied offline build flags:

   ```bash
   cd /home/sglang-das
   pip3 install -e "python[all_hip]" --no-deps --no-build-isolation --no-index
   ```

Do not replace the final command with a normal dependency-resolving install. The dependency installation is intentionally handled by `requirements_hcu.txt`.

## Verification

After all build commands succeed, verify both the editable package and the installed kernel distribution:

```bash
cd /home/sglang-das
python3 -c "import sglang; print('sglang import: OK')"
pip3 show sglang-kernel
```

Report the source directory, Python executable/version, and the result of both checks. A successful pip build is not enough to report success if either verification command fails.

## Safety

- Only run the documented `pip3 uninstall -y sglang-kernel` removal; do not uninstall other packages.
- Do not run the build as a background job, hide output, or continue after a failed step.
- Do not modify unrelated files in the workspace.
- If `--no-index` fails because required wheels are missing, report the missing package and the exact failing command rather than silently changing the install policy.