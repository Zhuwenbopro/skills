export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_MARLIN_W16A16_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LAYER_NORM_FWD=1
export TRITON_FAST_JIT=1
export TRITON_HCUTUNE=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_BOLT_RECOMPUTE_W_U=0
export SGLANG_USE_BOLT_MAMBA_STATE_SCATTER=0
unset NCCL_TOPO_FILE
sglang serve --model-path /models/Qwen3.6-35B-A3B \
    --attention-backend fa3 \
    --mem-fraction-static 0.9 \
    --tp-size 2 --pp-size 1 \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code \
    --chunked-prefill-size -1 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4