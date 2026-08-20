export HIP_VISIBLE_DEVICES=4,5
export SGLANG_ENABLE_SPEC_V2=1
unset NCCL_TOPO_FILE
python -m sglang.launch_server \
    --model-path /models/qwen3.6/Qwen3.6-35B-A3B \
    --tp-size 2 \
    --mem-fraction-static 0.8 \
    --context-length 8196 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --linear-attn-backend triton \
    --chunked-prefill-size -1 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4