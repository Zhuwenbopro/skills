export HIP_VISIBLE_DEVICES=4,5
unset NCCL_TOPO_FILE
python -m sglang.launch_server \
    --model-path /models/qwen3.6/Qwen3.6-35B-A3B \
    --tp-size 2 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --linear-attn-backend triton