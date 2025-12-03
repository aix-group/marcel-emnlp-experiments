#!/bin/bash
module purge
module load miniconda
eval "$(conda shell.bash activate)"

vllm_version=0.11.0
vllm_conda_env="vllm-$vllm_version"

setup_environment() {
    if conda env list | grep -q "^$vllm_conda_env"; then
        echo "Using conda environment $vllm_conda_env"
    else
        echo "Creating environment $vllm_conda_env..."
        conda create -n "$vllm_conda_env" python=3.13 pip -y
        conda run -n "$vllm_conda_env" pip install "vllm==$vllm_version"
    fi
}

setup_environment
conda activate $vllm_conda_env

cleanup() {
    echo "Stopping vllm..."
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT SIGINT SIGTERM SIGHUP

VLLM_MODEL="${VLLM_MODEL:-neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_API_KEY=token-abc123
VLLM_PORT="${VLLM_PORT:-8000}"

# use all available gpus
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

vllm serve $VLLM_MODEL \
    --tensor_parallel_size $num_gpus \
    --max_model_len $VLLM_MAX_MODEL_LEN \
    --port $VLLM_PORT \
    --disable-log-requests \
    > vllm-$(hostname)-$VLLM_PORT.log 2>&1 \
    &

VLLM_PID=$!
echo "Started vllm (PID $VLLM_PID) on port $(hostname):$VLLM_PORT"

echo "Waiting for /health endpoint on port $VLLM_PORT..."
until curl -s -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" | grep -q "200"; do
    sleep 10
done

export LLM_BASE_URL="http://localhost:$VLLM_PORT/v1"
export LLM_API_KEY=$VLLM_API_KEY
export OPENAI_BASE_URL="http://localhost:$VLLM_PORT/v1"
export OPENAI_API_KEY=$VLLM_API_KEY
echo "vLLM is healthy."
