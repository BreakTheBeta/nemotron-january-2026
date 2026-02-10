#!/bin/bash
# Native startup script for RTX 4090 with Qwen3-8B
#
# Runs all services natively (no Docker) on a single RTX 4090:
#   - ASR: Nemotron Speech Streaming (0.6B) - port 8080
#   - TTS: Magpie TTS multilingual (357M) - port 8001
#   - LLM: Qwen3-8B Q4 via llama.cpp - port 8000
#
# VRAM Budget (24GB):
#   ASR: ~2.4GB, TTS: ~1.4GB, LLM: ~5GB model + ~3GB KV cache = ~12GB total
#   Headroom: ~12GB
#
# Usage:
#   bash scripts/start_native_4090.sh
#
# Environment variables:
#   LLAMA_MODEL   - Path to GGUF model (default: auto-detect Qwen3-8B Q4)
#   LLM_CTX_SIZE  - Context size (default: 8192)
#   BOT_PORT      - Bot WebRTC server port (default: 7777)
#   BOT_HOST      - Bot listen host (default: 0.0.0.0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
LLM_CTX_SIZE="${LLM_CTX_SIZE:-8192}"
BOT_PORT="${BOT_PORT:-7777}"
BOT_HOST="${BOT_HOST:-0.0.0.0}"
SERVICE_TIMEOUT="${SERVICE_TIMEOUT:-120}"

# Auto-detect Qwen3-8B GGUF model
if [ -z "$LLAMA_MODEL" ]; then
    LLAMA_MODEL="$(find "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3-8B-GGUF" -name "*Q4_K_M*.gguf" 2>/dev/null | head -1)"
    if [ -z "$LLAMA_MODEL" ]; then
        echo "ERROR: No Qwen3-8B GGUF model found."
        echo "Download with: uv run huggingface-cli download unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf"
        exit 1
    fi
    echo "Auto-detected model: $LLAMA_MODEL"
fi

# Log directory
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Starting Voice Pipeline (Native RTX 4090)"
echo "============================================"
echo "  Date: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""
echo "  Services:"
echo "    ASR: port 8080 (Nemotron Speech 0.6B)"
echo "    TTS: port 8001 (Magpie TTS 357M)"
echo "    LLM: port 8000 (Qwen3-8B Q4 via llama.cpp)"
echo ""
echo "  Model: $LLAMA_MODEL"
echo "  Context: $LLM_CTX_SIZE tokens"
echo "  Bot: $BOT_HOST:$BOT_PORT"
echo "  Logs: $LOG_DIR/"
echo "============================================"

# Track PIDs for cleanup
ASR_PID=""
TTS_PID=""
LLM_PID=""

cleanup() {
    echo ""
    echo "Shutting down services..."
    [ -n "$ASR_PID" ] && kill -TERM $ASR_PID 2>/dev/null && echo "  Stopping ASR..."
    [ -n "$TTS_PID" ] && kill -TERM $TTS_PID 2>/dev/null && echo "  Stopping TTS..."
    [ -n "$LLM_PID" ] && kill -TERM $LLM_PID 2>/dev/null && echo "  Stopping LLM..."
    sleep 2
    [ -n "$ASR_PID" ] && kill -9 $ASR_PID 2>/dev/null || true
    [ -n "$TTS_PID" ] && kill -9 $TTS_PID 2>/dev/null || true
    [ -n "$LLM_PID" ] && kill -9 $LLM_PID 2>/dev/null || true
    echo "All services stopped."
}

trap cleanup EXIT INT TERM

# Health check helper
wait_for_service() {
    local name=$1
    local url=$2
    local timeout=$3
    local pid=$4

    echo -n "  Waiting for $name"
    for i in $(seq 1 $timeout); do
        if ! kill -0 $pid 2>/dev/null; then
            echo " FAILED (process exited)"
            return 1
        fi
        if curl -sf "$url" >/dev/null 2>&1; then
            echo " ready (${i}s)"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " TIMEOUT"
    return 1
}

# =============================================================================
# Step 1: Start TTS (smallest model, loads first)
# =============================================================================
echo "[1/3] Starting TTS server on port 8001..."
cd "$PROJECT_DIR"
uv run python3 -m nemotron_speech.tts_server --port 8001 > "$LOG_DIR/tts.log" 2>&1 &
TTS_PID=$!
echo "  TTS started (PID $TTS_PID)"

# =============================================================================
# Step 2: Start ASR
# =============================================================================
echo "[2/3] Starting ASR server on port 8080..."
uv run python3 -m nemotron_speech.server --port 8080 > "$LOG_DIR/asr.log" 2>&1 &
ASR_PID=$!
echo "  ASR started (PID $ASR_PID)"

# Wait for TTS and ASR to claim GPU memory before LLM
echo ""
if ! wait_for_service "TTS" "http://localhost:8001/health" "$SERVICE_TIMEOUT" "$TTS_PID"; then
    echo "TTS failed to start. Check $LOG_DIR/tts.log"
    exit 1
fi

# ASR is WebSocket-only, wait for log message
echo -n "  Waiting for ASR"
for i in $(seq 1 $SERVICE_TIMEOUT); do
    if ! kill -0 $ASR_PID 2>/dev/null; then
        echo " FAILED (process exited)"
        echo "Check $LOG_DIR/asr.log"
        exit 1
    fi
    if grep -q "GPU memory claimed" "$LOG_DIR/asr.log" 2>/dev/null; then
        echo " ready (${i}s)"
        break
    fi
    if [ $i -eq $SERVICE_TIMEOUT ]; then
        echo " TIMEOUT"
        echo "Check $LOG_DIR/asr.log"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# =============================================================================
# Step 3: Start LLM (llama.cpp with Qwen3-8B)
# =============================================================================
echo ""
echo "[3/3] Starting LLM server on port 8000..."
llama-server \
    -m "${LLAMA_MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --n-gpu-layers 99 \
    --ctx-size "${LLM_CTX_SIZE}" \
    --flash-attn on \
    --parallel 1 \
    --cache-ram 0 \
    --reasoning-budget 0 \
    -ctk q8_0 -ctv q8_0 \
    > "$LOG_DIR/llm.log" 2>&1 &
LLM_PID=$!
echo "  LLM started (PID $LLM_PID)"

echo ""
if ! wait_for_service "LLM" "http://localhost:8000/health" 60 "$LLM_PID"; then
    echo "LLM failed to start. Check $LOG_DIR/llm.log"
    exit 1
fi

# =============================================================================
# All services ready
# =============================================================================
echo ""
echo "============================================"
echo "All services started!"
echo "============================================"
echo "  ASR: ws://localhost:8080 (PID $ASR_PID)"
echo "  TTS: http://localhost:8001 (PID $TTS_PID)"
echo "  LLM: http://localhost:8000 (PID $LLM_PID)"
echo ""
echo "  Logs: $LOG_DIR/{asr,tts,llm}.log"
echo ""
echo "  To run the bot:"
echo "    cd $PROJECT_DIR"
echo "    uv run pipecat_bots/bot_interleaved_streaming.py -t webrtc --port $BOT_PORT --host $BOT_HOST"
echo ""
echo "Press Ctrl+C to stop all services."
echo "============================================"

# Wait for any process to exit
wait -n $ASR_PID $TTS_PID $LLM_PID
echo "A service exited unexpectedly."
