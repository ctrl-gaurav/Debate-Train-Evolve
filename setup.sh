#!/usr/bin/env bash
# ============================================================================
#  DTE Framework - Automated Setup Script
#  Debate, Train, Evolve: Self-Evolution of Language Model Reasoning
#  EMNLP 2025 | https://aclanthology.org/2025.emnlp-main.1666/
# ============================================================================
set -e

# ── Colors & formatting ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

ENV_NAME="${DTE_ENV_NAME:-dte}"
PYTHON_VERSION="${DTE_PYTHON_VERSION:-3.11}"

# ── Helper functions ─────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║                                                              ║"
    echo "  ║        ██████╗ ████████╗███████╗                             ║"
    echo "  ║        ██╔══██╗╚══██╔══╝██╔════╝                             ║"
    echo "  ║        ██║  ██║   ██║   █████╗                               ║"
    echo "  ║        ██║  ██║   ██║   ██╔══╝                               ║"
    echo "  ║        ██████╔╝   ██║   ███████╗                             ║"
    echo "  ║        ╚═════╝    ╚═╝   ╚══════╝                             ║"
    echo "  ║                                                              ║"
    echo "  ║   Debate, Train, Evolve  ·  v0.1.0                          ║"
    echo "  ║   Self-Evolution of Language Model Reasoning                 ║"
    echo "  ║   EMNLP 2025 Main Conference                                ║"
    echo "  ║                                                              ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

step() {
    echo ""
    echo -e "${BLUE}${BOLD}  ▸ $1${NC}"
}

success() {
    echo -e "    ${GREEN}✓${NC} $1"
}

warn() {
    echo -e "    ${YELLOW}!${NC} $1"
}

fail() {
    echo -e "    ${RED}✗${NC} $1"
}

info() {
    echo -e "    ${DIM}$1${NC}"
}

# ── Print header ─────────────────────────────────────────────────────────────
print_header

# ── Check for conda ──────────────────────────────────────────────────────────
step "Checking for conda installation"

if ! command -v conda &> /dev/null; then
    fail "conda is not installed or not in PATH"
    echo ""
    echo -e "    Please install conda first:"
    echo -e "    ${DIM}https://docs.conda.io/en/latest/miniconda.html${NC}"
    exit 1
fi
success "conda found: $(conda --version)"

# ── Check if env already exists ──────────────────────────────────────────────
step "Setting up conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})"

if conda env list | grep -qw "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists"
    echo ""
    echo -e "    ${YELLOW}Options:${NC}"
    echo -e "      1) ${BOLD}Update${NC} the existing environment (default)"
    echo -e "      2) ${BOLD}Remove${NC} and recreate from scratch"
    echo -e "      3) ${BOLD}Abort${NC}"
    echo ""
    read -r -p "    Choose [1/2/3]: " choice
    case "$choice" in
        2)
            info "Removing existing environment..."
            conda deactivate 2>/dev/null || true
            conda env remove -n "${ENV_NAME}" -y
            info "Creating fresh environment..."
            conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y -q
            ;;
        3)
            echo ""
            echo -e "  ${DIM}Aborted.${NC}"
            exit 0
            ;;
        *)
            info "Updating existing environment..."
            ;;
    esac
else
    info "Creating new environment..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y -q
fi

success "Environment '${ENV_NAME}' ready"

# ── Activate environment ─────────────────────────────────────────────────────
step "Activating environment"

# Source conda.sh to enable conda activate in scripts
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

PYTHON_BIN=$(which python)
PY_VER=$(python --version 2>&1)
success "Python: ${PY_VER} (${PYTHON_BIN})"

# ── Install PyTorch with CUDA ────────────────────────────────────────────────
step "Installing PyTorch with CUDA support"

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' || echo "")
    if [ -n "$CUDA_VER" ]; then
        success "CUDA detected: ${CUDA_VER}"
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

        if [ "$CUDA_MAJOR" -ge 12 ]; then
            info "Installing PyTorch for CUDA 12.x..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -1
        elif [ "$CUDA_MAJOR" -ge 11 ]; then
            info "Installing PyTorch for CUDA 11.x..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q 2>&1 | tail -1
        else
            warn "CUDA ${CUDA_VER} detected. Installing CPU-only PyTorch."
            pip install torch torchvision torchaudio -q 2>&1 | tail -1
        fi
    else
        warn "nvidia-smi found but could not detect CUDA version. Installing default PyTorch."
        pip install torch torchvision torchaudio -q 2>&1 | tail -1
    fi
else
    warn "No NVIDIA GPU detected. Installing CPU-only PyTorch."
    pip install torch torchvision torchaudio -q 2>&1 | tail -1
fi

# Verify torch
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAILED")
if [ "$TORCH_VER" = "FAILED" ]; then
    fail "PyTorch installation failed"
    exit 1
fi
success "PyTorch ${TORCH_VER} installed"

CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAIL" = "True" ]; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    success "CUDA available: ${GPU_COUNT} GPU(s) - ${GPU_NAME}"
else
    warn "CUDA not available (CPU-only mode)"
fi

# ── Install DTE framework ───────────────────────────────────────────────────
step "Installing DTE framework and dependencies"

pip install -e ".[dev]" -q 2>&1 | tail -1
success "DTE framework installed"

# Verify import
DTE_VER=$(python -c "import dte; print(dte.__version__)" 2>/dev/null || echo "FAILED")
if [ "$DTE_VER" = "FAILED" ]; then
    fail "DTE import failed"
    exit 1
fi
success "DTE v${DTE_VER} importable"

# ── Run verification tests ──────────────────────────────────────────────────
step "Running verification tests"

info "Running unit tests (config, answer extraction, reward model)..."
UNIT_RESULT=$(python -m pytest tests/test_config.py tests/test_answer_extraction.py tests/test_reward_model.py -q --tb=line 2>&1)
UNIT_PASSED=$(echo "$UNIT_RESULT" | grep -oP '\d+ passed' | head -1 || echo "0 passed")
UNIT_FAILED=$(echo "$UNIT_RESULT" | grep -oP '\d+ failed' | head -1 || echo "")

if echo "$UNIT_RESULT" | grep -q "failed"; then
    warn "Unit tests: ${UNIT_PASSED}, ${UNIT_FAILED}"
    echo "$UNIT_RESULT" | grep "FAILED" | head -5
else
    success "Unit tests: ${UNIT_PASSED}"
fi

# Run a quick GPU smoke test if CUDA is available
if [ "$CUDA_AVAIL" = "True" ]; then
    info "Running GPU smoke test (quick model load + inference)..."
    GPU_RESULT=$(python -c "
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
print(f'  Loading {model_name}...', end=' ', flush=True)
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
model.eval()
print('OK')

print('  Generating response...', end=' ', flush=True)
msgs = [{'role': 'user', 'content': 'What is 2+2? Answer briefly.'}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
response = tok.decode(out[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(f'OK -> \"{response.strip()[:60]}\"')

del model, tok
torch.cuda.empty_cache()
print('  GPU smoke test passed!')
" 2>&1)

    if echo "$GPU_RESULT" | grep -q "GPU smoke test passed"; then
        success "GPU smoke test passed"
        echo "$GPU_RESULT" | grep -v "^$" | while IFS= read -r line; do
            info "$line"
        done
    else
        warn "GPU smoke test encountered issues:"
        echo "$GPU_RESULT" | tail -5
    fi
fi

# ── Print startup report ────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║                   Setup Complete!                            ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "  ${BOLD}System Report${NC}"
echo -e "  ─────────────────────────────────────────────────────────────"
echo -e "  ${DIM}Environment:${NC}    ${ENV_NAME}"
echo -e "  ${DIM}Python:${NC}         ${PY_VER}"
echo -e "  ${DIM}PyTorch:${NC}        ${TORCH_VER}"
echo -e "  ${DIM}CUDA:${NC}           ${CUDA_AVAIL}"
echo -e "  ${DIM}DTE Version:${NC}    ${DTE_VER}"
if [ "$CUDA_AVAIL" = "True" ]; then
    echo -e "  ${DIM}GPUs:${NC}           ${GPU_COUNT} x ${GPU_NAME}"
fi
echo ""

echo -e "  ${BOLD}Quick Start${NC}"
echo -e "  ─────────────────────────────────────────────────────────────"
echo -e "  ${DIM}Activate env:${NC}   conda activate ${ENV_NAME}"
echo -e "  ${DIM}Run debate:${NC}     python -c \"import dte; r = dte.debate('What is 15*24?'); print(r.final_answer)\""
echo -e "  ${DIM}CLI help:${NC}       python main.py --help"
echo -e "  ${DIM}Run tests:${NC}      python -m pytest tests/ -v"
echo -e "  ${DIM}GPU tests:${NC}      python -m pytest tests/ -v -m gpu"
echo ""

echo -e "  ${BOLD}Resources${NC}"
echo -e "  ─────────────────────────────────────────────────────────────"
echo -e "  ${DIM}Paper:${NC}          https://aclanthology.org/2025.emnlp-main.1666/"
echo -e "  ${DIM}Website:${NC}        https://debate-train-evolve.github.io/"
echo -e "  ${DIM}GitHub:${NC}         https://github.com/ctrl-gaurav/Debate-Train-Evolve"
echo -e "  ${DIM}Docs:${NC}           docs/"
echo ""

echo -e "  ${GREEN}${BOLD}Ready to evolve your models!${NC}"
echo ""
