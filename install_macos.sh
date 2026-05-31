#!/bin/bash
set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${BOLD}HARTA - Epicardial Fat Segmentation Software${NC}"
echo "============================================"
echo ""

# ── 1. Check macOS ──────────────────────────────────────────────────────────
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This installer is for macOS only.${NC}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 2. Find Python 3.8–3.12 ─────────────────────────────────────────────────
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(sys.version_info[:2])")
            if [[ "$ver" == "(3, 8)" || "$ver" == "(3, 9)" || "$ver" == "(3, 10)" || \
                  "$ver" == "(3, 11)" || "$ver" == "(3, 12)" ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

echo -e "${BOLD}[1/5] Checking Python...${NC}"
PYTHON=$(find_python) || {
    echo -e "${RED}Python 3.8–3.12 not found.${NC}"
    echo ""
    echo "Install Python from https://www.python.org/downloads/"
    echo "or via Homebrew:  brew install python@3.12"
    exit 1
}
PYVER=$("$PYTHON" --version)
echo -e "  ${GREEN}✓${NC} Found $PYVER at $(which $PYTHON)"

# ── 3. Create virtual environment ───────────────────────────────────────────
echo ""
echo -e "${BOLD}[2/5] Creating virtual environment...${NC}"
if [[ -d "venv" ]]; then
    echo -e "  ${YELLOW}→${NC} Existing venv found, skipping creation."
else
    "$PYTHON" -m venv venv
    echo -e "  ${GREEN}✓${NC} Virtual environment created."
fi

PIP="venv/bin/pip"
VENV_PYTHON="venv/bin/python"
"$PIP" install --upgrade pip --quiet

# ── 4. Install dependencies ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/5] Installing dependencies (this may take a few minutes)...${NC}"

"$PIP" install --only-binary=:all: PyQt5 --quiet 2>/dev/null || \
"$PIP" install PyQt5 --quiet

"$PIP" install \
    numpy \
    opencv-python \
    pydicom \
    matplotlib \
    scipy \
    scikit-image \
    scikit-learn \
    "scikit-fuzzy>=0.4" \
    SimpleITK \
    Pillow \
    imageio \
    pandas \
    "fuzzy-c-means>=0.0.6" \
    pylibjpeg \
    pylibjpeg-libjpeg \
    python-gdcm \
    --quiet

echo -e "  ${GREEN}✓${NC} All dependencies installed."

# ── 5. Create required directories ──────────────────────────────────────────
echo ""
echo -e "${BOLD}[4/5] Creating working directories...${NC}"
mkdir -p aux_img/combined aux_img/slices aux_img/contours aux_img/fat
echo -e "  ${GREEN}✓${NC} Directories ready."

# ── 6. Create run script ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[5/5] Creating launcher...${NC}"

cat > run_harta.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./venv/bin/python harta.py
EOF
chmod +x run_harta.sh
echo -e "  ${GREEN}✓${NC} Launcher created: run_harta.sh"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Installation complete!${NC}"
echo ""
echo "To run HARTA:"
echo -e "  ${BOLD}cd $(pwd) && ./run_harta.sh${NC}"
echo ""
echo "HARTA accepts cardiac CT datasets in DICOM format (.dcm)"
echo "Sample data: https://visual.ic.uff.br/en/cardio/ctfat/"
echo ""
