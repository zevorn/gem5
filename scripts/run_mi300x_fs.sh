#!/bin/bash
# =============================================================================
# gem5 MI300X Full-System GPU Simulation - Complete Setup & Run Script
#
# This script handles:
#   1. Building gem5 (VEGA_X86)
#   2. Building the disk image (Ubuntu 24.04 + ROCm 7.0)
#   3. Building GPU test applications
#   4. Running MI300X full-system simulation
#
# Requirements:
#   - x86_64 Linux host with KVM (/dev/kvm)
#   - qemu-system-x86_64 (for disk image build)
#   - Docker (for GPU app compilation)
#   - ~60GB free disk space
#   - ~16GB RAM
#
# Usage:
#   ./scripts/run_mi300x_fs.sh build-all     # Full setup from scratch
#   ./scripts/run_mi300x_fs.sh build-gem5    # Build gem5 only
#   ./scripts/run_mi300x_fs.sh build-disk    # Build disk image only
#   ./scripts/run_mi300x_fs.sh build-app     # Build GPU test app only
#   ./scripts/run_mi300x_fs.sh run [app]     # Run simulation
#   ./scripts/run_mi300x_fs.sh run-atomic [app]  # Run without KVM (slow)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEM5_DIR="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="${GEM5_DIR}/../gem5-resources"

# Paths to built artifacts
DISK_IMAGE="${RESOURCES_DIR}/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-rocm70"
KERNEL="${RESOURCES_DIR}/src/x86-ubuntu-gpu-ml/vmlinux-rocm70"
SQUARE_APP="${RESOURCES_DIR}/src/gpu/square/bin.default/square.default"
GEM5_BIN="${GEM5_DIR}/build/VEGA_X86/gem5.opt"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ==========================
# Step 0: Check prerequisites
# ==========================
check_prerequisites() {
    info "Checking prerequisites..."

    if [ "$(uname -m)" != "x86_64" ]; then
        error "x86_64 host required (current: $(uname -m))"
    fi

    local missing=()
    command -v git >/dev/null || missing+=(git)
    command -v python3 >/dev/null || missing+=(python3)
    command -v scons >/dev/null || missing+=(scons)

    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing required tools: ${missing[*]}"
    fi

    info "Prerequisites OK"
}

check_kvm() {
    if [ ! -e /dev/kvm ]; then
        warn "KVM not available (/dev/kvm missing)"
        warn "Disk image build requires KVM"
        warn "Simulation can run with --cpu-type=AtomicSimpleCPU (very slow)"
        return 1
    fi
    info "KVM available"
    return 0
}

# ==========================
# Step 1: Build gem5
# ==========================
build_gem5() {
    info "Building gem5 (VEGA_X86)..."

    if [ -f "$GEM5_BIN" ]; then
        info "gem5 binary already exists: $GEM5_BIN"
        read -p "Rebuild? [y/N] " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && return 0
    fi

    cd "$GEM5_DIR"
    scons build/VEGA_X86/gem5.opt -j"$(nproc)" 2>&1 | tail -20

    if [ ! -f "$GEM5_BIN" ]; then
        error "gem5 build failed"
    fi
    info "gem5 built successfully: $GEM5_BIN"
}

# ==========================
# Step 2: Get gem5-resources
# ==========================
get_resources() {
    if [ -d "$RESOURCES_DIR" ]; then
        info "gem5-resources already exists: $RESOURCES_DIR"
        return 0
    fi

    info "Cloning gem5-resources..."
    git clone --depth 1 https://github.com/gem5/gem5-resources.git "$RESOURCES_DIR"
    info "gem5-resources cloned"
}

# ==========================
# Step 3: Build disk image
# ==========================
build_disk_image() {
    info "Building disk image (Ubuntu 24.04 + ROCm 7.0)..."
    info "This will take ~30 minutes and requires ~60GB disk space"

    command -v qemu-system-x86_64 >/dev/null || \
        error "qemu-system-x86_64 not found. Install: sudo apt install qemu-system-x86"
    command -v unzip >/dev/null || \
        error "unzip not found. Install: sudo apt install unzip"
    check_kvm || error "KVM required for disk image build"

    if [ -f "$DISK_IMAGE" ]; then
        info "Disk image already exists: $DISK_IMAGE"
        read -p "Rebuild? [y/N] " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && return 0
    fi

    cd "${RESOURCES_DIR}/src/x86-ubuntu-gpu-ml"
    ./build.sh

    if [ ! -f "$DISK_IMAGE" ]; then
        error "Disk image build failed"
    fi

    info "Disk image built: $DISK_IMAGE"
    info "Kernel extracted: $KERNEL"
}

# ==========================
# Step 4: Build GPU test app
# ==========================
build_gpu_app() {
    info "Building GPU test application (square)..."

    if [ -f "$SQUARE_APP" ]; then
        info "Square app already exists: $SQUARE_APP"
        return 0
    fi

    cd "${RESOURCES_DIR}/src/gpu/square"

    if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
        info "Using Docker to build..."
        docker run --rm -u "$(id -u):$(id -g)" \
            -v "$PWD:$PWD" -w "$PWD" \
            ghcr.io/gem5/gpu-fs make -f Makefile.default
    elif command -v hipcc >/dev/null; then
        info "Using local hipcc to build..."
        make -f Makefile.default
    else
        error "Need either Docker or hipcc (ROCm) to build GPU apps"
    fi

    if [ ! -f "$SQUARE_APP" ]; then
        error "GPU app build failed"
    fi
    info "GPU app built: $SQUARE_APP"
}

# ==========================
# Step 5: Run simulation
# ==========================
run_simulation() {
    local app="${1:-$SQUARE_APP}"
    local cpu_type="${2:-X86KvmCPU}"

    info "Running MI300X full-system simulation"
    info "  CPU type: $cpu_type"
    info "  App: $app"

    [ -f "$GEM5_BIN" ] || error "gem5 not built. Run: $0 build-gem5"
    [ -f "$DISK_IMAGE" ] || error "Disk image not found. Run: $0 build-disk"
    [ -f "$KERNEL" ] || error "Kernel not found. Run: $0 build-disk"
    [ -f "$app" ] || error "App not found: $app"

    if [ "$cpu_type" = "X86KvmCPU" ]; then
        check_kvm || error "KVM required for X86KvmCPU. Use: $0 run-atomic [app]"
    fi

    cd "$GEM5_DIR"

    # Method 1: Using gem5 stdlib config (recommended)
    info "Starting gem5... Output will appear in m5out/system.pc.com_1.device"
    "$GEM5_BIN" \
        configs/example/gem5_library/x86-mi300x-gpu.py \
        --image "$DISK_IMAGE" \
        --kernel "$KERNEL" \
        --app "$app"

    info "Simulation complete. Output:"
    echo "========================================"
    cat m5out/system.pc.com_1.device 2>/dev/null || warn "No output file found"
    echo "========================================"
}

run_simulation_legacy() {
    local app="${1:-$SQUARE_APP}"

    info "Running MI300X simulation (legacy config)..."

    [ -f "$GEM5_BIN" ] || error "gem5 not built"
    [ -f "$DISK_IMAGE" ] || error "Disk image not found"
    [ -f "$KERNEL" ] || error "Kernel not found"
    [ -f "$app" ] || error "App not found: $app"
    check_kvm || error "KVM required"

    cd "$GEM5_DIR"

    "$GEM5_BIN" \
        configs/example/gpufs/mi300.py \
        --disk-image "$DISK_IMAGE" \
        --kernel "$KERNEL" \
        --app "$app"
}

# ==========================
# Main
# ==========================
main() {
    local cmd="${1:-help}"
    shift || true

    check_prerequisites

    case "$cmd" in
        build-all)
            build_gem5
            get_resources
            build_disk_image
            build_gpu_app
            info "All components built successfully!"
            info "Run simulation: $0 run"
            ;;
        build-gem5)
            build_gem5
            ;;
        build-disk)
            get_resources
            build_disk_image
            ;;
        build-app)
            get_resources
            build_gpu_app
            ;;
        run)
            run_simulation "${1:-$SQUARE_APP}" "X86KvmCPU"
            ;;
        run-atomic)
            warn "AtomicSimpleCPU mode: ~100x slower than KVM"
            run_simulation "${1:-$SQUARE_APP}" "AtomicSimpleCPU"
            ;;
        run-legacy)
            run_simulation_legacy "${1:-$SQUARE_APP}"
            ;;
        status)
            echo "=== gem5 MI300X FS Status ==="
            [ -f "$GEM5_BIN" ] && info "gem5 binary: OK" || warn "gem5 binary: MISSING"
            [ -d "$RESOURCES_DIR" ] && info "gem5-resources: OK" || warn "gem5-resources: MISSING"
            [ -f "$DISK_IMAGE" ] && info "Disk image: OK" || warn "Disk image: MISSING"
            [ -f "$KERNEL" ] && info "Kernel: OK" || warn "Kernel: MISSING"
            [ -f "$SQUARE_APP" ] && info "Square app: OK" || warn "Square app: MISSING"
            check_kvm || true
            ;;
        help|*)
            cat <<'USAGE'
gem5 MI300X Full-System GPU Simulation

Commands:
  build-all      Full setup: gem5 + disk image + GPU app
  build-gem5     Build gem5 (VEGA_X86) only
  build-disk     Build disk image (Ubuntu 24.04 + ROCm 7.0)
  build-app      Build GPU test application (square)
  run [app]      Run simulation with KVM CPU
  run-atomic [app]  Run simulation without KVM (very slow)
  run-legacy [app]  Run with legacy gpufs/mi300.py config
  status         Show build status

Requirements:
  - x86_64 Linux host
  - /dev/kvm (for run and build-disk)
  - qemu-system-x86_64 (for build-disk)
  - Docker (for build-app)
  - ~60GB free disk space

Quick start:
  ./scripts/run_mi300x_fs.sh build-all
  ./scripts/run_mi300x_fs.sh run
USAGE
            ;;
    esac
}

main "$@"
