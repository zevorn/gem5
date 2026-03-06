#!/bin/bash
# ==========================================================================
# QEMU + gem5 MI300X Co-simulation Launcher
#
# This script starts gem5 (GPU back-end) and QEMU (host front-end) for
# MI300X co-simulation.  QEMU runs the host Linux kernel with amdgpu
# driver inside a Q35 machine; gem5 provides the GPU compute model.
#
# Prerequisites:
#   1. Build gem5:  scons build/VEGA_X86/gem5.opt -j$(nproc)
#   2. Build QEMU:  (in qemu-mi300x/) mkdir build && cd build &&
#                    ../configure --target-list=x86_64-softmmu && make -j$(nproc)
#   3. Prepare disk image & kernel using gem5-resources:
#        git clone https://github.com/gem5/gem5-resources.git
#        cd gem5-resources/src/x86-ubuntu-gpu-ml
#        export PACKER_GITHUB_API_TOKEN=xxxxxxx
#        ./build.sh
#
# Usage:
#   ./scripts/cosim_launch.sh [options]
#
# The script will:
#   a) Start gem5 in the background (listens on the Unix socket)
#   b) Wait for the socket to appear
#   c) Start QEMU (connects to gem5 via the socket)
#   d) On Ctrl-C, shut down both processes gracefully
# ==========================================================================

set -euo pipefail

# ---- Configurable paths ----

GEM5_DIR="${GEM5_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
GEM5_BIN="${GEM5_BIN:-${GEM5_DIR}/build/VEGA_X86/gem5.opt}"
GEM5_CONFIG="${GEM5_CONFIG:-${GEM5_DIR}/configs/example/gpufs/mi300_cosim.py}"

QEMU_BIN="${QEMU_BIN:-qemu-system-x86_64}"
DISK_IMAGE="${DISK_IMAGE:-}"
KERNEL="${KERNEL:-}"

SOCKET_PATH="${SOCKET_PATH:-/tmp/gem5-mi300x.sock}"
SHMEM_PATH="${SHMEM_PATH:-/mi300x-vram}"
SHMEM_HOST_PATH="${SHMEM_HOST_PATH:-/cosim-guest-ram}"
SHMEM_FILE="/dev/shm${SHMEM_PATH}"
SHMEM_HOST_FILE="/dev/shm${SHMEM_HOST_PATH}"

VRAM_SIZE="${VRAM_SIZE:-16GiB}"
NUM_CUS="${NUM_CUS:-40}"
HOST_MEM="${HOST_MEM:-8G}"
HOST_CPUS="${HOST_CPUS:-4}"

# ---- Argument parsing ----

usage() {
    cat <<EOF
Usage: $0 [options]

Required:
  --disk-image PATH     Path to x86-ubuntu-gpu-ml disk image
  --kernel PATH         Path to vmlinux-gpu-ml kernel

Optional:
  --gem5-bin PATH       gem5 binary (default: build/VEGA_X86/gem5.opt)
  --qemu-bin PATH       QEMU binary (default: qemu-system-x86_64)
  --socket-path PATH    Unix socket path (default: /tmp/gem5-mi300x.sock)
  --shmem-path NAME     POSIX shm name (default: /mi300x-vram)
  --vram-size SIZE      GPU VRAM size (default: 16GiB)
  --num-cus N           Number of compute units (default: 40)
  --host-mem SIZE       Host memory (default: 8G)
  --host-cpus N         Host CPUs (default: 4)
  --enable-kvm          Enable KVM acceleration (default: auto-detect)
  --gem5-debug FLAGS    gem5 debug flags (e.g. MI300XCosim,AMDGPUDevice)
  -h, --help            Show this help
EOF
    exit 0
}

ENABLE_KVM=""
GEM5_DEBUG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --disk-image)   DISK_IMAGE="$2"; shift 2 ;;
        --kernel)       KERNEL="$2"; shift 2 ;;
        --gem5-bin)     GEM5_BIN="$2"; shift 2 ;;
        --qemu-bin)     QEMU_BIN="$2"; shift 2 ;;
        --socket-path)  SOCKET_PATH="$2"; shift 2 ;;
        --shmem-path)   SHMEM_PATH="$2"; shift 2 ;;
        --vram-size)    VRAM_SIZE="$2"; shift 2 ;;
        --num-cus)      NUM_CUS="$2"; shift 2 ;;
        --host-mem)     HOST_MEM="$2"; shift 2 ;;
        --host-cpus)    HOST_CPUS="$2"; shift 2 ;;
        --enable-kvm)   ENABLE_KVM="yes"; shift ;;
        --gem5-debug)   GEM5_DEBUG="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

# ---- Validation ----

if [[ -z "$DISK_IMAGE" ]]; then
    echo "ERROR: --disk-image is required"
    echo "  Build one with gem5-resources/src/x86-ubuntu-gpu-ml/build.sh"
    exit 1
fi

if [[ -z "$KERNEL" ]]; then
    echo "ERROR: --kernel is required"
    echo "  Extract from gem5-resources disk image build"
    exit 1
fi

if [[ ! -f "$GEM5_BIN" ]]; then
    echo "ERROR: gem5 binary not found: $GEM5_BIN"
    echo "  Build with: scons build/VEGA_X86/gem5.opt -j\$(nproc)"
    exit 1
fi

if ! command -v "$QEMU_BIN" &>/dev/null; then
    echo "ERROR: QEMU binary not found: $QEMU_BIN"
    exit 1
fi

if [[ ! -f "$DISK_IMAGE" ]]; then
    echo "ERROR: Disk image not found: $DISK_IMAGE"
    exit 1
fi

if [[ ! -f "$KERNEL" ]]; then
    echo "ERROR: Kernel not found: $KERNEL"
    exit 1
fi

# ---- KVM detection ----

if [[ -z "$ENABLE_KVM" ]]; then
    if [[ -r /dev/kvm ]]; then
        ENABLE_KVM="yes"
    else
        ENABLE_KVM="no"
        echo "WARNING: /dev/kvm not available, running without KVM (will be slow)"
    fi
fi

KVM_OPTS=""
if [[ "$ENABLE_KVM" == "yes" ]]; then
    KVM_OPTS="-enable-kvm -cpu host"
fi

# ---- Cleanup handler ----

GEM5_PID=""
QEMU_PID=""

cleanup() {
    echo ""
    echo "Shutting down co-simulation..."
    [[ -n "$QEMU_PID" ]] && kill "$QEMU_PID" 2>/dev/null || true
    [[ -n "$GEM5_PID" ]] && kill "$GEM5_PID" 2>/dev/null || true
    rm -f "$SOCKET_PATH"
    rm -f "$SHMEM_FILE"
    rm -f "$SHMEM_HOST_FILE"
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ---- Clean up stale state ----

rm -f "$SOCKET_PATH"
rm -f "$SHMEM_FILE"
rm -f "$SHMEM_HOST_FILE"

# ---- Start gem5 ----

echo "============================================================"
echo "Starting gem5 MI300X co-simulation server..."
echo "  Binary:    $GEM5_BIN"
echo "  Socket:    $SOCKET_PATH"
echo "  VRAM SHM:  $SHMEM_PATH"
echo "  Host SHM:  $SHMEM_HOST_PATH"
echo "  VRAM:      $VRAM_SIZE"
echo "  Host RAM:  $HOST_MEM"
echo "  CUs:       $NUM_CUS"
echo "============================================================"

GEM5_CMD=(
    "$GEM5_BIN"
)

if [[ -n "$GEM5_DEBUG" ]]; then
    GEM5_CMD+=(--debug-flags="$GEM5_DEBUG")
fi

GEM5_CMD+=(
    "$GEM5_CONFIG"
    --socket-path="$SOCKET_PATH"
    --shmem-path="$SHMEM_PATH"
    --shmem-host-path="$SHMEM_HOST_PATH"
    --dgpu-mem-size="$VRAM_SIZE"
    --num-compute-units="$NUM_CUS"
    --mem-size="$HOST_MEM"
)

"${GEM5_CMD[@]}" &
GEM5_PID=$!

# Wait for the socket to appear
echo "Waiting for gem5 socket at $SOCKET_PATH ..."
WAIT_COUNT=0
while [[ ! -S "$SOCKET_PATH" ]]; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [[ $WAIT_COUNT -ge 60 ]]; then
        echo "ERROR: gem5 socket did not appear after 60 seconds"
        exit 1
    fi
    # Check if gem5 is still alive
    if ! kill -0 "$GEM5_PID" 2>/dev/null; then
        echo "ERROR: gem5 process exited unexpectedly"
        wait "$GEM5_PID" || true
        exit 1
    fi
done
echo "gem5 socket ready."

# ---- Start QEMU ----

# Kernel command line for the guest
#   - blacklist amdgpu initially (load manually after boot for control)
#   - serial console for gem5 terminal access
KCMDLINE="console=ttyS0,9600 root=/dev/sda1 drm_kms_helper.fbdev_emulation=0"

echo ""
echo "============================================================"
echo "Starting QEMU with MI300X co-simulation device..."
echo "  Machine: Q35"
echo "  CPUs:    $HOST_CPUS"
echo "  Memory:  $HOST_MEM"
echo "  KVM:     $ENABLE_KVM"
echo "  Disk:    $DISK_IMAGE"
echo "  Kernel:  $KERNEL"
echo "============================================================"

QEMU_CMD=(
    "$QEMU_BIN"
    -machine q35
    $KVM_OPTS
    -smp "$HOST_CPUS"
    # Shared memory backend for guest RAM - allows gem5 to DMA directly
    -object "memory-backend-file,id=mem0,size=${HOST_MEM},mem-path=${SHMEM_HOST_FILE},share=on"
    -numa "node,memdev=mem0"
    -kernel "$KERNEL"
    -append "$KCMDLINE"
    -drive "file=$DISK_IMAGE,format=raw,if=virtio"
    -device "mi300x-gem5,gem5-socket=$SOCKET_PATH,shmem-path=$SHMEM_FILE,vram-size=$((16 * 1024 * 1024 * 1024))"
    -serial mon:stdio
    -nographic
    -no-reboot
)

"${QEMU_CMD[@]}" &
QEMU_PID=$!

echo ""
echo "Co-simulation running.  Press Ctrl-C to stop."
echo "  gem5 PID:  $GEM5_PID"
echo "  QEMU PID:  $QEMU_PID"
echo ""

# Wait for either process to exit
wait -n "$GEM5_PID" "$QEMU_PID" 2>/dev/null || true
echo "One process exited, shutting down..."
