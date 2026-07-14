/*
 * Copyright (c) 2024-2025 The gem5 Contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "dev/amdgpu/mi300x_vfio_user.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/MI300XCosim.hh"
#include "dev/amdgpu/amdgpu_device.hh"
#include "mem/packet.hh"
#include "mem/packet_access.hh"
#include "mem/request.hh"
#include "sim/byteswap.hh"
#include "sim/sim_exit.hh"

namespace gem5
{

// ======================================================================
// Construction / Destruction
// ======================================================================

MI300XVfioUser::MI300XVfioUser(const Params &p)
    : SimObject(p),
      gpuDevice(p.gpu_device),
      socketPath(p.socket_path),
      shmemPath(p.shmem_path),
      vramSize(p.vram_size),
      keepaliveEvent([this] { processKeepalive(); }, name())
{}

MI300XVfioUser::~MI300XVfioUser()
{
    cleanupSharedMemory();

    if (vfuCtx) {
        vfu_destroy_ctx(vfuCtx);
        vfuCtx = nullptr;
    }

    if (!socketPath.empty()) {
        unlink(socketPath.c_str());
    }
}

void
MI300XVfioUser::processKeepalive()
{
    // Process any pending vfio-user messages on each keepalive tick.
    // PollEvent alone may not fire frequently enough for QEMU's timeout.
    if (vfuCtx) {
        processVfuEvents();
    }
    schedule(keepaliveEvent, curTick() + KEEPALIVE_INTERVAL);
}

// ======================================================================
// SimObject startup — initialize vfio-user server
// ======================================================================

void
MI300XVfioUser::startup()
{
    if (gpuDevice) {
        gpuDevice->setCosimBridge(this);
    }

    if (socketPath.empty()) {
        warn("MI300XVfioUser: No socket path configured, disabled.\n");
        return;
    }

    // Set up VRAM shared memory first (needed for BAR0 mmap fd)
    if (!shmemPath.empty()) {
        setupSharedMemory();
    }

    initVfuContext();

    schedule(keepaliveEvent, curTick() + KEEPALIVE_INTERVAL);
}

// ======================================================================
// libvfio-user context initialization
// ======================================================================

void
MI300XVfioUser::initVfuContext()
{
    // Remove stale socket
    unlink(socketPath.c_str());

    vfuCtx =
        vfu_create_ctx(VFU_TRANS_SOCK, socketPath.c_str(),
                       LIBVFIO_USER_FLAG_ATTACH_NB, this, VFU_DEV_TYPE_PCI);
    fatal_if(!vfuCtx, "MI300XVfioUser: vfu_create_ctx failed: %s",
             strerror(errno));

    // PCI identity: AMD MI300X
    int ret =
        vfu_pci_init(vfuCtx, VFU_PCI_TYPE_EXPRESS, PCI_HEADER_TYPE_NORMAL, 0);
    fatal_if(ret < 0, "MI300XVfioUser: vfu_pci_init failed: %s",
             strerror(errno));

    // Set PCI IDs in config space
    vfu_pci_set_id(vfuCtx, 0x1002, 0x74A0, 0x1002, 0x74A0);
    vfu_pci_set_class(vfuCtx, 0x03, 0x00, 0x00);

    setupBars();
    setupInterrupts();
    setupDma();

    // Device reset callback
    ret = vfu_setup_device_reset_cb(vfuCtx, resetCb);
    fatal_if(ret < 0, "MI300XVfioUser: reset_cb setup failed");

    // PCIe Express capability (required for amdgpu driver)
    struct pxcap px = {};
    px.hdr.id = PCI_CAP_ID_EXP;
    px.pxcaps.ver = 2;  // PCIe capability version 2
    px.pxcaps.dpt = 0;  // Endpoint device
    px.pxdcap.mps = 2;  // Max payload 512 bytes
    px.pxdcap.flrc = 1; // Function-level reset capable
    ret = vfu_pci_add_capability(vfuCtx, 0, 0, &px);
    fatal_if(ret < 0, "MI300XVfioUser: add PCIe cap failed: %s",
             strerror(errno));

    // MSI-X capability — table and PBA in BAR4
    struct msixcap msix = {};
    msix.hdr.id = PCI_CAP_ID_MSIX;
    msix.mxc.ts = NUM_MSIX_VECTORS - 1;           // table size (N-1)
    msix.mtab.tbir = 4;                           // table in BAR4
    msix.mtab.to = 0;                             // table at offset 0
    msix.mpba.pbir = 4;                           // PBA in BAR4
    msix.mpba.pbao = (NUM_MSIX_VECTORS * 16) / 8; // PBA after table
    // Flag 0: let libvfio-user handle MSI-X reads/writes internally
    // via cap_write_msix (tracks enable/mask state). VFU_CAP_FLAG_CALLBACK
    // would redirect to our cfgAccessCb which doesn't handle MSI-X.
    ret = vfu_pci_add_capability(vfuCtx, 0, 0, &msix);
    fatal_if(ret < 0, "MI300XVfioUser: add MSI-X cap failed: %s",
             strerror(errno));

    // Finalize
    ret = vfu_realize_ctx(vfuCtx);
    fatal_if(ret < 0, "MI300XVfioUser: vfu_realize_ctx failed: %s",
             strerror(errno));

    // Fix BAR register type bits after vfu_realize_ctx().
    // vfu_realize_ctx() incorrectly sets the IO bit on BARs without a
    // registered region (BAR1, BAR3 = upper halves of 64-bit pairs).
    // Clear those and set correct type bits for 64-bit BARs.
    auto *cfg = vfu_pci_get_config_space(vfuCtx);

    // BAR0+BAR1: 64-bit prefetchable memory (VRAM 16GB)
    cfg->hdr.bars[0].mem.locatable = 2; // 64-bit
    cfg->hdr.bars[0].mem.prefetchable = 1;
    cfg->hdr.bars[1].raw = 0; // upper half, must be zero

    // BAR2+BAR3: 64-bit non-prefetchable memory (Doorbell)
    cfg->hdr.bars[2].mem.locatable = 2; // 64-bit
    cfg->hdr.bars[3].raw = 0;           // upper half, must be zero

    inform("MI300XVfioUser: BAR type bits set: "
           "BAR0=0x%x BAR1=0x%x BAR2=0x%x BAR3=0x%x",
           cfg->hdr.bars[0].raw, cfg->hdr.bars[1].raw, cfg->hdr.bars[2].raw,
           cfg->hdr.bars[3].raw);

    // Register poll fd with gem5 event loop
    vfuPollFd = vfu_get_poll_fd(vfuCtx);
    fatal_if(vfuPollFd < 0, "MI300XVfioUser: vfu_get_poll_fd failed");

    vfuPollEvent = std::make_unique<VfuPollEvent>(vfuPollFd, this);
    pollQueue.schedule(vfuPollEvent.get());

    inform("MI300XVfioUser: listening on %s (vfio-user protocol)", socketPath);
}

void
MI300XVfioUser::setupBars()
{
    int ret;

    // BAR0: VRAM — mmap-able via shared memory fd for zero-copy
    if (shmemFd >= 0) {
        iovec mmap_areas;
        mmap_areas.iov_base = 0;
        mmap_areas.iov_len = vramSize;
        ret = vfu_setup_region(vfuCtx, VFU_PCI_DEV_BAR0_REGION_IDX, vramSize,
                               bar0AccessCb,
                               VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM,
                               &mmap_areas, 1, shmemFd, 0);
    } else {
        ret = vfu_setup_region(
            vfuCtx, VFU_PCI_DEV_BAR0_REGION_IDX, vramSize, bar0AccessCb,
            VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM, NULL, 0, -1, 0);
    }
    fatal_if(ret < 0, "MI300XVfioUser: BAR0 (VRAM) setup failed: %s",
             strerror(errno));

    // BAR2: Doorbell — callback only
    ret = vfu_setup_region(
        vfuCtx, VFU_PCI_DEV_BAR2_REGION_IDX, BAR2_SIZE, bar2AccessCb,
        VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM, NULL, 0, -1, 0);
    fatal_if(ret < 0, "MI300XVfioUser: BAR2 (Doorbell) setup failed");

    // BAR4: MSI-X table and PBA — handled internally by libvfio-user
    ret = vfu_setup_region(vfuCtx, VFU_PCI_DEV_BAR4_REGION_IDX, BAR4_SIZE,
                           NULL, VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM,
                           NULL, 0, -1, 0);
    fatal_if(ret < 0, "MI300XVfioUser: BAR4 (MSI-X) setup failed");

    // BAR5: MMIO registers — callback only
    ret = vfu_setup_region(
        vfuCtx, VFU_PCI_DEV_BAR5_REGION_IDX, BAR5_SIZE, bar5AccessCb,
        VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM, NULL, 0, -1, 0);
    fatal_if(ret < 0, "MI300XVfioUser: BAR5 (MMIO) setup failed");

    // Config space — let libvfio-user handle standard PCI header fields
    // internally (vendor/device ID, BARs, etc. are set via vfu_pci_set_id).
    // Our callback handles non-standard writes (driver config).
    ret = vfu_setup_region(vfuCtx, VFU_PCI_DEV_CFG_REGION_IDX,
                           PCI_CFG_SPACE_EXP_SIZE, cfgAccessCb,
                           VFU_REGION_FLAG_RW, NULL, 0, -1, 0);
    fatal_if(ret < 0, "MI300XVfioUser: Config space setup failed");
}

void
MI300XVfioUser::setupInterrupts()
{
    int ret;

    // MSI-X support
    ret = vfu_setup_device_nr_irqs(vfuCtx, VFU_DEV_MSIX_IRQ, NUM_MSIX_VECTORS);
    fatal_if(ret < 0, "MI300XVfioUser: MSI-X setup failed");

    // INTx fallback
    ret = vfu_setup_device_nr_irqs(vfuCtx, VFU_DEV_INTX_IRQ, 1);
    fatal_if(ret < 0, "MI300XVfioUser: INTx setup failed");
}

void
MI300XVfioUser::setupDma()
{
    int ret = vfu_setup_device_dma(vfuCtx, dmaRegisterCb, dmaUnregisterCb);
    fatal_if(ret < 0, "MI300XVfioUser: DMA setup failed");
}

// ======================================================================
// Event loop integration
// ======================================================================

MI300XVfioUser::VfuPollEvent::VfuPollEvent(int fd, MI300XVfioUser *s)
    : PollEvent(fd, POLLIN), server(s)
{}

void
MI300XVfioUser::VfuPollEvent::process(int revent)
{
    server->processVfuEvents();
}

void
MI300XVfioUser::processVfuEvents()
{
    if (!clientAttached) {
        // Try to accept a client connection
        int ret = vfu_attach_ctx(vfuCtx);
        if (ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return; // No client yet
            }
            warn("MI300XVfioUser: vfu_attach_ctx failed: %s", strerror(errno));
            return;
        }

        clientAttached = true;
        inform("MI300XVfioUser: client connected (vfio-user)");

        // Re-register with the new poll fd (changes after attach)
        int newFd = vfu_get_poll_fd(vfuCtx);
        if (newFd >= 0 && newFd != vfuPollFd) {
            vfuPollFd = newFd;
            // Defer re-registration to avoid destroying the PollEvent
            // we're currently inside of. The keepalive will still
            // process messages in the meantime.
            pollQueue.remove(vfuPollEvent.get());
            vfuPollEvent = std::make_unique<VfuPollEvent>(newFd, this);
            pollQueue.schedule(vfuPollEvent.get());
        }
        // Fall through to process any messages already pending
    }

    // Process pending messages
    while (true) {
        int ret = vfu_run_ctx(vfuCtx);
        if (ret == 0) {
            continue; // Processed a message, try next
        }
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break; // No more messages
            }
            if (errno == ENOTCONN) {
                inform("MI300XVfioUser: client disconnected");
                clientAttached = false;
                exitSimLoop("QEMU disconnected", 0);
                return;
            }
            warn("MI300XVfioUser: vfu_run_ctx error: %s", strerror(errno));
            break;
        }
        break;
    }
}

// ======================================================================
// BAR access callbacks
// ======================================================================

ssize_t
MI300XVfioUser::bar0AccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                             loff_t offset, bool is_write)
{
    auto *self = static_cast<MI300XVfioUser *>(vfu_get_private(ctx));
    return self->handleFrameAccess(buf, count, offset, is_write);
}

ssize_t
MI300XVfioUser::bar2AccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                             loff_t offset, bool is_write)
{
    auto *self = static_cast<MI300XVfioUser *>(vfu_get_private(ctx));
    return self->handleDoorbellAccess(buf, count, offset, is_write);
}

ssize_t
MI300XVfioUser::bar5AccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                             loff_t offset, bool is_write)
{
    auto *self = static_cast<MI300XVfioUser *>(vfu_get_private(ctx));
    return self->handleMmioAccess(buf, count, offset, is_write);
}

// ======================================================================
// BAR5: MMIO register access
// ======================================================================

ssize_t
MI300XVfioUser::handleMmioAccess(char *buf, size_t count, loff_t offset,
                                 bool is_write)
{
    size_t done = 0;
    while (done < count) {
        size_t chunk = std::min(count - done, (size_t)4);
        loff_t cur_off = offset + done;

        if (is_write) {
            uint32_t data = 0;
            memcpy(&data, buf + done, chunk);
            DPRINTF(MI300XCosim,
                    "vfio-user MMIO Write: offset=0x%lx size=%lu "
                    "data=0x%x\n",
                    (uint64_t)cur_off, chunk, data);
            gpuMmioWrite(cur_off, chunk, data);
        } else {
            uint64_t data = gpuMmioRead(cur_off, chunk);
            memcpy(buf + done, &data, chunk);
            DPRINTF(MI300XCosim,
                    "vfio-user MMIO Read: offset=0x%lx size=%lu "
                    "data=0x%lx\n",
                    (uint64_t)cur_off, chunk, data);
        }
        done += chunk;
    }
    return count;
}

// ======================================================================
// BAR2: Doorbell access
// ======================================================================

ssize_t
MI300XVfioUser::handleDoorbellAccess(char *buf, size_t count, loff_t offset,
                                     bool is_write)
{
    size_t done = 0;
    while (done < count) {
        // Doorbell accesses can be 4 or 8 bytes.
        // AMDGPUDevice::writeDoorbell expects 8-byte packets.
        size_t chunk = std::min(count - done, (size_t)8);
        loff_t cur_off = offset + done;

        if (is_write) {
            uint64_t data = 0;
            memcpy(&data, buf + done, chunk);
            DPRINTF(MI300XCosim,
                    "vfio-user Doorbell Write: offset=0x%lx size=%lu "
                    "data=0x%lx\n",
                    (uint64_t)cur_off, chunk, data);
            gpuDoorbellWrite(cur_off, chunk, data);
        } else {
            uint64_t data = gpuDoorbellRead(cur_off, chunk);
            memcpy(buf + done, &data, chunk);
        }
        done += chunk;
    }
    return count;
}

// ======================================================================
// BAR0: VRAM access (fallback for non-mmap'd paths)
// ======================================================================

ssize_t
MI300XVfioUser::handleFrameAccess(char *buf, size_t count, loff_t offset,
                                  bool is_write)
{
    // If VRAM is mmap'd, most accesses bypass this callback.
    // This handles the fallback case.
    if (shmemPtr != MAP_FAILED && (uint64_t)(offset + count) <= vramSize) {
        uint8_t *vram = static_cast<uint8_t *>(shmemPtr) + offset;
        if (is_write) {
            memcpy(vram, buf, count);
        } else {
            memcpy(buf, vram, count);
        }
        return count;
    }

    // Fall back to per-register access for small accesses
    if (count <= 8) {
        if (is_write) {
            uint64_t data = 0;
            memcpy(&data, buf, std::min(count, sizeof(data)));
            gpuFrameWrite(offset, count, data);
        } else {
            uint64_t data = gpuFrameRead(offset, count);
            memcpy(buf, &data, std::min(count, sizeof(data)));
        }
        return count;
    }

    warn("MI300XVfioUser: large Frame access without mmap, "
         "offset=0x%lx count=%lu",
         (uint64_t)offset, count);
    memset(buf, 0, count);
    return count;
}

// ======================================================================
// Config space access
// ======================================================================

ssize_t
MI300XVfioUser::cfgAccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                            loff_t offset, bool is_write)
{
    auto *self = static_cast<MI300XVfioUser *>(vfu_get_private(ctx));
    return self->handleCfgAccess(buf, count, offset, is_write);
}

ssize_t
MI300XVfioUser::handleCfgAccess(char *buf, size_t count, loff_t offset,
                                bool is_write)
{
    // PCI config space: split bulk accesses into dword (4-byte) chunks
    // since gem5's PCI config read/write only supports 1/2/4/8 byte ops.
    size_t done = 0;
    while (done < count) {
        size_t chunk = std::min(count - done, (size_t)4);
        loff_t cur_off = offset + done;

        if (is_write) {
            uint32_t data = 0;
            memcpy(&data, buf + done, chunk);
            DPRINTF(MI300XCosim,
                    "vfio-user Config Write: offset=0x%lx size=%lu "
                    "data=0x%x\n",
                    (uint64_t)cur_off, chunk, data);
            gpuConfigWrite(cur_off, chunk, data);
        } else {
            uint64_t data = gpuConfigRead(cur_off, chunk);
            memcpy(buf + done, &data, chunk);
            DPRINTF(MI300XCosim,
                    "vfio-user Config Read: offset=0x%lx size=%lu "
                    "data=0x%lx\n",
                    (uint64_t)cur_off, chunk, data);
        }
        done += chunk;
    }
    DPRINTF(MI300XCosim, "vfio-user Config %s: offset=0x%lx total=%lu\n",
            is_write ? "Write" : "Read", (uint64_t)offset, count);
    return count;
}

// ======================================================================
// DMA callbacks — guest RAM mapping
// ======================================================================

void
MI300XVfioUser::dmaRegisterCb(vfu_ctx_t *ctx, vfu_dma_info_t *info)
{
    auto *self = static_cast<MI300XVfioUser *>(vfu_get_private(ctx));
    self->handleDmaRegister(info);
}

void
MI300XVfioUser::dmaUnregisterCb(vfu_ctx_t *ctx, vfu_dma_info_t *info)
{
    auto *self = static_cast<MI300XVfioUser *>(vfu_get_private(ctx));
    self->handleDmaUnregister(info);
}

void
MI300XVfioUser::handleDmaRegister(vfu_dma_info_t *info)
{
    uint64_t iova = (uint64_t)info->iova.iov_base;
    uint64_t size = info->iova.iov_len;

    DPRINTF(MI300XCosim, "vfio-user DMA_MAP: iova=0x%lx size=0x%lx vaddr=%p\n",
            iova, size, info->vaddr);

    if (info->vaddr) {
        dmaMappings.push_back({iova, size, info->vaddr});

        // If this covers the guest RAM range, update GPU VM
        inform("MI300XVfioUser: DMA region mapped iova=0x%lx size=0x%lx", iova,
               size);
    }
}

void
MI300XVfioUser::handleDmaUnregister(vfu_dma_info_t *info)
{
    uint64_t iova = (uint64_t)info->iova.iov_base;

    DPRINTF(MI300XCosim, "vfio-user DMA_UNMAP: iova=0x%lx\n", iova);

    dmaMappings.erase(
        std::remove_if(dmaMappings.begin(), dmaMappings.end(),
                       [iova](const DmaMapping &m) { return m.iova == iova; }),
        dmaMappings.end());
}

uint8_t *
MI300XVfioUser::getGuestRamPtr(uint64_t gpa, uint64_t len) const
{
    for (const auto &m : dmaMappings) {
        if (gpa >= m.iova && gpa + len <= m.iova + m.size) {
            return static_cast<uint8_t *>(m.vaddr) + (gpa - m.iova);
        }
    }
    return nullptr;
}

// ======================================================================
// Device reset
// ======================================================================

int
MI300XVfioUser::resetCb(vfu_ctx_t *ctx, vfu_reset_type_t type)
{
    if (type == VFU_RESET_LOST_CONN) {
        return 0;
    }

    inform("MI300XVfioUser: device reset type=%d", type);
    return 0;
}

// ======================================================================
// Interrupt delivery
// ======================================================================

bool
MI300XVfioUser::sendIrqRaise(uint32_t vector)
{
    if (!vfuCtx || !clientAttached) {
        return false;
    }

    DPRINTF(MI300XCosim, "vfio-user IRQ raise vector=%u\n", vector);

    int ret = vfu_irq_trigger(vfuCtx, vector);
    if (ret < 0) {
        warn("MI300XVfioUser: vfu_irq_trigger(%u) failed: %s", vector,
             strerror(errno));
        return false;
    }
    return true;
}

bool
MI300XVfioUser::sendIrqLower(uint32_t vector)
{
    // vfio-user MSI-X is edge-triggered; lower is implicit
    return true;
}

// ======================================================================
// Shared memory (VRAM)
// ======================================================================

void
MI300XVfioUser::setupSharedMemory()
{
    shmemFd = shm_open(shmemPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (shmemFd < 0) {
        warn("MI300XVfioUser: shm_open(%s) failed: %s\n", shmemPath.c_str(),
             strerror(errno));
        return;
    }

    if (ftruncate(shmemFd, vramSize) < 0) {
        warn("MI300XVfioUser: ftruncate failed: %s\n", strerror(errno));
        close(shmemFd);
        shmemFd = -1;
        return;
    }

    shmemPtr = mmap(nullptr, vramSize, PROT_READ | PROT_WRITE, MAP_SHARED,
                    shmemFd, 0);
    if (shmemPtr == MAP_FAILED) {
        warn("MI300XVfioUser: mmap failed: %s\n", strerror(errno));
        close(shmemFd);
        shmemFd = -1;
        return;
    }

    inform("MI300XVfioUser: VRAM shared memory %s mapped at %p (%lu bytes)",
           shmemPath, shmemPtr, (unsigned long)vramSize);

    gpuDevice->getVM().vramShmemPtr = static_cast<uint8_t *>(shmemPtr);
    gpuDevice->getVM().vramShmemSize = vramSize;
    gpuDevice->getVM().setupWalkerCosim();
}

void
MI300XVfioUser::cleanupSharedMemory()
{
    if (shmemPtr != MAP_FAILED && shmemPtr != nullptr) {
        munmap(shmemPtr, vramSize);
        shmemPtr = MAP_FAILED;
    }
    if (shmemFd >= 0) {
        close(shmemFd);
        shmemFd = -1;
    }
    if (!shmemPath.empty()) {
        shm_unlink(shmemPath.c_str());
    }
}

// ======================================================================
// GPU device forwarding
// ======================================================================

uint64_t
MI300XVfioUser::gpuMmioRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    if (size == sizeof(uint32_t)) {
        return gpuDevice->getRegVal(offset);
    }
    if (size == sizeof(uint64_t)) {
        uint64_t lo = gpuDevice->getRegVal(offset);
        uint64_t hi = gpuDevice->getRegVal(offset + 4);
        return lo | (hi << 32);
    }

    uint32_t val = gpuDevice->getRegVal(offset & ~0x3ULL);
    uint32_t shift = (offset & 0x3) * 8;
    uint64_t mask = (1ULL << (size * 8)) - 1;
    return (val >> shift) & mask;
}

void
MI300XVfioUser::gpuMmioWrite(uint64_t offset, uint32_t size, uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    if (size == sizeof(uint32_t)) {
        gpuDevice->setRegVal(offset, static_cast<uint32_t>(data));
        return;
    }
    if (size == sizeof(uint64_t)) {
        gpuDevice->setRegVal(offset, static_cast<uint32_t>(data));
        gpuDevice->setRegVal(offset + 4, static_cast<uint32_t>(data >> 32));
        return;
    }

    uint32_t cur = gpuDevice->getRegVal(offset & ~0x3ULL);
    uint32_t shift = (offset & 0x3) * 8;
    uint64_t mask = (1ULL << (size * 8)) - 1;
    cur &= ~(mask << shift);
    cur |= (data & mask) << shift;
    gpuDevice->setRegVal(offset & ~0x3ULL, cur);
}

uint64_t
MI300XVfioUser::gpuDoorbellRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    uint64_t pkt_data = 0;
    RequestPtr req = std::make_shared<Request>(offset, size, 0,
                                               gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createRead(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->readDoorbell(pkt, offset);

    uint64_t value = pkt->getUintX(ByteOrder::little);
    delete pkt;
    return value;
}

void
MI300XVfioUser::gpuDoorbellWrite(uint64_t offset, uint32_t size, uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    // AMDGPUDevice::writeDoorbell uses pkt->getLE<uint64_t>(),
    // so always create an 8-byte packet regardless of access size.
    uint64_t pkt_data = htole(data);
    RequestPtr req =
        std::make_shared<Request>(offset, 8, 0, gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createWrite(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->writeDoorbell(pkt, offset);
    delete pkt;
}

uint64_t
MI300XVfioUser::gpuFrameRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    uint64_t pkt_data = 0;
    RequestPtr req = std::make_shared<Request>(offset, size, 0,
                                               gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createRead(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->readFrame(pkt, offset);

    uint64_t value = pkt->getUintX(ByteOrder::little);
    delete pkt;
    return value;
}

void
MI300XVfioUser::gpuFrameWrite(uint64_t offset, uint32_t size, uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    uint64_t pkt_data = htole(data);
    RequestPtr req = std::make_shared<Request>(offset, size, 0,
                                               gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createWrite(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->writeFrame(pkt, offset);
    delete pkt;
}

uint64_t
MI300XVfioUser::gpuConfigRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    uint64_t pkt_data = 0;
    RequestPtr req = std::make_shared<Request>(offset, size, 0,
                                               gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createRead(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->readConfig(pkt);

    uint64_t value = pkt->getUintX(ByteOrder::little);
    delete pkt;
    return value;
}

void
MI300XVfioUser::gpuConfigWrite(uint64_t offset, uint32_t size, uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XVfioUser: gpu_device not set");

    uint64_t pkt_data = htole(data);
    RequestPtr req = std::make_shared<Request>(offset, size, 0,
                                               gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createWrite(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->writeConfig(pkt);
    delete pkt;
}

} // namespace gem5
