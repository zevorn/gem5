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

#ifndef __DEV_AMDGPU_MI300X_VFIO_USER_HH__
#define __DEV_AMDGPU_MI300X_VFIO_USER_HH__

#include <sys/mman.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#define _Static_assert static_assert
#include <libvfio-user.h>

#undef _Static_assert

#include "base/pollevent.hh"
#include "base/types.hh"
#include "dev/amdgpu/amdgpu_defines.hh"
#include "dev/amdgpu/cosim_bridge.hh"
#include "params/MI300XVfioUser.hh"
#include "sim/eventq.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class AMDGPUDevice;

/**
 * MI300XVfioUser: vfio-user server exposing gem5's AMDGPUDevice
 * as a standard PCI device to QEMU's built-in vfio-user-pci client.
 *
 * Replaces the custom cosim socket protocol with the standard vfio-user
 * protocol. QEMU side uses the upstream vfio-user-pci device — no custom
 * QEMU code needed.
 *
 * BAR layout exposed to QEMU (matches MI300X hardware):
 *   BAR0: VRAM (16GB, mmap-able via shared memory fd)
 *   BAR2: Doorbell (2MB, callback-driven)
 *   BAR4: MSI-X table (4KB)
 *   BAR5: MMIO registers (512KB, callback-driven)
 *
 * Interrupt: MSI-X via eventfd (KVM direct injection)
 * DMA: Guest RAM fd passed by QEMU via DMA_MAP, mmap'd for zero-copy
 */
class MI300XVfioUser : public SimObject, public CosimBridge
{
  public:
    PARAMS(MI300XVfioUser);

    explicit MI300XVfioUser(const Params &p);
    ~MI300XVfioUser();

    void startup() override;

    // Called by AMDGPUDevice when GPU needs to send an interrupt
    bool sendIrqRaise(uint32_t vector) override;
    bool sendIrqLower(uint32_t vector) override;

    // DMA access to guest RAM (zero-copy via mmap)
    uint8_t *getGuestRamPtr(uint64_t gpa, uint64_t len) const;

  private:
    // -- libvfio-user context setup --
    void initVfuContext();
    void setupBars();
    void setupInterrupts();
    void setupDma();

    // -- Per-BAR region callbacks (static, forwarded to instance) --
    static ssize_t bar0AccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                                loff_t offset, bool is_write);
    static ssize_t bar2AccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                                loff_t offset, bool is_write);
    static ssize_t bar5AccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                                loff_t offset, bool is_write);

    // BAR5: MMIO registers
    ssize_t handleMmioAccess(char *buf, size_t count, loff_t offset,
                             bool is_write);
    // BAR2: Doorbell
    ssize_t handleDoorbellAccess(char *buf, size_t count, loff_t offset,
                                 bool is_write);
    // BAR0: VRAM (fallback for non-mmap'd access)
    ssize_t handleFrameAccess(char *buf, size_t count, loff_t offset,
                              bool is_write);

    // -- Config space callback --
    static ssize_t cfgAccessCb(vfu_ctx_t *ctx, char *buf, size_t count,
                               loff_t offset, bool is_write);
    ssize_t handleCfgAccess(char *buf, size_t count, loff_t offset,
                            bool is_write);

    // -- DMA callbacks --
    static void dmaRegisterCb(vfu_ctx_t *ctx, vfu_dma_info_t *info);
    static void dmaUnregisterCb(vfu_ctx_t *ctx, vfu_dma_info_t *info);
    void handleDmaRegister(vfu_dma_info_t *info);
    void handleDmaUnregister(vfu_dma_info_t *info);

    // -- Device reset callback --
    static int resetCb(vfu_ctx_t *ctx, vfu_reset_type_t type);

    // -- GPU device forwarding (reused from cosim) --
    uint64_t gpuMmioRead(uint64_t offset, uint32_t size);
    void gpuMmioWrite(uint64_t offset, uint32_t size, uint64_t data);
    uint64_t gpuDoorbellRead(uint64_t offset, uint32_t size);
    void gpuDoorbellWrite(uint64_t offset, uint32_t size, uint64_t data);
    uint64_t gpuFrameRead(uint64_t offset, uint32_t size);
    void gpuFrameWrite(uint64_t offset, uint32_t size, uint64_t data);
    uint64_t gpuConfigRead(uint64_t offset, uint32_t size);
    void gpuConfigWrite(uint64_t offset, uint32_t size, uint64_t data);

    // -- Shared memory (VRAM) --
    void setupSharedMemory();
    void cleanupSharedMemory();

    // -- Event loop integration --
    class VfuPollEvent : public PollEvent
    {
      public:
        VfuPollEvent(int fd, MI300XVfioUser *server);
        void process(int revent) override;

      private:
        MI300XVfioUser *server;
    };

    void processVfuEvents();

    // -- Members --
    AMDGPUDevice *gpuDevice;

    std::string socketPath;
    std::string shmemPath;
    Addr vramSize;

    // libvfio-user context
    vfu_ctx_t *vfuCtx = nullptr;
    std::unique_ptr<VfuPollEvent> vfuPollEvent;
    int vfuPollFd = -1;
    bool clientAttached = false;

    // VRAM shared memory
    void *shmemPtr = MAP_FAILED;
    int shmemFd = -1;

    // Guest RAM DMA mappings (GPA -> host ptr)
    struct DmaMapping
    {
        uint64_t iova;
        uint64_t size;
        void *vaddr;
    };
    std::vector<DmaMapping> dmaMappings;

    // BAR sizes
    static constexpr size_t BAR0_SIZE = 16ULL * 1024 * 1024 * 1024; // VRAM
    static constexpr size_t BAR2_SIZE = 2 * 1024 * 1024;            // Doorbell
    static constexpr size_t BAR4_SIZE = 4096;                       // MSI-X
    static constexpr size_t BAR5_SIZE = 512 * 1024;                 // MMIO
    static constexpr int NUM_MSIX_VECTORS = 256;

    // Keepalive event
    static constexpr Tick KEEPALIVE_INTERVAL = 1000000;
    EventFunctionWrapper keepaliveEvent;
    void processKeepalive();
};

} // namespace gem5

#endif // __DEV_AMDGPU_MI300X_VFIO_USER_HH__
