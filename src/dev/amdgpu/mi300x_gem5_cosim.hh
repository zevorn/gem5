/*
 * Copyright (c) 2024 The gem5 Contributors
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

#ifndef __DEV_AMDGPU_MI300X_GEM5_COSIM_HH__
#define __DEV_AMDGPU_MI300X_GEM5_COSIM_HH__

#include <sys/mman.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "base/pollevent.hh"
#include "base/types.hh"
#include "dev/amdgpu/amdgpu_defines.hh"
#include "params/MI300XGem5Cosim.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class AMDGPUDevice;

/**
 * Protocol message types matching the QEMU mi300x-gem5 PCIe device.
 * These must be kept in sync with the QEMU-side definitions.
 */
enum class CosimMsgType : uint32_t
{
    MmioRead       = 0x0001,
    MmioWrite      = 0x0002,
    MmioReadResp   = 0x0003,
    MmioWriteResp  = 0x0004,
    DmaRead        = 0x0005,
    DmaWrite       = 0x0006,
    DmaReadResp    = 0x0007,
    DmaWriteResp   = 0x0008,
    Interrupt      = 0x0009,
    InterruptResp  = 0x000a,
    SyncReq        = 0x000b,
    SyncResp       = 0x000c,
    Hello          = 0x00ff,
    HelloResp      = 0x0100,
};

/**
 * Protocol message header for QEMU <-> gem5 co-simulation.
 * Must match the MI300XGem5MsgHeader in QEMU's mi300x_gem5.h.
 *
 * All fields are in little-endian (native x86) byte order since
 * communication is via Unix domain socket on the same host.
 */
struct CosimMsgHeader
{
    uint32_t magic;       // 0x47454D35 ("GEM5")
    uint32_t version;     // Protocol version (1)
    uint32_t msg_type;    // CosimMsgType
    uint32_t msg_id;      // Sequence number for request/response matching
    uint32_t bar;         // BAR number (0, 2, 4)
    uint64_t addr;        // Address/offset within the BAR
    uint32_t size;        // Data size in bytes
    uint32_t status;      // 0 = success, non-zero = error
    uint8_t  data[256];   // Inline data for small transfers
} __attribute__((packed));

static constexpr uint32_t COSIM_MAGIC   = 0x47454D35; // "GEM5"
static constexpr uint32_t COSIM_VERSION = 1;

/**
 * BAR numbers matching the QEMU device layout:
 *   BAR0: MMIO registers (256 MB)
 *   BAR2: VRAM (shared memory, 16 GB)
 *   BAR4: Doorbell (8 MB)
 */
static constexpr uint32_t COSIM_BAR_MMIO     = 0;
static constexpr uint32_t COSIM_BAR_VRAM     = 2;
static constexpr uint32_t COSIM_BAR_DOORBELL = 4;

/**
 * MI300XGem5Cosim: Socket server that bridges QEMU's mi300x-gem5 PCIe
 * device to gem5's AMDGPUDevice for MI300X GPU simulation.
 *
 * Architecture:
 *   QEMU (guest + ROCm driver)
 *       |
 *       | Unix domain socket
 *       v
 *   MI300XGem5Cosim (this class, in gem5)
 *       |
 *       | forward MMIO / doorbell / framebuffer
 *       v
 *   AMDGPUDevice (existing gem5 MI300X model)
 *
 * VRAM is shared via mmap'd /dev/shm region for zero-copy access.
 */
class MI300XGem5Cosim : public SimObject
{
  public:
    PARAMS(MI300XGem5Cosim);

    explicit MI300XGem5Cosim(const Params &p);
    ~MI300XGem5Cosim();

    void startup() override;

  private:
    // -- Socket event handlers --

    class ListenEvent : public PollEvent
    {
      public:
        ListenEvent(int fd, MI300XGem5Cosim *cosim);
        void process(int revent) override;
        int getFd() const { return pfd.fd; }
      private:
        MI300XGem5Cosim *cosim;
    };

    class ClientEvent : public PollEvent
    {
      public:
        ClientEvent(int fd, MI300XGem5Cosim *cosim);
        void process(int revent) override;
      private:
        MI300XGem5Cosim *cosim;
    };

    void acceptConnection();
    void handleClientData(int fd);
    void closeClient(int fd);

    // -- Message processing --

    void processMessage(int fd, const CosimMsgHeader &msg);
    void handleMmioRead(int fd, const CosimMsgHeader &msg);
    void handleMmioWrite(int fd, const CosimMsgHeader &msg);
    void handleHello(int fd, const CosimMsgHeader &msg);
    void handleSync(int fd, const CosimMsgHeader &msg);

    // -- I/O helpers --

    bool sendAll(int fd, const void *buf, size_t len);
    bool recvAll(int fd, void *buf, size_t len);
    void sendResponse(int fd, CosimMsgHeader &resp);

    // -- MMIO forwarding --

    uint64_t forwardMmioRead(uint32_t bar, uint64_t addr, uint32_t size);
    void forwardMmioWrite(uint32_t bar, uint64_t addr, uint32_t size,
                          uint64_t data);
    uint64_t readFromGpuDevice(uint64_t offset, uint32_t size, int gem5Bar);
    void writeToGpuDevice(uint64_t offset, uint32_t size, uint64_t data,
                          int gem5Bar);

    // -- Shared memory (VRAM) --

    void setupSharedMemory();
    void cleanupSharedMemory();

    // -- DMA and interrupt (gem5 -> QEMU) --
  public:
    bool sendDmaRead(uint64_t addr, uint32_t size, const uint8_t *data);
    bool sendDmaWrite(uint64_t addr, uint32_t size, const uint8_t *data);
    bool sendInterrupt(uint32_t vector);

  private:
    AMDGPUDevice *gpuDevice;

    std::string socketPath;
    std::string shmemPath;
    Addr vramSize;

    std::unique_ptr<ListenEvent> listenEvent;
    std::unordered_map<int, std::unique_ptr<ClientEvent>> clientEvents;

    int primaryClientFd = -1;
    uint32_t nextMsgId = 1;

    // Shared memory for VRAM
    void *shmemPtr = MAP_FAILED;
    int shmemFd = -1;

    bool connected = false;
};

} // namespace gem5

#endif // __DEV_AMDGPU_MI300X_GEM5_COSIM_HH__
