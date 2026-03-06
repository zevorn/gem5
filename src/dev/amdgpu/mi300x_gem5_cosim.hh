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
 * These values MUST be kept in sync with QEMU's MI300XGem5MsgType
 * in include/hw/misc/mi300x_gem5.h.
 */
enum class CosimMsgType : uint32_t
{
    /* QEMU -> gem5 */
    MmioRead       = 0x01,
    MmioWrite      = 0x02,
    DoorbellRead   = 0x03,
    DoorbellWrite  = 0x04,
    DmaReq         = 0x05,
    Init           = 0x06,
    Shutdown       = 0x07,

    /* gem5 -> QEMU */
    MmioResp       = 0x81,
    IrqRaise       = 0x82,
    IrqLower       = 0x83,
    DmaRead        = 0x84,
    DmaWrite       = 0x85,
    InitResp       = 0x86,
};

/**
 * Wire-format message header for QEMU <-> gem5 co-simulation.
 * Must match MI300XGem5MsgHeader in QEMU's mi300x_gem5.h exactly.
 *
 * All fields are little-endian (native x86 byte order on same host).
 * Total size: 32 bytes.
 */
struct CosimMsgHeader
{
    uint32_t type;          /* CosimMsgType */
    uint32_t size;          /* payload size in bytes (after header) */
    uint64_t addr;          /* address for MMIO/DMA operations */
    uint64_t data;          /* data value or DMA length */
    uint32_t access_size;   /* 1/2/4/8 byte access width */
    uint32_t id;            /* transaction ID for request/response matching */
} __attribute__((packed));

static_assert(sizeof(CosimMsgHeader) == 32,
              "CosimMsgHeader must be 32 bytes to match QEMU");

static constexpr size_t COSIM_MSG_HDR_SIZE = sizeof(CosimMsgHeader);
static constexpr size_t COSIM_DMA_BUF_SIZE = 4 * 1024 * 1024; /* 4MB */

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
    void handleDoorbellRead(int fd, const CosimMsgHeader &msg);
    void handleDoorbellWrite(int fd, const CosimMsgHeader &msg);
    void handleInit(int fd, const CosimMsgHeader &msg);
    void handleShutdown(int fd, const CosimMsgHeader &msg);

    // -- I/O helpers --

    bool sendAll(int fd, const void *buf, size_t len);
    bool recvAll(int fd, void *buf, size_t len);
    void sendMsg(int fd, const CosimMsgHeader &msg);

    // -- MMIO forwarding --

    uint64_t readFromGpuDevice(uint64_t offset, uint32_t size, int gem5Bar);
    void writeToGpuDevice(uint64_t offset, uint32_t size, uint64_t data,
                          int gem5Bar);

    // -- Shared memory (VRAM) --

    void setupSharedMemory();
    void cleanupSharedMemory();

    // -- DMA and interrupt (gem5 -> QEMU) --
  public:
    bool sendDmaRead(uint64_t addr, uint64_t len);
    bool sendDmaWrite(uint64_t addr, uint64_t len, const uint8_t *data);
    bool sendIrqRaise(uint32_t vector);
    bool sendIrqLower(uint32_t vector);

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

    // DMA staging buffer
    uint8_t *dmaBuf = nullptr;

    bool connected = false;
};

} // namespace gem5

#endif // __DEV_AMDGPU_MI300X_GEM5_COSIM_HH__
