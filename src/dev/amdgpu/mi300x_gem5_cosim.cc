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

#include "dev/amdgpu/mi300x_gem5_cosim.hh"

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
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

MI300XGem5Cosim::MI300XGem5Cosim(const Params &p)
    : SimObject(p),
      gpuDevice(p.gpu_device),
      socketPath(p.socket_path),
      shmemPath(p.shmem_path),
      vramSize(p.vram_size),
      keepaliveEvent([this] { processKeepalive(); }, name())
{
    dmaBuf = new uint8_t[COSIM_DMA_BUF_SIZE];
}

void
MI300XGem5Cosim::processKeepalive()
{
    schedule(keepaliveEvent, curTick() + KEEPALIVE_INTERVAL);
}

MI300XGem5Cosim::~MI300XGem5Cosim()
{
    cleanupSharedMemory();

    for (auto &[fd, event] : clientEvents) {
        close(fd);
    }
    clientEvents.clear();

    delete[] dmaBuf;
    dmaBuf = nullptr;
}

void
MI300XGem5Cosim::startup()
{
    // Register this cosim bridge with the GPU device so it can route
    // interrupts and other operations through the cosim socket.
    if (gpuDevice) {
        gpuDevice->setCosimBridge(this);
    }

    if (socketPath.empty()) {
        warn("MI300XGem5Cosim: No socket path configured, "
             "co-simulation disabled.\n");
        return;
    }

    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    fatal_if(listen_fd < 0, "MI300XGem5Cosim: socket() failed: %s",
             strerror(errno));

    unlink(socketPath.c_str());

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

    int ret = bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr));
    fatal_if(ret < 0, "MI300XGem5Cosim: bind(%s) failed: %s",
             socketPath.c_str(), strerror(errno));

    ret = listen(listen_fd, 1);
    fatal_if(ret < 0, "MI300XGem5Cosim: listen() failed: %s",
             strerror(errno));

    listenEvent = std::make_unique<ListenEvent>(listen_fd, this);
    pollQueue.schedule(listenEvent.get());

    inform("MI300XGem5Cosim: listening on %s", socketPath);

    if (!shmemPath.empty()) {
        setupSharedMemory();
    }

    // Schedule keepalive to keep the event queue alive.
    // Without this, m5.simulate() returns immediately when timer devices
    // (RTC/PIT) are disabled for cosim.
    schedule(keepaliveEvent, curTick() + KEEPALIVE_INTERVAL);
}

// ---- Shared Memory ----

void
MI300XGem5Cosim::setupSharedMemory()
{
    shmemFd = shm_open(shmemPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (shmemFd < 0) {
        warn("MI300XGem5Cosim: shm_open(%s) failed: %s, "
             "VRAM sharing disabled.\n",
             shmemPath.c_str(), strerror(errno));
        return;
    }

    if (ftruncate(shmemFd, vramSize) < 0) {
        warn("MI300XGem5Cosim: ftruncate failed: %s\n", strerror(errno));
        close(shmemFd);
        shmemFd = -1;
        return;
    }

    shmemPtr = mmap(nullptr, vramSize, PROT_READ | PROT_WRITE,
                    MAP_SHARED, shmemFd, 0);
    if (shmemPtr == MAP_FAILED) {
        warn("MI300XGem5Cosim: mmap failed: %s\n", strerror(errno));
        close(shmemFd);
        shmemFd = -1;
        return;
    }

    inform("MI300XGem5Cosim: VRAM shared memory %s mapped at %p (%lu bytes)",
           shmemPath, shmemPtr, (unsigned long)vramSize);

    // Enable GART PTE fallback from shared VRAM in cosim mode
    gpuDevice->getVM().vramShmemPtr = static_cast<uint8_t *>(shmemPtr);
    gpuDevice->getVM().vramShmemSize = vramSize;
}

void
MI300XGem5Cosim::cleanupSharedMemory()
{
    if (shmemPtr != MAP_FAILED && shmemPtr != nullptr) {
        munmap(shmemPtr, vramSize);
        shmemPtr = MAP_FAILED;
    }
    if (shmemFd >= 0) {
        close(shmemFd);
        shmemFd = -1;
    }
}

// ---- Listen/Client Events ----

MI300XGem5Cosim::ListenEvent::ListenEvent(int fd, MI300XGem5Cosim *c)
    : PollEvent(fd, POLLIN), cosim(c)
{
}

void
MI300XGem5Cosim::ListenEvent::process(int revent)
{
    cosim->acceptConnection();
}

MI300XGem5Cosim::ClientEvent::ClientEvent(int fd, MI300XGem5Cosim *c)
    : PollEvent(fd, POLLIN), cosim(c)
{
}

void
MI300XGem5Cosim::ClientEvent::process(int revent)
{
    if (revent & (POLLHUP | POLLERR | POLLNVAL)) {
        cosim->closeClient(pfd.fd);
        return;
    }
    cosim->handleClientData(pfd.fd);
}

void
MI300XGem5Cosim::acceptConnection()
{
    int cli_fd = accept(listenEvent->getFd(), nullptr, nullptr);
    if (cli_fd < 0) {
        warn("MI300XGem5Cosim: accept() failed: %s", strerror(errno));
        return;
    }

    inform("MI300XGem5Cosim: accepted connection fd=%d", cli_fd);

    clientEvents[cli_fd] = std::make_unique<ClientEvent>(cli_fd, this);
    pollQueue.schedule(clientEvents[cli_fd].get());

    // First connection = MMIO (sync), second = events (async)
    if (mmioClientFd < 0) {
        mmioClientFd = cli_fd;
        inform("MI300XGem5Cosim: MMIO connection established fd=%d", cli_fd);
    } else if (eventClientFd < 0) {
        eventClientFd = cli_fd;
        connected = true;
        inform("MI300XGem5Cosim: Event connection established fd=%d", cli_fd);
    }
}

void
MI300XGem5Cosim::closeClient(int fd)
{
    inform("MI300XGem5Cosim: closing connection fd=%d", fd);
    close(fd);
    clientEvents.erase(fd);

    if (fd == mmioClientFd) {
        mmioClientFd = -1;
        connected = false;
    } else if (fd == eventClientFd) {
        eventClientFd = -1;
        connected = false;
    }
}

// ---- I/O Helpers ----

bool
MI300XGem5Cosim::sendAll(int fd, const void *buf, size_t len)
{
    const char *ptr = reinterpret_cast<const char *>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t ret = ::send(fd, ptr + sent, len - sent, MSG_NOSIGNAL);
        if (ret < 0) {
            if (errno == EINTR)
                continue;
            warn("MI300XCosim: send failed: %s", strerror(errno));
            return false;
        }
        sent += ret;
    }
    return true;
}

bool
MI300XGem5Cosim::recvAll(int fd, void *buf, size_t len)
{
    char *ptr = reinterpret_cast<char *>(buf);
    size_t received = 0;
    while (received < len) {
        ssize_t ret = ::recv(fd, ptr + received, len - received, 0);
        if (ret == 0) {
            return false;
        }
        if (ret < 0) {
            if (errno == EINTR)
                continue;
            warn("MI300XCosim: recv failed: %s", strerror(errno));
            return false;
        }
        received += ret;
    }
    return true;
}

void
MI300XGem5Cosim::sendMsg(int fd, const CosimMsgHeader &msg)
{
    if (!sendAll(fd, &msg, COSIM_MSG_HDR_SIZE)) {
        closeClient(fd);
    }
}

// ---- Client Data Handling ----

void
MI300XGem5Cosim::handleClientData(int fd)
{
    // Drain all pending messages. FASYNC/SIGIO is edge-triggered: if
    // multiple messages arrive before the first is processed, only one
    // signal fires. We must read all available data to avoid deadlock
    // (e.g., a fire-and-forget write followed by a blocking read).
    struct pollfd pfd;
    do {
        CosimMsgHeader msg;
        if (!recvAll(fd, &msg, COSIM_MSG_HDR_SIZE)) {
            closeClient(fd);
            return;
        }

        processMessage(fd, msg);

        pfd = {fd, POLLIN, 0};
    } while (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN));
}

// ---- Message Processing ----

void
MI300XGem5Cosim::processMessage(int fd, const CosimMsgHeader &msg)
{
    CosimMsgType type = static_cast<CosimMsgType>(msg.type);

    DPRINTF(MI300XCosim,
            "Received msg type=0x%x id=%u addr=0x%lx data=0x%lx "
            "access_size=%u size=%u\n",
            msg.type, msg.id, msg.addr, msg.data,
            msg.access_size, msg.size);

    switch (type) {
      case CosimMsgType::MmioRead:
        handleMmioRead(fd, msg);
        break;
      case CosimMsgType::MmioWrite:
        handleMmioWrite(fd, msg);
        break;
      case CosimMsgType::DoorbellRead:
        handleDoorbellRead(fd, msg);
        break;
      case CosimMsgType::DoorbellWrite:
        handleDoorbellWrite(fd, msg);
        break;
      case CosimMsgType::DmaReq:
        handleDmaReq(fd, msg);
        break;
      case CosimMsgType::Init:
        handleInit(fd, msg);
        break;
      case CosimMsgType::Shutdown:
        handleShutdown(fd, msg);
        break;
      case CosimMsgType::ConfigRead:
        handleConfigRead(fd, msg);
        break;
      case CosimMsgType::ConfigWrite:
        handleConfigWrite(fd, msg);
        break;
      case CosimMsgType::FrameRead:
        handleFrameRead(fd, msg);
        break;
      case CosimMsgType::FrameWrite:
        handleFrameWrite(fd, msg);
        break;
      default:
        warn("MI300XCosim: unknown message type 0x%x", msg.type);
        break;
    }
}

void
MI300XGem5Cosim::handleInit(int fd, const CosimMsgHeader &msg)
{
    uint64_t qemuVramSize = msg.data;
    inform("MI300XGem5Cosim: INIT from QEMU (vram_size=%" PRIu64 " bytes)",
           qemuVramSize);

    CosimMsgHeader resp{};
    resp.type = static_cast<uint32_t>(CosimMsgType::InitResp);
    resp.id = msg.id;
    resp.data = vramSize;
    resp.size = 0;
    sendMsg(fd, resp);
}

void
MI300XGem5Cosim::handleShutdown(int fd, const CosimMsgHeader &msg)
{
    inform("MI300XGem5Cosim: SHUTDOWN from QEMU, exiting simulation");
    closeClient(fd);
    exitSimLoop("QEMU shutdown request", 0);
}

void
MI300XGem5Cosim::handleMmioRead(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    // QEMU BAR5 (MMIO) -> gem5 readMMIO/writeMMIO (MMIO_BAR=5)
    uint64_t value = gpuMmioRead(msg.addr, accessSize);

    DPRINTF(MI300XCosim,
            "MMIO Read: addr=0x%lx access_size=%u -> value=0x%lx\n",
            msg.addr, accessSize, value);

    CosimMsgHeader resp{};
    resp.type = static_cast<uint32_t>(CosimMsgType::MmioResp);
    resp.id = msg.id;
    resp.addr = msg.addr;
    resp.data = value;
    resp.access_size = accessSize;
    resp.size = 0;
    sendMsg(fd, resp);
}

void
MI300XGem5Cosim::handleMmioWrite(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    DPRINTF(MI300XCosim,
            "MMIO Write: addr=0x%lx access_size=%u value=0x%lx\n",
            msg.addr, accessSize, msg.data);

    // QEMU BAR5 (MMIO) -> gem5 readMMIO/writeMMIO (MMIO_BAR=5)
    gpuMmioWrite(msg.addr, accessSize, msg.data);

    // MMIO writes are fire-and-forget from QEMU's perspective
    // (QEMU's mi300x_send_msg passes NULL response for writes)
}

void
MI300XGem5Cosim::handleDoorbellRead(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    // QEMU BAR2 (Doorbell) -> gem5 readDoorbell/writeDoorbell (DOORBELL_BAR=2)
    uint64_t value = gpuDoorbellRead(msg.addr, accessSize);

    DPRINTF(MI300XCosim,
            "Doorbell Read: addr=0x%lx access_size=%u -> value=0x%lx\n",
            msg.addr, accessSize, value);

    CosimMsgHeader resp{};
    resp.type = static_cast<uint32_t>(CosimMsgType::MmioResp);
    resp.id = msg.id;
    resp.addr = msg.addr;
    resp.data = value;
    resp.access_size = accessSize;
    resp.size = 0;
    sendMsg(fd, resp);
}

void
MI300XGem5Cosim::handleDoorbellWrite(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    DPRINTF(MI300XCosim,
            "Doorbell Write: addr=0x%lx access_size=%u value=0x%lx\n",
            msg.addr, accessSize, msg.data);

    // QEMU BAR2 (Doorbell) -> gem5 readDoorbell/writeDoorbell (DOORBELL_BAR=2)
    gpuDoorbellWrite(msg.addr, accessSize, msg.data);

    // Doorbell writes are fire-and-forget (QEMU sends NULL response)
}

void
MI300XGem5Cosim::handleDmaReq(int fd, const CosimMsgHeader &msg)
{
    // DmaReq from QEMU: addr = system memory address, data = length,
    // size = payload size following header (for writes).
    //
    // For a DMA write (QEMU->gem5 VRAM): payload follows the header.
    // For a DMA read (gem5 VRAM->QEMU): gem5 reads VRAM and sends back.
    uint64_t addr = msg.addr;
    uint64_t len = msg.data;

    DPRINTF(MI300XCosim,
            "DMA Request: addr=0x%lx len=%lu payload_size=%u\n",
            addr, len, msg.size);

    if (msg.size > 0) {
        // DMA write: QEMU sends payload to write into VRAM via shared memory
        uint32_t payloadSize = msg.size;
        if (payloadSize > COSIM_DMA_BUF_SIZE) {
            warn("MI300XCosim: DMA write payload too large: %u", payloadSize);
            return;
        }
        if (!recvAll(fd, dmaBuf, payloadSize)) {
            closeClient(fd);
            return;
        }

        // If shared memory is available, copy into VRAM region
        if (shmemPtr != MAP_FAILED && addr < vramSize) {
            size_t copyLen = std::min((size_t)payloadSize,
                                      (size_t)(vramSize - addr));
            memcpy(static_cast<uint8_t *>(shmemPtr) + addr, dmaBuf, copyLen);
        }
    } else {
        // DMA read: gem5 reads from VRAM shared memory and sends back
        if (len > COSIM_DMA_BUF_SIZE)
            len = COSIM_DMA_BUF_SIZE;

        if (shmemPtr != MAP_FAILED && addr < vramSize) {
            size_t copyLen = std::min((size_t)len,
                                      (size_t)(vramSize - addr));
            memcpy(dmaBuf, static_cast<uint8_t *>(shmemPtr) + addr, copyLen);
        } else {
            memset(dmaBuf, 0, len);
        }

        CosimMsgHeader resp{};
        resp.type = static_cast<uint32_t>(CosimMsgType::MmioResp);
        resp.id = msg.id;
        resp.addr = addr;
        resp.data = len;
        resp.size = len;

        if (!sendAll(fd, &resp, COSIM_MSG_HDR_SIZE)) {
            closeClient(fd);
            return;
        }
        if (!sendAll(fd, dmaBuf, len)) {
            closeClient(fd);
            return;
        }
    }
}

void
MI300XGem5Cosim::handleConfigRead(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    uint64_t value = gpuConfigRead(msg.addr, accessSize);

    DPRINTF(MI300XCosim,
            "Config Read: offset=0x%lx access_size=%u -> value=0x%lx\n",
            msg.addr, accessSize, value);

    CosimMsgHeader resp{};
    resp.type = static_cast<uint32_t>(CosimMsgType::MmioResp);
    resp.id = msg.id;
    resp.addr = msg.addr;
    resp.data = value;
    resp.access_size = accessSize;
    resp.size = 0;
    sendMsg(fd, resp);
}

void
MI300XGem5Cosim::handleConfigWrite(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    DPRINTF(MI300XCosim,
            "Config Write: offset=0x%lx access_size=%u value=0x%lx\n",
            msg.addr, accessSize, msg.data);

    gpuConfigWrite(msg.addr, accessSize, msg.data);
}

void
MI300XGem5Cosim::handleFrameRead(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    uint64_t value = gpuFrameRead(msg.addr, accessSize);

    DPRINTF(MI300XCosim,
            "Frame Read: offset=0x%lx access_size=%u -> value=0x%lx\n",
            msg.addr, accessSize, value);

    CosimMsgHeader resp{};
    resp.type = static_cast<uint32_t>(CosimMsgType::MmioResp);
    resp.id = msg.id;
    resp.addr = msg.addr;
    resp.data = value;
    resp.access_size = accessSize;
    resp.size = 0;
    sendMsg(fd, resp);
}

void
MI300XGem5Cosim::handleFrameWrite(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    DPRINTF(MI300XCosim,
            "Frame Write: offset=0x%lx access_size=%u value=0x%lx\n",
            msg.addr, accessSize, msg.data);

    gpuFrameWrite(msg.addr, accessSize, msg.data);
}

// ======================================================================
// GPU Device Forwarding
//
// These methods create proper gem5 Packet objects and call the
// AMDGPUDevice's BAR-specific methods directly, matching exactly
// what readDevice/writeDevice do in amdgpu_device.cc.
//
// BAR mapping (QEMU -> gem5):
//   QEMU BAR0 (VRAM)     -> gem5 readFrame/writeFrame (FRAMEBUFFER_BAR=0)
//   QEMU BAR2 (Doorbell) -> gem5 readDoorbell/writeDoorbell (DOORBELL_BAR=2)
//   QEMU BAR5 (MMIO)     -> gem5 readMMIO/writeMMIO (MMIO_BAR=5)
// ======================================================================

uint64_t
MI300XGem5Cosim::gpuMmioRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    // For 4-byte reads, use getRegVal which handles MMIO dispatch
    if (size == sizeof(uint32_t)) {
        return gpuDevice->getRegVal(offset);
    }

    // For 8-byte reads, compose two 32-bit reads
    if (size == sizeof(uint64_t)) {
        uint64_t lo = gpuDevice->getRegVal(offset);
        uint64_t hi = gpuDevice->getRegVal(offset + 4);
        return lo | (hi << 32);
    }

    // Sub-dword: read 32-bit and mask
    uint32_t val = gpuDevice->getRegVal(offset & ~0x3ULL);
    uint32_t shift = (offset & 0x3) * 8;
    uint64_t mask = (1ULL << (size * 8)) - 1;
    return (val >> shift) & mask;
}

void
MI300XGem5Cosim::gpuMmioWrite(uint64_t offset, uint32_t size, uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    if (size == sizeof(uint32_t)) {
        gpuDevice->setRegVal(offset, static_cast<uint32_t>(data));
        return;
    }

    if (size == sizeof(uint64_t)) {
        gpuDevice->setRegVal(offset, static_cast<uint32_t>(data));
        gpuDevice->setRegVal(offset + 4,
                             static_cast<uint32_t>(data >> 32));
        return;
    }

    // Sub-dword: read-modify-write
    uint32_t cur = gpuDevice->getRegVal(offset & ~0x3ULL);
    uint32_t shift = (offset & 0x3) * 8;
    uint64_t mask = (1ULL << (size * 8)) - 1;
    cur &= ~(mask << shift);
    cur |= (data & mask) << shift;
    gpuDevice->setRegVal(offset & ~0x3ULL, cur);
}

uint64_t
MI300XGem5Cosim::gpuDoorbellRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    // Create a proper packet to call readDoorbell
    uint64_t pkt_data = 0;
    RequestPtr req = std::make_shared<Request>(
        offset, size, 0, gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createRead(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->readDoorbell(pkt, offset);

    uint64_t value = pkt->getUintX(ByteOrder::little);
    delete pkt;

    return value;
}

void
MI300XGem5Cosim::gpuDoorbellWrite(uint64_t offset, uint32_t size,
                                  uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    // Create a proper packet to call writeDoorbell
    // writeDoorbell uses pkt->getLE<uint64_t>() for doorbell value
    uint64_t pkt_data = htole(data);
    RequestPtr req = std::make_shared<Request>(
        offset, size, 0, gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createWrite(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->writeDoorbell(pkt, offset);
    delete pkt;
}

uint64_t
MI300XGem5Cosim::gpuFrameRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    uint64_t pkt_data = 0;
    RequestPtr req = std::make_shared<Request>(
        offset, size, 0, gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createRead(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->readFrame(pkt, offset);

    uint64_t value = pkt->getUintX(ByteOrder::little);
    delete pkt;

    return value;
}

void
MI300XGem5Cosim::gpuFrameWrite(uint64_t offset, uint32_t size, uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    uint64_t pkt_data = htole(data);
    RequestPtr req = std::make_shared<Request>(
        offset, size, 0, gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createWrite(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->writeFrame(pkt, offset);
    delete pkt;
}

uint64_t
MI300XGem5Cosim::gpuConfigRead(uint64_t offset, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    uint64_t pkt_data = 0;
    RequestPtr req = std::make_shared<Request>(
        offset, size, 0, gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createRead(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->readConfig(pkt);

    uint64_t value = pkt->getUintX(ByteOrder::little);
    delete pkt;
    return value;
}

void
MI300XGem5Cosim::gpuConfigWrite(uint64_t offset, uint32_t size, uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    uint64_t pkt_data = htole(data);
    RequestPtr req = std::make_shared<Request>(
        offset, size, 0, gpuDevice->vramRequestorId());
    PacketPtr pkt = Packet::createWrite(req);
    pkt->dataStatic(reinterpret_cast<uint8_t *>(&pkt_data));

    gpuDevice->writeConfig(pkt);
    delete pkt;
}

// ---- DMA and Interrupt (gem5 -> QEMU) ----

bool
MI300XGem5Cosim::sendDmaRead(uint64_t addr, uint64_t len)
{
    if (eventClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.type = static_cast<uint32_t>(CosimMsgType::DmaRead);
    msg.id = nextMsgId++;
    msg.addr = addr;
    msg.data = len;
    msg.size = 0;

    if (!sendAll(eventClientFd, &msg, COSIM_MSG_HDR_SIZE)) {
        return false;
    }

    // QEMU event thread reads guest memory and sends back
    // a MmioResp with payload following the header
    CosimMsgHeader resp;
    if (!recvAll(eventClientFd, &resp, COSIM_MSG_HDR_SIZE)) {
        return false;
    }

    uint32_t payloadSize = resp.size;
    if (payloadSize > 0 && payloadSize <= COSIM_DMA_BUF_SIZE) {
        if (!recvAll(eventClientFd, dmaBuf, payloadSize)) {
            return false;
        }
    }

    return true;
}

bool
MI300XGem5Cosim::sendDmaWrite(uint64_t addr, uint64_t len,
                              const uint8_t *data)
{
    if (eventClientFd < 0)
        return false;

    if (len > COSIM_DMA_BUF_SIZE) {
        warn("MI300XCosim: DMA write too large: %" PRIu64, len);
        return false;
    }

    CosimMsgHeader msg{};
    msg.type = static_cast<uint32_t>(CosimMsgType::DmaWrite);
    msg.id = nextMsgId++;
    msg.addr = addr;
    msg.data = len;
    msg.size = len;

    if (!sendAll(eventClientFd, &msg, COSIM_MSG_HDR_SIZE)) {
        return false;
    }
    if (data && len > 0) {
        if (!sendAll(eventClientFd, data, len)) {
            return false;
        }
    }

    return true;
}

bool
MI300XGem5Cosim::sendIrqRaise(uint32_t vector)
{
    if (eventClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.type = static_cast<uint32_t>(CosimMsgType::IrqRaise);
    msg.id = nextMsgId++;
    msg.data = vector;
    msg.size = 0;

    return sendAll(eventClientFd, &msg, COSIM_MSG_HDR_SIZE);
}

bool
MI300XGem5Cosim::sendIrqLower(uint32_t vector)
{
    if (eventClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.type = static_cast<uint32_t>(CosimMsgType::IrqLower);
    msg.id = nextMsgId++;
    msg.data = vector;
    msg.size = 0;

    return sendAll(eventClientFd, &msg, COSIM_MSG_HDR_SIZE);
}

} // namespace gem5
