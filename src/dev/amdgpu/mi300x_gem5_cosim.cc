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

namespace gem5
{

MI300XGem5Cosim::MI300XGem5Cosim(const Params &p)
    : SimObject(p),
      gpuDevice(p.gpu_device),
      socketPath(p.socket_path),
      shmemPath(p.shmem_path),
      vramSize(p.vram_size)
{
}

MI300XGem5Cosim::~MI300XGem5Cosim()
{
    cleanupSharedMemory();

    for (auto &[fd, event] : clientEvents) {
        close(fd);
    }
    clientEvents.clear();
}

void
MI300XGem5Cosim::startup()
{
    // Build the listen socket for Unix domain communication
    if (socketPath.empty()) {
        warn("MI300XGem5Cosim: No socket path configured, "
             "co-simulation disabled.\n");
        return;
    }

    // Create Unix domain socket manually since we need a specific path
    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    fatal_if(listen_fd < 0, "MI300XGem5Cosim: socket() failed: %s",
             strerror(errno));

    // Remove any stale socket file
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

    // Setup shared memory for VRAM
    if (!shmemPath.empty()) {
        setupSharedMemory();
    }
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

    if (primaryClientFd < 0) {
        primaryClientFd = cli_fd;
        connected = true;
    }
}

void
MI300XGem5Cosim::closeClient(int fd)
{
    inform("MI300XGem5Cosim: closing connection fd=%d", fd);
    close(fd);
    clientEvents.erase(fd);

    if (fd == primaryClientFd) {
        primaryClientFd = -1;
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
            // Connection closed
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
MI300XGem5Cosim::sendResponse(int fd, CosimMsgHeader &resp)
{
    resp.magic = COSIM_MAGIC;
    resp.version = COSIM_VERSION;
    if (!sendAll(fd, &resp, sizeof(resp))) {
        closeClient(fd);
    }
}

// ---- Client Data Handling ----

void
MI300XGem5Cosim::handleClientData(int fd)
{
    CosimMsgHeader msg;
    if (!recvAll(fd, &msg, sizeof(msg))) {
        closeClient(fd);
        return;
    }

    if (msg.magic != COSIM_MAGIC) {
        warn("MI300XCosim: bad magic 0x%x from fd=%d, expected 0x%x",
             msg.magic, fd, COSIM_MAGIC);
        closeClient(fd);
        return;
    }

    processMessage(fd, msg);
}

// ---- Message Processing ----

void
MI300XGem5Cosim::processMessage(int fd, const CosimMsgHeader &msg)
{
    CosimMsgType type = static_cast<CosimMsgType>(msg.msg_type);

    DPRINTF(MI300XCosim,
            "Received msg type=%u id=%u bar=%u addr=0x%lx size=%u\n",
            msg.msg_type, msg.msg_id, msg.bar, msg.addr, msg.size);

    switch (type) {
      case CosimMsgType::MmioRead:
        handleMmioRead(fd, msg);
        break;
      case CosimMsgType::MmioWrite:
        handleMmioWrite(fd, msg);
        break;
      case CosimMsgType::Hello:
        handleHello(fd, msg);
        break;
      case CosimMsgType::SyncReq:
        handleSync(fd, msg);
        break;
      default:
        warn("MI300XCosim: unknown message type 0x%x", msg.msg_type);
        break;
    }
}

void
MI300XGem5Cosim::handleHello(int fd, const CosimMsgHeader &msg)
{
    inform("MI300XGem5Cosim: HELLO from QEMU (version %u)", msg.version);

    CosimMsgHeader resp{};
    resp.msg_type = static_cast<uint32_t>(CosimMsgType::HelloResp);
    resp.msg_id = msg.msg_id;
    resp.version = COSIM_VERSION;
    resp.status = 0;

    // Encode VRAM size in the response data field
    if (sizeof(uint64_t) <= sizeof(resp.data)) {
        uint64_t vram = vramSize;
        memcpy(resp.data, &vram, sizeof(vram));
        resp.size = sizeof(vram);
    }

    sendResponse(fd, resp);
}

void
MI300XGem5Cosim::handleSync(int fd, const CosimMsgHeader &msg)
{
    DPRINTF(MI300XCosim, "Sync request id=%u\n", msg.msg_id);

    CosimMsgHeader resp{};
    resp.msg_type = static_cast<uint32_t>(CosimMsgType::SyncResp);
    resp.msg_id = msg.msg_id;
    resp.status = 0;
    sendResponse(fd, resp);
}

void
MI300XGem5Cosim::handleMmioRead(int fd, const CosimMsgHeader &msg)
{
    uint32_t size = msg.size;
    if (size > sizeof(CosimMsgHeader::data)) {
        warn("MI300XCosim: MMIO read size %u too large", size);
        size = sizeof(CosimMsgHeader::data);
    }

    uint64_t value = forwardMmioRead(msg.bar, msg.addr, size);

    DPRINTF(MI300XCosim,
            "MMIO Read: bar=%u addr=0x%lx size=%u -> value=0x%lx\n",
            msg.bar, msg.addr, size, value);

    CosimMsgHeader resp{};
    resp.msg_type = static_cast<uint32_t>(CosimMsgType::MmioReadResp);
    resp.msg_id = msg.msg_id;
    resp.bar = msg.bar;
    resp.addr = msg.addr;
    resp.size = size;
    resp.status = 0;
    memcpy(resp.data, &value, size);
    sendResponse(fd, resp);
}

void
MI300XGem5Cosim::handleMmioWrite(int fd, const CosimMsgHeader &msg)
{
    uint32_t size = msg.size;
    if (size > sizeof(CosimMsgHeader::data)) {
        warn("MI300XCosim: MMIO write size %u too large", size);
        size = sizeof(CosimMsgHeader::data);
    }

    uint64_t value = 0;
    memcpy(&value, msg.data, size);

    DPRINTF(MI300XCosim,
            "MMIO Write: bar=%u addr=0x%lx size=%u value=0x%lx\n",
            msg.bar, msg.addr, size, value);

    forwardMmioWrite(msg.bar, msg.addr, size, value);

    CosimMsgHeader resp{};
    resp.msg_type = static_cast<uint32_t>(CosimMsgType::MmioWriteResp);
    resp.msg_id = msg.msg_id;
    resp.bar = msg.bar;
    resp.addr = msg.addr;
    resp.size = size;
    resp.status = 0;
    sendResponse(fd, resp);
}

// ---- MMIO Forwarding to AMDGPUDevice ----

uint64_t
MI300XGem5Cosim::forwardMmioRead(uint32_t bar, uint64_t addr, uint32_t size)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    // Map QEMU BAR numbers to gem5 BAR numbers:
    //   QEMU BAR0 (MMIO regs)   -> gem5 BAR5 (MMIO_BAR)
    //   QEMU BAR2 (VRAM)        -> gem5 BAR0 (FRAMEBUFFER_BAR)
    //   QEMU BAR4 (Doorbell)    -> gem5 BAR2 (DOORBELL_BAR)
    //
    // For VRAM reads via shared memory, we can read directly from shmem.
    if (bar == COSIM_BAR_VRAM) {
        // VRAM access: use shared memory if available
        if (shmemPtr != MAP_FAILED && addr + size <= vramSize) {
            uint64_t value = 0;
            memcpy(&value, static_cast<uint8_t *>(shmemPtr) + addr, size);
            return value;
        }
        // Fallback: forward as framebuffer read to GPU device
        return readFromGpuDevice(addr, size, FRAMEBUFFER_BAR);
    }

    // For MMIO and Doorbell: create a gem5 packet and forward to GPU device
    int gem5Bar;
    if (bar == COSIM_BAR_MMIO) {
        gem5Bar = MMIO_BAR;
    } else if (bar == COSIM_BAR_DOORBELL) {
        gem5Bar = DOORBELL_BAR;
    } else {
        warn("MI300XCosim: unknown BAR %u for read", bar);
        return 0;
    }

    return readFromGpuDevice(addr, size, gem5Bar);
}

void
MI300XGem5Cosim::forwardMmioWrite(uint32_t bar, uint64_t addr, uint32_t size,
                                  uint64_t data)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    if (bar == COSIM_BAR_VRAM) {
        // VRAM write: use shared memory if available
        if (shmemPtr != MAP_FAILED && addr + size <= vramSize) {
            memcpy(static_cast<uint8_t *>(shmemPtr) + addr, &data, size);
            return;
        }
        writeToGpuDevice(addr, size, data, FRAMEBUFFER_BAR);
        return;
    }

    int gem5Bar;
    if (bar == COSIM_BAR_MMIO) {
        gem5Bar = MMIO_BAR;
    } else if (bar == COSIM_BAR_DOORBELL) {
        gem5Bar = DOORBELL_BAR;
    } else {
        warn("MI300XCosim: unknown BAR %u for write", bar);
        return;
    }

    writeToGpuDevice(addr, size, data, gem5Bar);
}

uint64_t
MI300XGem5Cosim::readFromGpuDevice(uint64_t offset, uint32_t size,
                                   int gem5Bar)
{
    // Use the AMDGPUDevice's register read interface for MMIO BAR.
    // For other BARs we need to construct packets, but the existing
    // getRegVal/setRegVal only handle 32-bit MMIO reads.
    //
    // For MMIO reads, we can use getRegVal which handles the full
    // MMIO dispatch path.
    if (gem5Bar == MMIO_BAR && size == sizeof(uint32_t)) {
        return gpuDevice->getRegVal(offset);
    }

    // For non-standard sizes or other BARs, construct a packet.
    // The BAR base address must be added to get the full PCI address,
    // but since we're calling readMMIO/readDoorbell/readFrame directly
    // through getRegVal/setRegVal with offsets, we handle 32-bit MMIO
    // specially and warn for others.
    if (gem5Bar == MMIO_BAR) {
        // Handle different read sizes by composing 32-bit reads
        uint64_t value = 0;
        uint32_t remaining = size;
        uint64_t cur_addr = offset;
        uint32_t byte_offset = 0;

        while (remaining >= 4) {
            uint32_t reg_val = gpuDevice->getRegVal(cur_addr);
            memcpy(reinterpret_cast<uint8_t *>(&value) + byte_offset,
                   &reg_val, sizeof(uint32_t));
            cur_addr += 4;
            byte_offset += 4;
            remaining -= 4;
        }

        if (remaining > 0) {
            uint32_t reg_val = gpuDevice->getRegVal(cur_addr);
            memcpy(reinterpret_cast<uint8_t *>(&value) + byte_offset,
                   &reg_val, remaining);
        }

        return value;
    }

    // For doorbell and framebuffer, use getRegVal as a best-effort approach.
    // A full implementation would construct proper packets with BAR addresses.
    DPRINTF(MI300XCosim,
            "Read from gem5 BAR %d offset 0x%lx size %u "
            "(using register interface)\n",
            gem5Bar, offset, size);

    if (size <= sizeof(uint32_t)) {
        return gpuDevice->getRegVal(offset);
    }

    // For larger reads, compose from 32-bit reads
    uint64_t value = 0;
    for (uint32_t i = 0; i < size; i += 4) {
        uint32_t partial = gpuDevice->getRegVal(offset + i);
        uint32_t copySize = std::min((uint32_t)4, size - i);
        memcpy(reinterpret_cast<uint8_t *>(&value) + i, &partial, copySize);
    }
    return value;
}

void
MI300XGem5Cosim::writeToGpuDevice(uint64_t offset, uint32_t size,
                                  uint64_t data, int gem5Bar)
{
    if (gem5Bar == MMIO_BAR && size == sizeof(uint32_t)) {
        gpuDevice->setRegVal(offset, static_cast<uint32_t>(data));
        return;
    }

    if (gem5Bar == MMIO_BAR) {
        // Handle different write sizes by decomposing into 32-bit writes
        uint32_t remaining = size;
        uint64_t cur_addr = offset;
        uint32_t byte_offset = 0;

        while (remaining >= 4) {
            uint32_t reg_val;
            memcpy(&reg_val,
                   reinterpret_cast<const uint8_t *>(&data) + byte_offset,
                   sizeof(uint32_t));
            gpuDevice->setRegVal(cur_addr, reg_val);
            cur_addr += 4;
            byte_offset += 4;
            remaining -= 4;
        }

        if (remaining > 0) {
            // For sub-dword writes, read-modify-write
            uint32_t reg_val = gpuDevice->getRegVal(cur_addr);
            memcpy(&reg_val,
                   reinterpret_cast<const uint8_t *>(&data) + byte_offset,
                   remaining);
            gpuDevice->setRegVal(cur_addr, reg_val);
        }
        return;
    }

    DPRINTF(MI300XCosim,
            "Write to gem5 BAR %d offset 0x%lx size %u value 0x%lx "
            "(using register interface)\n",
            gem5Bar, offset, size, data);

    if (size <= sizeof(uint32_t)) {
        gpuDevice->setRegVal(offset, static_cast<uint32_t>(data));
        return;
    }

    for (uint32_t i = 0; i < size; i += 4) {
        uint32_t partial;
        uint32_t copySize = std::min((uint32_t)4, size - i);
        memcpy(&partial, reinterpret_cast<const uint8_t *>(&data) + i,
               copySize);
        gpuDevice->setRegVal(offset + i, partial);
    }
}

// ---- DMA and Interrupt (gem5 -> QEMU) ----

bool
MI300XGem5Cosim::sendDmaRead(uint64_t addr, uint32_t size,
                             const uint8_t *data)
{
    if (primaryClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.msg_type = static_cast<uint32_t>(CosimMsgType::DmaRead);
    msg.msg_id = nextMsgId++;
    msg.addr = addr;
    msg.size = std::min(size, (uint32_t)sizeof(msg.data));
    if (data) {
        memcpy(msg.data, data, msg.size);
    }
    sendResponse(primaryClientFd, msg);

    // Wait for response
    CosimMsgHeader resp;
    if (!recvAll(primaryClientFd, &resp, sizeof(resp))) {
        return false;
    }

    return resp.status == 0;
}

bool
MI300XGem5Cosim::sendDmaWrite(uint64_t addr, uint32_t size,
                              const uint8_t *data)
{
    if (primaryClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.msg_type = static_cast<uint32_t>(CosimMsgType::DmaWrite);
    msg.msg_id = nextMsgId++;
    msg.addr = addr;
    msg.size = std::min(size, (uint32_t)sizeof(msg.data));
    if (data) {
        memcpy(msg.data, data, msg.size);
    }
    sendResponse(primaryClientFd, msg);

    CosimMsgHeader resp;
    if (!recvAll(primaryClientFd, &resp, sizeof(resp))) {
        return false;
    }

    return resp.status == 0;
}

bool
MI300XGem5Cosim::sendInterrupt(uint32_t vector)
{
    if (primaryClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.msg_type = static_cast<uint32_t>(CosimMsgType::Interrupt);
    msg.msg_id = nextMsgId++;
    msg.addr = vector;
    msg.size = 0;
    msg.status = 0;
    sendResponse(primaryClientFd, msg);

    CosimMsgHeader resp;
    if (!recvAll(primaryClientFd, &resp, sizeof(resp))) {
        return false;
    }

    return resp.status == 0;
}

} // namespace gem5
