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
    dmaBuf = new uint8_t[COSIM_DMA_BUF_SIZE];
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
    CosimMsgHeader msg;
    if (!recvAll(fd, &msg, COSIM_MSG_HDR_SIZE)) {
        closeClient(fd);
        return;
    }

    processMessage(fd, msg);
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
      case CosimMsgType::Init:
        handleInit(fd, msg);
        break;
      case CosimMsgType::Shutdown:
        handleShutdown(fd, msg);
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
    inform("MI300XGem5Cosim: SHUTDOWN from QEMU");
    closeClient(fd);
}

void
MI300XGem5Cosim::handleMmioRead(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    uint64_t value = readFromGpuDevice(msg.addr, accessSize, MMIO_BAR);

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

    writeToGpuDevice(msg.addr, accessSize, msg.data, MMIO_BAR);

    // MMIO writes are fire-and-forget from QEMU's perspective
    // (QEMU's mi300x_send_msg passes NULL response for writes)
}

void
MI300XGem5Cosim::handleDoorbellRead(int fd, const CosimMsgHeader &msg)
{
    uint32_t accessSize = msg.access_size;
    if (accessSize == 0 || accessSize > 8)
        accessSize = 4;

    uint64_t value = readFromGpuDevice(msg.addr, accessSize, DOORBELL_BAR);

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

    writeToGpuDevice(msg.addr, accessSize, msg.data, DOORBELL_BAR);

    // Doorbell writes are fire-and-forget (QEMU sends NULL response)
}

// ---- MMIO Forwarding to AMDGPUDevice ----

uint64_t
MI300XGem5Cosim::readFromGpuDevice(uint64_t offset, uint32_t size,
                                   int gem5Bar)
{
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    if (gem5Bar == MMIO_BAR) {
        if (size <= sizeof(uint32_t)) {
            return gpuDevice->getRegVal(offset);
        }

        // Handle 8-byte reads by composing two 32-bit reads
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

    // For doorbell and framebuffer BARs
    DPRINTF(MI300XCosim,
            "Read from gem5 BAR %d offset 0x%lx size %u\n",
            gem5Bar, offset, size);

    if (size <= sizeof(uint32_t)) {
        return gpuDevice->getRegVal(offset);
    }

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
    fatal_if(!gpuDevice, "MI300XGem5Cosim: gpu_device not set");

    if (gem5Bar == MMIO_BAR) {
        if (size <= sizeof(uint32_t)) {
            gpuDevice->setRegVal(offset, static_cast<uint32_t>(data));
            return;
        }

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
            uint32_t reg_val = gpuDevice->getRegVal(cur_addr);
            memcpy(&reg_val,
                   reinterpret_cast<const uint8_t *>(&data) + byte_offset,
                   remaining);
            gpuDevice->setRegVal(cur_addr, reg_val);
        }
        return;
    }

    DPRINTF(MI300XCosim,
            "Write to gem5 BAR %d offset 0x%lx size %u value 0x%lx\n",
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
MI300XGem5Cosim::sendDmaRead(uint64_t addr, uint64_t len)
{
    if (primaryClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.type = static_cast<uint32_t>(CosimMsgType::DmaRead);
    msg.id = nextMsgId++;
    msg.addr = addr;
    msg.data = len;
    msg.size = 0;

    if (!sendAll(primaryClientFd, &msg, COSIM_MSG_HDR_SIZE)) {
        return false;
    }

    // The QEMU event thread will read guest memory and send back
    // a MmioResp with the payload following the header
    CosimMsgHeader resp;
    if (!recvAll(primaryClientFd, &resp, COSIM_MSG_HDR_SIZE)) {
        return false;
    }

    // Read payload data if present
    uint32_t payloadSize = resp.size;
    if (payloadSize > 0 && payloadSize <= COSIM_DMA_BUF_SIZE) {
        if (!recvAll(primaryClientFd, dmaBuf, payloadSize)) {
            return false;
        }
    }

    return true;
}

bool
MI300XGem5Cosim::sendDmaWrite(uint64_t addr, uint64_t len,
                              const uint8_t *data)
{
    if (primaryClientFd < 0)
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

    // Send header followed by payload
    if (!sendAll(primaryClientFd, &msg, COSIM_MSG_HDR_SIZE)) {
        return false;
    }
    if (data && len > 0) {
        if (!sendAll(primaryClientFd, data, len)) {
            return false;
        }
    }

    return true;
}

bool
MI300XGem5Cosim::sendIrqRaise(uint32_t vector)
{
    if (primaryClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.type = static_cast<uint32_t>(CosimMsgType::IrqRaise);
    msg.id = nextMsgId++;
    msg.data = vector;
    msg.size = 0;

    return sendAll(primaryClientFd, &msg, COSIM_MSG_HDR_SIZE);
}

bool
MI300XGem5Cosim::sendIrqLower(uint32_t vector)
{
    if (primaryClientFd < 0)
        return false;

    CosimMsgHeader msg{};
    msg.type = static_cast<uint32_t>(CosimMsgType::IrqLower);
    msg.id = nextMsgId++;
    msg.data = vector;
    msg.size = 0;

    return sendAll(primaryClientFd, &msg, COSIM_MSG_HDR_SIZE);
}

} // namespace gem5
