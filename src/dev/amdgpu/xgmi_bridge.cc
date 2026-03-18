/*
 * Copyright (c) 2025 The gem5 Contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "dev/amdgpu/xgmi_bridge.hh"

#include "base/trace.hh"
#include "debug/XGMIBridge.hh"
#include "dev/amdgpu/amdgpu_device.hh"
#include "sim/system.hh"

namespace gem5
{

XGMIBridge::XGMIBridge(const Params &p)
    : SimObject(p),
      gpuDevice(p.gpu_device),
      gpuId(p.gpu_id),
      numGpus(p.num_gpus),
      bandwidthBps(p.bandwidth),
      linkLatency(p.latency),
      numLanes(p.num_lanes),
      maxLinks(p.max_links),
      creditCount(p.credit_count),
      vramSizePerGpu(p.vram_size_per_gpu),
      deliveryEvent([this] { /* batch event placeholder */ }, name())
{
    DPRINTF(XGMIBridge,
            "GPU %d: BW=%lu B/s, latency=%lu ticks, "
            "credits=%d, VRAM/GPU=%lu\n",
            gpuId, bandwidthBps, linkLatency, creditCount, vramSizePerGpu);
}

void
XGMIBridge::init()
{
    SimObject::init();

    // Build peer table from configuration parameter
    credits.resize(numGpus, creditCount);
    sendQueues.resize(numGpus);

    for (auto *peer : params().peers) {
        int peerId = peer->gpuId;
        if (peerId >= 0 && peerId < numGpus && peerId != gpuId) {
            if (static_cast<int>(peers.size()) <= peerId) {
                peers.resize(peerId + 1, nullptr);
            }
            peers[peerId] = peer;
            DPRINTF(XGMIBridge, "GPU %d: peer GPU %d registered\n", gpuId,
                    peerId);
        }
    }
}

bool
XGMIBridge::isRemoteAddr(Addr addr) const
{
    uint64_t localBase = static_cast<uint64_t>(gpuId) * vramSizePerGpu;
    uint64_t localEnd = localBase + vramSizePerGpu;
    return (addr < localBase || addr >= localEnd);
}

int
XGMIBridge::getDestGpu(Addr addr) const
{
    return static_cast<int>(addr / vramSizePerGpu);
}

bool
XGMIBridge::sendPacket(XGMIPacket pkt)
{
    int dst = pkt.dstGpu;

    if (dst < 0 || dst >= numGpus || dst == gpuId) {
        warn("XGMIBridge GPU %d: invalid destination GPU %d\n", gpuId, dst);
        return false;
    }

    if (dst >= static_cast<int>(peers.size()) || !peers[dst]) {
        warn("XGMIBridge GPU %d: no peer for GPU %d\n", gpuId, dst);
        return false;
    }

    if (credits[dst] <= 0) {
        DPRINTF(XGMIBridge, "GPU %d -> GPU %d: stalled (no credits)\n", gpuId,
                dst);
        return false;
    }

    credits[dst]--;

    // Calculate transfer time based on bandwidth
    Tick transferTime = linkLatency;
    if (bandwidthBps > 0 && pkt.size > 0) {
        Tick dataTime = (static_cast<Tick>(pkt.size) * sim_clock::as_int::s) /
                        bandwidthBps;
        transferTime += dataTime;
    }

    DPRINTF(XGMIBridge,
            "GPU %d -> GPU %d: addr=0x%lx size=%u "
            "credits=%d xfer_ticks=%lu\n",
            gpuId, dst, pkt.addr, pkt.size, credits[dst], transferTime);

    // Schedule delivery event on the destination bridge
    peers[dst]->scheduleDelivery(std::move(pkt));

    return true;
}

void
XGMIBridge::scheduleDelivery(XGMIPacket pkt)
{
    Tick deliverAt = curTick() + linkLatency;
    if (bandwidthBps > 0 && pkt.size > 0) {
        deliverAt += (static_cast<Tick>(pkt.size) * sim_clock::as_int::s) /
                     bandwidthBps;
    }

    // Store packet and schedule event
    sendQueues[pkt.srcGpu].push(std::move(pkt));

    // Use a per-packet event (simplified; production would use a queue drain)
    auto *event = new EventFunctionWrapper(
        [this, srcGpu = sendQueues[pkt.srcGpu].front().srcGpu] {
            if (!sendQueues[srcGpu].empty()) {
                auto p = std::move(sendQueues[srcGpu].front());
                sendQueues[srcGpu].pop();
                deliverPacket(std::move(p));
            }
        },
        name(), true /* auto-delete */
    );
    schedule(event, deliverAt);
}

void
XGMIBridge::deliverPacket(XGMIPacket pkt)
{
    DPRINTF(XGMIBridge,
            "GPU %d: delivering %s from GPU %d, "
            "addr=0x%lx size=%u\n",
            gpuId,
            pkt.type == XGMIPacketType::ReadReq ? "ReadReq" : "WriteReq",
            pkt.srcGpu, pkt.addr, pkt.size);

    auto *system = gpuDevice->CP()->shader()->gpuCmdProc.system();

    if (pkt.type == XGMIPacketType::WriteReq && !pkt.payload.empty()) {
        // Write payload to local VRAM
        RequestPtr req = std::make_shared<Request>(
            pkt.addr, pkt.size, 0, gpuDevice->vramRequestorId());
        PacketPtr writePkt = Packet::createWrite(req);
        writePkt->dataDynamic(new uint8_t[pkt.size]);
        std::memcpy(writePkt->getPtr<uint8_t>(), pkt.payload.data(), pkt.size);

        auto *devMem = system->getDeviceMemory(writePkt);
        if (devMem) {
            devMem->access(writePkt);
        }
        delete writePkt;

        // Return credit after write completion
        returnCredit(pkt.srcGpu);

    } else if (pkt.type == XGMIPacketType::ReadReq) {
        // Read data from local VRAM and send response back
        RequestPtr req = std::make_shared<Request>(
            pkt.addr, pkt.size, 0, gpuDevice->vramRequestorId());
        PacketPtr readPkt = Packet::createRead(req);
        uint8_t *dataPtr = new uint8_t[pkt.size];
        readPkt->dataDynamic(dataPtr);

        auto *devMem = system->getDeviceMemory(readPkt);
        if (devMem) {
            devMem->access(readPkt);

            // Build read response with data
            XGMIPacket resp;
            resp.type = XGMIPacketType::ReadResp;
            resp.srcGpu = gpuId;
            resp.dstGpu = pkt.srcGpu;
            resp.addr = pkt.addr;
            resp.size = pkt.size;
            resp.transactionId = pkt.transactionId;
            resp.payload.resize(pkt.size);
            std::memcpy(resp.payload.data(), dataPtr, pkt.size);

            // Send response back to requester
            if (pkt.srcGpu < static_cast<int>(peers.size()) &&
                peers[pkt.srcGpu]) {
                peers[pkt.srcGpu]->scheduleDelivery(std::move(resp));
            }
        }
        delete readPkt;

        // Return credit after read completes (data sent back)
        returnCredit(pkt.srcGpu);

    } else if (pkt.type == XGMIPacketType::ReadResp) {
        // Read response arrived — data is in payload
        DPRINTF(XGMIBridge,
                "GPU %d: read response from GPU %d, "
                "txn=%lu, %u bytes\n",
                gpuId, pkt.srcGpu, pkt.transactionId, pkt.size);

        // Credit returned by the responder side
        returnCredit(pkt.srcGpu);
    }
}

void
XGMIBridge::returnCredit(int senderGpuId)
{
    if (senderGpuId >= 0 && senderGpuId < static_cast<int>(peers.size()) &&
        peers[senderGpuId]) {
        peers[senderGpuId]->credits[gpuId]++;
        DPRINTF(XGMIBridge,
                "GPU %d: returned credit to GPU %d "
                "(now %d)\n",
                gpuId, senderGpuId, peers[senderGpuId]->credits[gpuId]);
    }
}

int
XGMIBridge::availableCredits(int dstGpu) const
{
    if (dstGpu >= static_cast<int>(credits.size())) {
        return 0;
    }
    return credits[dstGpu];
}

} // namespace gem5
