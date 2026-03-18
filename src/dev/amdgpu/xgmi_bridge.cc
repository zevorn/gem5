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

    for (auto *peer : p.peers) {
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
            "GPU %d: delivering packet from GPU %d, "
            "addr=0x%lx size=%u\n",
            gpuId, pkt.srcGpu, pkt.addr, pkt.size);

    // Write payload to destination VRAM via device memory system
    if (!pkt.payload.empty() && gpuDevice) {
        Addr localAddr = gpuDevice->globalToLocalVRAM(pkt.addr);
        auto *system = gpuDevice->cp->shader()->gpuCmdProc.system();

        RequestPtr req = std::make_shared<Request>(
            pkt.addr, pkt.size, 0, gpuDevice->vramRequestorId());
        PacketPtr writePkt = Packet::createWrite(req);
        writePkt->dataDynamic(new uint8_t[pkt.size]);
        std::memcpy(writePkt->getPtr<uint8_t>(), pkt.payload.data(), pkt.size);

        auto *devMem = system->getDeviceMemory(writePkt);
        if (devMem) {
            devMem->access(writePkt);
            DPRINTF(XGMIBridge,
                    "GPU %d: wrote %u bytes to VRAM "
                    "addr=0x%lx (local=0x%lx)\n",
                    gpuId, pkt.size, pkt.addr, localAddr);
        } else {
            warn("XGMIBridge GPU %d: no device memory for addr 0x%lx\n", gpuId,
                 pkt.addr);
        }
        delete writePkt;
    }

    // Return credit to sender after delivery completes
    returnCredit(pkt.srcGpu);
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
