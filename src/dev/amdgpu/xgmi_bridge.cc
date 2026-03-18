/*
 * Copyright (c) 2025 The gem5 Contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "dev/amdgpu/xgmi_bridge.hh"

#include "base/trace.hh"
#include "debug/XGMIBridge.hh"
#include "dev/amdgpu/amdgpu_device.hh"

namespace gem5
{

XGMIBridge::XGMIBridge(const Params &p)
    : SimObject(p),
      gpuDevice(p.gpu_device),
      gpuId(p.gpu_id),
      bandwidth(p.bandwidth),
      latencyTicks(p.latency),
      numLanes(p.num_lanes),
      maxLinks(p.max_links),
      creditCount(p.credit_count),
      vramSizePerGpu(p.vram_size_per_gpu)
{
    DPRINTF(XGMIBridge, "XGMIBridge: GPU %d, BW=%lu B/s, lat=%lu ticks\n",
            gpuId, bandwidth, latencyTicks);
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

void
XGMIBridge::addPeer(XGMIBridge *peer)
{
    int peerId = peer->gpuId;
    if (static_cast<int>(peers.size()) <= peerId) {
        peers.resize(peerId + 1, nullptr);
        credits.resize(peerId + 1, creditCount);
    }
    peers[peerId] = peer;
    DPRINTF(XGMIBridge, "XGMIBridge: GPU %d added peer GPU %d\n", gpuId,
            peerId);
}

void
XGMIBridge::sendPacket(const XGMIPacket &pkt, const uint8_t *data)
{
    if (pkt.dstGpu >= static_cast<int>(peers.size()) || !peers[pkt.dstGpu]) {
        warn("XGMIBridge: GPU %d cannot reach GPU %d\n", gpuId, pkt.dstGpu);
        return;
    }

    if (credits[pkt.dstGpu] <= 0) {
        DPRINTF(XGMIBridge,
                "XGMIBridge: GPU %d -> GPU %d: "
                "back-pressure (no credits)\n",
                gpuId, pkt.dstGpu);
        // Stall — caller must retry; packets are never dropped.
        return;
    }

    credits[pkt.dstGpu]--;
    DPRINTF(XGMIBridge,
            "XGMIBridge: GPU %d -> GPU %d: "
            "addr=0x%lx size=%u credits=%d\n",
            gpuId, pkt.dstGpu, pkt.addr, pkt.size, credits[pkt.dstGpu]);

    peers[pkt.dstGpu]->recvPacket(pkt, data);
}

void
XGMIBridge::recvPacket(const XGMIPacket &pkt, const uint8_t *data)
{
    DPRINTF(XGMIBridge,
            "XGMIBridge: GPU %d received from GPU %d: "
            "addr=0x%lx size=%u\n",
            gpuId, pkt.srcGpu, pkt.addr, pkt.size);

    // Return credit to sender
    if (pkt.srcGpu < static_cast<int>(peers.size()) && peers[pkt.srcGpu]) {
        peers[pkt.srcGpu]->credits[gpuId]++;
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
