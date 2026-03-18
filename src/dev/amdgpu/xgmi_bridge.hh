/*
 * Copyright (c) 2025 The gem5 Contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __DEV_AMDGPU_XGMI_BRIDGE_HH__
#define __DEV_AMDGPU_XGMI_BRIDGE_HH__

#include <cstdint>
#include <vector>

#include "params/XGMIBridge.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class AMDGPUDevice;

/**
 * xGMI packet header for inter-GPU communication.
 */
struct XGMIPacket
{
    uint8_t srcGpu;
    uint8_t dstGpu;
    uint64_t addr;
    uint32_t size;
    // payload follows header in actual transport
};

/**
 * xGMI interconnect bridge attached to a GPU's L2 (TCC) egress.
 *
 * Routes remote VRAM accesses through the modeled xGMI link.
 * Local VRAM accesses bypass the bridge entirely.
 */
class XGMIBridge : public SimObject
{
  public:
    PARAMS(XGMIBridge);
    XGMIBridge(const Params &p);
    ~XGMIBridge() override = default;

    /** Check if an address targets a remote GPU's VRAM. */
    bool isRemoteAddr(Addr addr) const;

    /** Get the destination GPU ID for a remote address. */
    int getDestGpu(Addr addr) const;

    /** Register a peer bridge for direct communication. */
    void addPeer(XGMIBridge *peer);

    /** Send a packet to a remote GPU via xGMI. */
    void sendPacket(const XGMIPacket &pkt, const uint8_t *data);

    /** Receive a packet from a remote GPU. */
    void recvPacket(const XGMIPacket &pkt, const uint8_t *data);

    /** Available flow-control credits for a given destination. */
    int availableCredits(int dstGpu) const;

  private:
    AMDGPUDevice *gpuDevice;
    int gpuId;

    uint64_t bandwidth;    // bytes per second
    uint64_t latencyTicks; // link latency in ticks
    int numLanes;
    int maxLinks;
    int creditCount;

    uint64_t vramSizePerGpu;

    /** Peer bridges indexed by GPU ID. */
    std::vector<XGMIBridge *> peers;

    /** Per-destination credit counters. */
    std::vector<int> credits;
};

} // namespace gem5

#endif // __DEV_AMDGPU_XGMI_BRIDGE_HH__
