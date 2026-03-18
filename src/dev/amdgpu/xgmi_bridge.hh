/*
 * Copyright (c) 2025 The gem5 Contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __DEV_AMDGPU_XGMI_BRIDGE_HH__
#define __DEV_AMDGPU_XGMI_BRIDGE_HH__

#include <cstdint>
#include <queue>
#include <vector>

#include "params/XGMIBridge.hh"
#include "sim/eventq.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class AMDGPUDevice;

/**
 * xGMI packet header for inter-GPU communication.
 */
enum class XGMIPacketType : uint8_t
{
    WriteReq = 0,   // Write data to remote VRAM
    ReadReq = 1,    // Read data from remote VRAM
    ReadResp = 2,   // Response with data from remote read
    Completion = 3, // Write completion acknowledgment
};

struct XGMIPacket
{
    XGMIPacketType type = XGMIPacketType::WriteReq;
    uint8_t srcGpu;
    uint8_t dstGpu;
    uint64_t addr;
    uint32_t size;
    uint64_t transactionId = 0;
    std::vector<uint8_t> payload;
};

/**
 * xGMI interconnect bridge attached to a GPU's L2 (TCC) egress.
 *
 * Routes remote VRAM accesses through the modeled xGMI link with
 * bandwidth throttling, latency modeling via gem5 events, and
 * credit-based flow control.
 */
class XGMIBridge : public SimObject
{
  public:
    PARAMS(XGMIBridge);
    XGMIBridge(const Params &p);
    ~XGMIBridge() override = default;

    void init() override;

    /** Check if an address targets a remote GPU's VRAM. */
    bool isRemoteAddr(Addr addr) const;

    /** Get the destination GPU ID for a remote address. */
    int getDestGpu(Addr addr) const;

    /**
     * Enqueue a packet for transmission to a remote GPU.
     * Returns true if accepted, false if back-pressured (no credits).
     */
    bool sendPacket(XGMIPacket pkt);

    /** Available flow-control credits for a given destination. */
    int availableCredits(int dstGpu) const;

  private:
    AMDGPUDevice *gpuDevice;
    int gpuId;
    int numGpus;

    uint64_t bandwidthBps;
    Tick linkLatency;
    int numLanes;
    int maxLinks;
    int creditCount;

    uint64_t vramSizePerGpu;

    /** Peer bridges from configuration (indexed by position, not GPU ID). */
    std::vector<XGMIBridge *> peers;

    /** Per-destination credit counters (indexed by GPU ID). */
    std::vector<int> credits;

    /** Per-destination send queues. */
    std::vector<std::queue<XGMIPacket>> sendQueues;

    /** Schedule delivery of a packet after link latency. */
    void scheduleDelivery(XGMIPacket pkt);

    /** Called by event: deliver packet to destination VRAM. */
    void deliverPacket(XGMIPacket pkt);

    /** Called by event: return credit to sender after completion. */
    void returnCredit(int senderGpuId);

    /** Delivery event wrapper. */
    EventFunctionWrapper deliveryEvent;
};

} // namespace gem5

#endif // __DEV_AMDGPU_XGMI_BRIDGE_HH__
