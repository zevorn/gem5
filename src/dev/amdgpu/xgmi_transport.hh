/*
 * Copyright (c) 2025 The gem5 Contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __DEV_AMDGPU_XGMI_TRANSPORT_HH__
#define __DEV_AMDGPU_XGMI_TRANSPORT_HH__

/**
 * Abstract xGMI transport interface.
 *
 * Allows swapping between:
 *   - InProcessTransport: direct function calls (Milestone 1-2, single gem5)
 *   - IPCTransport: Unix socket / shmem ring buffer (Milestone 3, multi-gem5)
 *   - SSTTransport: SST Merlin network (Milestone 4-5)
 */

#include <cstdint>
#include <functional>
#include <vector>

namespace gem5
{

struct XGMIPacket;

class XGMITransport
{
  public:
    virtual ~XGMITransport() = default;

    /**
     * Send a packet to a remote GPU.
     * Returns true if accepted, false if back-pressured.
     */
    virtual bool send(int dstGpu, const XGMIPacket &pkt) = 0;

    /**
     * Register a callback for incoming packets.
     */
    using RecvCallback = std::function<void(XGMIPacket)>;
    virtual void setRecvCallback(RecvCallback cb) = 0;

    /** Poll for incoming data (for IPC backends). */
    virtual void
    poll()
    {}

    /** Available credits for a destination. */
    virtual int credits(int dstGpu) const = 0;
};

/**
 * In-process transport: direct peer pointer calls.
 * Used in single-gem5-process multi-GPU mode (Milestone 1-2).
 */
class InProcessTransport : public XGMITransport
{
  public:
    void
    setPeers(std::vector<InProcessTransport *> peers)
    {
        peers_ = std::move(peers);
    }

    bool send(int dstGpu, const XGMIPacket &pkt) override;
    void
    setRecvCallback(RecvCallback cb) override
    {
        recvCb_ = std::move(cb);
    }
    int credits(int dstGpu) const override;

    void
    deliver(XGMIPacket pkt)
    {
        if (recvCb_) {
            recvCb_(std::move(pkt));
        }
    }

  private:
    std::vector<InProcessTransport *> peers_;
    RecvCallback recvCb_;
    std::vector<int> credits_;
};

/**
 * IPC transport stub for multi-process gem5 (Milestone 3).
 *
 * Each GPU runs in a separate gem5 process/container.  The transport
 * uses Unix domain sockets with a fixed simulation quantum for
 * global time synchronization.
 *
 * Protocol:
 *   1. Each process opens N-1 socket connections to peer GPUs
 *   2. At each quantum boundary, all processes exchange xGMI packets
 *   3. A coordinator process (or barrier) advances global time
 *
 * Implementation is a skeleton — the actual socket I/O and quantum
 * sync are marked as TODO for the next implementation phase.
 */
class IPCTransport : public XGMITransport
{
  public:
    IPCTransport(int gpuId, int numGpus, uint64_t quantumTicks)
        : gpuId_(gpuId), numGpus_(numGpus), quantum_(quantumTicks)
    {}

    bool send(int dstGpu, const XGMIPacket &pkt) override;
    void
    setRecvCallback(RecvCallback cb) override
    {
        recvCb_ = std::move(cb);
    }
    void poll() override;
    int credits(int dstGpu) const override;

    /** Connect to peer GPU's transport socket. */
    void connectPeer(int peerId, const std::string &socketPath);

    /** Synchronize at quantum boundary. */
    void syncBarrier();

  private:
    int gpuId_;
    int numGpus_;
    uint64_t quantum_;
    RecvCallback recvCb_;
    // Socket fds and send/recv buffers would go here
};

} // namespace gem5

#endif // __DEV_AMDGPU_XGMI_TRANSPORT_HH__
