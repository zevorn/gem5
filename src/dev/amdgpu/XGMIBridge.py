# Copyright (c) 2025 The gem5 Contributors
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from m5.objects.AMDGPU import AMDGPUDevice
from m5.params import *
from m5.SimObject import SimObject


class XGMIBridge(SimObject):
    """xGMI interconnect bridge between MI300X GPU instances.

    Attached to each GPU's L2 cache (TCC) egress port.  Routes memory
    requests whose target address falls in a remote GPU's VRAM range
    through the modeled xGMI link instead of local memory.

    The bridge is instantiated once per GPU and connects to peer bridges
    via the 'peers' parameter (set during configuration).
    """

    type = "XGMIBridge"
    cxx_header = "dev/amdgpu/xgmi_bridge.hh"
    cxx_class = "gem5::XGMIBridge"

    gpu_device = Param.AMDGPUDevice("The GPU this bridge is attached to")
    gpu_id = Param.Int(0, "ID of the GPU this bridge belongs to")
    num_gpus = Param.Int(1, "Total number of GPUs in the system")

    peers = VectorParam.XGMIBridge([], "Peer bridges for xGMI topology")

    bandwidth = Param.MemoryBandwidth(
        "128GBps", "Per-link bandwidth (default: 128 GB/s)"
    )
    latency = Param.Latency("100ns", "Per-hop link latency")
    num_lanes = Param.Int(16, "Number of xGMI lanes per link")
    max_links = Param.Int(7, "Maximum number of xGMI links per GPU")
    credit_count = Param.Int(32, "Flow control credits per link")

    vram_size_per_gpu = Param.MemorySize(
        "16GiB", "VRAM size per GPU (for address range calculation)"
    )
