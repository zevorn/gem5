# Copyright (c) 2024 The gem5 Contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""MI300X co-simulation configuration for QEMU + gem5.

This script starts gem5 with a full MI300X GPU model (compute units, SDMA
engines, PM4 command processors, memory hierarchy) and a co-simulation
socket bridge (MI300XGem5Cosim).  The HOST side (CPU, kernel, ROCm driver)
runs inside QEMU with a Q35 chipset, and QEMU's "mi300x-gem5" PCIe device
forwards MMIO/Doorbell/DMA over a Unix domain socket to this gem5 process.

Usage:
    # Terminal 1 - start gem5 (waits for QEMU to connect):
    build/VEGA_X86/gem5.opt configs/example/gpufs/mi300_cosim.py \\
        --socket-path /tmp/gem5-mi300x.sock \\
        --shmem-path  /mi300x-vram

    # Terminal 2 - start QEMU (connects to gem5):
    qemu-system-x86_64 -machine q35 -enable-kvm -m 8G -smp 4 \\
        -device mi300x-gem5,gem5-socket=/tmp/gem5-mi300x.sock,\\
                shmem-path=/dev/shm/mi300x-vram \\
        -drive file=disk-image.qcow2,format=qcow2 \\
        -kernel vmlinux-gpu-ml \\
        -append "console=ttyS0 root=/dev/sda1" \\
        -nographic

See scripts/cosim_launch.sh for a helper that starts both processes.
"""

import argparse
import math

import m5
from m5.objects import *
from m5.util import addToPath

addToPath("../../")
from amd import AmdGPUOptions
from common import (
    GPUTLBConfig,
    GPUTLBOptions,
    ObjectList,
    Options,
)
from ruby import Ruby

from example.gpufs.Disjoint_VIPER import *


def addCosimOptions(parser):
    parser.add_argument(
        "--socket-path",
        type=str,
        default="/tmp/gem5-mi300x.sock",
        help="Unix domain socket path for QEMU connection",
    )
    parser.add_argument(
        "--shmem-path",
        type=str,
        default="/mi300x-vram",
        help="POSIX shared memory name for VRAM (e.g. /mi300x-vram)",
    )
    parser.add_argument(
        "--dgpu-mem-size",
        type=str,
        default="16GiB",
        help="dGPU memory size",
    )
    parser.add_argument(
        "--dgpu-num-dirs",
        type=int,
        default=1,
        help="Number of dGPU directories (memory controllers)",
    )
    parser.add_argument(
        "--dgpu-mem-type",
        default="HBM_1000_4H_1x128",
        choices=ObjectList.mem_list.get_names(),
        help="Type of dGPU memory",
    )
    parser.add_argument(
        "--gpu-device",
        default="MI300X",
        choices=["MI300X", "MI355X"],
        help="GPU model (MI300X=gfx942, MI355X=gfx950)",
    )
    parser.add_argument(
        "--gpu-topology",
        type=str,
        default="Crossbar",
        help="Network topology for GPU side",
    )


def buildCosimSystem(args):
    """Build a minimal gem5 system with the MI300X GPU model and cosim bridge.

    Unlike the standard GPUFS config (mi300.py), this does NOT create a
    full x86 host system.  Instead, it creates:
      - A minimal System with the Ruby GPU memory hierarchy
      - The AMDGPUDevice with all its IP blocks (SDMA, PM4, IH, CU, etc.)
      - An MI300XGem5Cosim bridge listening on a Unix socket
    QEMU provides the host CPU, kernel, and amdgpu driver.
    """

    # ----------------------------------------------------------------
    # System skeleton
    # ----------------------------------------------------------------
    system = System()
    system.mem_mode = "atomic_noncaching"
    system.cache_line_size = args.cacheline_size

    system.voltage_domain = VoltageDomain(voltage=args.sys_voltage)
    system.clk_domain = SrcClockDomain(
        clock=args.sys_clock, voltage_domain=system.voltage_domain
    )

    # We still need a memory bus for DMA devices
    system.membus = SystemXBar()

    # Host memory range (for DMA from GPU to guest RAM via cosim bridge)
    system.mem_ranges = [AddrRange(args.mem_size)]
    system.memories = [SimpleMemory(range=system.mem_ranges[0])]
    system.memories[0].port = system.membus.mem_side_ports

    # IO bus for PIO devices
    system.iobus = IOXBar()
    system.bridge = Bridge(delay="50ns")
    system.bridge.mem_side_port = system.iobus.cpu_side_ports
    system.bridge.cpu_side_port = system.membus.mem_side_ports
    system.bridge.ranges = [
        AddrRange(0xC0000000, size="256MiB"),  # PCI config space
        AddrRange(0xE0000000, size="256MiB"),  # HSAPP map area
    ]

    # ----------------------------------------------------------------
    # GPU shader (compute units)
    # ----------------------------------------------------------------
    n_cu = args.num_compute_units
    args.num_sqc = int(math.ceil(float(n_cu) / args.cu_per_sqc))
    args.num_scalar_cache = int(
        math.ceil(float(n_cu) / args.cu_per_scalar_cache)
    )

    shader = Shader(
        n_wf=args.wfs_per_simd,
        cu_per_sqc=args.cu_per_sqc,
        timing=True,
        clk_domain=system.clk_domain,
    )
    shader.impl_kern_launch_acq = True
    shader.impl_kern_end_rel = False

    per_lane = args.TLB_config == "perLane"

    compute_units = []
    for i in range(n_cu):
        compute_units.append(
            ComputeUnit(
                cu_id=i,
                perLaneTLB=per_lane,
                num_SIMDs=args.simds_per_cu,
                wf_size=args.wf_size,
                spbypass_pipe_length=args.sp_bypass_path_length,
                dpbypass_pipe_length=args.dp_bypass_path_length,
                issue_period=args.issue_period,
                coalescer_to_vrf_bus_width=args.glbmem_rd_bus_width,
                vrf_to_coalescer_bus_width=args.glbmem_wr_bus_width,
                num_global_mem_pipes=args.glb_mem_pipes_per_cu,
                num_shared_mem_pipes=args.shr_mem_pipes_per_cu,
                n_wf=args.wfs_per_simd,
                execPolicy=args.CUExecPolicy,
                localMemBarrier=args.LocalMemBarrier,
                countPages=args.countPages,
                memtime_latency=args.memtime_latency,
                max_cu_tokens=args.max_cu_tokens,
                vrf_lm_bus_latency=args.vrf_lm_bus_latency,
                mem_req_latency=args.mem_req_latency,
                mem_resp_latency=args.mem_resp_latency,
                scalar_mem_req_latency=args.scalar_mem_req_latency,
                scalar_mem_resp_latency=args.scalar_mem_resp_latency,
                mfma_scale=args.mfma_scale,
                localDataStore=LdsState(
                    banks=args.numLdsBanks,
                    bankConflictPenalty=args.ldsBankConflictPenalty,
                    size=args.lds_size,
                ),
            )
        )

        wavefronts = []
        vrfs = []
        vrf_pool_mgrs = []
        srfs = []
        rfcs = []
        srf_pool_mgrs = []
        for j in range(args.simds_per_cu):
            for k in range(shader.n_wf):
                wavefronts.append(
                    Wavefront(simdId=j, wf_slot_id=k, wf_size=args.wf_size)
                )
            if args.reg_alloc_policy == "simple":
                vrf_pool_mgrs.append(
                    SimplePoolManager(
                        pool_size=args.vreg_file_size,
                        min_alloc=args.vreg_min_alloc,
                    )
                )
                srf_pool_mgrs.append(
                    SimplePoolManager(
                        pool_size=args.sreg_file_size,
                        min_alloc=args.vreg_min_alloc,
                    )
                )
            elif args.reg_alloc_policy == "dynamic":
                vrf_pool_mgrs.append(
                    DynPoolManager(
                        pool_size=args.vreg_file_size,
                        min_alloc=args.vreg_min_alloc,
                    )
                )
                srf_pool_mgrs.append(
                    DynPoolManager(
                        pool_size=args.sreg_file_size,
                        min_alloc=args.vreg_min_alloc,
                    )
                )

            vrfs.append(
                VectorRegisterFile(
                    simd_id=j,
                    wf_size=args.wf_size,
                    num_regs=args.vreg_file_size,
                )
            )
            srfs.append(
                ScalarRegisterFile(
                    simd_id=j,
                    wf_size=args.wf_size,
                    num_regs=args.sreg_file_size,
                )
            )
            rfcs.append(
                RegisterFileCache(
                    simd_id=j, cache_size=args.register_file_cache_size
                )
            )

        compute_units[-1].wavefronts = wavefronts
        compute_units[-1].vector_register_file = vrfs
        compute_units[-1].scalar_register_file = srfs
        compute_units[-1].register_file_cache = rfcs
        compute_units[-1].register_manager = RegisterManager(
            policy=args.registerManagerPolicy,
            vrf_pool_managers=vrf_pool_mgrs,
            srf_pool_managers=srf_pool_mgrs,
        )

        compute_units[-1].ldsPort = compute_units[-1].ldsBus.cpu_side_port
        compute_units[-1].ldsBus.mem_side_port = (
            compute_units[-1].localDataStore.cuPort
        )

    shader.CUs = compute_units
    shader.eventq_index = 0

    # ----------------------------------------------------------------
    # AMD GPU device
    # ----------------------------------------------------------------
    gpu_dev = AMDGPUDevice(pci_func=0, pci_dev=8)
    gpu_dev.device_name = args.gpu_device

    if args.gpu_device == "MI300X":
        gpu_dev.DeviceID = 0x74A1
        gpu_dev.BAR5 = PciMemBar(size="2MiB")
    elif args.gpu_device == "MI355X":
        gpu_dev.DeviceID = 0x75A0
        gpu_dev.BAR5 = PciMemBar(size="2MiB")

    gpu_dev.SubsystemVendorID = 0x1002
    gpu_dev.SubsystemID = 0x0C34
    gpu_dev.Status = 0x0290
    gpu_dev.PXCAPBaseOffset = 0x80
    gpu_dev.CapabilityPtr = 0x80
    gpu_dev.PXCAPCapId = 0x10
    gpu_dev.PXCAPDevCap2 = 0x00000180
    gpu_dev.PXCAPDevCtrl2 = 0x0040

    # HSA Packet Processor
    hsapp_gpu_map_paddr = 0xE0000000
    hsapp_pt_walker = VegaPagetableWalker()
    gpu_hsapp = HSAPacketProcessor(
        pioAddr=hsapp_gpu_map_paddr,
        numHWQueues=args.num_hw_queues,
        walker=hsapp_pt_walker,
    )

    dispatcher = GPUDispatcher()
    cp_pt_walker = VegaPagetableWalker()
    gpu_cmd_proc = GPUCommandProcessor(
        hsapp=gpu_hsapp,
        dispatcher=dispatcher,
        walker=cp_pt_walker,
    )
    shader.dispatcher = dispatcher
    shader.gpu_cmd_proc = gpu_cmd_proc
    gpu_dev.cp = gpu_cmd_proc

    # GPU Interrupt Handler
    device_ih = AMDGPUInterruptHandler()
    gpu_dev.device_ih = device_ih

    # SDMA engines (MI300X has 16)
    sdma_bases = [
        0x4980, 0x6180, 0x65000, 0x66000,
        0x84980, 0x86180, 0xE5000, 0xE6000,
        0x104980, 0x106180, 0x165000, 0x166000,
        0x184980, 0x186180, 0x1E5000, 0x1E6000,
    ]
    num_sdmas = 16
    sdma_sizes = [0x1000] * num_sdmas

    sdma_pt_walkers = []
    sdma_engines = []
    for idx in range(num_sdmas):
        w = VegaPagetableWalker()
        e = SDMAEngine(
            walker=w,
            mmio_base=sdma_bases[idx],
            mmio_size=sdma_sizes[idx],
        )
        sdma_pt_walkers.append(w)
        sdma_engines.append(e)

    gpu_dev.sdmas = sdma_engines

    # PM4 packet processors (8 for MI300X / MI355X)
    pm4_procs = []
    pm4_ranges = [
        (0xC000, 0xD000), (0x4C000, 0x4D000),
        (0x8C000, 0x8D000), (0xCC000, 0xCD000),
        (0x10C000, 0x10D000), (0x14C000, 0x14D000),
        (0x18C000, 0x18D000), (0x1CC000, 0x1CD000),
    ]
    for ip_id, (start, end) in enumerate(pm4_ranges):
        pm4_procs.append(
            PM4PacketProcessor(
                ip_id=ip_id,
                mmio_range=AddrRange(start=start, end=end),
            )
        )
    gpu_dev.pm4_pkt_procs = pm4_procs

    # GPU memory manager
    gpu_mem_mgr = AMDGPUMemoryManager(cache_line_size=args.cacheline_size)
    gpu_dev.memory_manager = gpu_mem_mgr

    # CPU-side system hub
    system_hub = AMDGPUSystemHub()
    shader.system_hub = system_hub

    # ----------------------------------------------------------------
    # Wire up the GPU to the system
    # ----------------------------------------------------------------
    system.gpu_dev = gpu_dev
    system.shader = shader

    # DMA ports
    system._dma_ports = []
    system._dma_ports.append(gpu_hsapp)
    system._dma_ports.append(gpu_cmd_proc)
    for sdma in sdma_engines:
        system._dma_ports.append(sdma)
    system._dma_ports.append(device_ih)
    for pm4 in pm4_procs:
        system._dma_ports.append(pm4)
    system._dma_ports.append(system_hub)
    system._dma_ports.append(gpu_mem_mgr)
    system._dma_ports.append(hsapp_pt_walker)
    system._dma_ports.append(cp_pt_walker)
    for w in sdma_pt_walkers:
        system._dma_ports.append(w)

    # PIO ports
    gpu_hsapp.pio = system.iobus.mem_side_ports
    gpu_cmd_proc.pio = system.iobus.mem_side_ports
    for sdma in sdma_engines:
        sdma.pio = system.iobus.mem_side_ports
    device_ih.pio = system.iobus.mem_side_ports
    for pm4 in pm4_procs:
        pm4.pio = system.iobus.mem_side_ports
    system_hub.pio = system.iobus.mem_side_ports

    # TLB hierarchy
    args.full_system = True
    shader_idx = 0  # shader is the only "CPU" in this config
    system.cpu = [shader]
    GPUTLBConfig.config_tlb_hierarchy(args, system, shader_idx, gpu_dev, True)

    # Ruby GPU memory hierarchy (disjoint VIPER)
    system.ruby = Disjoint_VIPER()
    system.ruby.create(args, system, system.iobus, system._dma_ports)
    system.ruby.clk_domain = SrcClockDomain(
        clock=args.ruby_clock, voltage_domain=system.voltage_domain
    )

    # Connect shader ports to Ruby
    gpu_port_idx = (
        len(system.ruby._cpu_ports)
        - n_cu
        - args.num_sqc
        - args.num_scalar_cache
    )
    gpu_port_idx -= args.num_cp * 2

    token_port_idx = 0
    for i in range(len(system.ruby._cpu_ports)):
        if isinstance(system.ruby._cpu_ports[i], VIPERCoalescer):
            shader.CUs[token_port_idx].gmTokenPort = (
                system.ruby._cpu_ports[i].gmTokenPort
            )
            token_port_idx += 1

    for i in range(n_cu):
        for j in range(args.wf_size):
            shader.CUs[i].memory_port[j] = (
                system.ruby._cpu_ports[gpu_port_idx].in_ports[j]
            )
        gpu_port_idx += 1

    for i in range(n_cu):
        if i > 0 and not i % args.cu_per_sqc:
            gpu_port_idx += 1
        shader.CUs[i].sqc_port = (
            system.ruby._cpu_ports[gpu_port_idx].in_ports
        )
    gpu_port_idx += 1

    for i in range(n_cu):
        if i > 0 and not i % args.cu_per_scalar_cache:
            gpu_port_idx += 1
        shader.CUs[i].scalar_port = (
            system.ruby._cpu_ports[gpu_port_idx].in_ports
        )

    # ----------------------------------------------------------------
    # Co-simulation bridge
    # ----------------------------------------------------------------
    system.cosim = MI300XGem5Cosim(
        gpu_device=gpu_dev,
        socket_path=args.socket_path,
        shmem_path=args.shmem_path,
        vram_size=args.dgpu_mem_size,
    )

    return system


if __name__ == "__m5_main__":
    parser = argparse.ArgumentParser()
    Options.addCommonOptions(parser)
    AmdGPUOptions.addAmdGPUOptions(parser)
    Ruby.define_options(parser)
    GPUTLBOptions.tlb_options(parser)
    addCosimOptions(parser)

    args = parser.parse_args()

    # Defaults for cosim mode
    args.num_cpus = 0
    args.mem_size = "8GiB"
    args.num_compute_units = 40
    args.gpu_topology = "Crossbar"

    system = buildCosimSystem(args)

    root = Root(full_system=False, system=system)

    m5.instantiate()

    print("=" * 60)
    print("gem5 MI300X co-simulation server ready")
    print(f"  Socket:  {args.socket_path}")
    print(f"  SHM:     {args.shmem_path}")
    print(f"  VRAM:    {args.dgpu_mem_size}")
    print(f"  CUs:     {args.num_compute_units}")
    print("Waiting for QEMU to connect...")
    print("=" * 60)

    exit_event = m5.simulate()

    while True:
        cause = exit_event.getCause()
        if cause in ("m5_exit instruction encountered",
                     "user interrupt received",
                     "simulate() limit reached"):
            break
        elif "GPU Kernel Completed" in cause:
            print(f"GPU kernel completed at tick {m5.curTick()}")
        elif "GPU Blit Kernel Completed" in cause:
            pass
        else:
            print(f"Exit event: {cause}. Continuing...")

        exit_event = m5.simulate(m5.MaxTick - m5.curTick())

    print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
