# Copyright (c) 2024 The gem5 Contributors
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MI300X co-simulation configuration for QEMU + gem5.

Builds a minimal gem5 system with ONLY the MI300X GPU model (no x86
kernel, no CPU execution).  QEMU handles all host-side simulation
(CPU, kernel boot, PCI enumeration, ROCm driver); gem5 provides
the GPU compute back-end via a Unix domain socket bridge.

Architecture:
    QEMU (guest Linux + amdgpu driver)
        |  Unix domain socket (MMIO, doorbell, DMA, IRQ)
        v
    MI300XGem5Cosim (this config, in gem5)
        |  forward via AMDGPUDevice read/write
        v
    AMDGPUDevice + Shader + Ruby GPU hierarchy

Shared memory:
    /dev/shm/cosim-guest-ram  - guest physical memory (QEMU + gem5 DMA)
    /dev/shm/mi300x-vram      - GPU VRAM (QEMU driver + gem5 shader)

Usage:
    # Terminal 1 - start gem5 (waits for QEMU to connect):
    build/VEGA_X86/gem5.opt configs/example/gpufs/mi300_cosim.py \
        --socket-path /tmp/gem5-mi300x.sock \
        --shmem-path /mi300x-vram \
        --shmem-host-path /cosim-guest-ram

    # Terminal 2 - start QEMU:
    qemu-system-x86_64 -machine q35 -enable-kvm -smp 4 -m 8G \
        -object memory-backend-file,id=mem0,size=8G,\
                mem-path=/dev/shm/cosim-guest-ram,share=on \
        -numa node,memdev=mem0 \
        -device mi300x-gem5,gem5-socket=/tmp/gem5-mi300x.sock,\
                shmem-path=/dev/shm/mi300x-vram \
        -drive file=disk-image,format=raw,if=virtio \
        -kernel vmlinux -append "console=ttyS0 root=/dev/vda1" \
        -nographic
"""

import argparse
import math

import m5
from m5.objects import *
from m5.util import (
    addToPath,
    convert,
)

addToPath("../../")
from amd import AmdGPUOptions
from common import (
    GPUTLBConfig,
    GPUTLBOptions,
    ObjectList,
    Options,
)
from example.gpufs.Disjoint_VIPER import *
from ruby import Ruby
from system.amdgpu import (
    connectGPU,
    createGPU,
)


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
        "--shmem-host-path",
        type=str,
        default="/cosim-guest-ram",
        help="POSIX shared memory name for host (guest) RAM",
    )


def buildCosimSystem(args):
    """Build a minimal gem5 GPU-only system for cosimulation with QEMU.

    No kernel, no x86 workload — just the MI300X GPU model with PCI
    infrastructure (Pc platform) and Ruby GPU cache hierarchy.
    QEMU handles all host-side simulation.
    """

    n_cu = args.num_compute_units
    args.num_sqc = int(math.ceil(float(n_cu) / args.cu_per_sqc))
    args.num_scalar_cache = int(
        math.ceil(float(n_cu) / args.cu_per_scalar_cache)
    )

    # Minimal system — StubWorkload by default, no kernel needed
    system = System()
    system.mem_mode = "atomic_noncaching"
    system.m5ops_base = 0xFFFF0000

    # Memory ranges — must match QEMU Q35's memory split so that the
    # shared_backstore file offsets agree.  Q35 places the PCI hole at
    # 2 GiB when total RAM >= 2.75 GiB, otherwise at 2.75 GiB.
    # See qemu/hw/i386/pc_q35.c q35_machine_init().
    total_mem = convert.toMemorySize(args.mem_size)
    lowmem_limit = (
        0x80000000  # 2 GiB — Q35 with large RAM
        if total_mem >= 0xB0000000  # 2.75 GiB threshold
        else 0xB0000000  # 2.75 GiB — Q35 with small RAM
    )
    below_4g = min(total_mem, lowmem_limit)
    above_4g = total_mem - below_4g

    if above_4g > 0:
        system.mem_ranges = [
            AddrRange(below_4g),
            AddrRange(Addr("4GiB"), size=above_4g),
        ]
    else:
        system.mem_ranges = [AddrRange(total_mem)]

    # Share host memory with QEMU via POSIX shared memory
    system.shared_backstore = args.shmem_host_path
    system.auto_unlink_shared_backstore = True

    # PCI infrastructure (Pc provides PciBus → PciUpstream for AMDGPUDevice)
    system.pc = Pc()

    # Disable SouthBridge timer events — cosim doesn't need them, and they
    # advance curTick until it overflows, crashing the event scheduler.
    system.pc.south_bridge.cmos.disable_rtc_events = True
    system.pc.south_bridge.pit.disable_timer_events = True

    # Clock domains
    system.voltage_domain = VoltageDomain(voltage=args.sys_voltage)
    system.clk_domain = SrcClockDomain(
        clock=args.sys_clock, voltage_domain=system.voltage_domain
    )
    system.cpu_voltage_domain = VoltageDomain()
    system.cpu_clk_domain = SrcClockDomain(
        clock=args.cpu_clock, voltage_domain=system.cpu_voltage_domain
    )

    # Bus setup (mirrors connectX86RubySystem)
    system.iobus = IOXBar()
    system._dma_ports = [system.pc.pci_host.up_request_port()]
    system.pc.attachIO(system.iobus, system._dma_ports)

    # Minimal dummy CPU — Shader needs cpu_pointer, Ruby needs CPU ports.
    # In cosim mode QEMU handles all CPU work; this CPU is structural only.
    system.cpu = [AtomicSimpleCPU(clk_domain=system.cpu_clk_domain, cpu_id=0)]

    # Create GPU shader and device using standard helpers
    shader = createGPU(system, args)
    connectGPU(system, args)

    # Share VRAM backing store with QEMU
    system.pc.south_bridge.gpu.vram_shared_backstore = args.shmem_path

    # The shader is appended as a "CPU" (needed by GPUTLBConfig)
    shader_idx = args.num_cpus
    system.cpu.append(shader)

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
    system.pc.south_bridge.gpu.cp = gpu_cmd_proc

    # GPU Interrupt Handler
    device_ih = AMDGPUInterruptHandler()
    system.pc.south_bridge.gpu.device_ih = device_ih

    # SDMA engines (MI300X: 16)
    sdma_bases = [
        0x4980,
        0x6180,
        0x65000,
        0x66000,
        0x84980,
        0x86180,
        0xE5000,
        0xE6000,
        0x104980,
        0x106180,
        0x165000,
        0x166000,
        0x184980,
        0x186180,
        0x1E5000,
        0x1E6000,
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
    system.pc.south_bridge.gpu.sdmas = sdma_engines

    # PM4 packet processors (8)
    pm4_procs = []
    pm4_ranges = [
        (0xC000, 0xD000),
        (0x4C000, 0x4D000),
        (0x8C000, 0x8D000),
        (0xCC000, 0xCD000),
        (0x10C000, 0x10D000),
        (0x14C000, 0x14D000),
        (0x18C000, 0x18D000),
        (0x1CC000, 0x1CD000),
    ]
    for ip_id, (start, end) in enumerate(pm4_ranges):
        pm4_procs.append(
            PM4PacketProcessor(
                ip_id=ip_id,
                mmio_range=AddrRange(start=start, end=end),
            )
        )
    system.pc.south_bridge.gpu.pm4_pkt_procs = pm4_procs

    # GPU memory manager and system hub
    gpu_mem_mgr = AMDGPUMemoryManager(cache_line_size=args.cacheline_size)
    system.pc.south_bridge.gpu.memory_manager = gpu_mem_mgr
    system_hub = AMDGPUSystemHub()
    shader.system_hub = system_hub

    # Attach GPU to PCI bus
    system.pc.attachPciDevice(system.pc.south_bridge.gpu)

    # DMA ports
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
    GPUTLBConfig.config_tlb_hierarchy(
        args, system, shader_idx, system.pc.south_bridge.gpu, True
    )

    # Ruby GPU memory hierarchy (disjoint VIPER)
    system.ruby = Disjoint_VIPER()
    system.ruby.create(args, system, system.iobus, system._dma_ports)
    system.ruby.clk_domain = SrcClockDomain(
        clock=args.ruby_clock, voltage_domain=system.voltage_domain
    )

    # Wire CPU ports
    for i in range(args.num_cpus):
        cpu = system.cpu[i]
        cpu.clk_domain = system.cpu_clk_domain
        cpu.createThreads()
        cpu.createInterruptController()
        system.ruby._cpu_ports[i].connectCpuPorts(cpu)

    # Wire GPU ports to Ruby
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
            system.cpu[shader_idx].CUs[token_port_idx].gmTokenPort = (
                system.ruby._cpu_ports[i].gmTokenPort
            )
            token_port_idx += 1

    for i in range(n_cu):
        for j in range(args.wf_size):
            system.cpu[shader_idx].CUs[i].memory_port[j] = (
                system.ruby._cpu_ports[gpu_port_idx].in_ports[j]
            )
        gpu_port_idx += 1

    for i in range(n_cu):
        if i > 0 and not i % args.cu_per_sqc:
            gpu_port_idx += 1
        system.cpu[shader_idx].CUs[i].sqc_port = system.ruby._cpu_ports[
            gpu_port_idx
        ].in_ports
    gpu_port_idx += 1

    for i in range(n_cu):
        if i > 0 and not i % args.cu_per_scalar_cache:
            gpu_port_idx += 1
        system.cpu[shader_idx].CUs[i].scalar_port = system.ruby._cpu_ports[
            gpu_port_idx
        ].in_ports

    # ----------------------------------------------------------------
    # Co-simulation bridge
    # ----------------------------------------------------------------
    system.cosim = MI300XGem5Cosim(
        gpu_device=system.pc.south_bridge.gpu,
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

    # GPU FS options (dgpu-mem-size, gpu-topology, etc.)
    from example.gpufs.runfs import addRunFSOptions

    addRunFSOptions(parser)

    args = parser.parse_args()

    # Cosim defaults — GPU-only, no real CPU execution
    args.num_cpus = 1
    args.cpu_type = "AtomicSimpleCPU"
    args.mem_size = "8GiB"
    args.num_compute_units = 40
    if not hasattr(args, "gpu_topology") or args.gpu_topology is None:
        args.gpu_topology = "Crossbar"
    if not hasattr(args, "cpu_topology") or args.cpu_topology is None:
        args.cpu_topology = "Crossbar"
    args.gpu_device = "MI300X"
    if not hasattr(args, "dgpu_mem_size") or args.dgpu_mem_size is None:
        args.dgpu_mem_size = "16GiB"

    system = buildCosimSystem(args)

    root = Root(full_system=True, system=system)

    m5.instantiate()

    print("=" * 60)
    print("gem5 MI300X co-simulation server ready")
    print(f"  Socket:     {args.socket_path}")
    print(f"  VRAM SHM:   {args.shmem_path}")
    print(f"  Host SHM:   {args.shmem_host_path}")
    print(f"  VRAM size:  {args.dgpu_mem_size}")
    print(f"  Host RAM:   {args.mem_size}")
    print(f"  CUs:        {args.num_compute_units}")
    print("Waiting for QEMU to connect...")
    print("=" * 60)

    exit_event = m5.simulate()

    while True:
        cause = exit_event.getCause()
        if cause in (
            "m5_exit instruction encountered",
            "user interrupt received",
            "simulate() limit reached",
        ):
            break
        elif "GPU Kernel Completed" in cause:
            print(f"GPU kernel completed at tick {m5.curTick()}")
        elif "GPU Blit Kernel Completed" in cause:
            pass
        else:
            print(f"Exit event: {cause}. Continuing...")

        exit_event = m5.simulate(m5.MaxTick - m5.curTick())

    print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
