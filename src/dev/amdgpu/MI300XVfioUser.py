# Copyright (c) 2024-2025 The gem5 Contributors
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

from m5.objects.AMDGPU import AMDGPUDevice
from m5.params import *
from m5.SimObject import SimObject


class MI300XVfioUser(SimObject):
    """vfio-user server exposing gem5 AMDGPUDevice as a standard PCI device.

    Uses the standard vfio-user protocol (libvfio-user) so QEMU can connect
    with its built-in vfio-user-pci device — no custom QEMU device needed.

    VRAM is shared via mmap'd /dev/shm for zero-copy access.
    Guest RAM is mapped via vfio-user DMA_MAP (fd passing + mmap).
    Interrupts use MSI-X eventfd for KVM direct injection.
    """

    type = "MI300XVfioUser"
    cxx_header = "dev/amdgpu/mi300x_vfio_user.hh"
    cxx_class = "gem5::MI300XVfioUser"

    gpu_device = Param.AMDGPUDevice("The AMDGPUDevice to forward MMIO to")

    socket_path = Param.String(
        "/tmp/gem5-mi300x.sock",
        "Unix domain socket path for vfio-user connection",
    )

    shmem_path = Param.String(
        "/mi300x-vram",
        "POSIX shared memory name for VRAM (e.g. /mi300x-vram)",
    )

    vram_size = Param.MemorySize("16GiB", "Size of VRAM shared memory region")
