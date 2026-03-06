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

from m5.objects.AMDGPU import AMDGPUDevice
from m5.params import *
from m5.SimObject import SimObject


class MI300XGem5Cosim(SimObject):
    """Socket server bridging QEMU mi300x-gem5 PCIe device to gem5 AMDGPUDevice.

    This SimObject listens on a Unix domain socket for MMIO requests from
    QEMU's mi300x-gem5 PCIe endpoint device. It forwards those requests to
    the gem5 AMDGPUDevice (MI300X model) and returns the results.

    VRAM is shared via mmap'd /dev/shm for zero-copy access between QEMU
    and gem5.
    """

    type = "MI300XGem5Cosim"
    cxx_header = "dev/amdgpu/mi300x_gem5_cosim.hh"
    cxx_class = "gem5::MI300XGem5Cosim"

    gpu_device = Param.AMDGPUDevice("The AMDGPUDevice to forward MMIO to")

    socket_path = Param.String(
        "/tmp/gem5-mi300x.sock",
        "Unix domain socket path for QEMU connection",
    )

    shmem_path = Param.String(
        "/mi300x-vram",
        "POSIX shared memory name for VRAM (e.g. /mi300x-vram)",
    )

    vram_size = Param.MemorySize(
        "16GiB", "Size of VRAM shared memory region"
    )
