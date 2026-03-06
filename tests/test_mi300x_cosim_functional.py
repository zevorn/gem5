#!/usr/bin/env python3
"""
QEMU-style functional tests for MI300X gem5-QEMU co-simulation.

Inspired by QEMU's functional test framework, these tests simulate
what a guest OS + amdgpu driver would do through the cosim bridge,
verifying the end-to-end behavior by comparing expected output strings
(similar to how QEMU functional tests SSH into a guest and compare
command output).

Each test simulates the QEMU side: connecting to a mock gem5 server,
sending protocol messages as if the amdgpu driver were probing and
initializing the MI300X, and verifying the responses match expected
behavior.

Usage:
    python3 tests/test_mi300x_cosim_functional.py
"""

import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest


# ======================================================================
# Protocol wire format (must match mi300x_gem5_cosim.hh)
# ======================================================================

MSG_HDR_FORMAT = '<IIQQiI'
MSG_HDR_SIZE = struct.calcsize(MSG_HDR_FORMAT)

# Message types
MSG_MMIO_READ    = 0x01
MSG_MMIO_WRITE   = 0x02
MSG_DB_READ      = 0x03
MSG_DB_WRITE     = 0x04
MSG_DMA_REQ      = 0x05
MSG_INIT         = 0x06
MSG_SHUTDOWN     = 0x07
MSG_CONFIG_READ  = 0x08
MSG_CONFIG_WRITE = 0x09
MSG_FRAME_READ   = 0x0A
MSG_FRAME_WRITE  = 0x0B

MSG_MMIO_RESP    = 0x81
MSG_IRQ_RAISE    = 0x82
MSG_IRQ_LOWER    = 0x83
MSG_DMA_READ     = 0x84
MSG_DMA_WRITE    = 0x85
MSG_INIT_RESP    = 0x86


def pack_msg(msg_type, size=0, addr=0, data=0, access_size=0, msg_id=0):
    return struct.pack(MSG_HDR_FORMAT,
                       msg_type, size, addr, data, access_size, msg_id)


def unpack_msg(buf):
    fields = struct.unpack(MSG_HDR_FORMAT, buf)
    return {
        'type': fields[0],
        'size': fields[1],
        'addr': fields[2],
        'data': fields[3],
        'access_size': fields[4],
        'id': fields[5],
    }


# ======================================================================
# Mock gem5 GPU device model
# ======================================================================

class MockGem5GPU:
    """Simulates the gem5 AMDGPUDevice's register/memory behavior.

    This provides the same register values that the real gem5 MI300X model
    would return, enabling end-to-end validation of the driver init sequence.
    """

    # MI300X PCI config space
    PCI_VENDOR_ID = 0x1002
    PCI_DEVICE_ID = 0x74A1
    PCI_SUBSYS_VENDOR_ID = 0x1002
    PCI_SUBSYS_ID = 0x0C34
    PCI_CLASS_CODE = 0x030000  # VGA controller

    # MMHUB aperture (matches gem5 AMDGPUDevice constructor for MI300X)
    # gem5: mmhubBase = 0x8000ULL << 24, register stores mmhubBase >> 24
    MMHUB_BASE = 0x8000  # (0x8000ULL << 24) >> 24
    MMHUB_TOP = 0x8000 + (16 * 1024 * 1024 * 1024 >> 24)  # + 16GB >> 24
    VRAM_SIZE_MB = 16 * 1024  # 16GB
    VRAM_SIZE_BYTES = 16 * 1024 * 1024 * 1024

    def __init__(self):
        # PCI config space (offset -> 32-bit value)
        self.config = {
            0x00: (self.PCI_DEVICE_ID << 16) | self.PCI_VENDOR_ID,
            0x04: 0x00100007,  # Status | Command (Mem+IO+BusMaster)
            0x08: self.PCI_CLASS_CODE,
            0x2C: (self.PCI_SUBSYS_ID << 16) | self.PCI_SUBSYS_VENDOR_ID,
        }

        # MMIO register map (offset -> 32-bit value)
        self.regs = {
            # PSP firmware status (MP0_SMN_C2PMSG_33)
            0x3B10C: 0x80000000,  # FFFFFFFF_80000000 = ready
            # MMHUB FB location
            0x60920: self.MMHUB_BASE,   # MC_VM_FB_LOCATION_BASE
            0x60924: self.MMHUB_TOP,    # MC_VM_FB_LOCATION_TOP
            0x60928: self.VRAM_SIZE_MB, # MC_VM_FB_SIZE_MB
            # GRBM status (idle)
            0xD000: 0x00000000,
            0xD004: 0x00000000,
            0xD010: 0x00000000,  # GRBM_STATUS_SE0
            # GC_VERSION (gfx942)
            0xD080: 0x00090402,
            # NBIO/BIF registers
            0x0000: 0x0000,
            0x0004: 0x0000,
            # SMU firmware version
            0x5A000: 0x00440023,  # SMU version 68.0.35
        }

        # VRAM backing store (small buffer for test)
        self.vram = bytearray(4096)

        # Doorbell state
        self.doorbells = {}

        # Interrupt state
        self.pending_irqs = []

    def config_read(self, offset, size=4):
        base = offset & ~0x3
        val = self.config.get(base, 0)
        shift = (offset & 0x3) * 8
        mask = (1 << (size * 8)) - 1
        return (val >> shift) & mask

    def config_write(self, offset, value, size=4):
        if size == 4:
            self.config[offset] = value & 0xFFFFFFFF
        else:
            base = offset & ~0x3
            cur = self.config.get(base, 0)
            shift = (offset & 0x3) * 8
            mask = (1 << (size * 8)) - 1
            cur &= ~(mask << shift)
            cur |= (value & mask) << shift
            self.config[base] = cur

    def mmio_read(self, offset, size=4):
        base = offset & ~0x3
        val = self.regs.get(base, 0)
        shift = (offset & 0x3) * 8
        mask = (1 << (size * 8)) - 1
        return (val >> shift) & mask

    def mmio_write(self, offset, value, size=4):
        if size == 4:
            self.regs[offset] = value & 0xFFFFFFFF
        else:
            base = offset & ~0x3
            cur = self.regs.get(base, 0)
            shift = (offset & 0x3) * 8
            mask = (1 << (size * 8)) - 1
            cur &= ~(mask << shift)
            cur |= (value & mask) << shift
            self.regs[base] = cur

    def frame_read(self, offset, size=4):
        if offset + size <= len(self.vram):
            val = int.from_bytes(self.vram[offset:offset+size], 'little')
            return val
        return 0

    def frame_write(self, offset, value, size=4):
        if offset + size <= len(self.vram):
            self.vram[offset:offset+size] = value.to_bytes(size, 'little')


class MockGem5Server:
    """Mock gem5 cosim server that uses MockGem5GPU for register values."""

    def __init__(self, sock_path, gpu=None):
        self.sock_path = sock_path
        self.gpu = gpu or MockGem5GPU()
        self.server_sock = None
        self.running = False
        self.thread = None
        self.mmio_conn = None
        self.event_conn = None
        self.transaction_log = []

    def start(self):
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.sock_path)
        self.server_sock.listen(2)
        self.server_sock.settimeout(5.0)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.mmio_conn:
            self.mmio_conn.close()
        if self.event_conn:
            self.event_conn.close()
        if self.server_sock:
            self.server_sock.close()
        if self.thread:
            self.thread.join(timeout=3)

    def _run(self):
        try:
            # Accept MMIO connection
            self.mmio_conn, _ = self.server_sock.accept()
            self.mmio_conn.settimeout(5.0)

            # Accept event connection
            try:
                self.event_conn, _ = self.server_sock.accept()
                self.event_conn.settimeout(5.0)
            except socket.timeout:
                pass  # Event connection is optional

            # Process messages on MMIO connection
            while self.running:
                try:
                    data = self.mmio_conn.recv(MSG_HDR_SIZE)
                    if not data:
                        break
                    if len(data) < MSG_HDR_SIZE:
                        break
                    msg = unpack_msg(data)
                    self.transaction_log.append(msg)
                    self._handle(self.mmio_conn, msg)
                except socket.timeout:
                    continue
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break
        except socket.timeout:
            pass

    def _handle(self, conn, msg):
        t = msg['type']
        if t == MSG_INIT:
            resp = pack_msg(msg_type=MSG_INIT_RESP, msg_id=msg['id'],
                           data=self.gpu.VRAM_SIZE_BYTES)
            conn.sendall(resp)

        elif t == MSG_MMIO_READ:
            val = self.gpu.mmio_read(msg['addr'], msg['access_size'] or 4)
            resp = pack_msg(msg_type=MSG_MMIO_RESP, msg_id=msg['id'],
                           addr=msg['addr'], data=val,
                           access_size=msg['access_size'] or 4)
            conn.sendall(resp)

        elif t == MSG_MMIO_WRITE:
            self.gpu.mmio_write(msg['addr'], msg['data'],
                               msg['access_size'] or 4)

        elif t == MSG_CONFIG_READ:
            val = self.gpu.config_read(msg['addr'], msg['access_size'] or 4)
            resp = pack_msg(msg_type=MSG_MMIO_RESP, msg_id=msg['id'],
                           addr=msg['addr'], data=val,
                           access_size=msg['access_size'] or 4)
            conn.sendall(resp)

        elif t == MSG_CONFIG_WRITE:
            self.gpu.config_write(msg['addr'], msg['data'],
                                 msg['access_size'] or 4)

        elif t == MSG_FRAME_READ:
            val = self.gpu.frame_read(msg['addr'], msg['access_size'] or 4)
            resp = pack_msg(msg_type=MSG_MMIO_RESP, msg_id=msg['id'],
                           addr=msg['addr'], data=val,
                           access_size=msg['access_size'] or 4)
            conn.sendall(resp)

        elif t == MSG_FRAME_WRITE:
            self.gpu.frame_write(msg['addr'], msg['data'],
                                msg['access_size'] or 4)

        elif t == MSG_DB_WRITE:
            self.gpu.doorbells[msg['addr']] = msg['data']

        elif t == MSG_DMA_REQ:
            if msg['size'] > 0:
                # DMA write: receive payload
                payload = conn.recv(msg['size'])
                off = msg['addr']
                end = min(off + len(payload), len(self.gpu.vram))
                if off < end:
                    self.gpu.vram[off:end] = payload[:end - off]
            else:
                # DMA read: send VRAM data
                length = min(msg['data'], len(self.gpu.vram) - msg['addr'])
                if length > 0:
                    chunk = bytes(self.gpu.vram[msg['addr']:
                                                msg['addr'] + length])
                else:
                    chunk = b'\x00' * msg['data']
                resp = pack_msg(msg_type=MSG_MMIO_RESP, msg_id=msg['id'],
                               addr=msg['addr'], data=len(chunk),
                               size=len(chunk))
                conn.sendall(resp)
                conn.sendall(chunk)

        elif t == MSG_SHUTDOWN:
            self.running = False

    def send_irq(self, vector):
        """Send IRQ on event connection (gem5 -> QEMU direction)."""
        if self.event_conn:
            irq = pack_msg(msg_type=MSG_IRQ_RAISE, data=vector)
            self.event_conn.sendall(irq)


class QEMUClient:
    """Simulates the QEMU side of the cosim bridge.

    This is what the amdgpu driver "sees" through QEMU's mi300x-gem5 device.
    Methods return string representations for SSH-style output comparison.
    """

    def __init__(self, sock_path):
        self.sock_path = sock_path
        self.mmio_sock = None
        self.event_sock = None
        self._msg_id = 0

    def _next_id(self):
        self._msg_id += 1
        return self._msg_id

    def connect(self):
        """Establish MMIO and event connections (like QEMU device init)."""
        self.mmio_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.mmio_sock.connect(self.sock_path)
        self.mmio_sock.settimeout(5.0)

        self.event_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.event_sock.connect(self.sock_path)
        self.event_sock.settimeout(5.0)

    def close(self):
        if self.mmio_sock:
            try:
                self.mmio_sock.sendall(
                    pack_msg(msg_type=MSG_SHUTDOWN, msg_id=self._next_id()))
            except Exception:
                pass
            self.mmio_sock.close()
        if self.event_sock:
            self.event_sock.close()

    def init_handshake(self, vram_size=16 * 1024**3):
        """Perform INIT handshake, returns (success, gem5_vram_size)."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_INIT, msg_id=mid, data=vram_size))
        resp = unpack_msg(self.mmio_sock.recv(MSG_HDR_SIZE))
        return resp['type'] == MSG_INIT_RESP, resp['data']

    def pci_config_read(self, offset, size=4):
        """Read PCI config space, returns integer value."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_CONFIG_READ, msg_id=mid,
            addr=offset, access_size=size))
        resp = unpack_msg(self.mmio_sock.recv(MSG_HDR_SIZE))
        return resp['data']

    def pci_config_write(self, offset, value, size=4):
        """Write PCI config space (fire-and-forget)."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_CONFIG_WRITE, msg_id=mid,
            addr=offset, data=value, access_size=size))

    def mmio_read(self, offset, size=4):
        """Read MMIO register, returns integer value."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_MMIO_READ, msg_id=mid,
            addr=offset, access_size=size))
        resp = unpack_msg(self.mmio_sock.recv(MSG_HDR_SIZE))
        return resp['data']

    def mmio_write(self, offset, value, size=4):
        """Write MMIO register (fire-and-forget)."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_MMIO_WRITE, msg_id=mid,
            addr=offset, data=value, access_size=size))

    def doorbell_write(self, offset, value, size=4):
        """Write doorbell (fire-and-forget)."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_DB_WRITE, msg_id=mid,
            addr=offset, data=value, access_size=size))

    def frame_read(self, offset, size=4):
        """Read framebuffer/VRAM, returns integer value."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_FRAME_READ, msg_id=mid,
            addr=offset, access_size=size))
        resp = unpack_msg(self.mmio_sock.recv(MSG_HDR_SIZE))
        return resp['data']

    def frame_write(self, offset, value, size=4):
        """Write framebuffer/VRAM (fire-and-forget)."""
        mid = self._next_id()
        self.mmio_sock.sendall(pack_msg(
            msg_type=MSG_FRAME_WRITE, msg_id=mid,
            addr=offset, data=value, access_size=size))

    def recv_irq(self, timeout=2.0):
        """Wait for IRQ on event connection, returns vector or None."""
        self.event_sock.settimeout(timeout)
        try:
            data = self.event_sock.recv(MSG_HDR_SIZE)
            if len(data) == MSG_HDR_SIZE:
                msg = unpack_msg(data)
                if msg['type'] == MSG_IRQ_RAISE:
                    return msg['data']
        except socket.timeout:
            pass
        return None

    # --- SSH-style string output helpers ---
    # These mimic what `cat /sys/class/drm/card0/device/...` would return
    # in a real guest, enabling QEMU functional test-style comparison.

    def ssh_lspci_device_id(self):
        """Simulate: lspci -nn | grep -i amd | awk '{print $NF}'"""
        vendor = self.pci_config_read(0x00) & 0xFFFF
        device = (self.pci_config_read(0x00) >> 16) & 0xFFFF
        return f"[{vendor:04x}:{device:04x}]"

    def ssh_cat_gpu_version(self):
        """Simulate: cat /sys/class/drm/card0/device/gpu_version"""
        gc_ver = self.mmio_read(0xD080)
        major = (gc_ver >> 16) & 0xFF
        minor = (gc_ver >> 8) & 0xFF
        stepping = gc_ver & 0xFF
        return f"gfx{major}{minor:01x}{stepping:01x}"

    def ssh_cat_vram_size(self):
        """Simulate: cat /sys/class/drm/card0/device/mem_info_vram_total"""
        size_mb = self.mmio_read(0x60928)
        size_bytes = size_mb * 1024 * 1024
        return str(size_bytes)

    def ssh_cat_fw_version(self):
        """Simulate: cat /sys/kernel/debug/dri/0/amdgpu_firmware_info"""
        psp_status = self.mmio_read(0x3B10C)
        return "PSP: 0x{:08x}".format(psp_status)

    def ssh_cat_gpu_busy(self):
        """Simulate: cat /sys/class/drm/card0/device/gpu_busy_percent"""
        grbm = self.mmio_read(0xD000)
        # GRBM_STATUS bits: if all zero, GPU is idle
        if grbm == 0:
            return "0"
        return "100"


# ======================================================================
# Functional Tests (QEMU-style: SSH command output string comparison)
# ======================================================================

class TestFunctionalDriverProbe(unittest.TestCase):
    """Functional test: simulate amdgpu driver probe and verify outputs.

    These tests follow the QEMU functional test pattern:
    1. Start a mock gem5 server
    2. Connect a QEMU client (simulating the mi300x-gem5 PCIe device)
    3. Run "commands" that simulate what SSH commands in the guest would do
    4. Compare output strings against expected values
    """

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock',
                                          prefix='cosim_func_')
        self.gpu = MockGem5GPU()
        self.server = MockGem5Server(self.sock_path, self.gpu)
        self.server.start()
        time.sleep(0.1)  # Wait for server to be ready

        self.client = QEMUClient(self.sock_path)
        self.client.connect()
        time.sleep(0.1)  # Wait for both connections

    def tearDown(self):
        self.client.close()
        time.sleep(0.1)
        self.server.stop()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_init_handshake(self):
        """QEMU functional: INIT handshake returns correct VRAM size."""
        success, vram = self.client.init_handshake()
        self.assertTrue(success)
        self.assertEqual(vram, 16 * 1024 * 1024 * 1024)

    def test_lspci_shows_mi300x(self):
        """QEMU functional: lspci shows AMD MI300X device ID.

        Expected SSH output: [1002:74a1]
        """
        output = self.client.ssh_lspci_device_id()
        self.assertEqual(output, "[1002:74a1]")

    def test_gpu_version_gfx942(self):
        """QEMU functional: GPU version register reports gfx942.

        Expected SSH output: gfx942
        """
        output = self.client.ssh_cat_gpu_version()
        self.assertEqual(output, "gfx942")

    def test_vram_total_16gb(self):
        """QEMU functional: VRAM total size is 16GB.

        Expected SSH output: 17179869184
        """
        output = self.client.ssh_cat_vram_size()
        self.assertEqual(output, str(16 * 1024 * 1024 * 1024))

    def test_psp_firmware_ready(self):
        """QEMU functional: PSP firmware status shows ready.

        Expected SSH output: PSP: 0x80000000
        """
        output = self.client.ssh_cat_fw_version()
        self.assertEqual(output, "PSP: 0x80000000")

    def test_gpu_idle(self):
        """QEMU functional: GPU reports 0% busy when idle.

        Expected SSH output: 0
        """
        output = self.client.ssh_cat_gpu_busy()
        self.assertEqual(output, "0")

    def test_pci_subsystem_ids(self):
        """QEMU functional: PCI subsystem IDs match MI300X.

        Expected:
          Subsystem vendor: 0x1002 (AMD)
          Subsystem device: 0x0C34
        """
        val = self.client.pci_config_read(0x2C)
        subsys_vendor = val & 0xFFFF
        subsys_device = (val >> 16) & 0xFFFF
        self.assertEqual(f"{subsys_vendor:04x}", "1002")
        self.assertEqual(f"{subsys_device:04x}", "0c34")


class TestFunctionalVRAMAccess(unittest.TestCase):
    """Functional test: VRAM read/write through framebuffer BAR."""

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock',
                                          prefix='cosim_vram_')
        self.gpu = MockGem5GPU()
        self.server = MockGem5Server(self.sock_path, self.gpu)
        self.server.start()
        time.sleep(0.1)

        self.client = QEMUClient(self.sock_path)
        self.client.connect()
        time.sleep(0.1)

    def tearDown(self):
        self.client.close()
        time.sleep(0.1)
        self.server.stop()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_vram_write_read_roundtrip(self):
        """Write a pattern to VRAM and read it back.

        Simulates: echo pattern > /dev/dri/card0 then reading back.
        """
        self.client.frame_write(0x100, 0xDEADBEEF)
        val = self.client.frame_read(0x100)
        self.assertEqual(val, 0xDEADBEEF)

    def test_vram_multiple_locations(self):
        """Write different values to multiple VRAM locations and verify."""
        offsets_values = [
            (0x000, 0x11111111),
            (0x100, 0x22222222),
            (0x200, 0x33333333),
            (0x300, 0x44444444),
        ]

        for off, val in offsets_values:
            self.client.frame_write(off, val)

        for off, expected in offsets_values:
            val = self.client.frame_read(off)
            self.assertEqual(val, expected,
                           f"VRAM[0x{off:x}] = 0x{val:x}, expected 0x{expected:x}")

    def test_vram_zero_on_init(self):
        """Fresh VRAM should read as zeros.

        Simulates: memset check after GPU initialization.
        """
        val = self.client.frame_read(0x500)
        self.assertEqual(val, 0)


class TestFunctionalMMIORegisterAccess(unittest.TestCase):
    """Functional test: MMIO register read/write patterns."""

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock',
                                          prefix='cosim_mmio_')
        self.gpu = MockGem5GPU()
        self.server = MockGem5Server(self.sock_path, self.gpu)
        self.server.start()
        time.sleep(0.1)

        self.client = QEMUClient(self.sock_path)
        self.client.connect()
        time.sleep(0.1)

    def tearDown(self):
        self.client.close()
        time.sleep(0.1)
        self.server.stop()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_mmhub_fb_location(self):
        """Read MMHUB framebuffer location registers.

        These registers tell the driver where VRAM is mapped.
        """
        base = self.client.mmio_read(0x60920)
        top = self.client.mmio_read(0x60924)
        self.assertGreater(top, base)

    def test_grbm_status_idle(self):
        """GRBM_STATUS should report idle on fresh GPU.

        Expected: all bits zero (no engines busy).
        """
        status = self.client.mmio_read(0xD000)
        self.assertEqual(status, 0)

    def test_mmio_write_read_custom_register(self):
        """Write a value to an MMIO register and read it back.

        Simulates driver configuring a GPU register.
        """
        self.client.mmio_write(0x1234, 0xABCD0000)
        val = self.client.mmio_read(0x1234)
        self.assertEqual(val, 0xABCD0000)

    def test_smu_firmware_version(self):
        """Read SMU firmware version register.

        Expected: non-zero value indicating SMU firmware is loaded.
        """
        smu_ver = self.client.mmio_read(0x5A000)
        self.assertNotEqual(smu_ver, 0)

    def test_gc_version_register(self):
        """Read GC (Graphics Core) version register.

        Expected: 0x00090402 for gfx942 (MI300X).
        """
        gc_ver = self.client.mmio_read(0xD080)
        self.assertEqual(gc_ver, 0x00090402)


class TestFunctionalInterrupt(unittest.TestCase):
    """Functional test: interrupt delivery from gem5 to QEMU."""

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock',
                                          prefix='cosim_irq_')
        self.gpu = MockGem5GPU()
        self.server = MockGem5Server(self.sock_path, self.gpu)
        self.server.start()
        time.sleep(0.1)

        self.client = QEMUClient(self.sock_path)
        self.client.connect()
        time.sleep(0.2)  # Extra time for event connection

    def tearDown(self):
        self.client.close()
        time.sleep(0.1)
        self.server.stop()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_irq_delivery(self):
        """gem5 sends IRQ, QEMU client receives it on event socket.

        Simulates: GPU completes a kernel and raises completion interrupt.
        """
        # gem5 sends IRQ on event connection
        self.server.send_irq(vector=3)

        # QEMU receives IRQ
        vector = self.client.recv_irq(timeout=3.0)
        self.assertIsNotNone(vector)
        self.assertEqual(vector & 0xFFFF, 3)

    def test_multiple_irqs(self):
        """Multiple IRQs delivered in order."""
        for i in range(5):
            self.server.send_irq(vector=i + 10)
            time.sleep(0.05)

        received = []
        for _ in range(5):
            v = self.client.recv_irq(timeout=2.0)
            if v is not None:
                received.append(v & 0xFFFF)

        self.assertEqual(received, [10, 11, 12, 13, 14])


class TestFunctionalDMATransfer(unittest.TestCase):
    """Functional test: DMA transfers between host and GPU VRAM."""

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock',
                                          prefix='cosim_dma_')
        self.gpu = MockGem5GPU()
        self.server = MockGem5Server(self.sock_path, self.gpu)
        self.server.start()
        time.sleep(0.1)

        self.client = QEMUClient(self.sock_path)
        self.client.connect()
        time.sleep(0.1)

    def tearDown(self):
        self.client.close()
        time.sleep(0.1)
        self.server.stop()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_dma_write_to_vram(self):
        """DMA write: transfer data from host to GPU VRAM.

        Simulates SDMA copying a buffer from system memory to VRAM.
        """
        payload = bytes(range(64))
        mid = self.client._next_id()
        self.client.mmio_sock.sendall(pack_msg(
            msg_type=MSG_DMA_REQ, msg_id=mid,
            addr=0x0, data=len(payload), size=len(payload)))
        self.client.mmio_sock.sendall(payload)

        time.sleep(0.1)

        # Verify data landed in VRAM
        for i in range(0, 64, 4):
            val = self.client.frame_read(i)
            expected = int.from_bytes(payload[i:i+4], 'little')
            self.assertEqual(val, expected,
                           f"VRAM[0x{i:x}] mismatch after DMA write")

    def test_dma_read_from_vram(self):
        """DMA read: transfer data from GPU VRAM to host.

        Simulates SDMA copying a buffer from VRAM to system memory.
        """
        # Pre-fill VRAM
        self.client.frame_write(0x0, 0xAABBCCDD)
        self.client.frame_write(0x4, 0x11223344)

        # DMA read request
        mid = self.client._next_id()
        self.client.mmio_sock.sendall(pack_msg(
            msg_type=MSG_DMA_REQ, msg_id=mid,
            addr=0x0, data=8, size=0))  # size=0 means read

        # Receive response header + payload
        resp_data = self.client.mmio_sock.recv(MSG_HDR_SIZE)
        resp = unpack_msg(resp_data)
        self.assertEqual(resp['type'], MSG_MMIO_RESP)

        payload_len = resp['size']
        self.assertEqual(payload_len, 8)

        payload = self.client.mmio_sock.recv(payload_len)
        val0 = int.from_bytes(payload[0:4], 'little')
        val1 = int.from_bytes(payload[4:8], 'little')
        self.assertEqual(val0, 0xAABBCCDD)
        self.assertEqual(val1, 0x11223344)


class TestFunctionalFullDriverInit(unittest.TestCase):
    """End-to-end functional test: complete amdgpu driver initialization.

    This simulates the full sequence of what the Linux amdgpu driver
    does when probing an MI300X through QEMU's mi300x-gem5 device:

    1. PCI config space enumeration
    2. INIT handshake
    3. PSP firmware status check
    4. MMHUB configuration
    5. GRBM status check
    6. GC version detection
    7. Doorbell initialization

    Each step verifies the SSH-style output string matches expected values.
    """

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock',
                                          prefix='cosim_full_')
        self.gpu = MockGem5GPU()
        self.server = MockGem5Server(self.sock_path, self.gpu)
        self.server.start()
        time.sleep(0.1)

        self.client = QEMUClient(self.sock_path)
        self.client.connect()
        time.sleep(0.1)

    def tearDown(self):
        self.client.close()
        time.sleep(0.1)
        self.server.stop()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def _ssh_run(self, command):
        """Simulate running a command via SSH and returning stdout.

        Maps common diagnostic commands to cosim protocol operations.
        """
        if command == "lspci -nn | grep -i display":
            return self.client.ssh_lspci_device_id()
        elif command == "cat /sys/class/drm/card0/device/gpu_version":
            return self.client.ssh_cat_gpu_version()
        elif command == "cat /sys/class/drm/card0/device/mem_info_vram_total":
            return self.client.ssh_cat_vram_size()
        elif command == "cat /sys/kernel/debug/dri/0/amdgpu_firmware_info":
            return self.client.ssh_cat_fw_version()
        elif command == "cat /sys/class/drm/card0/device/gpu_busy_percent":
            return self.client.ssh_cat_gpu_busy()
        else:
            return "UNKNOWN COMMAND"

    def test_full_driver_init_and_verify(self):
        """Complete driver init with SSH output verification.

        This is the main QEMU-style functional test that walks through
        the entire driver initialization and verifies each step's output.
        """
        results = []

        # Step 1: INIT handshake
        success, vram = self.client.init_handshake()
        self.assertTrue(success, "INIT handshake failed")
        results.append(("INIT", "OK" if success else "FAIL"))

        # Step 2: PCI enumeration
        output = self._ssh_run("lspci -nn | grep -i display")
        self.assertEqual(output, "[1002:74a1]")
        results.append(("lspci", output))

        # Step 3: Enable PCI BusMaster
        self.client.pci_config_write(0x04, 0x00100007)
        cmd_reg = self.client.pci_config_read(0x04)
        bus_master = bool(cmd_reg & 0x04)
        self.assertTrue(bus_master, "BusMaster not enabled")
        results.append(("PCI BusMaster", "enabled" if bus_master else "disabled"))

        # Step 4: GPU version
        output = self._ssh_run("cat /sys/class/drm/card0/device/gpu_version")
        self.assertEqual(output, "gfx942")
        results.append(("gpu_version", output))

        # Step 5: VRAM size
        output = self._ssh_run(
            "cat /sys/class/drm/card0/device/mem_info_vram_total")
        self.assertEqual(output, "17179869184")
        results.append(("vram_total", output))

        # Step 6: PSP firmware
        output = self._ssh_run(
            "cat /sys/kernel/debug/dri/0/amdgpu_firmware_info")
        self.assertEqual(output, "PSP: 0x80000000")
        results.append(("PSP", output))

        # Step 7: GPU busy
        output = self._ssh_run(
            "cat /sys/class/drm/card0/device/gpu_busy_percent")
        self.assertEqual(output, "0")
        results.append(("gpu_busy", output))

        # Step 8: Doorbell init (write doorbell page)
        self.client.doorbell_write(0x0, 0x0)
        results.append(("doorbell_init", "OK"))

        # Print summary (QEMU functional test style)
        print("\n  Driver init verification results:")
        for name, value in results:
            print(f"    {name:20s}: {value}")


if __name__ == '__main__':
    print("=" * 70)
    print("MI300X gem5-QEMU Co-simulation Functional Tests")
    print("(QEMU-style: SSH command output string comparison)")
    print("=" * 70)
    print()

    unittest.main(verbosity=2)
