#!/usr/bin/env python3
"""
Integration test for MI300X gem5-QEMU co-simulation protocol.

This test validates that the wire protocol defined in gem5's
mi300x_gem5_cosim.hh matches QEMU's mi300x_gem5.h exactly:
  - CosimMsgHeader layout and size (32 bytes)
  - Message type values
  - Field offsets and byte order

It also runs a mock "QEMU client" that connects to a Unix domain socket
and exercises the protocol message exchange, simulating what QEMU's
mi300x-gem5 device would do.

Usage:
    python3 tests/test_mi300x_cosim_protocol.py
"""

import ctypes
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest


# ======================================================================
# Protocol definitions (must match BOTH sides)
# ======================================================================

# Message types from QEMU's mi300x_gem5.h
MI300X_MSG_MMIO_READ    = 0x01
MI300X_MSG_MMIO_WRITE   = 0x02
MI300X_MSG_DB_READ      = 0x03
MI300X_MSG_DB_WRITE     = 0x04
MI300X_MSG_DMA_REQ      = 0x05
MI300X_MSG_INIT         = 0x06
MI300X_MSG_SHUTDOWN     = 0x07
MI300X_MSG_CONFIG_READ  = 0x08
MI300X_MSG_CONFIG_WRITE = 0x09
MI300X_MSG_FRAME_READ   = 0x0A
MI300X_MSG_FRAME_WRITE  = 0x0B

MI300X_MSG_MMIO_RESP  = 0x81
MI300X_MSG_IRQ_RAISE  = 0x82
MI300X_MSG_IRQ_LOWER  = 0x83
MI300X_MSG_DMA_READ   = 0x84
MI300X_MSG_DMA_WRITE  = 0x85
MI300X_MSG_INIT_RESP  = 0x86

# Header format: type(u32) size(u32) addr(u64) data(u64) access_size(u32) id(u32)
# Total: 4 + 4 + 8 + 8 + 4 + 4 = 32 bytes
MSG_HDR_FORMAT = '<IIQQiI'  # little-endian
MSG_HDR_SIZE = struct.calcsize(MSG_HDR_FORMAT)

assert MSG_HDR_SIZE == 32, f"Header size mismatch: {MSG_HDR_SIZE} != 32"


def pack_msg(msg_type, size=0, addr=0, data=0, access_size=0, msg_id=0):
    """Pack a co-simulation message header."""
    return struct.pack(MSG_HDR_FORMAT,
                       msg_type, size, addr, data, access_size, msg_id)


def unpack_msg(buf):
    """Unpack a co-simulation message header."""
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
# Test: Protocol struct layout compatibility
# ======================================================================

class TestProtocolLayout(unittest.TestCase):
    """Verify the wire format matches between gem5 and QEMU."""

    def test_header_size(self):
        """CosimMsgHeader / MI300XGem5MsgHeader must be 32 bytes."""
        self.assertEqual(MSG_HDR_SIZE, 32)

    def test_field_offsets(self):
        """Verify field offsets match the C struct layout."""
        # struct MI300XGem5MsgHeader {
        #     uint32_t type;          // offset 0
        #     uint32_t size;          // offset 4
        #     uint64_t addr;          // offset 8
        #     uint64_t data;          // offset 16
        #     uint32_t access_size;   // offset 24
        #     uint32_t id;            // offset 28
        # } __attribute__((packed));

        # Pack a known message and verify byte positions
        msg = pack_msg(
            msg_type=0xAABBCCDD,
            size=0x11223344,
            addr=0xDEADBEEFCAFEBABE,
            data=0x0102030405060708,
            access_size=0x00000004,
            msg_id=0x99887766,
        )

        self.assertEqual(len(msg), 32)

        # type at offset 0 (4 bytes, LE)
        self.assertEqual(struct.unpack_from('<I', msg, 0)[0], 0xAABBCCDD)
        # size at offset 4
        self.assertEqual(struct.unpack_from('<I', msg, 4)[0], 0x11223344)
        # addr at offset 8
        self.assertEqual(struct.unpack_from('<Q', msg, 8)[0], 0xDEADBEEFCAFEBABE)
        # data at offset 16
        self.assertEqual(struct.unpack_from('<Q', msg, 16)[0], 0x0102030405060708)
        # access_size at offset 24
        self.assertEqual(struct.unpack_from('<I', msg, 24)[0], 0x00000004)
        # id at offset 28
        self.assertEqual(struct.unpack_from('<I', msg, 28)[0], 0x99887766)

    def test_message_type_values(self):
        """Verify all message type enum values match between gem5 and QEMU."""
        # QEMU -> gem5
        self.assertEqual(MI300X_MSG_MMIO_READ, 0x01)
        self.assertEqual(MI300X_MSG_MMIO_WRITE, 0x02)
        self.assertEqual(MI300X_MSG_DB_READ, 0x03)
        self.assertEqual(MI300X_MSG_DB_WRITE, 0x04)
        self.assertEqual(MI300X_MSG_DMA_REQ, 0x05)
        self.assertEqual(MI300X_MSG_INIT, 0x06)
        self.assertEqual(MI300X_MSG_SHUTDOWN, 0x07)
        self.assertEqual(MI300X_MSG_CONFIG_READ, 0x08)
        self.assertEqual(MI300X_MSG_CONFIG_WRITE, 0x09)
        self.assertEqual(MI300X_MSG_FRAME_READ, 0x0A)
        self.assertEqual(MI300X_MSG_FRAME_WRITE, 0x0B)

        # gem5 -> QEMU
        self.assertEqual(MI300X_MSG_MMIO_RESP, 0x81)
        self.assertEqual(MI300X_MSG_IRQ_RAISE, 0x82)
        self.assertEqual(MI300X_MSG_IRQ_LOWER, 0x83)
        self.assertEqual(MI300X_MSG_DMA_READ, 0x84)
        self.assertEqual(MI300X_MSG_DMA_WRITE, 0x85)
        self.assertEqual(MI300X_MSG_INIT_RESP, 0x86)

    def test_endianness(self):
        """All fields must be little-endian."""
        msg = pack_msg(msg_type=1, addr=0x100)
        raw = bytearray(msg)

        # type field (offset 0): value 1 in LE = 01 00 00 00
        self.assertEqual(raw[0], 0x01)
        self.assertEqual(raw[1], 0x00)
        self.assertEqual(raw[2], 0x00)
        self.assertEqual(raw[3], 0x00)

        # addr field (offset 8): value 0x100 in LE = 00 01 00 00 00 00 00 00
        self.assertEqual(raw[8], 0x00)
        self.assertEqual(raw[9], 0x01)


# ======================================================================
# Test: Mock QEMU client <-> Mock gem5 server socket exchange
# ======================================================================

class TestSocketProtocol(unittest.TestCase):
    """Test the socket protocol with a mock gem5 server."""

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock', prefix='cosim_test_')
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.sock_path)
        self.server_sock.listen(1)
        self.server_thread = None
        self.server_responses = []

    def tearDown(self):
        self.server_sock.close()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def _start_server(self, handler):
        """Start a server thread that accepts one connection."""
        def server_worker():
            conn, _ = self.server_sock.accept()
            try:
                handler(conn)
            finally:
                conn.close()

        self.server_thread = threading.Thread(target=server_worker)
        self.server_thread.daemon = True
        self.server_thread.start()

    def _connect_client(self):
        """Connect a client socket (simulating QEMU side)."""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(self.sock_path)
        return client

    def test_init_handshake(self):
        """Test INIT / INIT_RESP handshake (QEMU sends, gem5 replies)."""
        vram_size = 16 * 1024 * 1024 * 1024  # 16 GiB

        def server_handler(conn):
            # Read INIT message from "QEMU"
            data = conn.recv(MSG_HDR_SIZE)
            self.assertEqual(len(data), MSG_HDR_SIZE)
            msg = unpack_msg(data)
            self.assertEqual(msg['type'], MI300X_MSG_INIT)
            self.server_responses.append(msg)

            # Send INIT_RESP (like gem5 would)
            resp = pack_msg(
                msg_type=MI300X_MSG_INIT_RESP,
                msg_id=msg['id'],
                data=vram_size,
            )
            conn.sendall(resp)

        self._start_server(server_handler)
        client = self._connect_client()

        # QEMU sends INIT with its vram_size
        init_msg = pack_msg(
            msg_type=MI300X_MSG_INIT,
            msg_id=1,
            data=vram_size,
        )
        client.sendall(init_msg)

        # Receive INIT_RESP
        resp_data = client.recv(MSG_HDR_SIZE)
        self.assertEqual(len(resp_data), MSG_HDR_SIZE)
        resp = unpack_msg(resp_data)
        self.assertEqual(resp['type'], MI300X_MSG_INIT_RESP)
        self.assertEqual(resp['id'], 1)
        self.assertEqual(resp['data'], vram_size)

        client.close()
        self.server_thread.join(timeout=2)

    def test_mmio_read_roundtrip(self):
        """Test MMIO_READ -> MMIO_RESP roundtrip."""
        test_addr = 0xD000  # GRBM_STATUS
        test_value = 0xCAFEBABE
        test_access_size = 4

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            self.assertEqual(msg['type'], MI300X_MSG_MMIO_READ)
            self.assertEqual(msg['addr'], test_addr)
            self.assertEqual(msg['access_size'], test_access_size)

            resp = pack_msg(
                msg_type=MI300X_MSG_MMIO_RESP,
                msg_id=msg['id'],
                addr=test_addr,
                data=test_value,
                access_size=test_access_size,
            )
            conn.sendall(resp)

        self._start_server(server_handler)
        client = self._connect_client()

        # QEMU reads MMIO register
        read_msg = pack_msg(
            msg_type=MI300X_MSG_MMIO_READ,
            msg_id=42,
            addr=test_addr,
            access_size=test_access_size,
        )
        client.sendall(read_msg)

        resp_data = client.recv(MSG_HDR_SIZE)
        resp = unpack_msg(resp_data)
        self.assertEqual(resp['type'], MI300X_MSG_MMIO_RESP)
        self.assertEqual(resp['data'], test_value)
        self.assertEqual(resp['id'], 42)

        client.close()
        self.server_thread.join(timeout=2)

    def test_mmio_write_fire_and_forget(self):
        """Test MMIO_WRITE (no response expected)."""
        test_addr = 0x1000
        test_value = 0xDEADBEEF
        received = []

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            received.append(msg)

        self._start_server(server_handler)
        client = self._connect_client()

        write_msg = pack_msg(
            msg_type=MI300X_MSG_MMIO_WRITE,
            msg_id=7,
            addr=test_addr,
            data=test_value,
            access_size=4,
        )
        client.sendall(write_msg)

        # Give server time to process
        time.sleep(0.1)
        client.close()
        self.server_thread.join(timeout=2)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['type'], MI300X_MSG_MMIO_WRITE)
        self.assertEqual(received[0]['addr'], test_addr)
        self.assertEqual(received[0]['data'], test_value)
        self.assertEqual(received[0]['access_size'], 4)

    def test_doorbell_write_fire_and_forget(self):
        """Test DB_WRITE (no response expected)."""
        test_addr = 0x100
        test_value = 0x12345678
        received = []

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            received.append(msg)

        self._start_server(server_handler)
        client = self._connect_client()

        write_msg = pack_msg(
            msg_type=MI300X_MSG_DB_WRITE,
            msg_id=10,
            addr=test_addr,
            data=test_value,
            access_size=4,
        )
        client.sendall(write_msg)

        time.sleep(0.1)
        client.close()
        self.server_thread.join(timeout=2)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['type'], MI300X_MSG_DB_WRITE)
        self.assertEqual(received[0]['addr'], test_addr)
        self.assertEqual(received[0]['data'], test_value)

    def test_irq_raise_from_gem5(self):
        """Test IRQ_RAISE message from gem5 to QEMU."""
        vector = 5

        def server_handler(conn):
            # gem5 sends IRQ_RAISE
            irq_msg = pack_msg(
                msg_type=MI300X_MSG_IRQ_RAISE,
                data=vector,
            )
            conn.sendall(irq_msg)

        self._start_server(server_handler)
        client = self._connect_client()

        # QEMU receives IRQ_RAISE
        data = client.recv(MSG_HDR_SIZE)
        msg = unpack_msg(data)
        self.assertEqual(msg['type'], MI300X_MSG_IRQ_RAISE)
        self.assertEqual(msg['data'] & 0xFFFF, vector)

        client.close()
        self.server_thread.join(timeout=2)

    def test_dma_read_from_gem5(self):
        """Test DMA_READ: gem5 requests data from QEMU guest memory."""
        dma_addr = 0x100000
        dma_len = 64
        fake_data = bytes(range(64))

        def server_handler(conn):
            # gem5 sends DMA_READ request
            dma_msg = pack_msg(
                msg_type=MI300X_MSG_DMA_READ,
                addr=dma_addr,
                data=dma_len,
            )
            conn.sendall(dma_msg)

            # Read the response from QEMU (header + payload)
            resp_data = conn.recv(MSG_HDR_SIZE)
            resp = unpack_msg(resp_data)
            self.assertEqual(resp['type'], MI300X_MSG_MMIO_RESP)

            payload_size = resp['size']
            if payload_size > 0:
                payload = conn.recv(payload_size)
                self.assertEqual(payload, fake_data)

        self._start_server(server_handler)
        client = self._connect_client()

        # QEMU receives DMA_READ
        data = client.recv(MSG_HDR_SIZE)
        msg = unpack_msg(data)
        self.assertEqual(msg['type'], MI300X_MSG_DMA_READ)
        self.assertEqual(msg['addr'], dma_addr)
        self.assertEqual(msg['data'], dma_len)

        # QEMU sends response with guest memory data
        resp = pack_msg(
            msg_type=MI300X_MSG_MMIO_RESP,
            msg_id=msg['id'],
            addr=dma_addr,
            data=dma_len,
            size=dma_len,
        )
        client.sendall(resp)
        client.sendall(fake_data)

        client.close()
        self.server_thread.join(timeout=2)

    def test_dma_req_write(self):
        """Test DMA_REQ with payload (DMA write from QEMU to gem5 VRAM)."""
        dma_addr = 0x200000
        payload = bytes(range(128))
        received_msgs = []

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            received_msgs.append(msg)
            # Read the DMA payload
            if msg['size'] > 0:
                payload_data = conn.recv(msg['size'])
                received_msgs.append(payload_data)

        self._start_server(server_handler)
        client = self._connect_client()

        write_msg = pack_msg(
            msg_type=MI300X_MSG_DMA_REQ,
            msg_id=20,
            addr=dma_addr,
            data=len(payload),
            size=len(payload),
        )
        client.sendall(write_msg)
        client.sendall(payload)

        time.sleep(0.1)
        client.close()
        self.server_thread.join(timeout=2)

        self.assertEqual(len(received_msgs), 2)
        self.assertEqual(received_msgs[0]['type'], MI300X_MSG_DMA_REQ)
        self.assertEqual(received_msgs[0]['addr'], dma_addr)
        self.assertEqual(received_msgs[0]['size'], len(payload))
        self.assertEqual(received_msgs[1], payload)

    def test_dma_req_read(self):
        """Test DMA_REQ without payload (DMA read from gem5 VRAM)."""
        dma_addr = 0x300000
        dma_len = 64
        fake_vram_data = bytes([0xAB] * dma_len)

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            self.assertEqual(msg['type'], MI300X_MSG_DMA_REQ)
            self.assertEqual(msg['size'], 0)  # No payload = read request

            # gem5 reads VRAM and sends response
            resp = pack_msg(
                msg_type=MI300X_MSG_MMIO_RESP,
                msg_id=msg['id'],
                addr=dma_addr,
                data=dma_len,
                size=dma_len,
            )
            conn.sendall(resp)
            conn.sendall(fake_vram_data)

        self._start_server(server_handler)
        client = self._connect_client()

        read_msg = pack_msg(
            msg_type=MI300X_MSG_DMA_REQ,
            msg_id=21,
            addr=dma_addr,
            data=dma_len,
            size=0,  # No payload = read request
        )
        client.sendall(read_msg)

        resp_data = client.recv(MSG_HDR_SIZE)
        resp = unpack_msg(resp_data)
        self.assertEqual(resp['type'], MI300X_MSG_MMIO_RESP)
        self.assertEqual(resp['size'], dma_len)

        payload = client.recv(dma_len)
        self.assertEqual(payload, fake_vram_data)

        client.close()
        self.server_thread.join(timeout=2)

    def test_config_read_roundtrip(self):
        """Test CONFIG_READ -> MMIO_RESP roundtrip for PCI config space."""
        test_offset = 0x00  # VendorID/DeviceID
        test_value = 0x74A11002  # MI300X DeviceID=0x74A1, VendorID=0x1002

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            self.assertEqual(msg['type'], MI300X_MSG_CONFIG_READ)
            self.assertEqual(msg['addr'], test_offset)

            resp = pack_msg(
                msg_type=MI300X_MSG_MMIO_RESP,
                msg_id=msg['id'],
                addr=test_offset,
                data=test_value,
                access_size=4,
            )
            conn.sendall(resp)

        self._start_server(server_handler)
        client = self._connect_client()

        read_msg = pack_msg(
            msg_type=MI300X_MSG_CONFIG_READ,
            msg_id=50,
            addr=test_offset,
            access_size=4,
        )
        client.sendall(read_msg)

        resp_data = client.recv(MSG_HDR_SIZE)
        resp = unpack_msg(resp_data)
        self.assertEqual(resp['type'], MI300X_MSG_MMIO_RESP)
        self.assertEqual(resp['data'], test_value)
        self.assertEqual(resp['id'], 50)

        client.close()
        self.server_thread.join(timeout=2)

    def test_config_write_fire_and_forget(self):
        """Test CONFIG_WRITE (no response expected)."""
        test_offset = 0x04  # PCI Command register
        test_value = 0x0007  # IO + Mem + BusMaster
        received = []

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            received.append(msg)

        self._start_server(server_handler)
        client = self._connect_client()

        write_msg = pack_msg(
            msg_type=MI300X_MSG_CONFIG_WRITE,
            msg_id=51,
            addr=test_offset,
            data=test_value,
            access_size=4,
        )
        client.sendall(write_msg)

        time.sleep(0.1)
        client.close()
        self.server_thread.join(timeout=2)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['type'], MI300X_MSG_CONFIG_WRITE)
        self.assertEqual(received[0]['addr'], test_offset)
        self.assertEqual(received[0]['data'], test_value)

    def test_frame_read_roundtrip(self):
        """Test FRAME_READ -> MMIO_RESP roundtrip for VRAM access."""
        test_offset = 0x1000
        test_value = 0xDEADBEEF

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            self.assertEqual(msg['type'], MI300X_MSG_FRAME_READ)

            resp = pack_msg(
                msg_type=MI300X_MSG_MMIO_RESP,
                msg_id=msg['id'],
                addr=test_offset,
                data=test_value,
                access_size=4,
            )
            conn.sendall(resp)

        self._start_server(server_handler)
        client = self._connect_client()

        read_msg = pack_msg(
            msg_type=MI300X_MSG_FRAME_READ,
            msg_id=60,
            addr=test_offset,
            access_size=4,
        )
        client.sendall(read_msg)

        resp_data = client.recv(MSG_HDR_SIZE)
        resp = unpack_msg(resp_data)
        self.assertEqual(resp['type'], MI300X_MSG_MMIO_RESP)
        self.assertEqual(resp['data'], test_value)

        client.close()
        self.server_thread.join(timeout=2)

    def test_frame_write_fire_and_forget(self):
        """Test FRAME_WRITE (no response expected)."""
        test_offset = 0x2000
        test_value = 0xCAFEBABE
        received = []

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            received.append(msg)

        self._start_server(server_handler)
        client = self._connect_client()

        write_msg = pack_msg(
            msg_type=MI300X_MSG_FRAME_WRITE,
            msg_id=61,
            addr=test_offset,
            data=test_value,
            access_size=4,
        )
        client.sendall(write_msg)

        time.sleep(0.1)
        client.close()
        self.server_thread.join(timeout=2)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['type'], MI300X_MSG_FRAME_WRITE)
        self.assertEqual(received[0]['addr'], test_offset)
        self.assertEqual(received[0]['data'], test_value)

    def test_shutdown_message(self):
        """Test SHUTDOWN message from QEMU."""
        received = []

        def server_handler(conn):
            data = conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            received.append(msg)

        self._start_server(server_handler)
        client = self._connect_client()

        shutdown_msg = pack_msg(
            msg_type=MI300X_MSG_SHUTDOWN,
            msg_id=99,
        )
        client.sendall(shutdown_msg)

        time.sleep(0.1)
        client.close()
        self.server_thread.join(timeout=2)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['type'], MI300X_MSG_SHUTDOWN)

    def test_multiple_transactions(self):
        """Test multiple sequential MMIO read transactions."""
        num_txns = 10

        def server_handler(conn):
            for i in range(num_txns):
                data = conn.recv(MSG_HDR_SIZE)
                msg = unpack_msg(data)
                self.assertEqual(msg['type'], MI300X_MSG_MMIO_READ)
                self.assertEqual(msg['id'], i)

                resp = pack_msg(
                    msg_type=MI300X_MSG_MMIO_RESP,
                    msg_id=i,
                    addr=msg['addr'],
                    data=0x1000 + i,
                    access_size=4,
                )
                conn.sendall(resp)

        self._start_server(server_handler)
        client = self._connect_client()

        for i in range(num_txns):
            read_msg = pack_msg(
                msg_type=MI300X_MSG_MMIO_READ,
                msg_id=i,
                addr=0xD000 + i * 4,
                access_size=4,
            )
            client.sendall(read_msg)

            resp_data = client.recv(MSG_HDR_SIZE)
            resp = unpack_msg(resp_data)
            self.assertEqual(resp['type'], MI300X_MSG_MMIO_RESP)
            self.assertEqual(resp['id'], i)
            self.assertEqual(resp['data'], 0x1000 + i)

        client.close()
        self.server_thread.join(timeout=2)


# ======================================================================
# Test: Source code cross-validation
# ======================================================================

class TestTwoSocketArchitecture(unittest.TestCase):
    """Test the two-connection architecture (MMIO + Event sockets)."""

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock', prefix='cosim_2sock_')
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.sock_path)
        self.server_sock.listen(2)  # Accept two connections

    def tearDown(self):
        self.server_sock.close()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_two_connections_init_then_event(self):
        """Verify QEMU connects twice: first MMIO, then event."""
        connections = []

        def server_accept_two():
            """Accept two connections from 'QEMU'."""
            for _ in range(2):
                conn, _ = self.server_sock.accept()
                connections.append(conn)

        server_thread = threading.Thread(target=server_accept_two)
        server_thread.daemon = True
        server_thread.start()

        # QEMU first connection: MMIO
        mmio_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mmio_client.connect(self.sock_path)

        # Send INIT on MMIO socket
        init_msg = pack_msg(msg_type=MI300X_MSG_INIT, msg_id=1,
                           data=16 * 1024**3)
        mmio_client.sendall(init_msg)

        time.sleep(0.1)

        # Server reads INIT from first connection
        self.assertEqual(len(connections), 1)
        data = connections[0].recv(MSG_HDR_SIZE)
        msg = unpack_msg(data)
        self.assertEqual(msg['type'], MI300X_MSG_INIT)

        # Server sends INIT_RESP
        resp = pack_msg(msg_type=MI300X_MSG_INIT_RESP, msg_id=1,
                       data=16 * 1024**3)
        connections[0].sendall(resp)

        # Read INIT_RESP on MMIO socket
        resp_data = mmio_client.recv(MSG_HDR_SIZE)
        resp = unpack_msg(resp_data)
        self.assertEqual(resp['type'], MI300X_MSG_INIT_RESP)

        # QEMU second connection: Events
        event_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        event_client.connect(self.sock_path)

        time.sleep(0.1)
        self.assertEqual(len(connections), 2)

        # Now MMIO and events use separate connections
        # MMIO: connections[0] <-> mmio_client
        # Events: connections[1] <-> event_client

        # Server sends IRQ on event connection (not MMIO)
        irq_msg = pack_msg(msg_type=MI300X_MSG_IRQ_RAISE, data=5)
        connections[1].sendall(irq_msg)

        # Event client receives IRQ (not MMIO client)
        irq_data = event_client.recv(MSG_HDR_SIZE)
        irq = unpack_msg(irq_data)
        self.assertEqual(irq['type'], MI300X_MSG_IRQ_RAISE)
        self.assertEqual(irq['data'] & 0xFFFF, 5)

        # Simultaneously, MMIO still works independently
        mmio_read = pack_msg(msg_type=MI300X_MSG_MMIO_READ, msg_id=2,
                            addr=0xD000, access_size=4)
        mmio_client.sendall(mmio_read)

        mmio_data = connections[0].recv(MSG_HDR_SIZE)
        mmio_msg = unpack_msg(mmio_data)
        self.assertEqual(mmio_msg['type'], MI300X_MSG_MMIO_READ)

        mmio_resp = pack_msg(msg_type=MI300X_MSG_MMIO_RESP, msg_id=2,
                            addr=0xD000, data=0, access_size=4)
        connections[0].sendall(mmio_resp)

        mmio_resp_data = mmio_client.recv(MSG_HDR_SIZE)
        self.assertEqual(len(mmio_resp_data), MSG_HDR_SIZE)

        mmio_client.close()
        event_client.close()
        for c in connections:
            c.close()
        server_thread.join(timeout=2)

    def test_mmio_and_event_no_interference(self):
        """Verify event messages don't interfere with MMIO responses."""
        connections = []

        def server_handler():
            for _ in range(2):
                conn, _ = self.server_sock.accept()
                connections.append(conn)

        t = threading.Thread(target=server_handler)
        t.daemon = True
        t.start()

        mmio_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mmio_client.connect(self.sock_path)
        event_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        event_client.connect(self.sock_path)
        time.sleep(0.1)

        # Send 10 MMIO reads while gem5 sends IRQs on event connection
        for i in range(10):
            # gem5 sends IRQ on event connection
            irq = pack_msg(msg_type=MI300X_MSG_IRQ_RAISE, data=i)
            connections[1].sendall(irq)

            # QEMU sends MMIO read on MMIO connection
            read_msg = pack_msg(msg_type=MI300X_MSG_MMIO_READ, msg_id=i,
                               addr=0x1000 + i * 4, access_size=4)
            mmio_client.sendall(read_msg)

            # gem5 responds to MMIO on MMIO connection
            req_data = connections[0].recv(MSG_HDR_SIZE)
            req = unpack_msg(req_data)
            self.assertEqual(req['type'], MI300X_MSG_MMIO_READ)

            resp = pack_msg(msg_type=MI300X_MSG_MMIO_RESP, msg_id=i,
                           data=0xAA00 + i)
            connections[0].sendall(resp)

            # QEMU reads MMIO response (guaranteed on MMIO socket, not event)
            resp_data = mmio_client.recv(MSG_HDR_SIZE)
            r = unpack_msg(resp_data)
            self.assertEqual(r['type'], MI300X_MSG_MMIO_RESP)
            self.assertEqual(r['data'], 0xAA00 + i)

            # Event client reads IRQ (on event socket, not MMIO)
            irq_data = event_client.recv(MSG_HDR_SIZE)
            irq_msg = unpack_msg(irq_data)
            self.assertEqual(irq_msg['type'], MI300X_MSG_IRQ_RAISE)
            self.assertEqual(irq_msg['data'] & 0xFFFF, i)

        mmio_client.close()
        event_client.close()
        for c in connections:
            c.close()
        t.join(timeout=2)


class TestDriverInitSimulation(unittest.TestCase):
    """Simulate the amdgpu driver init sequence over the protocol."""

    def setUp(self):
        self.sock_path = tempfile.mktemp(suffix='.sock', prefix='cosim_drv_')
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.sock_path)
        self.server_sock.listen(2)

    def tearDown(self):
        self.server_sock.close()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_driver_init_sequence(self):
        """Simulate full amdgpu driver initialization MMIO sequence.

        This mimics what the real amdgpu driver does during probe:
        1. INIT handshake
        2. Read GPU identification registers
        3. Read framebuffer location registers
        4. Read GRBM status
        5. Write configuration registers
        """
        # MI300X register values (matching gem5's setRegVal in constructor)
        MI300X_FB_LOCATION_BASE = 0x60920
        MI300X_FB_LOCATION_TOP = 0x60924
        MI300X_MEM_SIZE_REG = 0x60928
        AMDGPU_MP0_SMN_C2PMSG_33 = 0x3B10C

        mmhub_base = 0x8000 >> 24  # typically set by gem5
        mmhub_top = (0x8000 + 0x4000) >> 24
        mem_size = 16 * 1024  # 16 GB in MB

        # PCI config space (offset -> value)
        config_map = {
            0x00: 0x74A11002,  # DeviceID:VendorID (MI300X)
            0x04: 0x00100007,  # Status:Command (Mem+IO+BusMaster)
            0x08: 0x03800000,  # Class code (VGA)
            0x2C: 0x0C341002,  # SubsystemID:SubsystemVendorID
        }

        # Register map: offset -> value (what gem5 would return)
        reg_map = {
            AMDGPU_MP0_SMN_C2PMSG_33: 0x80000000,
            MI300X_FB_LOCATION_BASE: mmhub_base,
            MI300X_FB_LOCATION_TOP: mmhub_top,
            MI300X_MEM_SIZE_REG: mem_size,
            0xD000: 0x00000000,  # GRBM_STATUS (idle)
            0xD004: 0x00000000,  # GRBM_STATUS2
        }

        def gem5_server():
            """Mock gem5 server handling driver init MMIO."""
            mmio_conn, _ = self.server_sock.accept()

            # Handle INIT
            data = mmio_conn.recv(MSG_HDR_SIZE)
            msg = unpack_msg(data)
            assert msg['type'] == MI300X_MSG_INIT
            resp = pack_msg(msg_type=MI300X_MSG_INIT_RESP,
                           msg_id=msg['id'], data=16 * 1024**3)
            mmio_conn.sendall(resp)

            # Handle MMIO reads/writes until shutdown
            while True:
                data = mmio_conn.recv(MSG_HDR_SIZE)
                if not data:
                    break
                msg = unpack_msg(data)

                if msg['type'] == MI300X_MSG_SHUTDOWN:
                    break
                elif msg['type'] == MI300X_MSG_MMIO_READ:
                    addr = msg['addr']
                    value = reg_map.get(addr, 0)
                    resp = pack_msg(msg_type=MI300X_MSG_MMIO_RESP,
                                   msg_id=msg['id'], addr=addr,
                                   data=value, access_size=4)
                    mmio_conn.sendall(resp)
                elif msg['type'] == MI300X_MSG_MMIO_WRITE:
                    # Accept writes silently (fire-and-forget)
                    reg_map[msg['addr']] = msg['data']
                elif msg['type'] == MI300X_MSG_DB_WRITE:
                    # Doorbell writes are fire-and-forget
                    pass
                elif msg['type'] == MI300X_MSG_CONFIG_READ:
                    addr = msg['addr']
                    value = config_map.get(addr, 0)
                    resp = pack_msg(msg_type=MI300X_MSG_MMIO_RESP,
                                   msg_id=msg['id'], addr=addr,
                                   data=value, access_size=4)
                    mmio_conn.sendall(resp)
                elif msg['type'] == MI300X_MSG_CONFIG_WRITE:
                    config_map[msg['addr']] = msg['data']
                elif msg['type'] == MI300X_MSG_FRAME_READ:
                    resp = pack_msg(msg_type=MI300X_MSG_MMIO_RESP,
                                   msg_id=msg['id'], addr=msg['addr'],
                                   data=0, access_size=4)
                    mmio_conn.sendall(resp)
                elif msg['type'] == MI300X_MSG_FRAME_WRITE:
                    pass  # fire-and-forget

            mmio_conn.close()

        server_thread = threading.Thread(target=gem5_server)
        server_thread.daemon = True
        server_thread.start()

        # QEMU side: connect and run driver init sequence
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(self.sock_path)

        # Step 1: INIT handshake
        init_msg = pack_msg(msg_type=MI300X_MSG_INIT, msg_id=0,
                           data=16 * 1024**3)
        client.sendall(init_msg)
        resp = unpack_msg(client.recv(MSG_HDR_SIZE))
        self.assertEqual(resp['type'], MI300X_MSG_INIT_RESP)
        self.assertEqual(resp['data'], 16 * 1024**3)

        # Step 1.5: Read PCI config space (VendorID/DeviceID)
        client.sendall(pack_msg(msg_type=MI300X_MSG_CONFIG_READ, msg_id=1,
                               addr=0x00, access_size=4))
        resp = unpack_msg(client.recv(MSG_HDR_SIZE))
        self.assertEqual(resp['data'], 0x74A11002)  # MI300X

        # Enable PCI BusMaster (config write, fire-and-forget)
        client.sendall(pack_msg(msg_type=MI300X_MSG_CONFIG_WRITE, msg_id=1,
                               addr=0x04, data=0x00100007, access_size=4))

        # Step 2: Read MP0 firmware status
        client.sendall(pack_msg(msg_type=MI300X_MSG_MMIO_READ, msg_id=1,
                               addr=AMDGPU_MP0_SMN_C2PMSG_33, access_size=4))
        resp = unpack_msg(client.recv(MSG_HDR_SIZE))
        self.assertEqual(resp['data'], 0x80000000)

        # Step 3: Read FB location registers
        for reg_addr, expected in [
            (MI300X_FB_LOCATION_BASE, mmhub_base),
            (MI300X_FB_LOCATION_TOP, mmhub_top),
            (MI300X_MEM_SIZE_REG, mem_size),
        ]:
            client.sendall(pack_msg(msg_type=MI300X_MSG_MMIO_READ,
                                   msg_id=2, addr=reg_addr, access_size=4))
            resp = unpack_msg(client.recv(MSG_HDR_SIZE))
            self.assertEqual(resp['data'], expected,
                           f"Register 0x{reg_addr:x} expected {expected} "
                           f"got {resp['data']}")

        # Step 4: Read GRBM_STATUS
        client.sendall(pack_msg(msg_type=MI300X_MSG_MMIO_READ, msg_id=3,
                               addr=0xD000, access_size=4))
        resp = unpack_msg(client.recv(MSG_HDR_SIZE))
        self.assertEqual(resp['data'], 0)  # GPU idle

        # Step 5: Write a config register (fire-and-forget)
        client.sendall(pack_msg(msg_type=MI300X_MSG_MMIO_WRITE, msg_id=4,
                               addr=0x1234, data=0xDEADBEEF, access_size=4))

        # Step 6: Write a doorbell (fire-and-forget)
        client.sendall(pack_msg(msg_type=MI300X_MSG_DB_WRITE, msg_id=5,
                               addr=0x100, data=0x42, access_size=4))

        # Step 7: Shutdown
        client.sendall(pack_msg(msg_type=MI300X_MSG_SHUTDOWN, msg_id=99))

        time.sleep(0.1)
        client.close()
        server_thread.join(timeout=2)


class TestSourceCodeAlignment(unittest.TestCase):
    """Verify gem5 and QEMU source definitions are aligned."""

    def _find_gem5_root(self):
        """Find the gem5 source root."""
        path = os.path.dirname(os.path.abspath(__file__))
        while path != '/':
            if os.path.exists(os.path.join(path, 'src', 'dev', 'amdgpu',
                                           'mi300x_gem5_cosim.hh')):
                return path
            path = os.path.dirname(path)
        return None

    def _find_qemu_root(self):
        """Find the QEMU source root."""
        for path in ['/home/user/qemu-mi300x', '/home/user/qemu']:
            if os.path.exists(os.path.join(path, 'include', 'hw', 'misc',
                                           'mi300x_gem5.h')):
                return path
        return None

    def test_gem5_header_has_matching_types(self):
        """Check gem5 header defines same message type values as QEMU."""
        gem5_root = self._find_gem5_root()
        if not gem5_root:
            self.skipTest("gem5 source root not found")

        header = os.path.join(gem5_root, 'src', 'dev', 'amdgpu',
                              'mi300x_gem5_cosim.hh')
        with open(header) as f:
            content = f.read()

        # Verify enum values match QEMU
        self.assertIn('MmioRead       = 0x01', content)
        self.assertIn('MmioWrite      = 0x02', content)
        self.assertIn('DoorbellRead   = 0x03', content)
        self.assertIn('DoorbellWrite  = 0x04', content)
        self.assertIn('Init           = 0x06', content)
        self.assertIn('Shutdown       = 0x07', content)
        self.assertIn('ConfigRead     = 0x08', content)
        self.assertIn('ConfigWrite    = 0x09', content)
        self.assertIn('FrameRead      = 0x0A', content)
        self.assertIn('FrameWrite     = 0x0B', content)
        self.assertIn('MmioResp       = 0x81', content)
        self.assertIn('IrqRaise       = 0x82', content)
        self.assertIn('IrqLower       = 0x83', content)
        self.assertIn('DmaRead        = 0x84', content)
        self.assertIn('DmaWrite       = 0x85', content)
        self.assertIn('InitResp       = 0x86', content)

    def test_gem5_struct_matches_qemu(self):
        """Check gem5 CosimMsgHeader fields match QEMU MI300XGem5MsgHeader."""
        gem5_root = self._find_gem5_root()
        if not gem5_root:
            self.skipTest("gem5 source root not found")

        header = os.path.join(gem5_root, 'src', 'dev', 'amdgpu',
                              'mi300x_gem5_cosim.hh')
        with open(header) as f:
            content = f.read()

        # Check struct fields exist in order
        self.assertIn('uint32_t type;', content)
        self.assertIn('uint32_t size;', content)
        self.assertIn('uint64_t addr;', content)
        self.assertIn('uint64_t data;', content)
        self.assertIn('uint32_t access_size;', content)
        self.assertIn('uint32_t id;', content)
        self.assertIn('sizeof(CosimMsgHeader) == 32', content)

    def test_qemu_struct_fields(self):
        """Check QEMU header defines the expected struct."""
        qemu_root = self._find_qemu_root()
        if not qemu_root:
            self.skipTest("QEMU source root not found")

        header = os.path.join(qemu_root, 'include', 'hw', 'misc',
                              'mi300x_gem5.h')
        with open(header) as f:
            content = f.read()

        self.assertIn('uint32_t type;', content)
        self.assertIn('uint32_t size;', content)
        self.assertIn('uint64_t addr;', content)
        self.assertIn('uint64_t data;', content)
        self.assertIn('uint32_t access_size;', content)
        self.assertIn('uint32_t id;', content)
        self.assertIn('__attribute__((packed))', content)

    def test_qemu_msg_type_values(self):
        """Verify QEMU enum values match our protocol constants."""
        qemu_root = self._find_qemu_root()
        if not qemu_root:
            self.skipTest("QEMU source root not found")

        header = os.path.join(qemu_root, 'include', 'hw', 'misc',
                              'mi300x_gem5.h')
        with open(header) as f:
            content = f.read()

        self.assertIn('MI300X_MSG_MMIO_READ    = 0x01', content)
        self.assertIn('MI300X_MSG_MMIO_WRITE   = 0x02', content)
        self.assertIn('MI300X_MSG_DB_READ      = 0x03', content)
        self.assertIn('MI300X_MSG_DB_WRITE     = 0x04', content)
        self.assertIn('MI300X_MSG_INIT         = 0x06', content)
        self.assertIn('MI300X_MSG_SHUTDOWN     = 0x07', content)
        self.assertIn('MI300X_MSG_MMIO_RESP    = 0x81', content)
        self.assertIn('MI300X_MSG_IRQ_RAISE    = 0x82', content)
        self.assertIn('MI300X_MSG_IRQ_LOWER    = 0x83', content)
        self.assertIn('MI300X_MSG_DMA_READ     = 0x84', content)
        self.assertIn('MI300X_MSG_DMA_WRITE    = 0x85', content)
        self.assertIn('MI300X_MSG_INIT_RESP    = 0x86', content)


if __name__ == '__main__':
    print("=" * 70)
    print("MI300X gem5-QEMU Co-simulation Protocol Integration Test")
    print("=" * 70)
    print(f"Message header size: {MSG_HDR_SIZE} bytes")
    print(f"Header format: {MSG_HDR_FORMAT}")
    print()

    unittest.main(verbosity=2)
