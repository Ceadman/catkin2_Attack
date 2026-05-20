#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简 TCP 吊舱调试脚本
1. 建立 TCP 连接
2. 周期发送一条 16 进制指令（可带累加帧计数与校验和）
3. 打印收到的原始报文
4. Ctrl-C 优雅退出
"""
import socket
import threading
import time
import signal
import sys

# ---------------- 用户可改 -----------------
TARGET_IP   = "192.168.144.119"
TARGET_PORT = 2000
TX_PERIOD_S = 0.2          # 发送周期
CMD_TEMPLATE = "55 AA 00 01 01 05 10 A2 00 00 00 B9 F0"  # 默认指令
# ----------------------------------------

def hex_str_to_bytes(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str.replace(" ", ""))

def build_frame(seq: int) -> bytes:
    """把模板第 3 字节换成 seq，并重新算校验和（倒数第 2 字节）"""
    raw = bytearray(hex_str_to_bytes(CMD_TEMPLATE))
    raw[2] = seq & 0xFF
    checksum = sum(raw[2:-2]) & 0xFF
    raw[-2] = checksum
    return bytes(raw)

class SimpleTCPTest:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((TARGET_IP, TARGET_PORT))
        self.sock.settimeout(1.0)
        self.running = True
        self.seq = 0

    def recv_thread(self):
        while self.running:
            try:
                data = self.sock.recv(1024)
                if not data:
                    print("[RX] peer closed")
                    break
                print("[RX]", data.hex(' ').upper())
            except socket.timeout:
                continue
            except Exception as e:
                print("[RX] err:", e)
                break

    def send_loop(self):
        while self.running:
            frame = build_frame(self.seq)
            self.seq = (self.seq + 1) & 0xFF
            try:
                self.sock.sendall(frame)
                print("[TX]", frame.hex(' ').upper())
            except Exception as e:
                print("[TX] err:", e)
                break
            time.sleep(TX_PERIOD_S)

    def close(self):
        self.running = False
        try:
            self.sock.shutdown(socket.SHUT_WR)
            self.sock.close()
        except Exception:
            pass

def main():
    tst = SimpleTCPTest()
    signal.signal(signal.SIGINT, lambda sig, frame: (print("\nCtrl-C 退出"), tst.close(), sys.exit(0)))

    t_rx = threading.Thread(target=tst.recv_thread, daemon=True)
    t_rx.start()
    tst.send_loop()          # 主线程阻塞在这里

if __name__ == "__main__":
    main()