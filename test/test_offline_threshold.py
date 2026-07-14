import socket
import os
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait
from threading import Thread
import process_lib.control_lib as ctrl
import struct
import time

config = ctrl.ConfigManager("config.json")
config_data = config.get_all()
require_refresh = False

message = []
for value in config_data.values():
    message.append(value)
server_socket = None
isConnected = False
pack = None

def update_message():
    global message
    global config_data
    global require_refresh
    message = []
    for key in config_data.keys():
        message.append(config_data[key])
    require_refresh = False

def _fatal_exit(reason, exc=None):
    if exc is not None:
        print(f"{reason}: {exc}")
    else:
        print(reason)
    # Called inside worker threads: force whole process to exit non-zero.
    os._exit(1)


def _init_pack():
    global pack
    if pack is not None:
        return pack
    try:
        pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)
    except Exception as exc:
        raise RuntimeError(f"无法打开串口: {exc}")
    return pack

def _send_by_firewater(data_list, socket):
    send_msg = ",".join(str(x) for x in data_list) + "\n"
    socket.send(send_msg.encode("utf8"))

def _send_by_justfloat(data_list, socket):
    format_string = '<' + 'f' * len(data_list)
    packed_data = struct.pack(format_string, *data_list)
    tail = b'\x00\x00\x80\x7f'
    socket.send(packed_data + tail)

def _send_thread(conns, method, socket):
    global pack
    global message
    global config_data
    global config
    global require_refresh
    global isConnected
    
    # normalize to list
    if not isinstance(conns, (list, tuple)):
        conns = [conns]
    while True:
        if require_refresh:
            config.update()
        config_data = config.get_all()
        update_message()
        try:
            if method == "firewater":
                _send_by_firewater(message, socket)
            elif method == "justfloat":
                _send_by_justfloat(message, socket)
            time.sleep(0.01)
        except Exception:
            print("客户端断开连接")
            # mark disconnected and notify producers if possible
            global isConnected
            isConnected = False
            for c in conns:
                try:
                    c.send(isConnected)
                except Exception:
                    pass
            return

def _recv_thread(conns, socket):
    global isConnected
    global config
    global pack
    global config_data
    global require_refresh

    if not isinstance(conns, (list, tuple)):
        conns = [conns]
    while isConnected:
        try:
            msg = socket.recv(1024).decode("utf8")
            if len(msg) == 0:
                break
            command, value = pack.parse_input(msg)
            if command == "start":
                config.update()
            else:
                original_value = config_data.get(command, None)
                if original_value is not None:
                    config.set_value(command, int(value))
                    config.save()
                    require_refresh = True
            for c in conns:
                try:
                    c.send(msg)
                except Exception:
                    pass
        except Exception:
            break

def _recv_serial_thread(conns):
    global config
    global pack
    global config_data
    global require_refresh
    if not isinstance(conns, (list, tuple)):
        conns = [conns]
    while True:
        try:
            pack.recv_packet()
            data = pack.get_recv_data()
            if data is None:
                continue
            command, value = pack.parse_input(data)
            if command == "start":
                config.update()
                pack.insert_byte(0x06)
                pack.insert_three_bytes(pack.num_to_bytes(1))
                for val in config_data.values():
                    pack.insert_three_bytes(pack.num_to_bytes(int(val)))
                pack.send_packet()
            else:
                original_value = config_data.get(command, None)
                if original_value is not None:
                    config.set_value(command, int(value))
                    config.save()
                    require_refresh = True
                    send_message = "@Get$#"
                    pack.send_char(send_message)

            for c in conns:
                try:
                    c.send(data)
                except Exception:
                    pass
        except Exception as e:
            print(f"串口接收线程异常: {e}")
            break

def Listen_Thread(connect_socket):
    global server_socket
    global isConnected
    connect_socket.listen(3)
    server_socket, client_addr = connect_socket.accept()
    isConnected = True

def Empty_Thread(conns):
    global isConnected
    global message
    global pack
    global require_refresh
    global config_data
    global config
    
    if not isinstance(conns, (list, tuple)):
        conns = [conns]
    while not isConnected:
        if require_refresh:
            config.update()
        config_data = config.get_all()
        update_message()
        try:
            ready = wait(conns, timeout=0.01)
        except Exception:
            ready = []
        for r in ready:
            try:
                msg = r.recv()
            except Exception:
                continue

def Send_Process(conn, method="justfloat"):
    global server_socket
    global isConnected
    if method not in ("justfloat", "firewater"):
        print("发送方式不正确")
        method = "justfloat"
        print("自动更改格式为justfloat")
    isConnected = False
    _init_pack()
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connect_socket.bind(("", 11451))
    conns = conn if isinstance(conn, (list, tuple)) else [conn]
    t_serial = Thread(target=_recv_serial_thread, args=(conns,))
    t_serial.start()
    try:
        while True:
            # normalize conn to list when starting helper threads
            t0 = Thread(target=Listen_Thread, args=(connect_socket,))
            t00 = Thread(target=Empty_Thread, args=(conns,))
            t00.start()
            t0.start()
            # wait for client connection (Listen_Thread will set isConnected)
            t0.join()
            # ensure Empty_Thread has processed up to connection time
            t00.join()
            isConnected = True
            print("客户端已连接")
            t1 = Thread(target=_send_thread, args=(conns, method, server_socket))
            t2 = Thread(target=_recv_thread, args=(conns, server_socket))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            isConnected = False
            try:
                server_socket.close()
            except Exception:
                pass
    finally:
        try:
            connect_socket.close()
            server_socket.close()
        except Exception:
            pass
    
if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    p_send = Process(target=Send_Process, args=(child_conn, "justfloat"))
    p_send.start()
    try:
        while True:
            msg = parent_conn.recv()
            print("接收到数据:", msg)
    except KeyboardInterrupt:
        print("主进程被用户中断")
    except Exception as e:
        print(f"主进程发生错误: {e}")
    finally:
        p_send.terminate()