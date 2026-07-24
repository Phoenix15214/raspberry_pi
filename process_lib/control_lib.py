import numpy as np
import socket
import serial
import struct
import time
import os
import json
import tempfile
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager
import multiprocessing.shared_memory as shared_memory
from threading import Thread
# PID相关函数

class PID_Inc():
    def __init__(self, kp, ki, kd, goal, frquency, integral_limit=2000, min_output=10):
        self.frequency = frquency
        self.kp = kp
        self.ki = ki / self.frequency
        self.kd = kd * self.frequency
        self.goal = goal
        self.integral_limit = integral_limit  # 输出限幅
        self.min_output = min_output  # 死区
        self.output_integral = 0.0  # 输出
        self.prev_error = 0.0
        self.prev_prev_error = 0.0

    def Set_Goal(self, goal):
        self.goal = goal

    def Set_Integral_Limit(self, limit):
        self.integral_limit = limit

    def Cal_PID(self, current_val):
        error = self.goal - current_val
        # 此处output是控制量的变化值（增量式pid）
        output = self.kp * (error - self.prev_error) + self.ki * error + self.kd * (error - 2 * self.prev_error + self.prev_prev_error)
        self.prev_prev_error = self.prev_error
        self.prev_error = error
        self.output_integral += output

        if self.integral_limit is None and abs(self.output_integral) > self.min_output:
            return self.output_integral
        elif abs(self.output_integral) < self.min_output:
            return 0
        else:
            return max(-self.integral_limit, min(self.integral_limit, self.output_integral))

class PID_Loc():
    def __init__(self, kp, ki, kd, goal, frequency, integral_limit=1000, output_limit=2000, min_output=10):
        self.frequency = frequency
        self.kp = kp
        self.ki = ki / self.frequency
        self.kd = kd * self.frequency
        self.goal = goal
        self.min_output = min_output
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0

    def Set_Goal(self, goal):
        self.goal = goal
    
    def Set_Limit(self, limit):
        self.integral_limit = limit

    def Set_Output_Limit(self, limit):
        self.output_limit = limit

    def Cal_PID(self, current_val):
        error = self.goal - current_val
        derivative = error - self.prev_error
        self.prev_error = error
        temp_output = self.kp * error + self.kd * derivative
        self.integral += error
        if temp_output + self.ki * self.integral > self.integral_limit and error > 0:
            self.integral -= error
        elif temp_output +self.ki * self.integral < -self.integral_limit and error < 0:
            self.integral -= error
        output = temp_output + self.ki * self.integral
        
        return max(-self.output_limit, min(self.output_limit, output)) if abs(output) > self.min_output else 0


# 获取最近目标相关函数

def Get_Closest_Target(location, targets):
    """
    找出与location坐标最近的目标。
    参数:
        location:中心坐标,应为一维numpy数组。
        targets:需要计算距离的点集,应为二维numpy数组。
    返回:
        与location距离最短的点的index值。
    """
    location = np.asarray(location)
    targets = np.asarray(targets)
    if location.ndim != 1:
        raise ValueError("location必须为一维数组")
    if targets.ndim != 2:
        raise ValueError("targets必须为二维数组")
    if location.shape[0] != targets.shape[1]:
        raise ValueError(f"location维度:{location.shape[0]}与targets维度{targets.shape[1]}不匹配")
    
    distances = np.linalg.norm(targets - location, axis=1)
    closest_index = np.argmin(distances)

    return closest_index

# VOFA+调试相关函数
def Parse_Input(msg):
    """
    解析VOFA+发送的设置数据
    参数:
        msg:解码后的消息。
    返回:
        command:要设置的变量。
        value:要设置的值。
    """
    start_flag = ":"
    end_flag = "\n"
    start_pos = msg.find(start_flag)
    content_pos = start_pos + len(start_flag)
    end_pos = msg.find(end_flag)
    if start_pos == -1 or end_pos == -1:
        return None, None
    if end_pos <= start_pos:
        return None, None
    command = msg[0:start_pos]
    value = msg[content_pos:end_pos]
    return command, value


def _send_by_firewater(data_list, socket):
    send_msg = ",".join(str(x) for x in data_list) + "\n"
    socket.send(send_msg.encode("utf8"))

def _send_by_justfloat(data_list, socket):
    format_string = '<' + 'f' * len(data_list)
    packed_data = struct.pack(format_string, *data_list)
    tail = b'\x00\x00\x80\x7f'
    socket.send(packed_data + tail)

def _send_thread(conn, method, socket):
    while True:
            msg = conn.recv()
            try:
                if method == "firewater":
                    _send_by_firewater(msg, socket)
                elif method == "justfloat":
                    _send_by_justfloat(msg, socket)
            except:
                print("客户端断开连接")
                isConnected = False
                conn.send(isConnected)
                break

def _recv_thread(conn, socket):
    while True:
        try:
            msg = socket.recv(1024).decode("utf8")
            if len(msg) == 0:
                break
            conn.send(msg)
        except:
            break

def Send_Process(conn, method="justfloat"):
    """
    在进程内使用,用于向VOFA+发送调试信息。
    参数:
        conn:与其他进程或主程序通信的pipe。
        method:发送数据的格式,有"firewater"和"justfloat"两种选择。
    返回:
        无,需要在进程中运行。
    """
    if method not in ("justfloat", "firewater"):
        print("发送方式不正确")
        method = "justfloat"
        print("自动更改格式为justfloat")
    isConnected = False
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_socket.bind(("", 11451))
    while True:    
        connect_socket.listen(3)
        server_socket, client_addr = connect_socket.accept()
        isConnected = True
        print("客户端已连接")
        conn.send(isConnected)
        t1 = Thread(target=_send_thread, args=(conn, method, server_socket))
        t2 = Thread(target=_recv_thread, args=(conn, server_socket))
        t1.start()
        t2.start()

# 串口通信相关内容
class SerialPacket:
    def __init__(self, port="COM6", baudrate=115200, timeout=0.1):
        self.header = bytearray([0xFF, 0xAA])
        self.tail = bytearray([0x55, 0xFE])
        self.recv_header = "@"
        self.recv_tail = "$#"
        self.recv_data = []  # 收到数据未经编码，默认已是UTF-8
        self.send_data = bytearray()
        self.buffer = bytearray()
        self.isOpened = False
        self.index = 0  # 数据插入位置
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            if self.ser.is_open:
                print(f"串口 {port} 已成功打开！")
                self.isOpened = True
            else:
                print(f"串口 {port} 打开失败！")
                self.isOpened = False
        except serial.SerialException as e:
            print(f"串口 {port} 打开失败，错误信息：{e}")
            self.isOpened = False
            raise serial.SerialException(f"串口 {port} 打开失败，错误信息：{e}")
        except Exception as e:
            print(f"发生未知错误：{e}")
            self.isOpened = False
            raise Exception(f"发生未知错误：{e}")

    def __clear_packet(self):
        self.send_data = bytearray()
        self.index = 0  # 数据插入位置清零

    def insert_byte(self, value):
        self.send_data.insert(self.index, value)  # 插入到数据部分开头（包头之后）
        self.index += 1

    def insert_two_bytes(self, values):
        self.send_data.insert(self.index, values[0])  # 插入到数据部分开头（包头之后）
        self.index += 1
        self.send_data.insert(self.index, values[1])  # 插入到数据部分开头（包头之后）
        self.index += 1

    def insert_three_bytes(self, values): # 三字节实际只有一条消息，有一字节为校验位
        check_byte = values[0] ^ values[1]
        bytes_to_insert = [values[0], values[1], check_byte]
        self.send_data[self.index:self.index] = bytes_to_insert # 使用切片赋值在指定位置一次性插入多个元素
        self.index += 3
        
    def insert_bytes(self, index, values):
        for i, val in enumerate(values):
            self.send_data.insert(index + i, val)
            self.index += 1

    def num_to_bytes(self, value):
        if not 0 <= value <= 0xFFFF:
            raise ValueError("输入值必须在 0~65535 之间")

        high_byte = (value >> 8) & 0xFF  # 高8位
        low_byte = value & 0xFF  # 低8位

        return [high_byte, low_byte]

    def __build_packet(self):
        return self.header + self.send_data + self.tail

    def send_packet(self):
        if self.isOpened:
            try:
                packet = self.__build_packet()
                self.ser.write(packet)
                self.__clear_packet()
            except serial.SerialException as e:
                print(f"串口发送数据失败，错误信息：{e}")
                self.isOpened = False
            except Exception as e:
                print(f"发生未知错误：{e}")
                self.isOpened = False
    
    def send_char(self, data):
        if self.isOpened:
            try:
                self.ser.write(data.encode("utf-8"))
            except serial.SerialException as e:
                print(f"串口发送数据失败，错误信息：{e}")
                self.isOpened = False
            except Exception as e:
                print(f"发生未知错误：{e}")
                self.isOpened = False
    
    def __parse_buffer(self):
        while True:
            start = self.buffer.decode("utf-8").find(self.recv_header)
            if start == -1:
                break
            end = self.buffer.decode("utf-8").find(self.recv_tail, start + len(self.recv_header))
            if end == -1:
                break
            self.recv_data.append(self.buffer[start + len(self.recv_header): end])
            self.buffer = self.buffer[end + len(self.recv_tail):]
        
    def recv_packet(self, timeout=None):
        if timeout is not None:
            self.ser.timeout = timeout
        if self.isOpened and self.ser.in_waiting > 0:
            packet = self.ser.read(self.ser.in_waiting)
            self.buffer += packet
            self.__parse_buffer()
            return True
        return False

    def get_recv_data(self, clear=True):
        recv_data = self.recv_data
        if recv_data:
            recv_data = recv_data[0].decode("utf-8")
        else:
            recv_data = None
        if clear:
            self.recv_data = []
        return recv_data

    def parse_input(self, msg):
        start_flag = ":"
        start_pos = msg.find(start_flag)
        content_pos = start_pos + len(start_flag)
        if start_pos == -1:
            return None, None
        command = msg[0:start_pos]
        value = msg[content_pos:]
        return command, value

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.isOpened = False

    def __enter__(self):
        return self
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        self.close()
        
class Timer:
    def __init__(self):
        self.last_time = time.time()

    def update(self):
        current = time.time()
        interval = current - self.last_time
        return interval
    
    def reset(self):
        self.last_time = time.time()

# 按键相关函数
class Button:
    def __init__(self, max_interval=3, long_push_interval=1, max_click=3):
        self.max_interval = max_interval # 判断多次按下时的最大间隔时间
        self.long_push_interval = long_push_interval # 判断长按的间隔时间
        self.last_time = time.time() # 上一次按下的时间
        self.min_interval = 0.01 # 最小间隔时间，防止抖动
        self.clicking = False
        self.max_click = max_click
        self.click_count = 0 # 多次按下的次数
        self.state = "release" # 当前按键的状态
        self.last_action = "standby" # 上一次按键的动作
    
    def pushed(self):
        current = time.time()
        interval = current - self.last_time
        if interval < self.min_interval:
            self.last_time = current
            return
        if interval > self.max_interval:
            self.click_count = 0
        self.last_time = current
        self.state = "push"

    def release(self):
        current = time.time()
        interval = current - self.last_time
        if interval < self.min_interval:
            self.state = "release"
            self.last_time = current
            return
        if interval > self.long_push_interval:
            self.last_action = "long_push"
        else:
            self.clicking = True
            self.click_count += 1 if self.click_count < self.max_click else 0
            self.last_action = "clicked"
        self.last_time = current
        self.state = "release"
        return interval

    def get_state(self):
        current = time.time()
        # 检查是否处于正在点击状态
        if self.clicking:
            if (current - self.last_time) < self.max_interval:
                return "clicking"
            else:
                self.clicking = False
                return "clicked"
        else:
            if self.last_action == "long_push":
                # 如果上一个动作是长按，则返回长按状态，并重置上一个动作
                result = "long_push"
                self.last_action = "standby"
                return result
            elif self.last_action == "standby":
                return "standby"
            elif self.last_action == "clicked":
                result = "clicked"
                self.last_action = "standby"
                return result
            else:
                print("未知状态,需要完善程序")
                return "standby"

# 顶点重排序函数
def Reorder_Vertex(vertices):
    sums = []
    sorted_vertices = []
    for vertex in vertices:
        sums.append(vertex[0] + vertex[1])
    max_index = sums.index(max(sums))
    sorted_vertices.append(vertices[(max_index + 2) % 4])  # 左上
    sorted_vertices.append(vertices[(max_index + 3) % 4])  # 右上
    sorted_vertices.append(vertices[max_index])  # 右下
    sorted_vertices.append(vertices[(max_index + 1) % 4])  # 左下
    sorted_vertices = np.array(sorted_vertices)
    return sorted_vertices

# 通过极坐标进行顶点重排序
def Reorder_Vertex_Pole(vertices):
    sums = []
    sum_x = []
    sum_y = []
    arctan = []
    sorted_vertices = []
    for vertex in vertices:
        sum_x.append(vertex[0])
        sum_y.append(vertex[1])
        sums.append(vertex[0] + vertex[1])
    max_index = sums.index(max(sums)) # 作为右下角
    center_x = np.mean(vertices[:, 0])
    center_y = np.mean(vertices[:, 1])
    for vertex in vertices:
        arctan.append(np.arctan2(vertex[1] - center_y, vertex[0] - center_x))
    # 根据 arctan 值对顶点进行排序
    sorted_indices = np.argsort(arctan)
    for index in sorted_indices:
        sorted_vertices.append(vertices[index])
    sorted_vertices = np.array(sorted_vertices)
    return sorted_vertices

# 文件化阈值保存
class ConfigManager:
    def __init__(self, config_file_path="config.json"):
        self.config_file_path = config_file_path
        self.config_data = {}
        self.load_config()
        
    def load_config(self):
        if not os.path.exists(self.config_file_path):
            print("配置文件不存在,使用默认配置")
            self.__set_default_value()
            return
        try:
            with open(self.config_file_path, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("加载配置文件失败,使用默认配置")
            self.__set_default_value()
    
    def save(self):
        # 创建临时文件，方便进行原子写入，防止突发事件导致文件损坏
        dir_path = os.path.dirname(self.config_file_path) or "."
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8", 
            dir=dir_path, 
            delete=False,
            suffix=".tmp"
        ) as tmp_file:
            json.dump(self.config_data, tmp_file, ensure_ascii=False, indent=2)
            tmp_path = tmp_file.name
        
        os.replace(tmp_path, self.config_file_path)
    
    def get(self, key, default=None):
        return self.config_data.get(key, default)

    def get_all(self):
        return self.config_data.copy()
    
    def del_data(self, key):
        if key in self.config_data:
            del self.config_data[key]
            self.save()

    def update(self):
        self.load_config()

    def set_value(self, key, value, auto_save=True):
        self.config_data[key] = value
        if auto_save:
            self.save()

    def __set_default_value(self):
        default_values = {}
        for key, value in default_values.items():
            self.set_value(key, value, auto_save=False)
        self.save()

# 共享内存相关函数
class MemoryShare:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.shm = None
        self.data = None
        self.create()

    def create(self):
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=np.prod(self.shape) * np.dtype(self.dtype).itemsize)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name)
        except Exception as e:
            print(f"创建共享内存失败: {e}")
            raise RuntimeError("共享内存创建失败。")
        self.data = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def connect(self):
        self.shm = shared_memory.SharedMemory(name=self.name)
        self.data = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def read(self):
        return self.data if self.data is not None else None

    def write(self, data):
        if self.data is not None:
            np.copyto(self.data, data)
        else:
            raise ValueError("共享内存未创建或连接。")

    def close(self):
        try:
            if self.shm is not None:
                self.shm.close()
                self.shm.unlink()
        except FileNotFoundError:
            print("共享内存文件未找到。")
        except Exception as e:
            print(f"关闭共享内存时发生错误: {e}")
