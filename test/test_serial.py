import process_lib.control_lib as ctrl
import time

def Parse_Input(msg):
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

def main():
    pack = ctrl.SerialPacket(port="/dev/serial0", baudrate=115200, timeout=0.1)
    if pack is None:
        print("Failed to initialize SerialPacket.")
        return
    while True:
        pack.insert_byte(0x04)
        for i in range(4):
            pack.insert_three_bytes(pack.num_to_bytes(i + 1))
        pack.send_packet()
        pack.recv_packet()
        recv_data = pack.get_recv_data()
        if recv_data:
            print(f"Received data: {recv_data}")
            command, value = Parse_Input(recv_data)
            if command is not None and value is not None:
                print(f"Parsed command: {command}, value: {value}")
        else:
            time.sleep(0.1)  # Sleep briefly to avoid busy waiting

if __name__ == "__main__":
    main()
