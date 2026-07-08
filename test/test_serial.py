import process_lib.control_lib as ctrl
import time

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
        if recv_data is not None:
            print(f"Received data: {recv_data}")
            command, value = pack.parse_input(recv_data)
            if command is not None and value is not None:
                print(f"Parsed command: {command}, value: {value}")
        else:
            time.sleep(0.1)  # Sleep briefly to avoid busy waiting

if __name__ == "__main__":
    main()
