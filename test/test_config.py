import process_lib.control_lib as ctrl
config = ctrl.ConfigManager("config.json")

def main():
    print(config.get("angle_shift"))
    print(config.get("offset"))

    config.set_value("angle_shift", 50)

    print(config.get("angle_shift"))

if __name__ == "__main__":
    main()
