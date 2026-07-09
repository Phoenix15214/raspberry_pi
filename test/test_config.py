import process_lib.control_lib as ctrl
config = ctrl.ConfigManager("config.json")
config2 = ctrl.ConfigManager("config.json")

def main():
    print(config.get("angle_shift"))
    print(config.get("offset"))

    config.set_value("angle_shift", 50)
    config2.set_value("offset", 20)
    config.update()  # Reload the configuration from the file
    config2.update()  # Reload the configuration from the file
    print(config.get("angle_shift"))
    print(config.get("offset"))

if __name__ == "__main__":
    main()
