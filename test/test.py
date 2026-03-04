try:
    with open("file.txt") as f:
        s = f.readline()
except FileNotFoundError:
    print("File not found")
finally:
    print("program ended")