# raspberry_pi
在树莓派上同步更新，用于树莓派视觉开发
# 注意！！！
由于在树莓派上使用VSCode运行程序，程序中的文件地址应填文件的绝对地址，而非相对地址
或在文件开头加上：
```python
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
```
