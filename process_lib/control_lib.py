import numpy as np
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
    def __init__(self, kp, ki, kd, goal, frequency, integral_limit=1000, output_limit=2000, min_output=10.0):
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
