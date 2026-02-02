import numpy as np

# PID相关函数

class PID_Inc():
    def __init__(self, kp, ki, kd, goal, frquency):
        self.frequency = frquency
        self.kp = kp
        self.ki = ki / self.frequency
        self.kd = kd * self.frequency
        self.goal = goal
        self.output_integral = 0.0
        self.integral_limit = 2000
        self.prev_error = 0.0
        self.prev_prev_error = 0.0

    def Set_Goal(self, goal):
        self.goal = goal

    def Set_Integral_Limit(self, limit):
        self.integral_limit = limit

    def Cal_PID(self, current_val):
        error = self.goal - current_val
        output = self.kp * (error - self.prev_error) + self.ki * error + self.kd * (error - 2 * self.prev_error + self.prev_prev_error)
        self.prev_prev_error = self.prev_error
        self.prev_error = error
        self.output_integral += output

        return self.output_integral if self.integral_limit is None else max(-self.integral_limit, min(self.integral_limit, self.output_integral))

class PID_Loc():
    def __init__(self, kp, ki, kd, goal, frequency):
        self.frequency = frequency
        self.kp = kp
        self.ki = ki / self.frequency
        self.kd = kd * self.frequency
        self.goal = goal
        self.integral_limit = 1000
        self.output_limit = 2000
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
        should_integrate = True
        if temp_output + self.ki * self.integral > self.output_limit and error > 0:
            should_integrate = False
        elif temp_output + self.ki * self.integral < -self.output_limit and error < 0:
            should_integrate = False
        if should_integrate:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error))
        output = temp_output + self.ki * self.integral
        
        return max(-self.output_limit, min(self.output_limit, output))

# 卡尔曼滤波相关函数


