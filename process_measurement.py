import numpy as np
from math import sqrt, fabs, sin, cos, atan2

"""
This script provides various process (motion) and measurement models to be used by a UKF 
"""

def ctrv(input_state, dt):
    """
    Implement constant turn-rate velocity (CTRV) process model
    Input: input_state: [px, py, v, yaw, yawd] or [px, py, v, yaw, yawd, nu_a, nu_yawdd] 
    Output: ouput_state: [px, py, v, yaw, yawd]
    """
    px = input_state[0]
    py = input_state[1]
    v = input_state[2]
    yaw = input_state[3]
    yawd = input_state[4]
    
    if fabs(yawd) > 0.001: # avoid division by zero due to very small yawd
        px_p = px + (v/yawd) * (sin(yaw + yawd*dt) - sin(yaw))
        py_p = py + (v/yawd) * (-cos(yaw + yawd * dt) + cos(yaw))
    else:
        px_p = px + v*cos(yaw)*dt
        py_p = py + v*sin(yaw)*dt
    v_p = v
    yaw_p = yaw + yawd*dt
    yawd_p = yawd
    
    if len(input_state) == 7: # When the effect of nu_a, and nu_yawdd need to be considered, i.e, for 
                              # non-additive noise, the state needs to be augmented.
        nu_a = input_state[5]
        nu_yawdd = input_state[6]
        
        px_p += 0.5 * nu_a * cos(yaw) * dt * dt
        py_p += 0.5 * nu_a * sin(yaw) * dt * dt
        
        yaw_p += 0.5 * nu_yawdd * dt * dt;
        yawd_p += nu_yawdd * dt;
         
    output_state=np.array([[px_p, py_p, v_p, yaw_p, yawd_p]])
   
    return output_state


def cv(input_state, dt):
    """
    Implement constant velocity (CV) process model
    Input: input_state: [px, py, v, yaw, yawd] or [px, py, v, yaw, yawd, nu_a, nu_yawdd] 
    Output: ouput_state: [px, py, v, yaw, yawd]
    """
    px = input_state[0]
    py = input_state[1]
    v = input_state[2]
    yaw = input_state[3]
    yawd = input_state[4]
    
    
    px_p = px + v*cos(yaw)*dt
    py_p = py + v*sin(yaw)*dt
    v_p = v # constant velocity
    yaw_p = yaw # no turn, so yaw state the same
    yawd_p = 0 # ? 
    
    if len(input_state) == 7: # When the effect of nu_a, and nu_yawdd need to be considered
        nu_a = input_state[5]
        nu_yawdd = input_state[6]
        
        px_p += 0.5 * nu_a * cos(yaw) * dt * dt
        py_p += 0.5 * nu_a * sin(yaw) * dt * dt
        
        yaw_p += 0.5 * nu_yawdd * dt * dt;
        yawd_p += nu_yawdd * dt;
         
    output_state=np.array([[px_p, py_p, v_p, yaw_p, yawd_p]])
   
    return output_state


def lidar(input_state):
    """
    LIDAR measurement model
    Input: CTRV state representation [p_x, p_y, v, yaw, yawd]
    Output: LIDAR measurement [p_x, p_y]
    """
    return input_state[0:2]


def radar(input_state):
    """
    RADAR measurement model 
    Input: CTRV state representation [p_x, p_y, v, yaw, yawd]
    Output: RADAR measurement [rho, phi, rho_dot]
    """
    p_x = input_state[0]
    p_y = input_state[1]
    v = input_state[2]
    yaw = input_state[3]
    
    v_x = cos(yaw)*v;
    v_y= sin(yaw)*v;
    c = sqrt(p_x*p_x+p_y*p_y);
    
    if fabs(c) < 0.0001: # warning for divsion by zero
        print("Error -Division by Zero")
        c = 0.0001;
    
    rho = c;
    phi = atan2(p_y, p_x);
    rho_dot = (p_x*v_x + p_y*v_y)/c
    
    return np.array([rho, phi, rho_dot])
