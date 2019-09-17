
# coding: utf-8

# In[1]:


from copy import deepcopy
from math import log, exp, sqrt, fabs, sin, cos, atan2, tan
import sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import cholesky
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.linalg import norm
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import scipy.stats as stats
from filterpy.stats import plot_covariance_ellipse

from scipy.linalg import inv, block_diag


# In[75]:


class UKF:
    def __init__(self, dim_x, dim_z, dt, fx, hx, alpha = 0.1, beta = 2.0, kappa = 1.0, sigma_mode = 1, 
                 augmentation = False, dim_xa = None, xa = None,  
                 x_resid=False, x_resid_indices = None, z_resid=False, z_resid_indices = None):
        """
        Class construction for initializing parameters and attributes
        """
        self.x = zeros(dim_x)  # state, used for prediction and measurement
        self.P = eye(dim_x)    # Covariance, used for prediciton and measurement
        self.x_prior = np.copy(self.x) # Prior state estimate, read only
        self.P_prior = np.copy(self.P) # Prior state covariance matrix, read only
        self.x_post = np.copy(self.x) # Posterior state estimate, read only
        self.P_post = np.copy(self.P) # Posteroir state covariance matrix, read only
        self.Q = eye(dim_x) # Process noise matrix
        self.Qa = None  # Process noise matrix if Sigma points augmentation is implemented
        self.R = eye(dim_z) # Measurement noise matrix
        self.augmentation = augmentation # If augmentation is implemented
        if not augmentation:  # The dimensions of Sigma points can be different, depending on augmentation
            self.sigma_pts = zeros([2*dim_x + 1, dim_x]) # sigma points
            self.sigma_pts_f = zeros([2*dim_x + 1, dim_x]) # sigma points transformed through process function f(x)
            self.sigma_pts_h = zeros([2*dim_x + 1, dim_z]) # sigma points transformed through measurement function h(x)
        else:
            self.sigma_pts = zeros([2*dim_xa + 1, dim_xa]) # sigma points
            self.sigma_pts_f = zeros([2*dim_xa + 1, dim_x]) # sigma points transformed through process function f(x)
            self.sigma_pts_h = zeros([2*dim_xa + 1, dim_z]) # sigma points transformed through measurement function h(x)
            
        self.alpha_ = alpha # alpha, used for calculating sigma points and associated weights
        self.beta_ = beta  # beta, used for calculating sigma points and associated weights
        self.kappa_ = kappa # kappa, used for calculating sigma points and associated weights
        self.sigma_mode = sigma_mode # different way to generate sigma points, currrently only Mode 1 and 2
        self.dim_x = dim_x # dimension of the state
        self.dim_xa = dim_xa # dimension of augmented state
        self.xa = xa #  augmented state vector
        self.dim_z = dim_z # dimension of the measurement
        self.dt = dt # duration of the time step
        self.hx = hx # mesurement function
        self.fx = fx # process function
        self.x_resid = x_resid # adjust the values of some residual of state
        self.x_resid_indices = x_resid_indices # the indices of the resid that need to be adjusted
        self.z_resid = z_resid # adjust the values of some residual of measurement
        self.z_resid_indices = z_resid_indices # the indices of the resid that need to be adjusted
        self.Wc = self.calculate_weights()[0] # weight for calculate covariance
        self.Wm = self.calculate_weights()[1] # weidht for calculate mean
        self.K = np.zeros((dim_x, dim_z))    # Kalman gain
        self.y = np.zeros((dim_z))           # residual
        self.z = np.array([[None]*dim_z]).T  # measurement
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty
    
    def calculate_weights(self):
        """
        Calculate the weights associated with sigma points. The weights depend on parameters dim_x, aplha, beta, 
        and gamma. The number of sigma points required is 2 * dim_x + 1
        """
        # dimension for detemining the number of sigma points generated
        dim = self.dim_x if not self.augmentation else self.dim_xa
        if self.sigma_mode == 1:
            lambda_ = self.alpha_**2 * (dim + self.kappa_) - dim
            Wc = np.full(2*dim + 1,  1. / (2*(dim + lambda_))) # 
            Wm = np.full(2*dim + 1,  1. / (2*(dim + lambda_)))
            Wc[0] = lambda_ / (dim + lambda_) + (1. - self.alpha_**2 + self.beta_)
            Wm[0] = lambda_ / (dim + lambda_)
        elif self.sigma_mode == 2:
            lambda_ = 3 - dim
            Wc = np.full(2*dim + 1, 1./ (2*(dim + lambda_)))
            Wm = np.full(2*dim + 1, 1./ (2*(dim + lambda_)))
            Wc[0], Wm[0] = lambda_ /(dim + lambda_), lambda_ /(dim + lambda_)
        
        return (Wc, Wm)
    
    def update_sigma_pts(self):
        """
        Create (update) Sigma points during the prediction stage
        """
        if not self.augmentation:
            dim, x, P = self.dim_x, self.x, self.P
        else:
            dim = self.dim_xa
            x = np.concatenate([self.x, self.xa])
            P = block_diag(self.P, self.Qa)
        if self.sigma_mode == 1:    
            lambda_ = self.alpha_**2 * (dim + self.kappa_) - dim
        elif self.sigma_mode == 2:
            lambda_ = 3 - dim
        U = cholesky((dim + lambda_) * P) 
        self.sigma_pts[0] = x
        for k in range (dim):
            self.sigma_pts[k + 1]  = x + U[k]
            self.sigma_pts[dim + k + 1] = x - U[k]
            
    def calculate_mean_covariance(self, sigma_pts, M, adjust=False, indices = None):
        """
        Utility functon to calculate the mean and covariance in both the prediciton and update stages
        Inupt: sigma_pts: sigma points transfored by the process or the measurement
               M: process or measurement noise Matrix (Q or R)
               adjust: Boolean, adjust some elements (angle/direction related) of the residual (y) to be in the range of 
                                [-np.pi, np.pi]
               indices: the indices of the elements that need to be adjusted if outside the range of [-np.pi, np.pi]                  
        Out: mean: mean
             cov: covariance
        """
        mean = np.dot(self.Wm, sigma_pts)
        n_sigmas, n = sigma_pts.shape
        cov = zeros((n, n))
        for k in range(n_sigmas):
            y = (sigma_pts[k] - mean)
            if adjust:
                y = self.residual(y, indices)
            y = y.reshape(n, 1) # need to convert into 2D array, for the transpose operation !!!
            cov += self.Wc[k] * np.dot(y, y.T) 
        cov += M
        return (mean, cov)

    def compute_process_sigma_pts(self, input_sigma_pts, **fx_args):
        """
        Calculate the sigam points transformed the process function fx
        Input: 
              input_sigma_pts: input sigma points
              **fx_args: keywords/arguments associated with process/system function defined as fx
        Output:      
              output_sigma_pts: sigma points transformed by the process
        """
        fx, dt = self.fx, self.dt
        n_sigmas, _ = input_sigma_pts.shape
        output_sigma_pts = zeros([n_sigmas, self.dim_x])    
        for i, s in enumerate(input_sigma_pts):
            output_sigma_pts[i] = fx(s, dt, **fx_args)
           
        return output_sigma_pts    
            
    def prediction(self, **fx_args):
        """
        Prediction, calculated the prior state estimate and covariance
        Input:
              **fx_args: keywords/arguments associated with process/system function defined as fx
        """
        fx = self.fx
        self.update_sigma_pts( )  # update the sigma points
        sigma_pts = self.sigma_pts
        
        process_sigma_pts = self.compute_process_sigma_pts(sigma_pts, **fx_args) # sigma points transformed by the process
        self.sigma_pts_f = process_sigma_pts
        
        # mean and covariance of sigma transformed mean and covariance
        if not self.x_resid:
            self.x, self.P = self.calculate_mean_covariance(process_sigma_pts, self.Q)  
        else:
            self.x, self.P = self.calculate_mean_covariance(process_sigma_pts, self.Q, adjust = True, indices = self.x_resid_indices)
        self.x_prior, self.P_prior = np.copy(self.x), np.copy(self.P)
       
        
    
    def compute_measurement_sigma_pts(self, input_sigma_pts, **hx_args):
        """
        Calculate the sigam points transformed by the measurement function hx
        Input: 
              input_sigma_pts: input sigma points
              **hx_args: keywords/arguments associated with measurement function defined in hx
        Output:      
              output_sigma_pts: sigma points transformed by the measurement
        """
        hx = self.hx
        n_sigmas, _ = input_sigma_pts.shape
        dim_z = self.dim_z
        output_sigma_pts = np.zeros([n_sigmas, dim_z])
        for i in range(n_sigmas):
            output_sigma_pts[i] = hx(input_sigma_pts[i], **hx_args)
        return output_sigma_pts    
            
            
            
    def update(self, z, **hx_args):
        
        """
        Update step, calculate the (new) posterior state and covariance
        Input:
             z: measuremnt
             **hx_args: keywords/arguments associated with measurement function defined in hx
        """
        
        hx = self.hx
        sigmas_f = self.sigma_pts_f
        n_sigmas = sigmas_f.shape[0]
        # Transform sigma points from state space to measurement space
        sigmas_h = self.compute_measurement_sigma_pts(sigmas_f, **hx_args) 
        self.sigma_pts_h = sigmas_h
        
        if not self.z_resid:
            zp, Pz = self.calculate_mean_covariance(sigmas_h, self.R)
        else:
            zp, Pz = self.calculate_mean_covariance(sigmas_h, self.R, adjust = True, 
                                                    indices = self.z_resid_indices)
        self.S = Pz
        
        # Compute cross variance of the state and the measurement
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(n_sigmas):
            x_r = sigmas_f[i] - self.x
            if self.x_resid:
                x_r = self.residual(x_r, self.x_resid_indices)
            z_r = sigmas_h[i] - zp
            if self.z_resid:
                z_r = self.residual(z_r, self.z_resid_indices)
            Pxz += self.Wc[i] * outer(x_r, z_r)
        self.SI = inv(Pz)
        
        K = dot(Pxz, inv(Pz)) # Kalman gain
        self.K = K
        
        # New state estimae and covariance maxtrix
        self.y = z - zp
        self.x = self.x + dot(K, z - zp)
        self.P = self.P - dot(K, Pz).dot(K.T)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
        self.z = deepcopy(z)
        
    def residual(self, y, indices):
        """
        Adjust the the element of residual (y) so the value is in the range of [-np.pi, np.pi]
        Input:
             y: the 1D numpy array
             indices: list of the indices of the elements whose value need to be in the range [-np.pi, np.pi]
        """
        y_tmp = y
        for idx in indices:
            while y_tmp[idx] > np.pi:
                  y_tmp[idx] -= 2 * np.pi
            while y_tmp[idx] < -np.pi:
                  y_tmp[idx] += 2 * np.pi     
        return y_tmp             
                    
                


# In[76]:


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
    
    if len(input_state) == 7: # When the effect of nu_a, and nu_yawdd need to be considered
        nu_a = input_state[5]
        nu_yawdd = input_state[6]
        
        px_p += 0.5 * nu_a * cos(yaw) * dt * dt
        py_p += 0.5 * nu_a * sin(yaw) * dt * dt
        
        yaw_p += 0.5 * nu_yawdd * dt * dt;
        yawd_p += nu_yawdd * dt;
         
    output_state=np.array([[px_p, py_p, v_p, yaw_p, yawd_p]])
   
    return output_state




# In[77]:


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


# In[78]:


def lidar(input_state):
    """
    LIDAR measurement model
    Input: CTRV state representation [p_x, p_y, v, yaw, yawd]
    Output: LIDAR measurement [p_x, p_y]
    """
    return input_state[0:2]


# In[79]:


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
        c=0.0001;
    
    rho = c;
    phi = atan2(p_y, p_x);
    rho_dot = (p_x*v_x + p_y*v_y)/c
    
    return np.array([rho, phi, rho_dot])


# In[80]:


gt_np = np.load('gt_car3.npz')['X']
print(gt_np.shape)
lidar_np = np.load('lidar_car3.npz')['X']
print(lidar_np.shape)
print(gt_np[0])
print(lidar_np[0])


# In[81]:


dt = (gt_np[1][0] - gt_np[0][0])/1e6
print(dt)


# In[82]:


ukf = UKFa(dim_x = 5, dim_z = 2, fx = ctrv, hx = lidar,
              dt = dt, alpha =.00001, beta = 2, kappa = 0, sigma_mode = 2,
              augmentation = True, dim_xa = 7, xa = np.array([0, 0]),
              x_resid = True, x_resid_indices = [3], z_resid = True, z_resid_indices = [1])
x0 = np.array([-11, 1, 0, 0, 0])
P0 = np.diag([1, 1, 1, 0.0255, 0.0255])
ukf.x = x0
ukf.P = P0
std_a = 0.5
std_yawdd = 0.3
estimate =[]
Qa = np.diag([std_a * std_a, std_yawdd * std_yawdd])
ukf.Qa = Qa
ukf.Q = np.zeros([5,5])
ukf.R = np.diag([0.15**2, 0.15**2])
for i in range(gt_np.shape[0]):
    ukf.prediction() 
    z0 = np.array(lidar_np[i][2:])
    ukf.update(z0)
    estimate.append([ukf.x_post[0], ukf.x_post[1],ukf.x_post[2]*cos(ukf.x_post[3]), ukf.x_post[2]*sin(ukf.x_post[3])])
estimate_np = np.array(estimate)


# In[65]:


estimate_np.shape


# In[83]:


x_gt = gt_np[:,2]
y_gt = gt_np[:,3]
x_e = estimate_np[:,0]
y_e = estimate_np[:,1]
x_ld = lidar_np[:,2]
y_ld = lidar_np[:,3]
plt.figure(figsize = [6, 10])
plt.plot(y_gt, x_gt)
plt.plot(y_e, x_e)
#plt.plot(y_ld, x_ld)
plt.xlabel("Y")
plt.ylabel("X")
plt.legend(['Ground Truth', 'Estimation', 'Measurement'])
plt.grid()
plt.show()


# In[69]:


plt.plot(x_gt)
plt.plot(x_ld)
plt.plot(x_e)
plt.show()


# In[70]:


plt.plot(y_gt)
plt.plot(y_ld)
plt.plot(y_e)
plt.show()

