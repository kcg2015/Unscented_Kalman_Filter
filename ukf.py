import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import sqrt, fabs, sin, cos, atan2
from numpy import eye, zeros, dot, outer
from scipy.linalg import inv, cholesky, block_diag

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
                    
                


