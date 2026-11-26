import numpy as np
from numpy.linalg import inv

# Task 3 
# Define Extended Kalman Filter algorithm

# Subtask 1
# Define parameters of Kalman Filter based on what you have learned in the lecture


# Jacobian of measurement model
# JH = np.array([[],
#                []])

# timestep
dt = 0.04

# For acceleration noise std, range [0.1, 1.0]
sigma_a = 1.0 

class EKF(object):
    def __init__(self, x) -> None: # Dunder method-> to initialize the object, None is just type hint indicating this method does not return value

        # initial state
        self.x = np.zeros(4) if x is None else x

        # state transition matrix for constant velocity model
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])


        # Covariance matrix of process noise
        # process_noise_std = np.array([0.1, 0.1, 0.1, np.deg2rad(0.1)])
        # self.Q = np.diag(process_noise_std ** 2)
        # Taking into consideration of acceleration uncertainty influences both position and velocity over time 
        self.Q = np.array([[0.25*dt**4, 0, 0.5*dt**3, 0],
                           [0, 0.25*dt**4, 0, 0.5*dt**3],
                           [0.5*dt**3, 0, dt**2, 0],
                           [0, 0.5*dt**3, 0, dt**2]])*sigma_a**2

        # Initial estimate covariance matrix
        self.P = np.diag([1000,1000,1000,1000]) # Moderate uncertainty

        # Covariance matrix of observation noise
        # Typically in the case of camera sensor, measurement noise variance = 0.25
        self.R = np.diag(np.array([0.5, 0.5])**2)

        # Measurement matrix (position measurement only) in the case of linear measurement model
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Identity matrix
        self.I = np.eye(4)

# Prediction step by Extended Kalman Filter

    def predict_ekf(self, v, h):

       x = self.x   
       Q = self.Q
       P = self.P

       # Jacobian of F (nonlinear):- Linearization
       JF = np.array([[1, 0, dt * np.cos(h), -v * dt * np.sin(h)], 
                      [0, 1, dt * np.sin(h),  v * dt * np.cos(h)],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], np.float64)
       
       # Input matrix B
       B = np.array([[np.cos(h) * dt, 0],
                     [np.sin(h) * dt, 0],
                     [0, dt],
                     [1, 0]], np.float64)
    
       # Define Control vector
       # u = np.array([[v, h]]).T

       # Nonlinear state prediction
       new_x = np.zeros(4)                                  # calculation of state etimate
       
       new_x[0] = x[0] + v * dt * np.cos(h)
       new_x[1] = x[1] + v * dt * np.sin(h)
       new_x[2] = v                                         # assumed Velocity is constant 
       new_x[3] = h                                         # assumed Heading angle is constant

       # Calculation of covariance matrix
       new_P = JF @ P @ JF.T + Q                            

       self.x = new_x
       self.P = new_P

       return self.x

# Update step by Extended Kalman Filter

    def update_ekf(self, coord):
       x = self.x
       P = self.P
       R = self.R
       I = self.I
       H = self.H

       S = H @ P @ H.T + R          # Calculate residual covariance matrix
       K = P @ H.T @ inv(S)         # Calculate Kalman Gain

       z = coord.reshape(2)
       y = z - (H @ x)                    # Calculate Innovation/residual 

       new_x = x + K @ y            # update state estimate
       new_P = (I - K @ H) @ P      # update covariance matrix

       self.x = new_x
       self.P = new_P

       return self.x


# Task 4 (Forecasting with EKF):

# Subtask 1
# Define a function for forecasting with Extended Kalman Filter
# In this task you must do forecasting for each point in measurement data 

def forecast_ekf(f_dt, meas, n):  # f_dt is timestep for forecasting

   u1 = []
   B1 = []

   for i in range(meas.shape[0]):
       u1.append(np.array([[meas[i,2,0], meas[i,3,0]]], np.float64).T)
       arr_u = np.concatenate(u1).reshape(len(u1), 2, 1)

       B1.append(np.array([[np.cos(meas[i,3,0])*f_dt,0],
                           [np.sin(meas[i,3,0])*f_dt,0],
                           [1, 0],
                           [0, f_dt]],np.float64))

#    F = np.array([[1, 0, f_dt, 0],
#                  [0, 1, f_dt, 0],
#                  [0, 0, 1, 0],
#                  [0, 0, 0, 1]])
                
   forecast_list = []
   
   # This for loop iterate over the measurment data, because we want to forecast next step for each individual point in measurement
   
   for m, b, u in zip(meas, B1, arr_u):
             
       lst = []
       current_state = m.copy()
       # This for loop performs the forecasting for n iterations
       for _ in range(n):
        # Write the equation for forecasting in this for loop
           forecast_point = np.zeros(4)
           forecast_point[0] = current_state[0] + current_state[2] * np.cos(current_state[3]) * f_dt
           forecast_point[1] = current_state[1] + current_state[2] * np.sin(current_state[3]) * f_dt
           forecast_point[2] = current_state[2]                                            # assuming constant velocity
           forecast_point[3] = current_state[3]                                            # assuming constant heading

           lst.append(forecast_point)
           current_state = forecast_point
        
       forecast_list.append(lst)

   forecast_array = np.stack(forecast_list)

   return forecast_array
