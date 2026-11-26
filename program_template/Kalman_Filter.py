import numpy as np 
from numpy.linalg import inv

# Task 2 
# Implement Linear Kalman Filter algorithm

# Subtask 1
# Define variables and algorithm of Linear Kalman Filter based on what you have learned in the lecture

# Timestep
dt = 0.04

# For acceleration noise std, range [0.1, 1.0]
sigma_a = 1.0 

class KF(object):
    def __init__(self, x) -> None:

        # initial state
         self.x = np.zeros(4) if x is None else x  
         
        # Covariance matrix P of state estimate
         self.P = np.diag([1000,1000,1000,1000]) 

        # State transition matrix F for constant velocity model
         self.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1,  0],
                            [0, 0, 0,  1]])

        # Covariance matrix of process noise 
        #self.Q = np.diag(np.array([0.1,0.1,0.1,0.1])**2)
        # Taking into consideration of acceleration uncertainty influences both position and velocity over time 
         self.Q = np.array([[0.25*dt**4, 0, 0.5*dt**3, 0],
                            [0, 0.25*dt**4, 0, 0.5*dt**3],
                            [0.5*dt**3, 0, dt**2, 0],
                            [0, 0.5*dt**3, 0, dt**2]])*sigma_a**2

        # Measurement matrix H - measurement measures positions x and y only
         self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Identity matrix
         self.I = np.eye(4)
        
        # Input matrix B (for acceleration input affecting positions and velocities)
         self.B = np.array([[(dt**(2))/2, 0],
                            [0, (dt**(2))/2],
                            [dt, 0],
                            [0, dt]])
         
         # self.B = np.array([[dt, 0],
         #                    [0, dt],
         #                    [1, 0],
         #                    [0, 1]])
        
        # Covariance matrix of measurement noise
        # Typically in the case of camera sensor, measurement noise variance = 0.25
         self.R = np.diag(np.array([0.5, 0.5])**2)

    # Prediction step by Kalman Filter

    def predict(self):
       x = self.x
       F = self.F
       P = self.P
       Q = self.Q
       B = self.B

       # Define control input vector acceleration in x and y
       # Control vector which affects the state transition, in this case set to zero due to unknown acceleration value
      #  u = np.array([[0, 0]]).T       

       new_x = F @ x    # Predict new state estimate                     
       new_P = F @ P @ F.T + Q  # Predict new covariance matrix

       self.x = new_x
       self.P = new_P

       return self.x

    # Update step by Kalman Filter

    def update(self, coord):
       
       x = self.x
       H = self.H
       R = self.R
       P = self.P
       I = self.I 

       S = H @ P @ H.T + R          # Calculate covariance matrix of residual
       K = P @ H.T @ inv(S)         # Calculate Kalman Gain
       z = coord                    # Taking new measurement only position x and y
       y = z - (H @ x)              # Calculate Innovation / Residual

       new_x = x + K @ y            # updated state estimate
       new_P = (I - K @ H) @ P      # equivalent formula (P - K @ S @ K.T) 

       self.x = new_x
       self.P = new_P

       return self.x

# Task 4 (Forecasting with linear KF):

# Subtask 1
# Define a function for forecasting with Linear Kalman Filter 
# In this task you must do forecasting for each point in measurement data

def forecast_kf(f_dt, meas, n): # f_dt is dt for forcasting
    # State transition matrix for forecasting time step f_dt

    F = np.array([[1, 0, f_dt, 0],
                  [0, 1, 0, f_dt],
                  [0, 0, 1,    0],
                  [0, 0, 0,    1]])

    
   #  B = np.array([[(f_dt**(2))/2, 0],
   #               [0, (f_dt**(2))/2 ],
   #               [f_dt, 0],
   #               [0, f_dt]])

    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    
    
    u1 = []
    for i in range(meas.shape[0]):
               # vx = meas[i, 2, 0]
               # vy = meas[i, 3, 0]
               u1.append(np.array([[0, 0]]).T) 
               arr_u = np.concatenate(u1).reshape(len(u1), 2, 1)
              

    forecast_list = []

    # This for loop iterate over the measurment data, because we want to forecast next step for each individual point in measurement
    # zip is being used to iterate over two iterables (measurement points and their corresponding control inputs)
    for m, u in zip(meas, arr_u):                       
        lst = []
        state = m.copy()

        # This for loop performs forecasting for n iterations
        for _ in range(n):
            # write the equation for forecasting in for loop
            forecast_point = F @ state + B @ u
            lst.append(forecast_point.flatten())
            state = forecast_point
        
        forecast_list.append(lst)
        
    forecast_array = np.stack(forecast_list)

    return forecast_array
