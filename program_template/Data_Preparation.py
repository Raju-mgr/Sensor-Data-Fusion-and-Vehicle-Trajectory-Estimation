# importing libraries 
import numpy as np 
import itertools

# Task 1 ( Reading the dataset )

# Subtask 1

def read(file):

    # use a function to read the data (e.g. with numpy)
    data = np.loadtxt(file, delimiter=',', skiprows = 0, dtype = float)
    data = [tuple(row) for row in data]

    return data

# Subtask 2
# Define a function that take the data as an input and creates measurements as a 2d Numpy array

def extract_meas(data, kf, car_id):

    # Group data by ID
    groups = itertools.groupby(data, key=lambda x: x[0])
    
    # make a condition whether its kf or ekf
    if kf == "kf" :

        # Extract elements that you need for the measurements (based on lecture, space state model for KF)
        
        lst = [[[item[2], item[3], item[4], item[5]] for item in group] for key, group in groups]

        # Extracts ego car (first car) and reshape the measurements                                                                                  
        meas = np.array(lst[car_id-1]).reshape(len(lst[car_id-1]), 4, 1)
        # hint: make sure the extracted data array looks exactly as in the instructions                                  

        return meas 

    else:
        # Do the same method for EKF,you also need heading value for calculation  

        lst = [[[item[2], item[3], item[4], item[5], item[6]] for item in group] for key, group in groups]
        
        # Extract elements from lst 
        x, y, vx, vy, h = np.array(lst[car_id-1]).T
        
        # Calculate the magnitude of velocity
        mag_v = np.sqrt(vx.astype(float)**2+vy.astype(float)**2)

        # Convert the heading angle value to radian
        h = np.deg2rad(h.astype(float))                                                   
        
        # Creates measurements array based on space state model for EKF then reshapes the data
        meas =  np.column_stack((x, y, mag_v, h)).reshape(len(x), 4, 1)
        # hint: make sure the extracted data array looks exactly as in the instructions                                     

        return  meas
 





