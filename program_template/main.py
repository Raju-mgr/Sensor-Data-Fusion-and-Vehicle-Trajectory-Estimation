from Data_Preparation import *
from Kalman_Filter import *
from Extended_KF import *
import Visualization as vs
from pathlib import Path

# Example Visualization (given later)
# from script_4 import Visualizer

# set base directory
base_dir = Path(r"C:\Users\Joel\Downloads\TUC_Summersemester\Multisensor_Praktikum\dataset")

# Define valid track numbers
valid_tracks = ["08", "26", "32"]

try:
    # Ask for the user input
    track_number = input(f"Enter the track number {valid_tracks}:").zfill(2)

    if track_number not in valid_tracks:
        raise ValueError("Invalid track number selected.")
    
    # Construct file paths
    csv_file = base_dir/f"tracks_{track_number}.csv"
    image_file = base_dir/f"{track_number}_background.png"

    # Ensure existance of files
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file.name}")
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file.name}")
    
    # Read Data
    data = read(csv_file)
    # print(data)

    # Image background path
    image_path = image_file

    # Ask for the user input to choose filter among (Kalman filter and extended Kalman filter)
    # Options are: "kf" or "ekf"
    filter_option = ["kf", "ekf"]
    kf = input(f"Choose the filter {filter_option}:")

    if kf not in filter_option:
        raise ValueError("Invalid filter selected.")
     
except ValueError as ve:
    print("Input Error:", ve)
except FileNotFoundError as fe:
    print("File Error:", fe)
except Exception as e:
    print("Unexpected Error:", e)

# Store measurement, filter data and forecast data of all car_ids of particular track number (Dict type)
all_meas = {}
all_filter_data = {}
all_forecast_data = {}

# Determine which car IDs are present
available_car_ids = sorted(set(int(row[0]) for row in data))   # store in set data type for unique value only in ascending order
print(f"Available car IDs in track {track_number}: {available_car_ids}")

for car_id in available_car_ids:
    try:
        meas = extract_meas(data, kf, car_id)
        all_meas[car_id] = meas

        # Creating an empty list to append filtered data by KF/EKF
        filter_data = []

        # condition if kf is true or false
        if kf == "kf":
             kf_class = KF(x = meas[0])   # Using Linear Kalman Filter

             # Loop through the rows of the measurement array
             for i in range(meas.shape[0]):
                 # Extract position (x,y) and velocity (vx,vy) from the current timestep measurement
                 x,y,vx,vy = meas[i,:,0]

                 # Predict the next state using Kalman Filter
                 filter_data.append(kf_class.predict())
    
                 # Update the current measurement by Kalman Filter
                 kf_class.update(np.array([[x],[y]])) # passing only position x and y as a new measurement 


        else:

             kf_class = EKF (x = meas[0]) # Using Extended Kalman Filter 

             # Loop through the rows of the measurement array
             for i in range(meas.shape[0]):
    
                  # Extract position (x,y) and magnitude of velocity (v) and heading angle (h) from the current timestep measurement
                  x,y,v,h = meas[i,:,0]

                  # Predict the next state using Kalman Filter
                  filter_data.append(kf_class.predict_ekf(v,h))
    
                  # Update the current measurement by Kalman Filter
                  kf_class.update_ekf(np.array([[x],[y]]))

        # Concatenate the list of filtered data into a NumPy array
        filter_data = np.concatenate(filter_data).reshape(len(filter_data), 4, 1)
        # print(f"Filtered data:\n {filter_data}")
        all_filter_data[car_id] = filter_data

        # Task 4, Subtask 1: Forecasting with Linear KF or EKF
        forecasting = True

        # timestep for forcasting
        f_dt = 0.2

        # number of iteration for forecasting 
        n = 5

        if forecasting:

             if kf == "kf":

                 forecast_data = forecast_kf(f_dt, meas, n)
        
             else:
    
                 forecast_data = forecast_ekf(f_dt, meas, n)

                #  print(f"Forcast data:\n {forecast_data}")
             all_forecast_data[car_id] = forecast_data
    except Exception as e:
        import traceback
        print(f"Skipping car ID {car_id} due to error:", e)
        traceback.print_exc()


# Task 5: ADE
# add your ADE calculation for the forecast here 
for car_id in all_meas:
     meas = all_meas[car_id]
     forecast_data = all_forecast_data[car_id]

     # Only use timesteps where ground truth for n steps ahead exits
     valid_len = meas.shape[0] - n
     forecast_positions = forecast_data[:valid_len, :, :2] # (valid_len, n, 2)

     # Ground truth for each forecasted step
     gt_positions = np.array([meas[i+1:i+1+n, :2, 0] for i in range(valid_len)]) # (valid_len, n, 2)

     # Compute Euclidean distances between forecasted and ground truth positions
     displacement_errors = np.linalg.norm(forecast_positions - gt_positions, axis = 2)

     # Compute Average Displacement Error
     ade = np.mean(displacement_errors)
     print(f"Car Id {car_id}:ADE:{ade}")


# Compute visualization
vs.visualizer(all_meas, all_filter_data, all_forecast_data, image_path, kf)



