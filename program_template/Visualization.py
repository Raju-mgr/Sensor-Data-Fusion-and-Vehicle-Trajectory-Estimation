import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation

# Visualisation

# Task 1,Subtask 3 (Visualization of measurements)

# Task 2&3,Subtask 3 (Visualization of filtered data)

# Task 4, Subtask 2 (Visualization of forecasted points)

def visualizer(all_meas, all_filter_data, all_forecast_data, path, kf):
    colors = ['r', 'c', 'm', 'y', 'orange', 'purple', 'pink']
    meters_per_pixel = 0.097752
    intervals = 40  # milliseconds
    
    img = np.array(Image.open(path))
    fig, ax = plt.subplots(figsize=(12, 10))
    
    lines = []
    forecasts = []
    texts = []
    velocities = []
    car_ids = list(all_meas.keys())
    
    def init():
      #  print("Initializing animation...")
       ax.clear()

       # Clear all global state
       lines.clear()
       forecasts.clear()
       texts.clear()
       velocities.clear()

       # Draw background
       ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])
       ax.set_xlim(0, img.shape[1])
       ax.set_ylim(0, img.shape[0])
       ax.set_title(f"Data Visualization using {kf}")
       ax.set_xlabel("Pixel X")
       ax.set_ylabel("Pixel Y")

       for idx, car_id in enumerate(car_ids):
            #  print(f"Initializing car {car_id}...")
             meas = all_meas[car_id]
             filter_meas = all_filter_data[car_id]
             forecast_data = all_forecast_data[car_id]

             x_pixels = meas[:, 0, 0] / meters_per_pixel
             y_pixels_transformed = img.shape[0] + meas[:, 1, 0] / meters_per_pixel

             x_filt = filter_meas[:, 0, 0] / meters_per_pixel
             y_filt = img.shape[0] + filter_meas[:, 1, 0] / meters_per_pixel

             x_forecast = forecast_data[:, :, 0] / meters_per_pixel
             y_forecast = img.shape[0] + forecast_data[:, :, 1] / meters_per_pixel

             # Plot initialization
             meas_line, = ax.plot([], [], color=colors[idx % len(colors)], linestyle='-', label=f'Meas {car_id}')
             filt_line, = ax.plot([], [], 'b-', label=f'Filt {car_id}')
             forecast_line, = ax.plot([], [], 'wo', markersize=3, label=f'Forecast {car_id}')

             text = ax.text(10, img.shape[0] - 10, '', color='white', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

             lines.append({
                  "x_meas": x_pixels,
                  "y_meas": y_pixels_transformed,
                  "meas_line": meas_line,
                  "x_filt": x_filt,
                  "y_filt": y_filt,
                  "filt_line": filt_line
             })

             forecasts.append({
                  "x": x_forecast,
                  "y": y_forecast,
                  "line": forecast_line
             })

             texts.append(text)

             if kf == "ekf":
                   velocities.append(filter_meas[:, 2, 0])

       ax.legend()
       ax.grid(True, alpha=0.3)
      #  print("Initialization complete")
       return []


    def update(frame):
      #   print(f"Processing frame {frame}...")
        all_artists = []

        for idx, car_id in enumerate(car_ids):
             data = lines[idx]
             forecast = forecasts[idx]

             x_meas = data["x_meas"]
             y_meas = data["y_meas"]
             x_filt = data["x_filt"]
             y_filt = data["y_filt"]
             meas_line = data["meas_line"]
             filt_line = data["filt_line"]

             # Clamp indices
             window_size = 20
             start_idx = max( 0, frame - window_size + 1)
             end_idx = min(frame + 1, len(x_meas))

             if frame < len(x_meas):
                   meas_line.set_data(x_meas[start_idx:end_idx], y_meas[start_idx:end_idx])
             all_artists.append(meas_line)

             if frame < len(x_filt):
                   filt_line.set_data(x_filt[start_idx:frame+1], y_filt[start_idx:frame+1])
            #  else:
            #        filt_line.set_data(x_filt, y_filt)
             all_artists.append(filt_line)

             # Forecast
             if frame < forecast["x"].shape[0]:
                   forecast["line"].set_data(forecast["x"][frame], forecast["y"][frame])
            #  else:
            #        forecast["line"].set_data([], [])
             all_artists.append(forecast["line"])

             # Text update
             if frame < len(x_filt):
                   texts[idx].set_position((x_filt[frame] + 30, y_filt[frame] - 5))
                   if kf == "ekf":
                         texts[idx].set_text(f"Car ID: {car_id}\n{velocities[idx][frame]:.2f} m/s")
                   else:
                         texts[idx].set_text(f"Car ID: {car_id}")
            #  else:
            #        texts[idx].set_text("")  # Hide text if out of range

             all_artists.append(texts[idx])

      #   print(f"Frame {frame} processed. Artists count: {len(all_artists)}")
        return all_artists
    
    max_frames = max(len(data) for data in all_meas.values())
    anim = animation.FuncAnimation(
        fig,
        update,
        frames= max_frames,
        interval=intervals,
        blit=True,
        repeat=False,
        init_func=init
    )

    plt.show()
    return anim
