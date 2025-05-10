# import cv2
# import numpy as np

# def detect_lanes(image_path):
#     # Load image
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur to reduce noise
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Apply Canny Edge Detection
#     edges = cv2.Canny(blur, 150, 100)

#     # Define Region of Interest (ROI)
#     height, width = edges.shape
#     mask = np.zeros_like(edges)
#     polygon = np.array([[
#         (0, height), (width, height), (width // 2, height // 2)
#     ]], np.int32)
#     cv2.fillPoly(mask, polygon, 255)
#     roi_edges = cv2.bitwise_and(edges, mask)

#     # Hough Line Transform to detect lane lines
#     lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=200)

#     # Draw detected lines on the original image
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

#     # Show the output
#     cv2.imshow("Lane Detection", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Run lane detection on an image
# detect_lanes("road_image.png")

# import cv2
# import numpy as np

# def detect_lanes_video(video_path, output_path=None):
#     # Open video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Define the codec and create a VideoWriter object if saving is needed
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit when the video ends

#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Apply Canny Edge Detection
#         edges = cv2.Canny(blur, 150, 100)

#         # Define Region of Interest (ROI)
#         mask = np.zeros_like(edges)
#         polygon = np.array([
#             [(0, height), (width, height), (width // 2, height // 2)]
#         ], np.int32)
#         cv2.fillPoly(mask, [polygon], 255)
#         roi_edges = cv2.bitwise_and(edges, mask)

#         # Hough Line Transform to detect lane lines
#         lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=200)

#         # Draw detected lines on the frame
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

#         # Show the processed frame
#         cv2.imshow("Lane Detection", frame)

#         # Write frame to output video if needed
#         if output_path:
#             out.write(frame)

#         # Press 'q' to exit early
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     if output_path:
#         out.release()
#     cv2.destroyAllWindows()

# # Run lane detection on a video
# detect_lanes_video("challenge.mp4", "output_video.mp4")


# import cv2
# import numpy as np

# def detect_lanes_video(video_path, output_path=None):
#     # Open video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Define the codec and create a VideoWriter object if saving is needed
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit when the video ends

#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Apply Canny Edge Detection
#         edges = cv2.Canny(blur, 150, 100)

#         # Define Region of Interest (ROI)
#         mask = np.zeros_like(edges)
#         polygon = np.array([
#     [(100, height), (width - 100, height), (width // 2 + 50, height // 2), (width // 2 - 50, height // 2)]
# ], np.int32)
#         cv2.fillPoly(mask, [polygon], 255)
#         roi_edges = cv2.bitwise_and(edges, mask)

#         # Hough Line Transform to detect lane lines
#         lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=200)

#         # Draw detected lines on the frame
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

#         # Show the processed frame
#         cv2.imshow("Lane Detection", frame)

#         # Write frame to output video if needed
#         if output_path:
#             out.write(frame)

#         # Press 'q' to exit early
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     if output_path:
#         out.release()
#     cv2.destroyAllWindows()

# # Run lane detection on a video
# detect_lanes_video("challenge.mp4", "output_video.mp4")

# CORRECTED CODE

# import cv2
# import numpy as np

# def detect_lanes_video(video_path, output_path=None):
#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit when the video ends

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blur, 50, 150)

#         # Define a trapezoidal ROI to focus on lane area
#         mask = np.zeros_like(edges)
#         roi_points = np.array([
#             [(100, height), (width - 100, height), (width // 2 + 50, height // 2 + 50), (width // 2 - 50, height // 2 + 50)]
#         ], np.int32)
#         cv2.fillPoly(mask, [roi_points], 255)
#         roi_edges = cv2.bitwise_and(edges, mask)

#         # Detect lines using Hough Transform
#         lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)

#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]

#                 # Calculate slope to filter out horizontal lines
#                 slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
#                 if abs(slope) > 0.5:  # Filter out nearly horizontal lines
#                     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

#         cv2.imshow("Lane Detection", frame)

#         if output_path:
#             out.write(frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     if output_path:
#         out.release()
#     cv2.destroyAllWindows()

# # Run lane detection on a video
# detect_lanes_video("challenge.mp4", "output_video.avi")


# import cv2
# import numpy as np

# # Store last detected lane region
# last_lane_region = None

# def apply_lane_mask(frame, lane_lines):
#     """Create a polygon mask for the detected lanes."""
#     global last_lane_region
#     mask = np.zeros_like(frame)

#     if len(lane_lines) < 4:  # Ensure we have at least two points per lane
#         if last_lane_region is not None:
#             lane_lines = last_lane_region  # Use the last detected lane region
#         else:
#             return frame  # No lanes detected yet

#     # Convert list of points to polygon
#     lane_region = np.array([lane_lines], dtype=np.int32)
#     last_lane_region = lane_region  # Store it for future frames

#     # Fill the lane region with white
#     cv2.fillPoly(mask, lane_region, (255, 255, 255))

#     # Apply the mask to extract the lane area
#     lane_only = cv2.bitwise_and(frame, mask)

#     return lane_only

# def detect_lanes_video(video_path, output_path=None):
#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit when video ends

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blur, 50, 150)

#         # Define ROI
#         mask = np.zeros_like(edges)
#         roi_points = np.array([
#             [(100, height), (width - 100, height), (width // 2 + 50, height // 2 + 50), (width // 2 - 50, height // 2 + 50)]
#         ], np.int32)
#         cv2.fillPoly(mask, [roi_points], 255)
#         roi_edges = cv2.bitwise_and(edges, mask)

#         # Detect lane lines
#         lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)

#         lane_lines = []
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
                
#                 if abs(slope) > 0.5:  # Ignore horizontal lines
#                     lane_lines.append((x1, y1))
#                     lane_lines.append((x2, y2))
#                     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green lane lines

#         if lane_lines:
#             segmented_frame = apply_lane_mask(frame, lane_lines)

#             # Overlay purple color only on the detected lane
#             overlay = np.zeros_like(frame)
#             overlay[:] = (128, 0, 128)  # Purple color
#             purple_segment = cv2.bitwise_and(overlay, segmented_frame)

#             # Blend with the original frame
#             frame = cv2.addWeighted(frame, 0.7, purple_segment, 0.3, 0)

#         cv2.imshow("Lane Segmentation", frame)

#         if output_path:
#             out.write(frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     if output_path:
#         out.release()
#     cv2.destroyAllWindows()

# # Run lane detection
# detect_lanes_video("challenge.mp4", "output_video.avi")


# import cv2
# import numpy as np

# def process_frame(frame):
#     """ Process a single frame to detect and segment lanes. """
    
#     height, width = frame.shape[:2]

#     # Convert to HSV for better color segmentation
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Mask yellow and white lanes
#     yellow_lower = np.array([18, 94, 140], np.uint8)
#     yellow_upper = np.array([48, 255, 255], np.uint8)
#     white_lower = np.array([0, 0, 200], np.uint8)
#     white_upper = np.array([255, 50, 255], np.uint8)

#     yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
#     white_mask = cv2.inRange(hsv, white_lower, white_upper)
#     lane_mask = cv2.bitwise_or(yellow_mask, white_mask)

#     # Apply Gaussian Blur to reduce noise
#     blurred = cv2.GaussianBlur(lane_mask, (5, 5), 0)

#     # Use Canny Edge Detection
#     edges = cv2.Canny(blurred, 50, 150)

#     # Define Region of Interest (ROI)
#     mask = np.zeros_like(edges)
#     roi_points = np.array([
#         [(100, height), (width - 100, height), (width // 2 + 50, height // 2), (width // 2 - 50, height // 2)]
#     ], np.int32)
#     cv2.fillPoly(mask, roi_points, 255)

#     # Apply mask to edges
#     roi_edges = cv2.bitwise_and(edges, mask)

#     # Use Hough Transform to detect lines
#     lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
#     line_image = np.zeros_like(frame)

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green lanes

#     # Blend the detected lanes with the original frame
#     output = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

#     return output

# def detect_lanes(video_path, output_path=None):
#     """ Detects and segments lanes in a video. """

#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Stop when video ends

#         # Process frame to get segmented lanes
#         output = process_frame(frame)

#         # Show the result
#         cv2.imshow("Lane Detection", output)

#         if output_path:
#             out.write(output)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     if output_path:
#         out.release()
#     cv2.destroyAllWindows()

# # Run lane detection
# detect_lanes("challenge.mp4", "output_video.avi")




# CURVED LANES DETECTION TOO

# import cv2
# import numpy as np

# # Smoothing using Exponential Moving Average (EMA)
# left_avg, right_avg = None, None
# alpha = 0.2  # Smoothing factor

# def curvature_threshold(poly_coeffs):
#     """Check if the curvature is significant enough to be considered a curve."""
#     if poly_coeffs is None or np.all(poly_coeffs == 0):
#         return False
#     return abs(poly_coeffs[0]) > 1e-6  # Adjust threshold as needed

# def process_frame(frame, prev_left, prev_right):
#     try:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Adaptive Canny Edge Detection
#         median_val = np.median(blurred)
#         lower = int(max(50, 0.8 * median_val))  
#         upper = int(min(255, 1.5 * median_val))

#         edges = cv2.Canny(blurred, lower, upper)

#         height, width = frame.shape[:2]
#         middle_x = width // 2  # Center of the frame

#         # Improved ROI for better lane capture
#         roi_vertices = np.array([[
#         (width * 0.1, height),  
#         (width * 0.9, height),  
#         (width * 0.6, height * 0.55),  
#         (width * 0.4, height * 0.55)
#         ]], dtype=np.int32)

            
#         mask = np.zeros_like(edges)
#         cv2.fillPoly(mask, roi_vertices, 255)
#         masked_edges = cv2.bitwise_and(edges, mask)

#         # Improved Hough Transform Parameters
#         lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=100)

#         left_points, right_points = [], []

#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 slope = (y2 - y1) / (x2 - x1 + 1e-6)
#                 if slope < -0.4 and x1 < middle_x and x2 < middle_x:
#                     left_points.append((x1, y1))
#                     left_points.append((x2, y2))
#                 elif slope > 0.4 and x1 > middle_x and x2 > middle_x:
#                     right_points.append((x1, y1))
#                     right_points.append((x2, y2))

#         def fit_curve(points, prev_avg):
#             if len(points) < 6:
#                 return prev_avg if prev_avg is not None else np.array([0, 0, 0])
#             points = np.array(points)
#             x, y = points[:, 0], points[:, 1]
#             poly_coeffs = np.polyfit(y, x, 2)

#             # Prevent premature bending by keeping the curve straight unless needed
#             if not curvature_threshold(poly_coeffs):
#                 poly_coeffs[0] = 0  # Force the highest order coefficient to 0 for a straight line

#             return alpha * poly_coeffs + (1 - alpha) * prev_avg if prev_avg is not None else poly_coeffs

#         left_curve = fit_curve(left_points, prev_left)
#         right_curve = fit_curve(right_points, prev_right)

#         def draw_curve(frame, poly_coeffs, color, max_y):
#             if poly_coeffs is None or np.all(poly_coeffs == 0):
#                 return
#             y_vals = np.linspace(max_y, height, num=50)
#             x_vals = np.polyval(poly_coeffs, y_vals)
#             x_vals = np.clip(x_vals, 0, width - 1)
#             points = np.array([np.column_stack((x_vals, y_vals))], dtype=np.int32)
#             cv2.polylines(frame, points, isClosed=False, color=color, thickness=5)

#         max_y = height * 0.6
#         draw_curve(frame, left_curve, (0, 255, 0), max_y)
#         draw_curve(frame, right_curve, (0, 255, 0), max_y)
    
#     except Exception as e:
#         print(f"Error in processing frame: {e}")
#         left_curve, right_curve = prev_left, prev_right

#     return frame, left_curve, right_curve

# cap = cv2.VideoCapture("harder_challenge_video.mp4")
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_delay = max(1, int(1000 / fps))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         break

#     processed_frame, left_avg, right_avg = process_frame(frame, left_avg, right_avg)
#     cv2.imshow("Lane Detection", processed_frame)

#     if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



########################  KALMAN FILTER #########################################################

# import cv2
# import numpy as np
    
# # --- Kalman Filter Class ---
# class KalmanFilter1D:
#     def __init__(self, initial_state, process_noise=1e-5, measurement_noise=1e-2):
#         self.state = np.array(initial_state, dtype=np.float32)
#         self.P = np.eye(len(initial_state))
#         self.Q = np.eye(len(initial_state)) * process_noise
#         self.R = np.eye(len(initial_state)) * measurement_noise
#         self.I = np.eye(len(initial_state))

#     def predict(self):
#         self.P = self.P + self.Q
#         return self.state

#     def correct(self, measurement):
#         measurement = np.array(measurement, dtype=np.float32)
#         S = self.P + self.R
#         K = self.P @ np.linalg.inv(S)
#         y = measurement - self.state
#         self.state = self.state + K @ y
#         self.P = (self.I - K) @ self.P
#         return self.state

# # --- Globals for Kalman Filters ---
# left_kf = None
# right_kf = None

# def curvature_threshold(poly_coeffs):
#     if poly_coeffs is None or np.all(poly_coeffs == 0):
#         return False
#     return abs(poly_coeffs[0]) > 1e-6

# def average_lane_line(lines):
#     if not lines:
#         return None
#     x_coords, y_coords = [], []
#     for x1, y1, x2, y2 in lines:
#         x_coords += [x1, x2]
#         y_coords += [y1, y2]
#     if len(x_coords) == 0:
#         return None
#     poly = np.polyfit(y_coords, x_coords, deg=2)  # Quadratic fit
#     return poly

# def draw_lane_line(frame, poly, color=(0, 255, 0)):
#     height = frame.shape[0]
#     if poly is not None:
#         y_vals = np.linspace(height * 0.6, height, num=100)
#         x_vals = np.polyval(poly, y_vals)
#         points = np.array(list(zip(x_vals, y_vals)), dtype=np.int32)
#         cv2.polylines(frame, [points], isClosed=False, color=color, thickness=4)

# def process_frame(frame):
#     global left_kf, right_kf

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Adaptive Canny Edge Detection
#     median_val = np.median(blurred)
#     lower = int(max(0, 0.66 * median_val))
#     upper = int(min(255, 1.33 * median_val))
#     edges = cv2.Canny(blurred, lower, upper)

#     height, width = frame.shape[:2]
#     middle_x = width // 2

#     roi_vertices = np.array([[
#         (50, height),
#         (width - 50, height),
#         (int(width * 0.65), int(height * 0.6)),
#         (int(width * 0.35), int(height * 0.6))
#     ]], dtype=np.int32)

#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, roi_vertices, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)

#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=25, minLineLength=40, maxLineGap=150)

#     left_lines, right_lines = [], []

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2 - y1) / (x2 - x1 + 1e-6)
#             if abs(slope) < 0.5:
#                 continue
#             if slope < 0 and x1 < middle_x and x2 < middle_x:
#                 left_lines.append((x1, y1, x2, y2))
#             elif slope > 0 and x1 > middle_x and x2 > middle_x:
#                 right_lines.append((x1, y1, x2, y2))

#     left_fit = average_lane_line(left_lines)
#     right_fit = average_lane_line(right_lines)

#     # Kalman Filter Setup
#     if left_fit is not None:
#         if left_kf is None:
#             left_kf = KalmanFilter1D(left_fit)
#         left_fit = left_kf.correct(left_fit)
#     elif left_kf is not None:
#         left_fit = left_kf.predict()

#     if right_fit is not None:
#         if right_kf is None:
#             right_kf = KalmanFilter1D(right_fit)
#         right_fit = right_kf.correct(right_fit)
#     elif right_kf is not None:
#         right_fit = right_kf.predict()

#     draw_lane_line(frame, left_fit, color=(255, 0, 0))   # Blue for left lane
#     draw_lane_line(frame, right_fit, color=(0, 255, 0))  # Green for right lane

#     return frame

# # --- Optional Testing ---
# if __name__ == "__main__":
#     cap = cv2.VideoCapture("80395-572395743_small.mp4")  # Replace with 0 for webcam

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         processed = process_frame(frame)
#         cv2.imshow("Lane Detection with Kalman Filter", processed)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # --- Kalman Filter Class ---
# class KalmanFilter1D:
#     def __init__(self, initial_state, process_noise=1e-5, measurement_noise=1e-2):
#         self.state = np.array(initial_state, dtype=np.float32)
#         self.P = np.eye(len(initial_state))
#         self.Q = np.eye(len(initial_state)) * process_noise
#         self.R = np.eye(len(initial_state)) * measurement_noise
#         self.I = np.eye(len(initial_state))

#     def predict(self):
#         self.P = self.P + self.Q
#         return self.state

#     def correct(self, measurement):
#         measurement = np.array(measurement, dtype=np.float32)
#         S = self.P + self.R
#         K = self.P @ np.linalg.inv(S)
#         y = measurement - self.state
#         self.state = self.state + K @ y
#         self.P = (self.I - K) @ self.P
#         return self.state

# # --- Globals for Kalman Filters ---
# left_kf = None
# right_kf = None

# # --- Helper Functions ---
# def curvature_threshold(poly_coeffs):
#     if poly_coeffs is None or np.all(poly_coeffs == 0):
#         return False
#     return abs(poly_coeffs[0]) > 1e-6  # Check the curvature (quadratic term)

# import warnings
# from numpy.polynomial import Polynomial
# from numpy import RankWarning

# def average_lane_line(lines, degree=1):
#     if not lines or len(lines) < 2:
#         return None

#     x_coords, y_coords = [], []
#     for x1, y1, x2, y2 in lines:
#         x_coords.extend([x1, x2])
#         y_coords.extend([y1, y2])

#     if len(set(y_coords)) < degree + 1 or len(set(x_coords)) < degree + 1:
#         return None  # Not enough unique points

#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter('error', RankWarning)
#             poly = np.polyfit(y_coords, x_coords, deg=degree)
#         return poly
#     except RankWarning:
#         return None


# def draw_lane_line(frame, poly, color=(0, 255, 0)):
#     height = frame.shape[0]
#     if poly is not None:
#         y_vals = np.linspace(height * 0.6, height, num=100)
#         x_vals = np.polyval(poly, y_vals)
#         points = np.array(list(zip(x_vals, y_vals)), dtype=np.int32)
#         cv2.polylines(frame, [points], isClosed=False, color=color, thickness=4)

# # --- Frame Processing ---
# def process_frame(frame):
#     global left_kf, right_kf

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Adaptive Canny Edge Detection
#     median_val = np.median(blurred)
#     lower = int(max(0, 0.66 * median_val))
#     upper = int(min(255, 1.33 * median_val))
#     edges = cv2.Canny(blurred, lower, upper)

#     height, width = frame.shape[:2]
#     middle_x = width // 2

#     roi_vertices = np.array([[
#         (50, height),
#         (width - 50, height),
#         (int(width * 0.65), int(height * 0.6)),
#         (int(width * 0.35), int(height * 0.6))
#     ]], dtype=np.int32)

#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, roi_vertices, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)

#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=25, minLineLength=40, maxLineGap=150)

#     left_lines, right_lines = [], []

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2 - y1) / (x2 - x1 + 1e-6)
#             if abs(slope) < 0.5:
#                 continue
#             if slope < 0 and x1 < middle_x and x2 < middle_x:
#                 left_lines.append((x1, y1, x2, y2))
#             elif slope > 0 and x1 > middle_x and x2 > middle_x:
#                 right_lines.append((x1, y1, x2, y2))

#     # Compute polynomial fit and apply curvature threshold
#     left_poly = average_lane_line(left_lines)
#     left_degree = 2 if curvature_threshold(left_poly) else 1
#     left_fit = average_lane_line(left_lines, degree=left_degree)

#     right_poly = average_lane_line(right_lines)
#     right_degree = 2 if curvature_threshold(right_poly) else 1
#     right_fit = average_lane_line(right_lines, degree=right_degree)

#     # Kalman Filter Smoothing
#     if left_fit is not None:
#         if left_kf is None:
#             left_kf = KalmanFilter1D(left_fit)
#         left_fit = left_kf.correct(left_fit)
#     elif left_kf is not None:
#         left_fit = left_kf.predict()

#     if right_fit is not None:
#         if right_kf is None:
#             right_kf = KalmanFilter1D(right_fit)
#         right_fit = right_kf.correct(right_fit)
#     elif right_kf is not None:
#         right_fit = right_kf.predict()

#     draw_lane_line(frame, left_fit, color=(255, 0, 0))   # Blue for left lane
#     draw_lane_line(frame, right_fit, color=(0, 255, 0))  # Green for right lane

#     return frame

# # --- Main Loop ---
# if __name__ == "__main__":
#     cap = cv2.VideoCapture("harder_challenge_video.mp4")  # Replace with 0 for webcam

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         processed = process_frame(frame)
#         cv2.imshow("Lane Detection with Kalman Filter", processed)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()




#########################################


# import cv2
# import numpy as np
# import warnings
# from numpy.polynomial import Polynomial
# from numpy import RankWarning

# # --- Kalman Filter Class ---
# class KalmanFilter1D:
#     def __init__(self, initial_state, process_noise=1e-5, measurement_noise=1e-2):
#         self.state = np.array(initial_state, dtype=np.float32)
#         self.P = np.eye(len(initial_state))
#         self.Q = np.eye(len(initial_state)) * process_noise
#         self.R = np.eye(len(initial_state)) * measurement_noise
#         self.I = np.eye(len(initial_state))

#     def predict(self):
#         self.P = self.P + self.Q
#         return self.state

#     def correct(self, measurement):
#         measurement = np.array(measurement, dtype=np.float32)
#         S = self.P + self.R
#         K = self.P @ np.linalg.inv(S)
#         y = measurement - self.state
#         self.state = self.state + K @ y
#         self.P = (self.I - K) @ self.P
#         return self.state

# # --- Globals ---
# left_kf = None
# right_kf = None

# # --- Lane helpers ---
# def curvature_threshold(poly_coeffs):
#     if poly_coeffs is None or np.all(poly_coeffs == 0):
#         return False
#     return abs(poly_coeffs[0]) > 1e-6

# def average_lane_line(lines, degree=1):
#     if not lines or len(lines) < 2:
#         return None

#     x_coords, y_coords = [], []
#     for x1, y1, x2, y2 in lines:
#         x_coords.extend([x1, x2])
#         y_coords.extend([y1, y2])

#     if len(set(y_coords)) < degree + 1 or len(set(x_coords)) < degree + 1:
#         return None

#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter('error', RankWarning)
#             poly = np.polyfit(y_coords, x_coords, deg=degree)
#         return poly
#     except RankWarning:
#         return None

# def draw_lane_line(frame, poly, color=(0, 255, 0)):
#     height = frame.shape[0]
#     if poly is not None:
#         y_vals = np.linspace(height * 0.6, height, num=100)
#         x_vals = np.polyval(poly, y_vals)
#         points = np.array(list(zip(x_vals, y_vals)), dtype=np.int32)
#         cv2.polylines(frame, [points], isClosed=False, color=color, thickness=4)

# def perspective_transform(frame):
#     height, width = frame.shape[:2]
#     src = np.float32([
#         [width * 0.45, height * 0.63],
#         [width * 0.55, height * 0.63],
#         [width - 50, height],
#         [50, height]
#     ])
#     dst = np.float32([
#         [width * 0.25, 0],
#         [width * 0.75, 0],
#         [width * 0.75, height],
#         [width * 0.25, height]
#     ])
#     M = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(frame, M, (width, height))
#     return warped

# def color_filter(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     white_mask = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))
#     yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
#     combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
#     return cv2.bitwise_and(frame, frame, mask=combined_mask)

# # --- Frame Processing ---
# def process_frame(frame):
#     global left_kf, right_kf

#     frame = perspective_transform(frame)
#     filtered = color_filter(frame)
#     gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Adaptive Canny Edge Detection
#     median_val = np.median(blurred)
#     lower = int(max(0, 0.66 * median_val))
#     upper = int(min(255, 1.33 * median_val))
#     edges = cv2.Canny(blurred, lower, upper)

#     height, width = frame.shape[:2]
#     middle_x = width // 2

#     roi_vertices = np.array([[ 
#         (50, height),
#         (width - 50, height),
#         (int(width * 0.65), int(height * 0.6)),
#         (int(width * 0.35), int(height * 0.6))
#     ]], dtype=np.int32)

#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, roi_vertices, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)

#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=60, maxLineGap=100)
#     left_lines, right_lines = [], []

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2 - y1) / (x2 - x1 + 1e-6)
#             if abs(slope) < 0.5:
#                 continue
#             if slope < 0 and x1 < middle_x and x2 < middle_x:
#                 left_lines.append((x1, y1, x2, y2))
#             elif slope > 0 and x1 > middle_x and x2 > middle_x:
#                 right_lines.append((x1, y1, x2, y2))

#     # Polynomial fitting with adaptive degree
#     left_poly = average_lane_line(left_lines)
#     left_degree = 2 if curvature_threshold(left_poly) else 1
#     left_fit = average_lane_line(left_lines, degree=left_degree)

#     right_poly = average_lane_line(right_lines)
#     right_degree = 2 if curvature_threshold(right_poly) else 1
#     right_fit = average_lane_line(right_lines, degree=right_degree)

#     # Kalman Filter Smoothing
#     if left_fit is not None:
#         if left_kf is None:
#             left_kf = KalmanFilter1D(left_fit)
#         left_fit = left_kf.correct(left_fit)
#     elif left_kf is not None:
#         left_fit = left_kf.predict()

#     if right_fit is not None:
#         if right_kf is None:
#             right_kf = KalmanFilter1D(right_fit)
#         right_fit = right_kf.correct(right_fit)
#     elif right_kf is not None:
#         right_fit = right_kf.predict()

#     draw_lane_line(frame, left_fit, color=(255, 0, 0))   # Blue for left lane
#     draw_lane_line(frame, right_fit, color=(0, 255, 0))  # Green for right lane

#     return frame

# # --- Main Loop ---
# if __name__ == "__main__":
#     cap = cv2.VideoCapture("challenge_video.mp4")  # Use 0 for webcam

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         processed = process_frame(frame)
#         cv2.imshow("Lane Detection Improved", processed)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()





################# COMBINATION OF BOTH OPENCV AND KALMAN FILTER ##############################


# import cv2
# import numpy as np

# # EMA smoothing buffers
# left_fit_avg, right_fit_avg = None, None
# alpha = 0.2

# def region_of_interest(img):
#     height, width = img.shape[:2]
#     mask = np.zeros_like(img)
#     polygon = np.array([[
#         (int(0.1 * width), height),
#         (int(0.45 * width), int(0.6 * height)),
#         (int(0.55 * width), int(0.6 * height)),
#         (int(0.9 * width), height)
#     ]], np.int32)
#     cv2.fillPoly(mask, polygon, 255)
#     return cv2.bitwise_and(img, mask)

# def canny_edge(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)
#     return edges

# def average_slope_intercept(lines):
#     left_lines, right_lines = [], []

#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         parameters = np.polyfit((x1, x2), (y1, y2), 1)  # fit line: y = mx + b
#         slope, intercept = parameters
#         if slope < 0:
#             left_lines.append((slope, intercept))
#         else:
#             right_lines.append((slope, intercept))

#     left_fit = np.mean(left_lines, axis=0) if left_lines else None
#     right_fit = np.mean(right_lines, axis=0) if right_lines else None

#     return left_fit, right_fit

# def apply_ema(avg_fit, new_fit):
#     if avg_fit is None:
#         return new_fit
#     return avg_fit * (1 - alpha) + new_fit * alpha

# def make_coordinates(img, line_params):
#     slope, intercept = line_params
#     y1 = img.shape[0]
#     y2 = int(y1 * 0.6)

#     if slope == 0:
#         slope = 0.1  # Avoid division by zero

#     x1 = int((y1 - intercept) / slope)
#     x2 = int((y2 - intercept) / slope)

#     return np.array([x1, y1, x2, y2])

# def display_lines(img, lines):
#     line_image = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             if line is not None:
#                 x1, y1, x2, y2 = line
#                 cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
#     return line_image

# def process_frame(frame):
#     global left_fit_avg, right_fit_avg
#     roi = region_of_interest(frame)
#     edges = canny_edge(roi)
#     lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 50,
#                             minLineLength=40, maxLineGap=100)

#     if lines is not None:
#         left_fit, right_fit = average_slope_intercept(lines)

#         if left_fit is not None:
#             left_fit_avg = apply_ema(left_fit_avg, left_fit)
#         if right_fit is not None:
#             right_fit_avg = apply_ema(right_fit_avg, right_fit)

#         left_line = make_coordinates(frame, left_fit_avg) if left_fit_avg is not None else None
#         right_line = make_coordinates(frame, right_fit_avg) if right_fit_avg is not None else None

#         line_image = display_lines(frame, [left_line, right_line])
#         final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#         return final_image

#     return frame
# cap = cv2.VideoCapture('harder_challenge_video.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     result = process_frame(frame)
#     cv2.imshow('Lane Detection', result)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#########################  recent code ######################

# import cv2
# import numpy as np

# # --- Kalman Filter Class ---
# class KalmanFilter1D:
#     def __init__(self, state, process_noise=1e-5, measurement_noise=1e-2):
#         self.state = np.array(state, dtype=np.float32)
#         self.P = np.eye(len(state))
#         self.Q = np.eye(len(state)) * process_noise
#         self.R = np.eye(len(state)) * measurement_noise
#         self.I = np.eye(len(state))

#     def predict(self):
#         self.P += self.Q
#         return self.state

#     def correct(self, measurement):
#         measurement = np.array(measurement, dtype=np.float32)
#         S = self.P + self.R
#         K = self.P @ np.linalg.inv(S)
#         y = measurement - self.state
#         self.state = self.state + K @ y
#         self.P = (self.I - K) @ self.P
#         return self.state

# # --- Globals for Kalman Filters ---
# left_kf = None
# right_kf = None

# # --- Lane Detection Utilities ---
# def region_of_interest(img):
#     height, width = img.shape[:2]
#     mask = np.zeros_like(img)
#     polygon = np.array([[ 
#         (int(0.1 * width), height), 
#         (int(0.45 * width), int(0.6 * height)),
#         (int(0.55 * width), int(0.6 * height)),
#         (int(0.9 * width), height)
#     ]], np.int32)
#     cv2.fillPoly(mask, polygon, 255)
#     return cv2.bitwise_and(img, mask)

# def canny_edge(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)
#     return edges

# def average_slope_intercept(lines):
#     left_lines, right_lines = [], []

#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         if x2 - x1 == 0:
#             continue
#         slope = (y2 - y1) / (x2 - x1)
#         intercept = y1 - slope * x1
#         if slope < -0.5:
#             left_lines.append((slope, intercept))
#         elif slope > 0.5:
#             right_lines.append((slope, intercept))

#     left_avg = np.mean(left_lines, axis=0) if left_lines else None
#     right_avg = np.mean(right_lines, axis=0) if right_lines else None

#     return left_avg, right_avg

# def make_coordinates(img, line_params):
#     if line_params is None:
#         return None
#     slope, intercept = line_params
#     y1 = img.shape[0]
#     y2 = int(y1 * 0.6)
#     if slope == 0:
#         slope = 1e-6
#     x1 = int((y1 - intercept) / slope)
#     x2 = int((y2 - intercept) / slope)
#     return np.array([x1, y1, x2, y2])

# def display_lines(img, lines):
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             if line is not None:
#                 x1, y1, x2, y2 = line
#                 cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)
#     return line_img

# # --- Frame Processing ---
# def process_frame(frame):
#     global left_kf, right_kf

#     edges = canny_edge(frame)
#     roi = region_of_interest(edges)
#     lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 50, minLineLength=40, maxLineGap=150)

#     if lines is not None:
#         left_fit, right_fit = average_slope_intercept(lines)

#         # Initialize Kalman Filters if needed
#         if left_fit is not None:
#             if left_kf is None:
#                 left_kf = KalmanFilter1D(left_fit)
#             left_fit = left_kf.correct(left_fit)
#         elif left_kf is not None:
#             left_fit = left_kf.predict()

#         if right_fit is not None:
#             if right_kf is None:
#                 right_kf = KalmanFilter1D(right_fit)
#             right_fit = right_kf.correct(right_fit)
#         elif right_kf is not None:
#             right_fit = right_kf.predict()

#         left_line = make_coordinates(frame, left_fit)
#         right_line = make_coordinates(frame, right_fit)
#         line_img = display_lines(frame, [left_line, right_line])
#         combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
#         return combo
#     else:
#         return frame

# # --- Main Loop ---
# cap = cv2.VideoCapture("harder_challenge_video.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     output = process_frame(frame)
#     cv2.imshow("Lane Detection with Kalman", output)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # --- Kalman Filter Class ---
# class KalmanFilter1D:
#     def __init__(self, state, process_noise=1e-5, measurement_noise=1e-2):   
#         self.state = np.array(state, dtype=np.float32)
#         self.P = np.eye(len(state))
#         self.Q = np.eye(len(state)) * process_noise
#         self.R = np.eye(len(state)) * measurement_noise
#         self.I = np.eye(len(state))

#     def predict(self):
#         self.P += self.Q
#         return self.state

#     def correct(self, measurement):
#         measurement = np.array(measurement, dtype=np.float32)
#         S = self.P + self.R
#         K = self.P @ np.linalg.inv(S)
#         y = measurement - self.state
#         self.state = self.state + K @ y
#         self.P = (self.I - K) @ self.P
#         return self.state

# # --- Globals for Kalman Filters ---
# left_kf = None
# right_kf = None

# # --- Lane Detection Utilities ---
# def region_of_interest(img):
#     height, width = img.shape[:2]   
#     mask = np.zeros_like(img)
#     polygon = np.array([[ 
#         (int(0.1 * width), height), 
#         (int(0.45 * width), int(0.6 * height)),
#         (int(0.55 * width), int(0.6 * height)),
#         (int(0.9 * width), height)
#     ]], np.int32)
#     cv2.fillPoly(mask, polygon, 255)
#     return cv2.bitwise_and(img, mask)

# def canny_edge(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
#     # Adaptive Canny Edge Detection                 
#     median_val = np.median(blur)                                # median_val = np.median(blur)    
#     lower = int(max(0, 0.70 * median_val))                      # lower = int(max(0, 0.66 * median_val)) 
#     upper = int(min(255, 1.37 * median_val))                    #  upper = int(min(255, 1.33 * median_val)) 
#     edges = cv2.Canny(blur, lower, upper)                       # edges = cv2.Canny(blur, lower, upper)
                     
#     height, width = frame.shape[:2]
#     # middle_x = width // 2  # Center of the frame

#     # Improved ROI for better lane capture
#     roi_vertices = np.array([[                                     # roi_vertices = np.array([[    
#             (50, height),                                            #             (50, height),   
#             (width - 50, height),                                     #          (width - 50, height), 
#             (width * 0.69, height * 1.0),  
#             (width * 0.39, height * 1.0)                               #                 (width * 0.35, height * 0.6)
#         ]], dtype=np.int32)                                             #            ]], dtype=np.int32)'''

#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, roi_vertices, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)

#     return edges

# def average_slope_intercept(lines):
#     left_lines, right_lines = [], []

#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         if x2 - x1 == 0:
#             continue
#         slope = (y2 - y1) / (x2 - x1)
#         intercept = y1 - slope * x1
#         if slope < -0.80:       # -0.50 by default
#             left_lines.append((slope, intercept))
#         elif slope > 0.8:        # 0.5 by default
#             right_lines.append((slope, intercept))

#     left_avg = np.mean(left_lines, axis=0) if left_lines else None
#     right_avg = np.mean(right_lines, axis=0) if right_lines else None

#     return left_avg, right_avg

# def make_coordinates(img, line_params):
#     if line_params is None:
#         return None
#     slope, intercept = line_params
#     y1 = img.shape[0]
#     y2 = int(y1 * 0.8)       #0.6  by default 
#     if slope == 0:
#         slope = 1e-6         
#     x1 = int((y1 - intercept) / slope)
#     x2 = int((y2 - intercept) / slope)
#     return np.array([x1, y1, x2, y2])

# def display_lines(img, lines):
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             if line is not None:
#                 x1, y1, x2, y2 = line
#                 cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)
#     return line_img

# # --- Frame Processing ---
# def process_frame(frame):
#     global left_kf, right_kf

#     edges = canny_edge(frame)
#     roi = region_of_interest(edges)
#     lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 50, minLineLength=30, maxLineGap=150)  #lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 50, minLineLength=40, maxLineGap=150)

#     if lines is not None:
#         left_fit, right_fit = average_slope_intercept(lines)

#         # Initialize Kalman Filters if needed
#         if left_fit is not None:
#             if left_kf is None:
#                 left_kf = KalmanFilter1D(left_fit)
#             left_fit = left_kf.correct(left_fit)
#         elif left_kf is not None:
#             left_fit = left_kf.predict()

#         if right_fit is not None:
#             if right_kf is None:
#                 right_kf = KalmanFilter1D(right_fit)
#             right_fit = right_kf.correct(right_fit)
#         elif right_kf is not None:
#             right_fit = right_kf.predict()

#         left_line = make_coordinates(frame, left_fit)
#         right_line = make_coordinates(frame, right_fit)
#         line_img = display_lines(frame, [left_line, right_line])
#         combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)  #combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
#         return combo
#     else:
#         return frame

# # --- Main Loop ---
# cap = cv2.VideoCapture("harder_challenge_video.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     output = process_frame(frame)
#     cv2.imshow("Lane Detection with Kalman", output)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



