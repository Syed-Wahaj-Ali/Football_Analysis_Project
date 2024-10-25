import pickle
import numpy as np
import cv2 as cv
import os
import sys
sys.path.append('E:\\Courses\\Local Machine Projects\\Football Analysis Project')
from utils.bbox_utils import measure_distance,measure_distance_cm

class CameraMovementEstimator():
    
    def __init__(self,frame):

        self.minimum_distance = 5    
        
        self.lk_params = dict(
            winSize = (21,21),
            maxLevel = 2,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,:20] = 255
        mask_features[:,900:1050] = 255
        
        self.features = dict(
            maxCorners = 200,
            qualityLevel = 0.03,
            minDistance = 2,
            blockSize = 7,
            mask = mask_features
        )

    def adjust_position_to_tracks(self,tracks,camera_movement_per_frame):

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read Stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Feature detection
        camera_movement = [[0, 0]] * len(frames)
        old_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        old_features = cv.goodFeaturesToTrack(old_gray, **self.features)

        # Check if good features were found
        if old_features is None:
            print("No good features to track in the first frame.")
            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv.cvtColor(frames[frame_num], cv.COLOR_BGR2GRAY)
            new_features, status, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            # Check if the optical flow calculation was successful
            if new_features is None or len(new_features) == 0:
                print(f"Frame {frame_num}: No new features detected.")
                continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_distance_cm(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movements(self,frames,camera_movement_per_frame):

        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()

            cv.rectangle(overlay,(0,0),(500,100),(255,255,255),cv.FILLED)
            alpha = 0.6 # Opacity factor
            cv.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement,  y_movement = camera_movement_per_frame[frame_num]
            frame = cv.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame)
        
        return output_frames