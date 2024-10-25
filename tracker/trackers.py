from re import L
from ultralytics import YOLO
import supervision as sv
import pickle
import pandas as pd # used to do interpolation
import os
import numpy as np
import cv2 as cv
import sys
sys.path.append('E:\\Courses\\Local Machine Projects\\Football Analysis Project')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # adding position w.r.t camera Movement
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    
    def interpolate_ball_position (self,ball_position):
        
        #converting the format to pandas dataframe
        ball_position = [x.get(1,{}).get('bbox',[]) for x in ball_position] 
        
        # Making datafarme
        df_ball_position = pd.DataFrame(ball_position,columns=['x1','y1','x2','y2'])

        # interpolate missing values
        df_ball_position = df_ball_position.interpolate()
        df_ball_position = df_ball_position.bfill() # just for the first frame in which ball is not detected
        
        # Converting back to the original format
        ball_position = [{1:{'bbox':x}} for x in df_ball_position.to_numpy().tolist()]

        return ball_position

    def detect_frame(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames,read_from_stub=False,stub_path=None):
        
        if read_from_stub == True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frame(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Get the class names mapping
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert 'goalkeeper' class to 'player'
            for object_ind, box in enumerate(detection.boxes):  # Iterate through the detection boxes
                class_id = int(box.cls)  # Get the class ID from the boxes
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']  # Replace 'goalkeeper' with 'player'
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Changing the format to useful format
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3].tolist()
                track_id = frame_detection[4].tolist()

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox':bbox}
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}
            
            if stub_path is not None:
                with open(stub_path,'wb') as f:
                    pickle.dump(tracks,f)
        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None): 
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox) # it'll helps us in ellipse radius
        
        # Drawing ellipse
        cv.ellipse(frame,
                   center= (x_center,y2),
                   axes=(int(width),int(0.35*width)),
                   angle=0.0,
                   startAngle=-45,
                   endAngle=235,
                   color=color,
                   thickness=2,
                   lineType=cv.LINE_4)
        
        # Assigning the Track ID number to each player 
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2-rectangle_height//2)+15
        y2_rect = (y2+rectangle_height//2)+15

        if track_id is not None:
            cv.rectangle(frame,
                         (int(x1_rect),int(y1_rect)),
                         (int(x2_rect),int(y2_rect)),
                         color,
                         cv.FILLED)
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10

            cv.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            ) 
        
        return frame
    
    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1]) # taking y1 for bottom point of inverted triangle
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x, y], [x-10, y-20], [x+10, y-20]])

        cv.drawContours(frame, [triangle_points],0,color,cv.FILLED)
        cv.drawContours(frame,[triangle_points],0,(0,0,0),2) # Boundary of triangle

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
    # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv.FILLED)
        alpha = 0.4  # This alpha is for transparency
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Now adding the writing material in the rectangle
        team_ball_control_till_frame = np.array(team_ball_control[:frame_num + 1])  # Ensure this is a NumPy array
        # print(f"Frame {frame_num}: Team Ball Control: {team_ball_control_till_frame}")
    # Get the number of times each team had ball control
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)  # Count occurrences of team 1
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)  # Count occurrences of team 2
        
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1_percentage = team_1_num_frames / total_frames
            team_2_percentage = team_2_num_frames / total_frames
        else:
            team_1_percentage = team_2_percentage = 0  # Avoid division by zero when no data is available

        # Display the percentages on the frame
        cv.putText(frame, f"Team 1 Ball Control {team_1_percentage * 100:.2f}%", (1400, 900), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv.putText(frame, f"Team 2 Ball Control {team_2_percentage * 100:.2f}%", (1400, 950), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotation(self,video_frames,tracks,team_ball_control):
        output_videos_frames = []

        for frame_num, frame in enumerate(video_frames):
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            # Draw around players
            for track_id, player in player_dict.items():
                color = player.get('team_color',(0,0,255))
                frame = self.draw_ellipse(frame,player['bbox'],color,track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))

            for _,referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee['bbox'],(0,255,255))
            
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball['bbox'],(255,0,0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)


            output_videos_frames.append(frame)
        return output_videos_frames