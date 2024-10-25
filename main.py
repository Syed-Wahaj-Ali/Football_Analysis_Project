import speed_distance_estimator
from utils import read_video, save_video
from tracker import Tracker
import cv2 as cv
from team_assigner import Team_assigner
from ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movements_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedDistanceEstimator


def main():
    # Read Video
    video_frame = read_video("input_videos/eagle_2.mp4")

    # Initializing the Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frame,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stub_full.pkl')
    # Get Object positions
    tracker.add_position_to_tracks(tracks)

    # Camera Movement Estimator
    camera_movements_estimator = CameraMovementEstimator(video_frame[0])
    camera_movement_per_frame = camera_movements_estimator.get_camera_movement(video_frame,
                                                                               read_from_stub=True,
                                                                               stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movements_estimator.adjust_position_to_tracks(tracks,camera_movement_per_frame)

    # View Transformen
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_track(tracks)
    
    # ball interpolation
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    # Speed And Distance Estimator
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_distance_to_tracks(tracks)



    # Assign player team 
    team_assigner = Team_assigner()
    team_assigner.assign_team_color(video_frame[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frame[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team']= team
            tracks['players'][frame_num][player_id]['team_color']= team_assigner.team_color[team]

    # assign the ball aquisition
    player_assigner = PlayerBallAssigner()
    # Ball Control
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no ball control and it's the first frame, assign a default value (e.g., None or -1)
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None)  # Or use any default value like -1 or 'unknown'
  

    # Drawing the annotation around players
    output_video_frames = tracker.draw_annotation(video_frame,tracks,team_ball_control)
    
    # Drawing camera movements 
    output_video_frames = camera_movements_estimator.draw_camera_movements(output_video_frames,camera_movement_per_frame)
    
    # Drawing the speed and distances
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    # Save video
    save_video(output_video_frames,"output_videos/out_video_1.avi")


if __name__ == '__main__':
    main()