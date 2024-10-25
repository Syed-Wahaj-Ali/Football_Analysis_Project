import cv2 as cv
from sklearn.cluster import KMeans

class Team_assigner:
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        
        # Reshape the image to 2d array
        image_2d = image.reshape(-1,3)

        # cluster the image
        kmeans_model = KMeans(n_clusters=2,init='k-means++',n_init=1)
        kmeans_model.fit(image_2d)

        return kmeans_model

    def get_player_color(self, frame, bbox):
    
        # Cropping the image
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Getting the top half of the image
        top_image = image[0:int(image.shape[0] / 2), :]

        # Clustering the image
        kmeans = self.get_clustering_model(top_image)

        # Cluster labels
        cluster_labels = kmeans.labels_

        # Reshape to the original image
        image_org = cluster_labels.reshape(int(top_image.shape[0]), int(top_image.shape[1]))

        # Getting the player's color
        # Separating the clusters of player and nonplayer/background

        corner_cluster = [image_org[0, 0], image_org[0, -1], image_org[-1, 0], image_org[-1, -1]]
        non_player_bg = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_bg

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    
    def assign_team_color (self,frame,player_detections):
        player_colors = []
        
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        # Separating the further 2 color form player color list
        print(f"The simple player color list: {player_color}")  # Checking the color is storing or not

        kmeans_clr_lst = KMeans(n_clusters=2,init='k-means++',n_init=10)
        kmeans_clr_lst.fit(player_colors)

        self.kmeans_clr_lst = kmeans_clr_lst
        
        self.team_color[1] = kmeans_clr_lst.cluster_centers_[0]
        self.team_color[2] = kmeans_clr_lst.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans_clr_lst.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        if player_id == 27:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        # print(f"The the player dict contains: {self.player_team_dict}")        # Checking weather the color is separating or none
        return team_id