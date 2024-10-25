# Football_Analysis_Project ⚽

#### Problem Definition:
> We're asked to do Football Analysis.

#### Tasks needed to be done
1. Object detection
2. Object Tracking
3. Manupulation of Bounding Boxes
4. View Transformation

#### Project Data:
The data is require to make object detecions of football players, referees and GoalKeeper. The data for football player detecion is taken from Roboflow
Link:[https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/12]

![image](https://github.com/user-attachments/assets/8aaeb450-4acc-446d-a5c0-d9492b99acd1)

#### Approach
I solved this problem by the following steps
1. Firstly I've trained an Object detection model using Yolo V8. It was necessary because I only want to detect players within the field not the outsiders. The Detection results were fine because of less data input.
2. Secondly by using the supervision and bounding boxes of each object I've done tracking.
3. thirdly by using OpenCV I've Manupulated BBOX, View Transformation, Camera Movements Estimation and viewing and Saving the results.

#### Results:
![image](https://github.com/user-attachments/assets/0f097aaf-cb7a-4da8-9c58-eca7a75673a2)
![image](https://github.com/user-attachments/assets/f3ec5a33-3ea3-4339-8c61-9c10140331d5)

#### How to get all these Material
1. Fork the Repo
2. Clone it to your Local Machine
3. Run the main.py code in vs Code

Remember: [Make Sure all the require Libraries intall in your machine and don't forget to change the name and path of stubs]
