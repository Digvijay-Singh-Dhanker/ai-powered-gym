# ai-powered-gym
Using computer vison this script counts the number of squats, curls and over head press you do.

## Frameworks used
1) OpenCV
2) Numpy
3) Mediapipe

With the help of OpenCV we are able to access the camera and using Google's Mediapipe we estimate the pose of the user in front of the camera. Using the data acquired by google mediapipe we calculate the angles with the help of which we calculate the number of repetitions done.

![Pose Model Detections 2021-12-29 08-55-37 (1)](https://user-images.githubusercontent.com/45396488/152279219-51697f0c-17c7-495f-bfb3-cfdb8e07aa38.gif)
