# Machine-Learning-based-LeArm-Robotic-Grasping
This project develops the process by which a LeArm robot can locate and grasp target objects accurately and efficiently in a 2D working space. The joint IDs of the LeArm can be found in JointIDs.png. The users can choose to achieve robotic grasping using conventional control-based approach, machine learning-based approach, or combine these two approaches to a hybrid approach.

The steps of running the project are as follows:

1. Run the takeimgs.py to take images for camera calibration. It is recommended to take more than 60 images to get an accurate result.
2. Run the camera_calibration.py to calibrate your camera based on the images taken and a 9*6 black and white chessboard. 
3. Use the code in Robot_control.zip and conventional_grasp.py to achieve conventional control-based robotic grasping. 
4. Use the code in Robot_control.zip and DataCollect.py to generate the training dataset for your MLP model. (An example of the training dataset can be found in MLPTrainingData.zip)
5. Run the MLP.py file to build your own MLP model and train it on the training dataset you generated. (It is recommended to copy and paste the code in  MLP.py to Google Colab and run it there to avoid any environment mismatch)
6. Run the ml-grasping.py to achieve machine learning-based robotic grasping based on the trained model.
7. Combine the code from the ml-grasping.py and conventional_grasp.py and adjust it according to your own needs to achieve hybrid robotic grasping.
