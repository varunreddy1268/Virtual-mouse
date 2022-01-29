import cv2
import numpy as np
import os
import mediapipe as mp
import uuid
"""ML pipeline, proven to be effective in our MediaPipe Hands and MediaPipe Face Mesh solutions. Using a detector, the pipeline first locates the person/pose region-of-interest (ROI) within the frame. The tracker subsequently predicts the pose landmarks and segmentation mask within the ROI using the ROI-cropped frame as input. Note that for video use cases the detector is invoked only as needed, i.e., for the very first frame and when the tracker could no longer identify body pose presence in the previous frame. For other frames the pipeline simply derives the ROI from the previous frame’s pose landmarks."""
"""The landmark model in MediaPipe Pose predicts the location of 33 pose landmarks (see figure below).
SCREENSHOT.JPG
"""
mp_drawing =mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_hands=mp.solutions.hands
"""mp.solutions.hands used for getting the tracing solutions based on the pre trained data points SCREENSHOT2.png"""


v_capture=cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5) as hands:
    """detection,tracking confidences for accurate image capturing"""
    while v_capture.isOpened():
        "once the camera gets started"
        x,y=v_capture.read()
        "x is object detected or not, image stores the results of object that got detected"
        if not x:
            print("Nothing is detected")
            continue
        image=cv2.cvtColor(y,cv2.COLOR_BGR2RGB)
        """changing default color scanning using opencv which is bgr to rgb"""
        image.flags.writeable= False
        """flag is used for not copying any other images other that hand"""
        image=cv2.flip(image,1)
        """just in case if you show the right hand flip it to the left hand"""
        results=hands.process(image)
        image.flags.writeable=True
        """now stop the flag"""
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        """reconvert the image into the same format"""

        """Land mass reduction"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                """starting loop for just taking the hand landmarks which we will get form the computer when a hand got captures infront of the camera"""
                mp_drawing.draw_landmarks(image, hand_landmarks,mp_hands.HAND_CONNECTIONS)
                """mp_drawings is used to draw the landmarks which is dettected, hand_connections is the lines which the connects all the points of the hand and forms a graphicall hand from the detected points"""
                cv2.imwrite(os.path.join("/Users/varunreddyseelam/Documents/output_image","{}.jpg".format(uuid.uuid1())),image)
                cv2.imshow("The hand is being trcked",image)
                if cv2.waitKey(1)&0xFF==ord('q'):
                    """to close the feed for the media pipe use the key Q"""
                    break
v_capture.release()
cv2.destroyAllWindows()
