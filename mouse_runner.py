import mediapipe as mp
import cv2
import numpy as np
import pyautogui as py
import os
v_cap=cv2.VideoCapture(0)
hands=mp.solutions.hands
m_hand=hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.9)
m_drawing=mp.solutions.drawing_utils
"""The input collected form the mouse.py is used by the landmark function for runnig the source pipe"""
def hand_landmarks(colorImg):
    landmarkList = []  # Default values if no landmarks are tracked

    landmarkPositions = m_hand.process(colorImg)  # Object for processing the video input
    landmarkCheck = landmarkPositions.multi_hand_landmarks  # Stores the out of the processing object (returns False on empty)
    if landmarkCheck:  # Checks if landmarks are tracked
        for hand in landmarkCheck:  # Landmarks for each hand
            for index, landmark in enumerate(hand.landmark):  # Loops through the 21 indexes and outputs their landmark coordinates (x, y, & z)
                m_drawing.draw_landmarks(img, hand, hands.HAND_CONNECTIONS)  # Draws each individual index on the hand with connections
                h, w, c = img.shape  # Height, width and channel on the image
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)  # Converts the decimal coordinates relative to the image for each index
                landmarkList.append([index, centerX, centerY])  # Adding index and its coordinates to a list

    return landmarkList

def fingertips(landmarks):
    fingertips_res=[]
    tipids=[4,8,12,16,20]
    """This are the tip values of the every finger when thevirtual ploting is done"""
    "For checking whetehr the thub is closed or open we need to compare the tip value of the thumb finger with the mid value of the thumb finger"
    "So if the thumb finger tip value which is 4 when it is opened, if it is closed then the thumb tip value becomes 0, is comapred against the mid value of te thumb finger"
    if landmarks[tipids[0]][1]>landmarks[tipids[0]-1][1]:
        fingertips_res.append(1)
    else:
        fingertips_res.append(0)
    "So, once we get whether the thumb is opened or closed we can get a sign that can be used as shortcutsign to do..."
    "Same goes with the res fingers to"
    for i in range(1,5):
        if landmarks[tipids[i]][2]<landmarks[tipids[i]-3][2]:
            fingertips_res.append(1)
        else:
            fingertips_res.append(0)
    return fingertips_res
"""Screen variables for mouse tracking"""
width_screen,height_screen=py.size()
print(width_screen,height_screen)
"""returns the screen size of the system"""
"""For coordinate grasping we need to create x,y variables to track the positions"""
current_x,current_y=0,0
previous_x,previous_y=0,0
def voiceover():
    import speech_recognition as sr
    from gtts import gTTS
    rec=sr.Recognizer()
    mytext=[]
    while (1):
        with sr.Microphone() as source:
            print("start talking")
            audio_text=recognizer.listen(source)
            try:
                print("Text:"+recognizer.recognize_google(audio_text))
                mytext=recognizer.recognizer_google(audio_text)
                print(mytext)
            except:
                pass
while True:
    check,img=v_cap.read()
    image_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    lmlist=hand_landmarks(image_rgb)
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        finger=fingertips(lmlist)
        current_x,current_y=0,0
        #[0,1,0,0,0]
        if finger[1]==1 and finger[2]==0:
            #intially the shape of the picture is 1440,900(width,Height)
            x3=np.interp(x1,(150,1440-350),(0,width_screen))
            y3=np.interp(y1,(150,900-350),(0,height_screen))
            current_x=previous_x+(x3-previous_x)/7
            current_y=previous_y+(y3-previous_y)/7
            py.moveTo(width_screen-current_x,current_y)
            previous_x,previous_y=current_x,current_y
        #[1,0,0,0,0]
        if finger[1]==0 and finger[0]==1:
            py.click()
        if finger[1]==0 and finger[4]==1:
            py.scroll(10)
        if finger[1]==0 and finger[3]==1:
            py.scroll(-10)
        #img=cv2.putText(img,"mouse",(current_x,current_y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imshow("Wbcame Turned On",img)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
