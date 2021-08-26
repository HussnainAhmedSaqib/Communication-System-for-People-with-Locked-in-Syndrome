from tkinter import *
from imutils import face_utils
from win32com.client import Dispatch
import cv2
import PIL.Image, PIL.ImageTk
from math import sqrt
import numpy as np
import dlib
import time


counter = 0

def speak(text):
    spk = Dispatch("SAPI.SpVoice")
    spk.Speak(text)


def on_click(alphabet):
    global counter
    if counter == 0:
        textBox.delete("1.0", END)
        counter = counter + 1
        textBox.insert(INSERT, str(alphabet))
    else:
        current = textBox.get("1.0", "end-1c")
        textBox.delete("1.0", END)
        textBox.insert(INSERT, str(current) + str(alphabet))
# add \n at the end if we not use end-1c

def space():
    current = textBox.get("1.0", "end-1c")
    textBox.delete("1.0", END)
    textBox.insert(INSERT, current+" ")


def backspace():
    str_to_insert = textBox.get("1.0", "end-1c")
    textBox.delete("1.0", END)
    str_to_insert = str_to_insert[:-1]
        textBox.insert(INSERT, str_to_insert)


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_euc_dist(a,b,c,d):
    dist=sqrt( (a-c)**2 + (b-d)**2 )
    return dist

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates

    A=get_euc_dist(eye[1][0],eye[1][1],eye[5][0],eye[5][1])
    B=get_euc_dist(eye[2][0],eye[2][1],eye[4][0],eye[4][1])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C=get_euc_dist(eye[0][0],eye[0][1],eye[3][0],eye[3][1])
    # compute the eye aspect ratiox
    ear = (A + B)/(2.0*C)
    return ear

def get_midpoint(point1, point2):
    midpoint=int((point1.x+point2.x)/2),int((point1.y+point2.y)/2)
    return midpoint
def calculate_gaze_ratio(eye_points,landmarks):
    # ler= LEFT EYE REGION
    ler = np.array([(landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y),
                                (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y),
                                (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y),
                                (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y),
                                (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y),
                                (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [ler], True, 255, 2)
    cv2.fillPoly(mask, [ler], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    xmin= np.min(ler[:, 0])
    xmax= np.max(ler[:, 0])
    ymin= np.min(ler[:, 1])
    ymax= np.max(ler[:, 1])
    gray_eye = eye[ymin: ymax, xmin: xmax]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 2
    elif right_side_white == 0:
        gaze_ratio = 2
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio
def is_left(gaze_ratio):
    if gaze_ratio >= 1.2:
        return TRUE
    else:
        return FALSE
def is_right(gaze_ratio):
    if gaze_ratio <= 0.8:
        return TRUE
    else:
        return FALSE


EYE_AR_THRESH = 0.17
EYE_AR_CONSEC_FRAMES = 3# for blink time
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 7
textbox_string = "Some text"
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
wink_counter = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
btn_arr = []
btn_arr.append([])
btn_arr.append([])
btn_arr.append([])
btn_arr.append([])
frame_arr=[]
btn_frame_arr = []
btn_frame_arr.append([])
btn_frame_arr.append([])
btn_frame_arr.append([])
btn_frame_arr.append([])
index_var = 0
num_frames= 0
frame_index = 0
row_selected = False
is_left_flag = FALSE
is_right_flag = FALSE
gaze_ratio=2
right_gaze_frames=0
left_gaze_frames=0
middle_gaze_frames=0
gaze_var=1
frame=0
gray=0
# class for video display



class iblink_app:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.video_source = video_source
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas_frame = Frame(window,padx=20,pady=20,bg='skyblue')
        self.canvas = Canvas(self.canvas_frame,width=self.vid.width, height=self.vid.height,bg='black')
        self.canvas.pack(fill=BOTH)
        self.canvas_frame.place(rely=0.45,relwidth=0.45,relheight=0.45,relx=0.5)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

 #       # Get a frame from the video source

    def update(self):
        # Get a frame from the video source
        global COUNTER, gaze_ratio, gray, right_gaze_frames, frame, left_gaze_frames, middle_gaze_frames, gaze_var, TOTAL, wink_counter, textbox_string,btn_arr, index_var,num_frames,frame_arr,frame_index,row_selected,TOTAL,is_left_flag, is_right_flag
        if num_frames==15:
            num_frames=0
            if TOTAL%2==0:
                frame_arr[frame_index].focus_set()
                frame_index+=1
                if frame_index==4:
                    frame_index=0
            else:
                btn_frame_arr[frame_index - 1][index_var].focus_set()
                index_var=index_var+gaze_var
                if index_var == (len(btn_arr[frame_index-1])):
                    index_var = 0
        ret, frame = self.vid.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(frame, 0)
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            landmarks=predictor(gray,rect)
            gaze_ratio_left_eye = calculate_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = calculate_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            is_left_flag = is_left(gaze_ratio)
            is_right_flag = is_right(gaze_ratio)
            if is_left_flag:
                left_gaze_frames += 1
                if left_gaze_frames == 20:
                    left_gaze_frames = 0
                    right_gaze_frames = 0
                    middle_gaze_frames = 0
                    is_right_flag = FALSE
                    is_left_flag= FALSE
                    gaze_var = -1
                    speak('left')
            elif is_right_flag:
                right_gaze_frames += 1
                if right_gaze_frames == 20:
                    left_gaze_frames = 0
                    right_gaze_frames = 0
                    middle_gaze_frames = 0
                    is_left_flag = FALSE
                    is_right_flag= FALSE
                    gaze_var = 1
                    speak('right')
            elif not is_left_flag and not is_right_flag:
                middle_gaze_frames += 1
                if middle_gaze_frames == 15:
                    left_gaze_frames = 0
                    right_gaze_frames = 0
                    middle_gaze_frames = 0
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            # difference of l_eye and r_eye ear
            ear_diff = np.abs(leftEAR - rightEAR)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    speak("blink")
                    if TOTAL%2==0:
                        if gaze_var==1:
                            btn_arr[frame_index - 1][index_var - 1].invoke()
                        elif gaze_var==-1:
                            btn_arr[frame_index - 1][index_var + 1].invoke()
                        #btn_arr[frame_index - 1][index_var - 1].invoke()
                        index_var = 0
                    # reset the eye frame counter
                COUNTER = 0
            if ear_diff > WINK_AR_DIFF_THRESH:
                # this condition means left eye blink
                if leftEAR < rightEAR:
                    wink_counter += 1
                    if wink_counter > WINK_CONSECUTIVE_FRAMES:
                        #textbox_string = textbox_string[:-1]
                        #print(textbox_string)
                        speak("left blink")
                        if TOTAL%2==1:
                            frame_arr[frame_index].focus_set()
                            TOTAL -= 1
                        else:
                            backspace()
                        wink_counter = 0
                elif leftEAR > rightEAR:
                    wink_counter += 1
                    if wink_counter > WINK_CONSECUTIVE_FRAMES:
                        #textbox_string = textbox_string + "_"
                        #print(textbox_string)
                        speak("right blink")
                        space()
                        wink_counter = 0
                else:
                    wink_counter = 0

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(3, 550)
        self.vid.set(4, 370)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            global num_frames
            num_frames += 1
            #print("num of frames = ", num_frames)
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# end of class



root = Tk()

root.title("Final Year Project")
root.configure(background='skyblue')
#root.geometry("1200x800")
root.state('zoomed')
height = root.winfo_screenheight()
width = root.winfo_screenwidth()

#print(height)
#print(width)

root.minsize(width, height)


main_screen = Frame(root,background='skyblue')
main_screen.place(relwidth=1.0,relheight=1.0)
splash_screen = Frame(root,background='skyblue')
splash_screen.place(relwidth=1.0,relheight=1.0)
lab = Label(splash_screen,text='iBlink',font=("Courier", 54),bg='skyblue',fg='black')
lab.place(relx=0.5, rely=0.6, anchor=CENTER)
img = PIL.ImageTk.PhotoImage(PIL.Image.open("icon.png"))
lab1 = Label(splash_screen,image=img,bg='skyblue')
lab1.place(rely=0.39,relx=0.5,anchor=CENTER)
splash_screen.update()
time.sleep(4)
#splash_screen.place.forget()
splash_screen.destroy()

main_screen.tkraise()


#frame for text box
box_frame = Frame(main_screen,padx=20,pady=20,bg='skyblue')
textBox = Text(box_frame, bg="white", font=("Helvetica", 14))
textBox.insert(INSERT, " Text Box ")
textBox.pack(fill=BOTH)
box_frame.place(rely=0.45,relwidth=0.45,relheight=0.45,relx=0.05)





f1= Frame(root, highlightcolor='red', highlightthickness=3,background='skyblue',highlightbackground='skyblue')
f2= Frame(root, highlightcolor='red', highlightthickness=3,background='skyblue',highlightbackground='skyblue')
f3= Frame(root, highlightcolor='red', highlightthickness=3,background='skyblue',highlightbackground='skyblue')
f4= Frame(root, highlightcolor='red', highlightthickness=3,background='skyblue',highlightbackground='skyblue')
frame_arr.append(f1)
frame_arr.append(f2)
frame_arr.append(f3)
frame_arr.append(f4)


for x in range(3):
    btn_frame_arr[0].append(
        Frame(f1, highlightcolor='white', highlightthickness=3, background='skyblue', highlightbackground='skyblue'))

frame_rows = 3
no_of_buttons = [10,10,6]

for row in range(frame_rows):
    for col in range(no_of_buttons[row]):
        btn_frame_arr[row+1].append(
            Frame(frame_arr[row+1], highlightcolor='white', highlightthickness=3, background='skyblue',
                  highlightbackground='skyblue'))

reciteButton = Button(btn_frame_arr[0][0], text="Recite", padx=20, pady=10, bg="black", fg='white',
                      command=lambda: speak(textBox.get("1.0", "end-1c")))
btn_arr[0].append(reciteButton)
space_button = Button(btn_frame_arr[0][1], text="  Space  ", bg="black", fg="white", padx=225, pady=10, command=space)
btn_arr[0].append(space_button)
back_space_button = Button(btn_frame_arr[0][2], text="<- Backspace", bg="black", fg="white", padx=20, pady=10,
                           command=backspace)
btn_arr[0].append(back_space_button)



btn_text = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
    'z'],['x', 'c', 'v', 'b', 'n', 'm']]
rowVar = 3
colVar = [10,10,6]
for row in range(rowVar):
    for col in range(colVar[row]):
        command = lambda a=btn_text[row][col]: on_click(a)
        b = Button(btn_frame_arr[row+1][col], text=btn_text[row][col], bg="black", fg="white", padx=20, pady=10, command=command)
        btn_arr[row+1].append(b)


# adding row one in frame one
# frame_arr[0].place(x=40,rely=0.525)
frame_arr[0].place(relx=0.5, rely=0.06, anchor=CENTER)
btn_frame_arr[0][0].grid(row=0, column=0, padx=4, pady=4)
btn_arr[0][0].grid(row=0, column=0)
btn_frame_arr[0][1].grid(row=0, column=1, padx=4, pady=4)
btn_arr[0][1].grid(row=0, column=0)
btn_frame_arr[0][2].grid(row=0, column=2, padx=4, pady=4)
btn_arr[0][2].grid(row=0, column=0)

# adding row two in frame two
# frame_arr[1].place(x=40,rely=0.625)
frame_arr[1].place(relx=0.5, rely=0.17, anchor=CENTER)
btn_frame_arr[1][0].grid(row=0, column=0, padx=4, pady=4)
btn_arr[1][0].grid(row=0, column=0)
btn_frame_arr[1][1].grid(row=0, column=1, padx=4, pady=4)
btn_arr[1][1].grid(row=0, column=0)
btn_frame_arr[1][2].grid(row=0, column=2, padx=4, pady=4)
btn_arr[1][2].grid(row=0, column=0)
btn_frame_arr[1][3].grid(row=0, column=3, padx=4, pady=4)
btn_arr[1][3].grid(row=0, column=0)
btn_frame_arr[1][4].grid(row=0, column=4, padx=4, pady=4)
btn_arr[1][4].grid(row=0, column=0)
btn_frame_arr[1][5].grid(row=0, column=5, padx=4, pady=4)
btn_arr[1][5].grid(row=0, column=0)
btn_frame_arr[1][6].grid(row=0, column=6, padx=4, pady=4)
btn_arr[1][6].grid(row=0, column=0)
btn_frame_arr[1][7].grid(row=0, column=7, padx=4, pady=4)
btn_arr[1][7].grid(row=0, column=0)
btn_frame_arr[1][8].grid(row=0, column=8, padx=4, pady=4)
btn_arr[1][8].grid(row=0, column=0)
btn_frame_arr[1][9].grid(row=0, column=9, padx=4, pady=4)
btn_arr[1][9].grid(row=0, column=0)

# adding row three in frame 3
# frame_arr[2].place(x=40,rely=0.725)
frame_arr[2].place(relx=0.5, rely=0.28, anchor=CENTER)
btn_frame_arr[2][0].grid(row=0, column=0, padx=4, pady=4)
btn_arr[2][0].grid(row=0, column=0)
btn_frame_arr[2][1].grid(row=0, column=1, padx=4, pady=4)
btn_arr[2][1].grid(row=0, column=0)
btn_frame_arr[2][2].grid(row=0, column=2, padx=4, pady=4)
btn_arr[2][2].grid(row=0, column=0)
btn_frame_arr[2][3].grid(row=0, column=3, padx=4, pady=4)
btn_arr[2][3].grid(row=0, column=0)
btn_frame_arr[2][4].grid(row=0, column=4, padx=4, pady=4)
btn_arr[2][4].grid(row=0, column=0)
btn_frame_arr[2][5].grid(row=0, column=5, padx=4, pady=4)
btn_arr[2][5].grid(row=0, column=0)
btn_frame_arr[2][6].grid(row=0, column=6, padx=4, pady=4)
btn_arr[2][6].grid(row=0, column=0)
btn_frame_arr[2][7].grid(row=0, column=7, padx=4, pady=4)
btn_arr[2][7].grid(row=0, column=0)
btn_frame_arr[2][8].grid(row=0, column=8, padx=4, pady=4)
btn_arr[2][8].grid(row=0, column=0)
btn_frame_arr[2][9].grid(row=0, column=9, padx=4, pady=4)
btn_arr[2][9].grid(row=0, column=0)

# adding row four in frame four
# frame_arr[3].place(x=170,rely=0.825)
frame_arr[3].place(relx=0.5, rely=0.39, anchor=CENTER)
btn_frame_arr[3][0].grid(row=0, column=0, padx=4, pady=4)
btn_arr[3][0].grid(row=0, column=0)
btn_frame_arr[3][1].grid(row=0, column=1, padx=4, pady=4)
btn_arr[3][1].grid(row=0, column=0)
btn_frame_arr[3][2].grid(row=0, column=2, padx=4, pady=4)
btn_arr[3][2].grid(row=0, column=0)
btn_frame_arr[3][3].grid(row=0, column=3, padx=4, pady=4)
btn_arr[3][3].grid(row=0, column=0)
btn_frame_arr[3][4].grid(row=0, column=4, padx=4, pady=4)
btn_arr[3][4].grid(row=0, column=0)
btn_frame_arr[3][5].grid(row=0, column=5, padx=4, pady=4)
btn_arr[3][5].grid(row=0, column=0)

iblink_app(main_screen, "Live video feed")

root.mainloop()
