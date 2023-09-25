from flask import Flask, render_template, request, Response, redirect
import numpy as np
import cv2
from ultralytics import YOLO
 
app = Flask(__name__)
 
video = cv2.VideoCapture('') #// if you have second camera you can set first parameter as 1
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("static/models/haarcascade_fullbody.xml")) 
PERSON_THRESHOLD_MED = 5
PERSON_THRESHOLD_MAX = 7
MODEL_FILE="static/models/yolov8n.pt"
COCO_FILE="static/models/coco.txt"

@app.route('/')
def index():
    return render_template('index.html')

VIDEO_EXTENSIONS = ['mp4']
PHOTO_EXTENSIONS = ['png','jpeg','jpg']
 
def fextension(filename):
    return filename.rsplit('.', 1)[1].lower()

@app.route('/upload', methods=['POST'])
def upload():
    global video
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype=fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/video/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/video/' + file.filename)
            return redirect('/video_feed_new')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/photo/' + file.filename)
            print('photo')
            return render_template('preview_photo.html', file_name=file.filename, type='image/'+exttype)
    return 'Invalid video file'

def gen(video):
    while True:
        success, image = video.read()
        frame = cv2.resize(frame, (720,480))
        try:
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            cv2.putText(image, "VIDEO ENDED, Please Go back", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        frame_gray = cv2.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray)


        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            #cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y+h, x:x+w]
        cv2.putText(image, "Persons: " + str(len(faces)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        no_faces = len(faces)
        status=""
        if no_faces < PERSON_THRESHOLD_MED:
            status = "Green"
            clor = (0, 255, 0)
        elif no_faces < PERSON_THRESHOLD_MAX:
            status = "Yellow"
            clor = (0, 255, 255)
        else:
            status = "Red"
            clor = (0, 0, 255)
        cv2.putText(image, "Status: ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.putText(image, status, (175, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, clor, 3)
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_new(video):
    # opening the file in read mode
    my_file = open(COCO_FILE, "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list = data.split("\n")
    my_file.close()

    # load a pretrained YOLOv8n model
    model = YOLO(MODEL_FILE, "v8") 
    while True:
        ret, frame = video.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, (720, 480))

        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        DP = detect_params[0].numpy()
        #print(DP)
        no_faces=0
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                print(i)

                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(frame,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),(255,255,255),3,)
                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(frame,class_list[int(clsID)],(int(bb[0]), int(bb[1]) - 10),font,0.5,(255, 255, 255),1,)
                if class_list[int(clsID)] == "person":
                    no_faces=no_faces+1


        status=""
        if no_faces < PERSON_THRESHOLD_MED:
            status = "Green"
            clor = (0, 255, 0)
        elif no_faces < PERSON_THRESHOLD_MAX:
            status = "Yellow"
            clor = (0, 255, 255)
        else:
            status = "Red"
            clor = (0, 0, 255)
        cv2.putText(frame, "Persons: " + str(no_faces), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.putText(frame, "Status: ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.putText(frame, status, (175, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, clor, 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_new')
def video_feed_new():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen_new(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed')
def camera_feed():
    global video
    video = cv2.VideoCapture(0)
    if not (video.isOpened()):
        return 'Could not connect to camera'
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)