from flask import Flask, render_template, Response, request, send_file
import os
import cv2
import torch
import magic

app = Flask(__name__)

# GLOBAL VARIABLE
videoFolderPath = 'static/Video File'
app.config['STATIC'] = 'static'
app.config['VIDEO_FOLDER'] = videoFolderPath
cameraSwitch = 0
imageList=[]

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'Model/bestYOLOV5m640.pt')

# Starting website and stream using camera
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(3))
frame_height = int(camera.get(4))
size = (frame_width, frame_height)

# Init Recorder for Streaming
streamRecordWriter = cv2.VideoWriter('detection.mp4',
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         20, size)

# Function to detect in real-time
def generate_frames():
    while True:
        # read the camera frame
        success, frame = camera.read()

        if not success:
            break
        else:
            result = model(frame)
            result.render()

            ret, buffer = cv2.imencode('.jpg', frame)

            frameResult = buffer.tobytes()

            streamRecordWriter.write(frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frameResult + b'\r\n')

# Function to check the file is video or not
def checkVideoFile(pathToVideo):
    mime = magic.Magic(mime=True)
    filename = mime.from_file(pathToVideo)
    if filename.find('video') != -1:
        return "correct"
    return "false"

# Function to start detection from video file
def videoDetection(videoFileName, listImage):
    vidImgDetCounter = 1

    video = cv2.VideoCapture(videoFolderPath + '/' + videoFileName)
    videoWriter = None

    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, -1)
            frameTime = "%.2f" % float(video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

            if videoWriter is None:
                videoWriter = cv2.VideoWriter( videoFolderPath + '/output.mp4',
                                              cv2.VideoWriter_fourcc(*'MP4V'),
                                              30, (int(frame.shape[1]), int(frame.shape[0])))
            result = model(frame)
            result.render()

            # Write the time of the frame from the video in top left image
            cv2.putText(frame, frameTime, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)

            if result.pandas().xyxy[0].empty:
                print("Kosong \n")
            else:
                savingDetectionVideoFrame(frame, listImage, frameTime, vidImgDetCounter)
                vidImgDetCounter+=1

            videoWriter.write(frame)
    videoWriter.release()

# Function to save the frame as image from video if the object exist
def savingDetectionVideoFrame(img, imgList, frameTime, imgCounter):
    imgName='static/Video File/Detection Images/' + str(imgCounter)+'.jpg'
    imgPath='Video File/Detection Images/' + str(imgCounter)+'.jpg'
    cv2.imwrite(imgName, img)
    helperList = [imgPath, frameTime]
    imgList.append(helperList)

# Function to delete the previous detection images from video
def deletePreviousDetectionImageFiles(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/requests',methods=['POST','GET'])
def task():
    global cameraSwitch, camera, streamRecordWriter

    if request.method == 'POST':
        # Start and Stop Connect to Camera
        if request.form.get('stop') == 'Stop/Start':
            if (cameraSwitch==0):
                if streamRecordWriter is not None:
                    streamRecordWriter.release()
                camera.release()
                cv2.destroyAllWindows()
                cameraSwitch = 1
            else:
                camera = cv2.VideoCapture(0)
                if streamRecordWriter is not None:
                    streamRecordWriter = cv2.VideoWriter('detection.mp4',
                                                         cv2.VideoWriter_fourcc(*'MP4V'),
                                                         20, size)
                cameraSwitch = 0

        # Download Detection Result using Camera
        elif request.form.get('get_DetStream') == 'Download Camera Result':
            streamRecordWriter.release()
            camera.release()
            cv2.destroyAllWindows()
            cameraSwitch = 1
            DetResult = "detection.mp4"
            return send_file(DetResult, as_attachment=True)

        # Start Detection using Video
        elif request.form.get('startVideoDet') == 'Start Video Detection':
            imageList = []

            inputVideo = request.files["video"]
            filename = inputVideo.filename

            checkingVideoPath=app.config['VIDEO_FOLDER']+"/"+filename
            inputVideo.save(os.path.join(app.config['VIDEO_FOLDER'], filename))

            # Checking if the file is video
            checkingVideoResult = checkVideoFile(checkingVideoPath)
            if(checkingVideoResult == "false"):
                return render_template("video.html", error="ONLY DETECT VIDEO FILE!!!")

            # Delete previous detection images
            deletePreviousDetectionImageFiles(videoFolderPath+"/Detection Images")
            imageList.clear()

            videoDetection(filename, imageList)

            return render_template("detection result.html", imageList=imageList)

        # Download Detection Result using Video
        elif request.form.get('getVideoDet') == 'Download Video Detection':
            return send_file( videoFolderPath+"/output.mp4", as_attachment=True)

    elif request.method == 'GET':
        return render_template('index.html')

    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videoIndex')
def videoIndex():
    global cameraSwitch

    if (cameraSwitch == 0):
        camera.release()
        cv2.destroyAllWindows()
        cameraSwitch = 1

    return render_template('video.html')

if __name__ == "__main__":
    app.run()