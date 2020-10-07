from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
import datetime
import uuid
from PIL import Image

app = Flask(__name__)
net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_testing.cfg")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Hello')
def hello_world():
    return 'Hello World!'


# @app.route('/upload')
# def upload_file():
# return render_template('index.html')

@app.route('/uploader_file', methods=['GET', 'POST'])
def uploader_file():
    print('HI')
    if request.method == 'POST':
        print('HI Post')
        check_roi = request.form.getlist('check')
        savedimages = []
        RICounts = 0
        lablesList = []
        f = request.files['file']
        uuidOne = str(uuid.uuid1())
        uploadedfilename = f.filename
        f.save('static/Original_' + uuidOne + '_' + f.filename)
        beforeImage = 'Original_' + uuidOne + '_' + f.filename

        for la in check_roi:

            # f.save('static/'+secure_filename(f.filename))

            bytefile = request.files['file'].read()  ## byte file

            with open("classes.txt", "r") as rf:
                classes = [line.strip() for line in rf.readlines()]
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))

            # # Load Imane

            img = Image.open(request.files['file'])
            img = np.array(img)
            # img = cv2.resize(img, None, fx=0.1, fy=0.1)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, None, fx=0.1, fy=0.1)
            height, width, channels = img.shape
            actimage = img
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            for b in blob:
                for n, img_blob in enumerate(b):
                    cv2.imshow(str(n), img_blob)

            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            # RICounts=len(boxes)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
            number_objects_detected = len(boxes)
            font = cv2.FONT_HERSHEY_PLAIN
            # lablesList=[]
            for i in range(len(boxes)):
                if i in indexes.flatten():

                    if la == classes[class_ids[i]]:
                        x, y, w, h = boxes[i]
                        label = classes[class_ids[i]]
                        color = colors[class_ids[i]]
                        lablesList.append(label)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, label, (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (250, 250, 250))

                    elif la == 'All':
                        x, y, w, h = boxes[i]
                        label = classes[class_ids[i]]
                        color = colors[class_ids[i]]
                        lablesList.append(label)
                        RICounts = len(lablesList)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, label, (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (250, 250, 250))

            afterImage = 'After_' + uuidOne + '_' + str(la) + '_' + f.filename
            savedimages.append(afterImage)

            cv2.imwrite('static/' + afterImage, img)
            cv2.destroyAllWindows()

    return render_template('index.html', after_Image=savedimages, user_image=beforeImage, find_RICounts=RICounts,
                           find_lablesList=lablesList, Upfilename=uploadedfilename)


#     return redirect(url_for('index'))

if __name__ == '__main__':
    from werkzeug.serving import run_simple

    # run_simple('localhost', 7788, app)
    app.run(debug=True)
