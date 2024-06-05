# from flask import Flask,request,jsonify
# from flask_cors import CORS
# import base64
import cv2
# import numpy as np
# import logging

# app= Flask(__name__)
# CORS(app)


# logging.basicConfig(level=logging.DEBUG)

def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame, bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


# @app.route('/detect', methods=['GET', 'POST'])

# def detect():
#     frame_data_base64 = request.json.get('frameData')

#     if not frame_data_base64 :
#         return jsonify(error="No frame data provided"), 400 
#     try:
#         # Decode base64-encoded image data
#         frame_data_bytes = base64.b64decode(frame_data_base64)
#     except Exception as e:
#         app.logger.error("Failed to decode image data: {}".format(str(e)))
#         return jsonify(error="Failed to decode image data: invalid base64 encoding"), 400

#     nparr= np.frombuffer(frame_data_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     if img is None:
#         return jsonify(error="Failed to decode image"), 400

#     frame_height, frame_width = img.shape[0], img.shape[1]
#     blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
#     faceNet.setInput(blob)
#     detections = faceNet.forward()


#     predictions = []

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.7:
#             x1 = int(detections[0, 0, i, 3] * frame_width)
#             y1 = int(detections[0, 0, i, 4] * frame_height)
#             x2 = int(detections[0, 0, i, 5] * frame_width)
#             y2 = int(detections[0, 0, i, 6] * frame_height)

#             face_img = img[y1:y2, x1:x2]


#             gender_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#             genderNet.setInput(gender_blob)
#             gender_preds = genderNet.forward()
#             gender = genderList[gender_preds[0].argmax()]


#             age_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#             ageNet.setInput(age_blob)
#             age_preds = ageNet.forward()
#             age = ageList[age_preds[0].argmax()]

#             predictions.append({'gender': gender, 'age': age})

#     return jsonify(predictions=predictions)

# if __name__ == '__main__':
#     app.run(debug=True)


video = cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    frame,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
        face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if face.size == 0:
            print("Warning: Detected face has zero size, skipping this face.")
            continue

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]


        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]


        label="{},{}".format(gender, age)
        cv2.rectangle(frame,(bbox[0],bbox[1]-30), (bbox[2],bbox[1]), (0,255,0),-1)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k ==ord('q'):
        break

video.release()
cv2.destroyAllWindows()