from flask import Flask, render_template
# import detect_face_roi
# from detect_face_roi.DetectFaceLivenessAnotherApproach import DetectFaceLivenessAnotherApproach
from detect_face_roi.DetectFaceLiveness import DetectFaceLiveness


app = Flask(__name__)

@app.route('/')
def start():
    return render_template('start1.html')

@app.route('/faceliveness/', methods=['POST'])
def faceliveness():  # put application's code here
    detect_eye_blink = DetectFaceLiveness("./model/shape_predictor_68_face_landmarks.dat")
    face_liveness = detect_eye_blink.count_eye_blinks()
    forward_message = ""
    if face_liveness:
        forward_message = 'The face under observation is a Live Face'
    else:
        forward_message = 'The face under observation is a Spoofed Face'

    print(forward_message)

    return render_template('result.html', forward_message=forward_message)



# ref: https://stackoverflow.com/questions/42601478/flask-calling-python-function-on-button-onclick-event

if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0')
