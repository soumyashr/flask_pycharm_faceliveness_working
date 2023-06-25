from flask import Flask



from detect_face_roi.DetectFaceLivenessAnotherApproach import DetectFaceLivenessAnotherApproach

app = Flask(__name__)



@app.route('/')
def detect_face_liveness():  # put application's code here
    detect_eye_blink = DetectFaceLivenessAnotherApproach("./model/shape_predictor_68_face_landmarks.dat")
    face_liveness = detect_eye_blink.count_eye_blinks()
    if face_liveness:
        msg = 'The face under observation is a Live Face'
    else:
        msg = 'The face under observation is a Spoofed Face'
    return msg


if __name__ == '__main__':
    context = ('local.crt', 'local.key')#certificate and key files
    # app.run(debug=True, ssl_context=context)
    app.run(port=443, ssl_context=("localhost+3.pem", "localhost+3-key.pem"))

