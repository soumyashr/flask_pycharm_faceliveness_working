import math

class Constants:


    # Dlib shape predictor ( http://dlib.net/ ) is a pre-trained model which detects 68 face landmarks and
    # was trained on "iBUG 300 - W face landmark dataset"
    MODEL_NAME = "./model/shape_predictor_68_face_landmarks.dat"

    # Threshold for Aspect Ratio, if ( aspect ratio <  EYE_AR_THRESH) , then it will suggest that eye blink happened
    EYE_AR_THRESH = 0.25

    #  SEBR The “normal” frequency (even without contacts) has been observed to be 16 bpm (blinks per minute).
    # Threshold for the number of consecutive frames the eye must be below the threshold
    # from testing, we are analyzing 464 frames in 20 seconds, ie ~ 23 frames per second
    EYE_AR_CONSEC_FRAMES = 3

    # Assumption: In a span of 4 sec, eyes are blinked at least once. This is based on Spontaneous Eyeblink Rate
    # - SEBR , published in https://pubmed.ncbi.nlm.nih.gov/11700965/ . Normally SEBR is 16
    # In how much time (s) you want to complete detection defined
    DETECTION_WINDOW = 20
    SEBR = 16
    BLINK_COUNT_THRESH = math.trunc(DETECTION_WINDOW * SEBR / 60)  # per second