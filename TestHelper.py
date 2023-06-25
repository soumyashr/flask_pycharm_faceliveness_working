import matplotlib.pyplot as plt
from detect_face_roi.DetectFaceLiveness import DetectFaceLiveness
from detect_face_roi.Constants import Constants


def plot_ear(snapshot_time_lst,ear_lst,custom_label):
    const = Constants()
    fig, ax = plt.subplots()
    ax.plot(snapshot_time_lst, ear_lst, label=custom_label)
    ax.hlines(y=const.EYE_AR_THRESH, xmin=0.01, xmax=30, linewidth=2, color='r', label='Threshold EAR value of 0.25')
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()


detect_eye_blink = DetectFaceLiveness("./model/shape_predictor_68_face_landmarks.dat")


# No vide file, ie use web cam
face_liveness,snapshot_time_lst,ear_lst,frame_cnt, ear_cnt_fav= detect_eye_blink.count_eye_blinks()
custom_label = 'EAR variations for 13 real eye blinks under normal light for ' +  str(frame_cnt)+ ' frames'
plot_ear(snapshot_time_lst,ear_lst,custom_label)
# favourable frames / total frames
print('With normal lighting, Success rate: ', ear_cnt_fav/frame_cnt)
print('\n\n\n')
#
# video_file = 'C:\\Users\\soums\\OneDrive\\Pictures\\Camera Roll\\13_Eyes_Blinks_Normal Light.mp4'
# face_liveness,snapshot_time_lst,ear_lst,frame_cnt, ear_cnt_fav= detect_eye_blink.count_eye_blinks(video_file)
# custom_label = 'EAR variations for 13 real eye blinks under normal light for ' +  str(frame_cnt)+ ' frames'
# plot_ear(snapshot_time_lst,ear_lst,custom_label)
# # favourable frames / total frames
# print('With normal lighting, Success rate: ', ear_cnt_fav/frame_cnt)
# print('\n\n\n')



# video_file = 'C:\\Users\\soums\\OneDrive\\Pictures\\Camera Roll\\15_Eye_Blinks_Low Light.mp4'
# face_liveness,snapshot_time_lst,ear_lst,frame_cnt, ear_cnt_fav = detect_eye_blink.count_eye_blinks(video_file)
# custom_label = 'EAR variations for 15 real eye blinks under low light for ' +  str(frame_cnt)+ ' frames'
# plot_ear(snapshot_time_lst,ear_lst,custom_label)
# # favourable frames / total frames
# print('With low lighting, Success rate: ', ear_cnt_fav/frame_cnt)
# print('\n\n\n')

# video_file = 'C:\\Users\\soums\\OneDrive\\Pictures\\Camera Roll\\24_Eyes_Blinks_Reflection Light.mp4'
# face_liveness,snapshot_time_lst,ear_lst,frame_cnt, ear_cnt_fav = detect_eye_blink.count_eye_blinks(video_file)
# custom_label = 'EAR variations for 24 real eye blinks with reflection lighting for ' +  str(frame_cnt)+ ' frames'
# plot_ear(snapshot_time_lst,ear_lst,custom_label)
# # favourable frames / total frames
# print('With reflection lighting, Success rate: ', ear_cnt_fav/frame_cnt)
# print('\n\n\n')

