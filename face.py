import sys
import os
from multiprocessing import Process

import cv2
import face_recognition
from pynput import keyboard
from pyzbar import pyzbar


######################################################################
#
# Init variables - used as ''global''
#
######################################################################

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
u1_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user1.jpg"))[0]
u2_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user2.jpg"))[0]
u3_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user3.jpg"))[0]
u4_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user4.jpg"))[0]
u5_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user5.jpg"))[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    u1_face_encoding,
    u2_face_encoding,
    u3_face_encoding,
    u4_face_encoding,
    u5_face_encoding,
]

known_face_names = [
    None,
    None,
    None,
    None,
    None,
]

######################################################################
#
# Definition of keyboard listener
#
######################################################################


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    global known_face_names
    print('{0} released'.format(
        key))
    key_as_int = int(key.char)
    print (key_as_int)
    print (key_as_int in [1,2,3,4,5])
    if key_as_int in [1,2,3,4,5]:
        if known_face_names[key_as_int - 1] is not None:
            known_face_names[key_as_int - 1] = None
        else:
            known_face_names[key_as_int - 1] = "user" + str(key_as_int)
    print(known_face_names)

listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)


def play_sys_audio():
    duration = 0.6  # second
    freq = 880  # Hz
    os.system('play -t alsa --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

######################################################################
#
# Definition of face recognition system
#
######################################################################

def face_rec(mode):
    print("face_rec start")
    print(mode)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    current_emoji = ''


    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which cv2 uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 10, bottom + 15), font, 1.0, (255, 255, 255), 1)

            unknown_face_emoji = ""

            if mode == "neutral":
                unknown_face_emoji = "images/neutral.png"
            else:
                unknown_face_emoji = "images/angry_face_imp.png"

            global face_change
            if name != "Unknown" and name is not None:
                current_emoji = 'good'
                image = cv2.imread("images/love.png", -1)
            else:
                # play just one time if face emoji is changing
                image = cv2.imread(unknown_face_emoji, -1)
                if mode != "neutral":
                    try:
                        play_sys_audio_thread.terminate()
                    except:
                        print("play_sys_audio_thread not started")
                    try:
                        play_sys_audio_thread = Process(target=play_sys_audio)
                        play_sys_audio_thread.start()
                    except:
                        print("play_sys_audio_thread not terminated")

                current_emoji = 'bad'

            emoji = cv2.resize(image, (right-left, bottom-top))

            x_offset=left-1
            y_offset=top-1
            y1, y2 = y_offset, y_offset + emoji.shape[0]
            x1, x2 = x_offset, x_offset + emoji.shape[1]

            alpha_s = emoji[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            try:
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])            
            except:
              print("An exception occurred")

        ######################################################################
        # Barcode detection
        ######################################################################
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw the
            # bounding box surrounding the barcode on the image
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)
            barcodeData = barcode.data.decode("utf-8")
            cv2.putText(frame, barcodeData + "-Validated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

            # TODO use regex to detect "userX" string
            if len(barcodeData) == 5 and barcodeData not in known_face_names:
                # if barcode data has a number ex. user3 -> 3
                index = int(list(filter(str.isdigit, barcodeData))[0])
                if index > 0 and index < 6:
                    known_face_names.insert(index-1, barcodeData)


        # Display the resulting image
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)

        key = cv2.waitKey(33)

        if key > 0:
            key_as_int = int(key) - 48
            if key_as_int in [1,2,3,4,5]:
                if known_face_names[key_as_int - 1] is not None:
                    known_face_names[key_as_int - 1] = None
                else:
                    known_face_names[key_as_int - 1] = "user" + str(key_as_int)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()



######################################################################
#
# Starting threads: face_recognition display / keyboard listener / flask server app
#
######################################################################


if __name__ == "__main__":

    if len(sys.argv) == 2:
        face_rec(str(sys.argv[1]))
    else:
        face_rec("")