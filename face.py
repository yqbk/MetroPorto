import threading

import face_recognition
from pyzbar import pyzbar
import cv2
from pynput import keyboard



# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires cv2 (the `cv2` library) to be installed only to read from your webcam.
# cv2 is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

######################################################################
#
# Init variables - used as ''global''
#
######################################################################

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
# obama_image = face_recognition.load_image_file("/images/user-2.jpg")
# obama_image = face_recognition.load_image_file("user-2.jpg")
u1_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user-1.jpg"))[0]
u2_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user-2.jpg"))[0]
u3_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user-3.jpg"))[0]
u4_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user-4.jpg"))[0]
u5_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("images/user-5.jpg"))[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    u1_face_encoding,
    u2_face_encoding,
    u3_face_encoding,
    u4_face_encoding,
    u5_face_encoding,
]

known_face_names = [
    "User 1",
    "User 2",
    "User 3",
    "User 4",
    "User 5",
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
            known_face_names[key_as_int - 1] = "User " + str(key_as_int)
    print(known_face_names)

listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)


######################################################################
#
# Definition of face recognition system
#
######################################################################

def face_rec():
    print("face_rec start")
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

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

            # Draw a box around the face
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 10, bottom + 15), font, 1.0, (255, 255, 255), 1)

            global face_change
            if name != "Unknown" and name is not None:
                image = cv2.imread("images/love.png", -1)
            else:
                image = cv2.imread("images/angry.png", -1)

            emoji = cv2.resize(image, (right-left, bottom-top))

            x_offset=left-1
            y_offset=top-1
            y1, y2 = y_offset, y_offset + emoji.shape[0]
            x1, x2 = x_offset, x_offset + emoji.shape[1]

            alpha_s = emoji[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

            # frame = cv2.add(frame, emoji)
            # x_offset=y_offset=50
            # frame[y_offset:y_offset+emoji.shape[0], x_offset:x_offset+emoji.shape[1]] = emoji

            # cv2.imwrite(emoji, frame)

                barcodes = pyzbar.decode(frame)

        # loop over the detected barcodes
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw the
            # bounding box surrounding the barcode on the image
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
            barcodeData = barcode.data.decode("utf-8")
        
            # draw the barcode data and barcode type on the image
            text = barcodeData 
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
        
            # print the barcode type and data to the terminal
            print(barcodeData)

        # Display the resulting image
        cv2.imshow('Video', frame)

        key = cv2.waitKey(33)

        if key > 0:
            key_as_int = int(key) - 48
            # print (key_as_int)
            # print (key_as_int in [1,2,3,4,5])
            if key_as_int in [1,2,3,4,5]:
                if known_face_names[key_as_int - 1] is not None:
                    known_face_names[key_as_int - 1] = None
                else:
                    known_face_names[key_as_int - 1] = "User " + str(key_as_int)
            print(known_face_names)

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
    face_rec()
    # face_rec_thread = threading.Thread(target=face_rec)
    # face_rec_thread.daemon = True

    # listener.start()
    # face_rec_thread.start()
    # server_app.start()

    # listener.join()
    # face_rec_thread.join()
    # server_app.join()