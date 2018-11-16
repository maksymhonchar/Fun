from PIL import Image
import face_recognition
import cv2.data

# image = face_recognition.load_image_file('data/1/1.jpg')
# face_locations = face_recognition.face_locations(image)
# print(face_locations)

# known_image = face_recognition.load_image_file("data/1/1.jpg")
# unknown_image = face_recognition.load_image_file("data/1/3.jpg")
# biden_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
# results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
# print(results)

# image = face_recognition.load_image_file("data/1/1.jpg")
# face_locations = face_recognition.face_locations(image)
# print("I found {} face(s) in this photograph.".format(len(face_locations)))
# for face_location in face_locations:
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
#               right))
#     face_image = image[top: bottom, left: right]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()

# video_capture = cv2.VideoCapture(0)
# face_locations = []
# while True:
#     ret, frame = video_capture.read()
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     face_locations = face_recognition.face_locations(small_frame, model="cnn")
#     for top, right, bottom, left in face_locations:
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4
#         face_image = frame[top:bottom, left:right]
#         face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
#         frame[top:bottom, left:right] = face_image
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()


