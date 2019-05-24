import face_recognition

# Step 1. Load images into numpy arrays.
obama_image = face_recognition.load_image_file("obama.jpeg")
merkel_image = face_recognition.load_image_file("merkel.jpg")
group_image = face_recognition.load_image_file("group.jpeg")

# Step 2. Get the face encodings for each face loaded earlier.
# face_encodings method returns a list of encodings for each face found in an image.
try:
    obama_image_encoding = face_recognition.face_encodings(obama_image)[0]
    merkel_image_encoding = face_recognition.face_encodings(merkel_image)[0]
    group_image_encoding = face_recognition.face_encodings(group_image)
except IndexError:
    print("Wasn't able to locate any faces in one of images. Check the image files.")
    quit()

# Step 3. Indicate unknown persons in group image.
known_faces = [
    obama_image_encoding,
    merkel_image_encoding
]
unknown_faces = group_image_encoding

for unknown_face in unknown_faces:
    recognition_result = face_recognition.compare_faces(known_faces, unknown_face)
    print('Is the unknown face a picture of Obama? {0}'.format(recognition_result[0]))
    print('Is the unknown face a picture of Merkel? {0}'.format(recognition_result[1]))
    print('Is the face even present in a current database? {0}'.format(not True in recognition_result))

print("End of the program")
