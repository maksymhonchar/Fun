import face_recognition


obama_image = face_recognition.load_image_file("obama.jpeg")
merkel_image = face_recognition.load_image_file("merkel.jpg")
grandpa_image = face_recognition.load_image_file("grandpa.jpg")
me2_image = face_recognition.load_image_file("me2.jpg")
me_image = face_recognition.load_image_file("me.jpg")

obama_image_encoding = face_recognition.face_encodings(obama_image)
merkel_image_encoding = face_recognition.face_encodings(merkel_image)
grandpa_image_encoding = face_recognition.face_encodings(grandpa_image)
me2_image_encoding = face_recognition.face_encodings(me2_image)
me_image_encoding = face_recognition.face_encodings(me_image)

known_faces = [
    obama_image_encoding[0],
    merkel_image_encoding[0],
    me2_image_encoding[0],
    grandpa_image_encoding[0]
]


recognition_result = face_recognition.compare_faces(known_faces, me_image_encoding[0])

faces_txt = ['obama', 'merkel', 'me', 'grandpa']
print('Is the face even present in a current database? {0}'.format(True in recognition_result))
for i, result in enumerate(recognition_result):
    print('Is the unknown face a picture of {0} ? - {1}'.format(faces_txt[i], result))

print("End of the program")
