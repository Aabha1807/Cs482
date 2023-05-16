import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def face_detection(image):
    # Read in the image
    img = mpimg.imread(image)

    # Detect the faces in image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)
    print(type(faces))
    print(faces)

    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)

    plt.imshow(img)
    plt.show()

    for x,y,w,h in faces:
       extracted_img = img[y:y+h, x:x+w]
    #    plt.imshow(extracted_img)
    #    plt.show()

    return extracted_img


