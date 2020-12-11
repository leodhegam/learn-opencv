from cv2 import cv2
i = cv2.imread('img/montedegente.jpg')

iPB = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

df = cv2.CascadeClassifier('xml/frontalface.xml')

faces = df.detectMultiScale(iPB,scaleFactor = 1.05, 
minNeighbors = 7,minSize =(30,30),flags = cv2.CASCADE_SCALE_IMAGE)

def escreve(img, texto, cor=(0,0,255)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (40,80), fonte, 1.8, cor, 4,
        cv2.LINE_AA)

for (x, y, w, h) in faces:
    cv2.rectangle(i, (x, y), (x + w, y + h), (0, 255, 255), 7)

escreve(i , str(len(faces)) + " faces encontradas!")
cv2.imwrite(str(len(faces))+' face(s)_encontrada(s).jpg', i)
cv2.waitKey(0)