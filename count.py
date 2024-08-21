import cv2

img = cv2.imread('C:/Users/simon/Desktop/d1f321ca2126a8a318910c4d97d5c89.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rec, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('binary', thresh)

backSubtractor = cv2.createBackgroundSubtractorKNN()
background = backSubtractor.apply(img)

foreground = backSubtractor.apply(img)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    count += 1
cv2.putText(img, "Count: " + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.imshow("foreground", foreground)
if cv2.waitKey(0):
    cv2.destroyAllWindows()


