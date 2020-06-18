import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imagenumber=3700
for i in range(imagenumber):
	if i%2==0:
		n=str(int(i))
		source='circle/'+n+'.png'
		n=str(int(i/2))
		dest='Train/circle/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
		source='square/'+n+'.png'
		dest='Train/square/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
		source='star/'+n+'.png'
		dest='Train/star/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
		source='triangle/'+n+'.png'
		dest='Train/triangle/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
i=0
for i in range(imagenumber):
	if i%2==1:
		n=str(int((i)))
		source='circle/'+n+'.png'
		n=str(int((i-1)/2))
		dest='Test/circle/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
		source='square/'+n+'.png'
		dest='Test/square/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
		source='star/'+n+'.png'
		dest='Test/star/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
		source='triangle/'+n+'.png'
		dest='Test/triangle/'+n+'.png'
		img = cv.imread(source,0)
		edges = cv.Canny(img,100,200)
		"""plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()
		"""
		resized=cv.resize(edges,(28,28),interpolation = cv.INTER_AREA)
		(thresh, blackAndWhiteImage) = cv.threshold(resized, 20, 255, cv.THRESH_BINARY)
		cv.imwrite(dest, blackAndWhiteImage)
