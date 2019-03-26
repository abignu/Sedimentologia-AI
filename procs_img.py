from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

#------filtros------#
#cargo imagen
img = Image.open('CMO-008x10b.JPG')
img = img.convert('L')

#rangos de la imagen
#width, height = img.size
#obtenemos el valor de los pixeles
imgData = np.asarray(img)
#thresholdedData = (imgData > THRESHOLD_VALUE) * 1.0
#print(list(img.getdata()))

#lo hacemos a mano el threshold
maximo = np.amax(imgData)
minimo = np.amin(imgData)
media = (maximo + minimo) / 2.0
thresholdedData = (imgData > media) * 1.0

width = thresholdedData.shape[0]
height = thresholdedData.shape[1]
elementos_totales = width * height
contador = 0
#saco porcentaje
for i in range(width):
	for j in range(height):
		if thresholdedData[i,j] == 1:
			contador += 1
			
terrigenos = contador/elementos_totales * 100
carbonatos = 100 - terrigenos
print('porcentaje terrigenos = {}, porcentaje carbonatos = {}'.format(terrigenos, carbonatos))
#muestro imágenes
#plt.imshow(thresholdedData)
plt.imshow(thresholdedData)
#plt.legend('Terrigenos', 'carbonatos')
plt.show()
#------filtros-----#