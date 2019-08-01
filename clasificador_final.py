import cv2
import glob, os
import numpy as np 

# directory
os.chdir('C:/Users/agustin/Desktop/clasificadas/')

#cantidad fosil/nofosil
fosil = 0
nofosil = 0

i = 0
for file in glob.glob('*.png'):

	# abro imagen
	img = cv2.imread(file)
	
	if 'nofosil' in file:
		nofosil += 1
	else:
		fosil += 1

	i += 1

#total imágenes
total_img = i

#cuentas finales 
porc_fosil = fosil / i
porc_nofosil = nofosil / i

print('Porcentaje fosil:', porc_fosil)
print('Porcentaje no fosil:', porc_nofosil)

cv2.waitKey(1)
cv2.destroyAllWindows()