import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import *


if __name__ == "__main__":

	if len(sys.argv) != 3:
		sys.exit('Insert a flower and a number')

	try:
		flower = sys.argv[1]
		number = int(sys.argv[2])

		files = os.listdir(os.path.join('flowers', flower))
		img_name = files[number]
		img = mpimg.imread(os.path.join('flowers', flower, img_name))

		# create colour histogram for the image
		hist = create_histogram_vector(img, 32)

		blue_end = len(hist)
		red_end = blue_end // 3
		green_end = red_end * 2 

		plt.figure(figsize=(10,4))
		ax1 = plt.subplot(121)
		ax1.bar(list(range(red_end)), hist[:red_end], color='red')
		ax1.bar(list(range(red_end, green_end)), hist[red_end:green_end], color='green')
		ax1.bar(list(range(green_end, blue_end)), hist[green_end:blue_end], color='blue')
		ax1.set_title('colour histogram')

		ax2 = plt.subplot(122)
		ax2.imshow(img)
		ax2.set_title(flower + ' image')
		plt.show()


	except ValueError:
		sys.exit('Invalid number !')
	except IndexError:
		sys.exit('The flower index is too big !')

		
