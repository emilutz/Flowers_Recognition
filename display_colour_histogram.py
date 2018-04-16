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
		hist_rgb = create_histogram_vector(img, 32)
		hist_hsv = create_hsv_histogram(img, 32)

		blue_end = len(hist_rgb)
		red_end = blue_end // 3
		green_end = red_end * 2 

		plt.figure(figsize=(12,3))
		ax1 = plt.subplot(131)
		ax1.bar(list(range(red_end)), hist_rgb[:red_end], color='red')
		ax1.bar(list(range(red_end, green_end)), hist_rgb[red_end:green_end], color='green')
		ax1.bar(list(range(green_end, blue_end)), hist_rgb[green_end:blue_end], color='blue')
		ax1.set_title('RGB histogram')

		ax2 = plt.subplot(132)
		ax2.bar(list(range(red_end)), hist_hsv[:red_end], color='red')
		ax2.bar(list(range(red_end, green_end)), hist_hsv[red_end:green_end], color='green')
		ax2.bar(list(range(green_end, blue_end)), hist_hsv[green_end:blue_end], color='blue')
		ax2.set_title('HSV histogram')

		ax3 = plt.subplot(133)
		ax3.imshow(img)
		ax3.set_title(flower + ' image')
		plt.show()


	except ValueError:
		sys.exit('Invalid number !')
	except IndexError:
		sys.exit('The flower index is too big !')

		
