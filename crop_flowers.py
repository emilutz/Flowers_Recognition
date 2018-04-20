import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_interest_area(img, new_size):
	width, height = img.size
	if width < height:
		resized_img = img.resize((new_size, height * new_size // width))
		crop_up = resized_img.size[1] // 2 - new_size // 2
		crop_down = resized_img.size[1] // 2 + new_size // 2
		return resized_img.crop((0, crop_up, resized_img.size[0], crop_down))
	else:
		resized_img = img.resize((width * new_size // height, new_size))
		crop_left = resized_img.size[0] // 2 - new_size // 2
		crop_right = resized_img.size[0] // 2 + new_size // 2
		return resized_img.crop((crop_left, 0, crop_right, resized_img.size[1]))



def process_images():

	read_dir = 'flowers'
	write_dir = 'flowers_cropped'
    
	# iterate over classes
	for clasz in os.listdir(read_dir):

		# iterate over images of a specific class
		for img_name in os.listdir(os.path.join(read_dir, clasz)):
			            
			# load the image
			img = Image.open(os.path.join(read_dir, clasz, img_name))

			# extract area of interest
			img = extract_interest_area(img, 150)

			# save the image
			img.save(os.path.join(write_dir, clasz, img_name))


process_images()