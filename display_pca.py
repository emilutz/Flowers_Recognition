import os
import sys
import numpy as np
from functions import display_pca


if __name__ == "__main__":

	if len(sys.argv) != 3:
		sys.exit('Insert a number between 2 and 3 and the feature type')

	try:
		dim = int(sys.argv[1])
		feature_type = sys.argv[2]

		if dim < 2 or dim > 3:
			sys.exit('Insert a number between 2 and 3')
	except ValueError:
		sys.exit('Insert a number between 2 and 3')


	# load the histograms and the labels
	histograms = None
	if feature_type == "rgb":
		histograms = np.load(os.path.join('colour_histograms','histograms.dat'))
	elif feature_type == "hsv":
		histograms = np.load(os.path.join('colour_histograms','hsv.dat'))
	else:
		sys.exit("Invalid feature type")
	labels = np.load(os.path.join('colour_histograms','labels.dat'))

	# call the PCA function
	display_pca(histograms, labels, dim)
