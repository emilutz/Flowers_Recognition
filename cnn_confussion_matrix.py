import numpy as np
import matplotlib.pyplot as plt


def display_confussion_matrix(conf_matrix):

	flower_dict = {
    0 : 'daisy',
    1 : 'dandelion',
    2 : 'rose',
    3 : 'sunflower',
    4 : 'tulip'
	}

	print("\n")
	print("{:^11}".format(" "), end="")
	for e in flower_dict.values():
		print("|{:^11}".format(e), end="")
	print("")

	for t in range(len(conf_matrix)):
		print("{:^11}".format(flower_dict[t]), end="")
		for p in range(len(conf_matrix)):
			print("|{:^11}".format(conf_matrix[t, p]), end="")
		print("")

	plt.imshow(conf_matrix, cmap='gray')
	plt.xticks(list(flower_dict.keys()), list(flower_dict.values()), rotation=45)
	plt.yticks(list(flower_dict.keys()), list(flower_dict.values()))
	[x.set_color("yellow") for x in plt.gca().get_xticklabels()]
	[y.set_color("yellow") for y in plt.gca().get_yticklabels()]
	plt.show()

