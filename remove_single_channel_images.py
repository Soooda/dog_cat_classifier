from PIL import Image
import os
import shutil

trash = '/Users/soda/.Trash/'
path = '/Users/soda/Desktop/Cat&Dog/train/Cat/'
l = os.listdir(path)

for f in l:
	p = os.path.join(path, f)
	image = Image.open(p)

	if len(image.getbands()) != 3:
		shutil.move(p, trash)
		print(f'Deleted: {p}')
