from skimage.feature import hog
from skimage import io
from scipy import misc
from PIL import Image
image = Image.open('0002.png')
imafter = image.resize((64, 64))
imafter = imafter.convert("L")
normalised_blocks, hog_image = hog(imafter, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualize=True)
print(normalised_blocks)
print(len(normalised_blocks))
hog_image = Image.fromarray(hog_image)
hog_image.show()

