#!/usr/bin/env python

""" 
https://en.wikipedia.org/wiki/Steganography
"""

# pip install Pillow
from PIL import Image
import io
import numpy as np

class Stegano:

	def __init__(self, img_path=None):
		self.set_img(img_path)


	def set_img(self, img_path):
		if img_path is not None:
			try:
				self.img = Image.open(img_path)
			except Exception as e:
				raise e


	def hide(self, img_hide_path):
		try:
			img_hide = Image.open(img_hide_path)
			
			pix = self.img.load()
			pix_hide = img_hide.load()
			x, y = self.img.size
			x_hide, y_hide = img_hide.size

			for i in range(x_hide):
				for j in range(y_hide):
					for byte in range(8):
						p = pix[(i*j+byte)/x, (i*j+byte)%y]
						p_h = pix_hide[i,j]
						a,b,c = p_h[(byte*3)//8] >> (7-(byte*3)//8) & 1, p_h[(byte*3+1)//8] >> (7-(byte*3+1)//8) & 1, p_h[(byte*3+2)//8] >> (7-(byte*3+2)//8) & 1
						pix[(i*j+byte)/x, (i*j+byte)%y] = (p[0], p[1], p[2])#(255,255,255)# 

			self.img.save('hidden.jpg')

		except Exception as e:
			raise e
	


if __name__=='__main__':

	#img = Image.open('test/lena.png')
	#rgb_img = img.convert('RGB')#.load()

	steg = Stegano('test/parrot.jpg')
	steg.hide('test/lena.png')
	
	'''
	im = Image.open('test/dog.jpg') # Can be many different formats.
	pix = im.load()
	x,y=1000,750
	print (im.size)  # Get the width and hight of the image for iterating over
	print (pix[x,y])  # Get the RGBA Value of the a pixel of an image
	print(type(pix[x,y][0]))
	pix[x,y] = (255,0,0)  # Set the RGBA Value of the image (tuple)
	#im.save('test/dog2.png')  # Save the modified pixels as .png
	'''

	'''
	img = Image.open('test/dog.jpg') # Can be many different formats.
	img_bytes = img.resize(img.size)
	buf = io.BytesIO()
	img_bytes.save(buf, format='JPEG')
	img_byte_arr = list(buf.getvalue())

	for x in range(len(img_byte_arr)):
		img_byte_arr[x] = 255

	i = Image.fromarray(np.asarray(img_byte_arr).reshape(img.size), mode='RGB')
	i.save('dog2.jpg', 'JPEG')
	'''

	'''
	pix = im.load()
	x,y=1000,750
	print (im.size)  # Get the width and hight of the image for iterating over
	print (pix[x,y])  # Get the RGBA Value of the a pixel of an image
	print(type(pix[x,y][0]))
	pix[x,y] = (255,0,0)  # Set the RGBA Value of the image (tuple)
	#im.save('test/dog2.png')  # Save the modified pixels as .png
	'''
