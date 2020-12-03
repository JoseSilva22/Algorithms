#!/usr/bin/env python

""" 
Floyd-Steinberg Dithering algorithm
http://en.wikipedia.org/wiki/Floyd-Steinberg
"""

# pip install Pillow
from PIL import Image


class Dither:

	def __init__(self, img, dims, quant_level):

		self.img = img
		self.x_size, self.y_size = dims
		self.quant_level = quant_level


	def _find_closest_palette_color(self, oldpixel): 
		r, g, b = oldpixel

		newR = round(self.quant_level * r / 255) * (255 / self.quant_level)
		newG = round(self.quant_level * g / 255) * (255 / self.quant_level)
		newB = round(self.quant_level * b / 255) * (255 / self.quant_level)
		
		return (round(newR), round(newG), round(newB))
	

	def _get_quant_error(self, oldpixel, newpixel):
		oldR, oldG, oldB = oldpixel
		newR, newG, newB = newpixel
		return [oldR-newR, oldG-newG, oldB-newB]


	def _add_quant_error(self, pixel, quant_error, spread):
		return tuple([round(color+i*spread) for color, i in zip(pixel, quant_error)])

	# floyd steinberg
	def fs_dither(self):

		for y in range(self.y_size):
			for x in range(self.x_size):
				oldpixel = self.img.getpixel((x, y))
				newpixel = self._find_closest_palette_color(oldpixel)
				self.img.putpixel((x, y), newpixel)
				quant_error = self._get_quant_error(oldpixel, newpixel)
				
				if (x < self.x_size - 1):
					self.img.putpixel((x+1, y), self._add_quant_error(self.img.getpixel((x+1, y)), quant_error, 7/16)) 
				if (x > 0) and (y < self.y_size - 1):
					self.img.putpixel((x-1, y+1), self._add_quant_error(self.img.getpixel((x-1, y+1)), quant_error, 3/16)) 
				if (y < self.y_size - 1):
					self.img.putpixel((x, y+1), self._add_quant_error(self.img.getpixel((x, y+1)), quant_error, 5/16)) 
				if (x < self.x_size - 1) and (y < self.y_size - 1):
					self.img.putpixel((x+1, y+1), self._add_quant_error(self.img.getpixel((x+1, y+1)), quant_error, 1/16)) 

		return self.img


if __name__=='__main__':

	img = Image.open('test/Michelangelo_David_original.png')
	rgb_img = img.convert('RGB')#.load()

	#TODO: Grayscale (é so 1 int)
	
	quant_level = 8
	D = Dither(rgb_img, img.size, quant_level)
	new_img = D.fs_dither()
	
	new_img.show()
	#input("Press Enter to continue...")
	

	