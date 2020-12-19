#include <stdio.h>

#include <stb/stb_include.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

//extern stbi_uc * stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);


int main(int argc, char const *argv[])
{
	int w, h, n;
	int comp = 3;
	stbi_uc *data = stbi_load("test/lena2.png", &w, &h, &n, comp); if (data) printf("%d %d %d\n", w, h, n); else printf("Failed &n\n");
	stbi_uc *aux = data;

	int bytes = w * h * comp;
	printf("bytes: %d\n", bytes);
	
	for (int i = 0; i < bytes; ++i)
	{
		data[i] = 225-(75*(i%3));
	}
	//printf("[%d]\n", *data);
	stbi_write_png("lena2red.png", w, h, comp, data, 6*3);
	free(data);
	return 0;
}