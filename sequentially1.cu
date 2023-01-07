#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 3
const int xSobel[FILTER_WIDTH*FILTER_WIDTH] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
const int ySobel[FILTER_WIDTH*FILTER_WIDTH] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

// void writePnm(uint32_t * pixels, int width, int height, char * fileName)
// {
// 	FILE * f = fopen(fileName, "w");
// 	if (f == NULL)
// 	{
// 		printf("Cannot write %s\n", fileName);
// 		exit(EXIT_FAILURE);
// 	}

// 	fprintf(f, "P2\n");
// 	fprintf(f, "%i\n%i\n255\n", width, height); 

// 	for (int i = 0; i < width * height; i++)
// 		fprintf(f, "%hhu\n", pixels[i]);

// 	fclose(f);
// }

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

void convertRgb2Gray(uchar3 *inPixels, int width, int height, uint32_t *outPixels)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int i = row * width + col;
            uint32_t r = inPixels[i].x;
            uint32_t g = inPixels[i].y;
            uint32_t b = inPixels[i].z;
            outPixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}

void calConvolution(uint32_t *inPixels, int width, int height, uint32_t *outPixels, const int* filter)
{
	for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	{
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
			uint32_t outPixel = 0;
			for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
			{
				for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
				{
					float filterVal = filter[filterR*FILTER_WIDTH + filterC];
					int inPixelsR = outPixelsR - FILTER_WIDTH/2 + filterR;
					int inPixelsC = outPixelsC - FILTER_WIDTH/2 + filterC;

					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);

					uint32_t inPixel = inPixels[inPixelsR * width + inPixelsC];
					outPixel += uint32_t(filterVal * inPixel);
				}
			}
			outPixels[outPixelsR*width + outPixelsC] = outPixel;
		}
	}
}

void calcPixelImportance(uint32_t *inPixels, int width, int height, uint32_t *importanceMatrix) 
{   	
    // Phát hiện cạnh theo chiều x: Convolution với bộ lọc x-sobel
    uint32_t *xEdge = (uint32_t*)malloc(width * height * sizeof(uint32_t));
    calConvolution(inPixels, width, height, xEdge, xSobel);

    // Phát hiện cạnh theo chiều y: Convolution với bộ lọc y-sobel
    uint32_t *yEdge = (uint32_t*)malloc(width * height * sizeof(uint32_t));
    calConvolution(inPixels, width, height, yEdge, ySobel);

    // Tính độ quan trọng của một pixel
    for (int i = 0; i < width * height; i++)
        importanceMatrix[i] = sqrt(xEdge[i]*xEdge[i] + yEdge[i]*yEdge[i]);
        // importanceMatrix[i] = abs(xEdge[i]) + abs(yEdge[i]);

    // Giải phóng vùng nhớ
    free(xEdge);
    free(yEdge);
}

int getMinNeighbourPos(int r, int c, uint32_t *minValMatrix, int width){
	int mid = minValMatrix[(r+1)*width + c];
	int left = (c == 0) ? INT_MAX : minValMatrix[(r+1)*width + (c - 1)];
	int right = (c == width - 1) ? INT_MAX : minValMatrix[(r+1)*width + (c + 1)];
	int minval = left;
	int minpos = -1;
	if (mid < minval) {
		minval = mid;
		minpos = 0;
	}
	if (right < minval) {
		minpos = 1;
	}
	return minpos;
}

void findLeastImportantSeam(uint32_t *importanceMatrix, int width, int height, uint32_t *seam)
{
    uint32_t *minValMatrix = importanceMatrix;
    int *trackCol = (int*)malloc(width * (height - 1) * sizeof(int));
    // Tính độ quan trọng ít nhất tính tới dưới cùng
    for (int r = height - 2; r >= 0; r--)
    {
        for (int c = 0; c < width; c++)
        {
			int minpos = getMinNeighbourPos(r, c, minValMatrix, width);
			trackCol[r*width + c] = minpos;
			minValMatrix[r*width + c] += minValMatrix[(r+1)*width + (c + minpos)];
        }
    }
	
	// Truy vết và tìm seam ít quan trọng nhất
    seam[0] = 0;
    for (int i = 1; i < width; i++)
        if (minValMatrix[i] < minValMatrix[seam[0]])
            seam[0] = i;
    for (int j = 1; j < height; j++)
        seam[j] = seam[j-1] + trackCol[(j-1)*width + seam[j-1]];

    // Free vùng nhớ
    free(trackCol);
}

void removeSeam(void *inPixels, int width, int height, uint32_t *seam, void *outPixels)
{
    for (int r = 0; r < height; r++){
		int outIdx = r * (width - 1);
		for (int c = seam[r] + 1; c < width; c++){
			// if (c == seam[r]){
			// 	continue;
			// }
			outPixels[outIdx] = inPixels[r*width + c];
			outIdx++;
		}
	}
}

void removeSeamGray(uint32_t *inPixels, int width, int height, uint32_t *seam, uint32_t *outPixels)
{
    for (int r = 0; r < height; r++){
		int outIdx = r * (width - 1);
		for (int c = 0; c < width; c++){
			if (c == seam[r]){
				continue;
			}
			outPixels[outIdx] = inPixels[r*width + c];
			outIdx++;
		}
	}
}

void seamCarvingByHost(uchar3 * inPixels, int width, int height, int nSeams, uchar3*& outPixels){
	uchar3 *src_img = inPixels;
	uchar3 *out = (uchar3 *)malloc((width - 1) * height * sizeof(uchar3));
	int expectWidth = width - nSeams;
	int temp = width;

	// Chuyển anh sang grayscale
	uint32_t *grayscale = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	convertRgb2Gray(src_img, width, height, grayscale);

	while (width > expectWidth) {
		if (width < temp){
			out = (uchar3 *)realloc(out, width * height * sizeof(uchar3));
		}
		// Khởi tạo tempGrayscale
		uint32_t *temp_grayscale = (uint32_t *)malloc(width * height * sizeof(uint32_t));

		// Tính toán ma trận Energy
		uint32_t *importanceMatrix = (uint32_t *)malloc(width * height * sizeof(uint32_t));
		calcPixelImportance(grayscale, width, height, importanceMatrix);
		
		// Tìm đường Seam ít quan trọng nhất
		uint32_t *seam = (uint32_t *)malloc(height*sizeof(uint32_t));
		findLeastImportantSeam(importanceMatrix, width, height, seam);
		
		// Xoá đường seam ra khỏi bức ảnh gốc
		removeSeam(src_img, width, height, seam, out);
		// Xoá đường seam ra khỏi ảnh grayscale
		removeSeamGray(grayscale, width, height, seam, temp_grayscale);
		
		src_img = out;
		grayscale = temp_grayscale;

		free(temp_grayscale);
		free(importanceMatrix);
		free(seam);
		width--;
	}
	outPixels = (uchar3 *)malloc(expectWidth * height * sizeof(uchar3));
	outPixels = out;
	free(grayscale);
	free(src_img);
	free(out);
}

int main(int argc, char ** argv)
{
	// Read input image file
	int width, height;
	uchar3* inPixels = NULL;
	readPnm(argv[1], width, height, inPixels)	;
	printf("\nImage size (width x height): %i x %i\n", width, height);

	int nSeams = atoi(argv[3]);
	uchar3* outPixels = NULL;
	seamCarvingByHost(inPixels, width, height, nSeams, outPixels);
	char * outFileNameBase = strtok(argv[2], ".");
	writePnm(outPixels, width - nSeams, height, concatStr(outFileNameBase, "_test.pnm"));
	free(inPixels);
	free(outPixels);
}