#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 3
#define BLOCK_SIZE 32
const int xSobel[FILTER_WIDTH*FILTER_WIDTH] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
const int ySobel[FILTER_WIDTH*FILTER_WIDTH] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

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


__global__ void convertRgb2GrayKernel(uchar3 *inPixels, int width, int height, uint32_t* outPixels)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width){
		int i = row * width + col;
		uint32_t r = inPixels[i].x;
		uint32_t g = inPixels[i].y;
		uint32_t b = inPixels[i].z;
		outPixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;
	}
}

__global__ void convertRgb2GrayKernel2(uchar3 *inPixels, int width, int height, uint32_t* outPixels)
{
	// TODO
	// Use streams to overlap memory transfer and computation
}


__global__ void calcPixelImportanceKernel(uint32_t* inPixels, int width, int height, 
	uint32_t* outPixels, int* xSobelFilter, int* ySobelFilter)
{   
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (c >= width || r >= height){
		return;
	}
	
	int32_t xEdge = 0, yEdge = 0;

	for (int filterR = 0; filterR < FILTER_WIDTH; filterR++){
		for (int filterC = 0; filterC < FILTER_WIDTH; filterC++){
			int xSobelVal = xSobelFilter[filterR*FILTER_WIDTH + filterC];
			int ySobelVal = ySobelFilter[filterR*FILTER_WIDTH + filterC];

			int inPixelsR = (r - FILTER_WIDTH/2)  + filterR;
			int inPixelsC = (c - FILTER_WIDTH/2) + filterC;
			
			inPixelsR = min(height - 1, max(0, inPixelsR));
			inPixelsC = min(width - 1, max(0, inPixelsC));
			
			int32_t inPixel = inPixels[inPixelsR * width + inPixelsC];
			xEdge += xSobelVal * inPixel;
			yEdge += ySobelVal * inPixel;
		}
	}
	
	outPixels[r * width + c] = fabsf(float(xEdge)) + fabsf(float(yEdge));
}

__global__ void calcPixelImportanceKernel2(uint32_t* inPixels, int width, int height, 
	uint32_t* outPixels, int* xSobelFilter, int* ySobelFilter)
{ 
	// TODO
	// Use SMEM
}

__global__ void calcPixelImportanceKernel3(uint32_t* inPixels, int width, int height, 
	uint32_t* outPixels, int* xSobelFilter, int* ySobelFilter)
{ 
	// TODO
	// Use SMEM
	// Use streams to overlap memory transfer and computation
}


// get Min Neighbour Position
__global__ void calcMinValAndKeepTrackKernel(uint32_t* below_row, uint32_t* row, int* trackRow, int width){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < width){
		int mid = below_row[col];
		int left = (col == 0) ? INT_MAX : below_row[col - 1];
		int right = (col == width - 1) ? INT_MAX : below_row[col + 1];
		int minval = left;
		int minpos = -1;
		if (mid < minval) {
			minval = mid;
			minpos = 0;
		}
		if (right < minval)
			minpos = 1;
		trackRow[col] =  minpos;
		row[col] += below_row[col + minpos];
	}
}

// get Min Neighbour Position
__global__ void calcMinValAndKeepTrackKernel(uint32_t* below_row, uint32_t* row, int* trackRow, int width){
	// TODO
	// Use streams to overlap memory transfer and computation
}


// Remove Seam for both original image and grayscale image
__global__ void removeSeamKernel(uchar3 *inPixels, uint32_t *grayscale int width, int height, uint32_t *seam, uchar3 *outPixels)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c >= width || r >= height){
		return;
	}

	// create a temp grayscale output
	extern __shared__ uint32_t tmp_grayScale[];
	
	if (c < seam[r]){ // copy the pixels before the seam
		outPixels[r*(width-1) + c] = inPixels[r*width + c];
		tmp_grayScale[r*(width-1) + c] = grayscale[r*width + c];
	}
	else if (c > seam[r]) { // copy the pixels after the seam
		outPixels[r*(width-1) + c - 1] = inPixels[r*width + c];
		tmp_grayScale[r*(width-1) + c - 1] = grayscale[r*width + c];
	}

	__syncthreads(); // Synchronize within each block
	// What if we should have synchronized across blocks?

	// copy the temp grayscale output to the original grayscale output
	grayscale[r*(width-1) + c] = tmp_grayScale[r*(width-1) + c];
}

__global__ void removeSeamKernel2(uchar3 *inPixels, int width, int height, uint32_t *seam, uchar3 *outPixels)
{
	// TODO
	// Use streams to overlap memory transfer and computation
}


void seamCarvingByDevice1(uchar3 * inPixels, int width, int height, int nSeams, uchar3*& outPixels){
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blockSizeMinVal(BLOCK_SIZE);
	dim3 gridSizeMinVal((width - 1) / blockSizeMinVal.x + 1);
	uchar3 *src_img = inPixels;
	uchar3 *out = (uchar3 *)malloc((width - 1) * height * sizeof(uchar3));
	int expectWidth = width - nSeams;
	int temp = width;

	// Chuyển anh sang grayscale
	uchar3* d_src;
	uint32_t* d_grayscale;
	CHECK(cudaMalloc(&d_src, width * height * sizeof(uchar3)));
	CHECK(cudaMalloc(&d_grayscale, width * height * sizeof(uint32_t)));
	CHECK(cudaMemcpy(d_src, src_img, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
	convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_src, width, height, d_grayscale);

	while (width > expectWidth) {
		if (width < temp){
			out = (uchar3 *)realloc(out, (width - 1) * height * sizeof(uchar3));
		}
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		// Tính toán ma trận Energy
		uint32_t *d_importanceMatrix;
		int* d_xSobel;
		int* d_ySobel;
		CHECK(cudaMalloc(&d_xSobel, 9 * sizeof(int)));
		CHECK(cudaMalloc(&d_ySobel, 9 * sizeof(int)));
		CHECK(cudaMalloc(&d_importanceMatrix, width * height * sizeof(uint32_t)));
		CHECK(cudaMemcpy(d_xSobel, xSobel, 9 * sizeof(int), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_ySobel, ySobel, 9 * sizeof(int), cudaMemcpyHostToDevice));
		calcPixelImportanceKernel<<<gridSize, blockSize>>>(d_grayscale, width, height, d_importanceMatrix, d_xSobel, d_ySobel);

		// Tìm đường Seam ít quan trọng nhất
		uint32_t* minValMatrix = (uint32_t*)malloc(width * height * sizeof(uint32_t));
		int *trackCol = (int*)malloc(width * (height - 1) * sizeof(int));
		CHECK(cudaMemcpy(minValMatrix, d_importanceMatrix, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		    // Tính độ quan trọng ít nhất của từng điểm từ trên xuống dưới
		for (int r = height - 2; r >= 0; r--){
			int offset = r * width;
			uint32_t* d_row;
			uint32_t* d_below_row;
			int *d_track_row;
			CHECK(cudaMalloc(&d_row, width * sizeof(uint32_t)));
			CHECK(cudaMalloc(&d_below_row, width * sizeof(uint32_t)));
			CHECK(cudaMalloc(&d_track_row, width * sizeof(int)));
			CHECK(cudaMemcpy(d_row, minValMatrix+offset, width * sizeof(uint32_t), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(d_below_row, minValMatrix+offset+width, width * sizeof(uint32_t), cudaMemcpyHostToDevice));
			calcMinValAndKeepTrackKernel<<<gridSizeMinVal, blockSizeMinVal>>>(d_below_row, d_row, d_track_row, width);
			CHECK(cudaMemcpy(minValMatrix+offset, d_row, width * sizeof(uint32_t), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(trackCol+offset, d_track_row, width * sizeof(uint32_t), cudaMemcpyDeviceToHost));
			CHECK(cudaFree(d_row));
			CHECK(cudaFree(d_below_row));
			CHECK(cudaFree(d_track_row));
		}
		// Truy vết và tìm seam ít quan trọng nhất
		uint32_t *seam = (uint32_t *)malloc(height*sizeof(uint32_t));
		seam[0] = 0;
		for (int i = 1; i < width; i++)
			if (minValMatrix[i] < minValMatrix[seam[0]])
				seam[0] = i;
		for (int j = 1; j < height; j++)
			seam[j] = seam[j-1] + trackCol[(j-1)*width + seam[j-1]];

		for (int j = 1; j < height; j++){
			printf("%i->", seam[j]);
		}
		printf("\n");

		// Xoá đường seam ra khỏi bức ảnh gốc
		uchar3* d_out;
		uint32_t* d_seam;
		CHECK(cudaMalloc(&d_out, (width - 1) * height * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_seam, height * sizeof(uint32_t)));
		CHECK(cudaMemcpy(d_seam, seam, height * sizeof(uint32_t), cudaMemcpyHostToDevice));
		removeSeamKernel<<<gridSize,blockSize>>>(d_src, width, height, d_seam, d_out);
		CHECK(cudaMemcpy(out, d_out, (width - 1) * height * sizeof(uchar3), cudaMemcpyDeviceToHost));
		src_img = out;
		CHECK(cudaFree(d_src));
		CHECK(cudaFree(d_grayscale));
		CHECK(cudaFree(d_importanceMatrix));
		CHECK(cudaFree(d_xSobel));
		CHECK(cudaFree(d_ySobel));
		CHECK(cudaFree(d_out));
		CHECK(cudaFree(d_seam));
		free(seam);
		width--;
	}
	outPixels = out;
	free(src_img);
	free(out);
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
	CHECK(cudaGetDeviceProperties(&devProv, 0));
	printf("**********GPU info**********\n");
	printf("Name: %s\n", devProv.name);
	printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
	printf("Num SMs: %d\n", devProv.multiProcessorCount);
	printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
	printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
	printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
	printf("CMEM: %lu bytes\n", devProv.totalConstMem);
	printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
	printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

	printf("****************************\n");

}

int main(int argc, char ** argv)
{
	printDeviceInfo();

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);
	
	int nSeams = atoi(argv[3]);
	uchar3* outPixels = (uchar3 *)malloc((width - nSeams) * height * sizeof(uchar3));
	GpuTimer timer;
	timer.Start();
	seamCarvingByDevice1(inPixels, width, height, nSeams, outPixels);
	timer.Stop();
	float time = timer.Elapsed();
	printf("Device version 1 - Number of seams = %i - Kernel time = %f ms\n", nSeams, time);
	char * outFileNameBase = strtok(argv[2], ".");
	writePnm(outPixels, width - nSeams, height, concatStr(outFileNameBase, "_test.pnm"));
	free(inPixels);
	free(outPixels);
}