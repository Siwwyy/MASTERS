#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#ifdef _DEBUG
#define DBG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#else
#define DBG_NEW new
#endif


#define TINYEXR_USE_OPENMP 1
#define TINYEXR_IMPLEMENTATION
#include "OpenExr.h"
#include "tinyexr.h"



void readRgba1(
	const char fileName[],
	Array2D<Rgba>& pixels,
	int& width,
	int& height)
{
	const char* err = nullptr;
	int ret = LoadEXR(reinterpret_cast<float**>(pixels.GetData()), pixels.GetWidth(), pixels.GetHeight(), fileName, &err);
	if (err) 
	{
		printf("Err %s\n", err);
		assert(false);
	}
	width = pixels.width();
	height = pixels.height();
}

void writeRgba1(
	const char fileName[],
	const Rgba* pixels,
	int width,
	int height)
{
	const char* err = nullptr;
	int ret = SaveEXR(reinterpret_cast<const float*>(pixels), width, height, 4, 0, fileName, &err);
	if (err) 
	{
		printf("Err %s\n", err);
		assert(false);
	}
}

#include <iostream>



int main(int argc, char * argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//_CrtSetBreakAlloc(3065);

	const char* InputFilename = argv[1];
	const char* OutputFilename = argv[2];
	//int SR = 3;
	//if (argc > 3)
	//{
	//	sscanf(argv[3], "%d", &SR);
	//}


	int InputWidth, InputHeight;
	Array2D<Rgba> InputPixels;
	readRgba1(InputFilename, InputPixels, InputWidth, InputHeight);

	int OutputWidth = InputWidth;
	int OutputHeight = InputHeight;
	//Array2D<Rgba> OutputPixels(OutputWidth, OutputHeight);
	Array2D<Rgba> OutputPixels(InputPixels);
	//OutputPixels = std::move(InputPixels);

	//writeRgba1(OutputFilename, &OutputPixels(0, 0), OutputPixels.width(), OutputPixels.height());
	//writeRgba1(OutputFilename, *InputPixels.GetData(), OutputPixels.width(), OutputPixels.height());
	writeRgba1(OutputFilename, &InputPixels(0, 0), InputPixels.width(), InputPixels.height());

	system("pause");
	return EXIT_SUCCESS;
}