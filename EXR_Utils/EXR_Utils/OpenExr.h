#ifndef OPEN_EXR_H_INCLUDED
#define OPEN_EXR_H_INCLUDED

#include <utility>

class Rgba {
public:
	float r;
	float g;
	float b;
	float a;
};


template < typename T >
class Array2D {
private:
	T* pData;
	int W;
	int H;
public:
	Array2D() :
		pData(nullptr),
		W(0),
		H(0)
	{}

	Array2D(int Width, int Height)
	{
		pData = new T[Width * Height];
		W = Width;
		H = Height;
	}
	
	~Array2D()
	{
		delete[] pData;
	}
	
	T** GetData() { return &pData; }
	int* GetWidth() { return &W; }
	int* GetHeight() { return &H; }
	int width() const { return W; }
	int height() const { return H; }
	T& operator()(int y, int x) {
		return *(pData + W * y + x);
	}

	//
	Array2D(const Array2D& Object) :
		W(Object.W),
		H(Object.H)
	{
		pData = new T[W * H];
		for (std::size_t i = 0; i < H * W; i++)
		{
			pData[i] = Object.pData[i];
		}
	}

	Array2D(Array2D&& Object) noexcept :
		W(std::exchange(Object.W, {})),
		H(std::exchange(Object.H, {})),
		pData(std::exchange(Object.pData, nullptr))
	{	}

	Array2D & operator=(const Array2D & Object)
	{
		if (this != &Object)
		{
			W = Object.W;
			H = Object.H;
			delete[] pData;
			
			pData = new T[W * H];
			for (std::size_t i = 0; i < H * W; i++)
			{
				pData[i] = Object.pData[i];
			}
		}
		return *this;
	}

	Array2D & operator=(Array2D && Object) noexcept
	{
		if (this != &Object)
		{
			W = std::exchange(Object.W, {});
			H = std::exchange(Object.H, {});
			delete[] pData;
			pData = std::exchange(Object.pData, nullptr);
		}
		return *this;
	}
};
#endif /* OPEN_EXR_H_INCLUDED */