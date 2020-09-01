//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <cuComplex.h>

//NPP
#include<npp.h>
#include<nppi_geometry_transforms.h>
#include<nppdefs.h>

//CUFFT
#include<cufft.h>
#include<cufftXt.h>

//OPENCV
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>

//THRUST
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/extrema.h>


//CPP
#include<stdio.h>
#include<iostream>
#include<chrono>
#include<vector>
#include<cmath>
#include<assert.h>
#include<stdexcept>
#include<sstream>
#include<type_traits>
#include<fstream>
#include<cstdlib>
#include<utility>

#include "helper_cuda.h"
#include "convolutionSeparable.h"

#include "phaseCorrelation.h"

int testfunc() {
	return 4;
}

__device__ __host__ inline cufftComplex operator*(const cufftComplex a, const cufftComplex b) {
	cufftComplex c;
	c.x = (a.x * b.x) - (a.y * b.y);
	c.y = (a.x * b.y) + (a.y * b.x);
	return c;
}

__device__ __host__ inline cufftComplex operator/(const cufftComplex a, const float b) {
	cufftComplex c;
	c.x = a.x / b;
	c.y = a.y / b;
	return c;
}

__device__ __host__ inline float mag(const cufftComplex a) {
	float xsq, ysq;
	xsq = a.x * a.x;
	ysq = a.y *a.y;
	return sqrtf(xsq + ysq);
}

__device__ __host__ inline cufftComplex cong(const cufftComplex a) {
	cufftComplex c;
	c.x = a.x;
	c.y = -a.y;
	return c;
}

__device__ __host__ inline cufftComplex zero() {
	cufftComplex c;
	c.x = 0.0;
	c.y = 0.0;
	return c;
}

struct bigger_real {
	__device__ __host__
	cufftComplex operator()(const cufftComplex a, const cufftComplex b) {
		if (a.x > b.x) return a;
		else return b;
	}
};

struct bigger_mag {
	__device__ __host__
	cufftComplex operator()(const cufftComplex a, const cufftComplex b) {
		if (mag(a) > mag(b)) return a;
		else return b;
	}
};

struct mag_comp {
	__device__ __host__
	bool operator()(const cufftComplex a, const cufftComplex b) {
		return (mag(a) < mag(b));
	}
};

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Assert Failed: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define NPP_CHECK(ans) { nppAssert((ans), __FILE__, __LINE__); }
inline void nppAssert(NppStatus code, const char* file, int line, bool abort = true) {
	if (code != NPP_SUCCESS) {
		std::cout << "CODE: " << code << std::endl;
		fprintf(stderr, "CUDA-NPP Assert Failed: %d %s %d\n", _cudaGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}

}

struct free_delete
{
	void operator()(void* x) {
		free(x);
	}
};

void writeBinary(std::string filename, float* data, const unsigned int n) {
	std::fstream file;
	file = std::fstream(filename, std::ios::out | std::ios::binary);
	file.write((char*)data, n * sizeof(float));
	file.close();
}

void writeBinary(std::string filename, cufftComplex* data, const unsigned int n) {
	std::fstream file;
	file = std::fstream(filename, std::ios::out | std::ios::binary);
	file.write((char*)data, n * sizeof(cufftComplex));
	file.close();
}

void writeBinary(std::string filename, Npp8u* data, const unsigned int n) {
	std::fstream file;
	file = std::fstream(filename, std::ios::out | std::ios::binary);
	file.write((char*)data, n * sizeof(Npp8u));
	file.close();
}

class Formatter
{
public:
	Formatter() {}
	~Formatter() {}

	template<typename Type>
	Formatter& operator << (const Type& value) {
		stream_ << value;
		return *this;
	}

	std::string str() const { return stream_.str(); }
	operator std::string() const { return stream_.str(); }

	enum ConvertToString {
		to_str
	};

	std::string operator >> (ConvertToString) { return stream_.str(); }

private:
	std::stringstream stream_;

	Formatter(const Formatter&);
	Formatter& operator = (Formatter&);

};


template<typename T>
class DualData {
public:
	T* host_data;
	T* device_data = NULL;
	bool hstate = false;
	bool dstate = false;

	size_t total_size = 0;
	size_t active_size = 0;

	size_t total_bytes = 0;
	size_t active_bytes = 0;

	std::string name = "Unnamed";

	DualData() {};

	//template<typename T, typename = std::enable_if_t<std::is_same<Npp8u, T>::value> >
	DualData(std::string name, cv::Mat Image) {
		cv::Size shape = Image.size();
		assert(Image.channels() == 1);
		size_t image_size = shape.area();

		this->total_size = image_size;
		this->active_size = image_size;

		this->active_bytes = image_size * sizeof(T);
		this->total_bytes = image_size * sizeof(T);

		this->host_data = static_cast<T*>(malloc(this->total_bytes));
		std::memcpy(this->host_data, static_cast<T*>(Image.data), this->active_bytes);
		this->hstate = true;

		this->name = name;
	}

	DualData(T* h_data, T* d_data, size_t size) {
		this->host_data = h_data;
		this->hstate = true;

		this->device_data = d_data;
		this->dstate = true;

		this->total_size = size;
		this->active_size = size;
	}

	DualData(T* h_data, T* d_data, size_t active_size, size_t total_size) {
		this->host_data = h_data;
		this->hstate = true;

		this->device_data = d_data;
		this->dstate = true;

		this->active_size = active_size;
		this->total_size = total_size;

		this->active_bytes = active_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);
	}

	DualData(std::string name, T* data, size_t total_size, std::string type) {
		if (type == "HOST") {
			this->host_data = data;
			this->hstate = true;
		}
		else if (type == "DEVICE") {
			this->device_data = data;
			this->dstate = true;
		}

		this->total_size = total_size;
		this->active_size = total_size;

		this->active_bytes = total_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);
		this->name = name;
	}

	DualData(T* data, size_t active_size, size_t total_size, std::string type) {
		if (type == "HOST") {
			this->host_data = data;
			this->hstate = true;
		}
		else if (type == "DEVICE") {
			this->device_data = data;
			this->dstate = true;
		}

		this->total_size = total_size;
		this->active_size = active_size;

		this->active_bytes = active_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);
	}

	DualData(std::string name, size_t total_size) {
		this->active_size = total_size;
		this->total_size = total_size;

		this->active_bytes = total_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);

		this->name = name;
	}

	void setDevice(T* ddata, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->device_data = ddata;
			this->dstate = true;

			this->active_size = total_size;
			this->total_size = total_size;

			this->active_bytes = total_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setDevice: Cannot set after either Host or Device data has been initialized");
		}
	}

	void setDevice(T* ddata, size_t active_size, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->host_data = ddata;
			this->dstate = true;

			this->active_size = active_size;
			this->total_size = total_size;

			this->active_bytes = active_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setDevice: Cannot set after either Host or Device data has been initialized");
		}
	}

	void setHost(T* hdata, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->host_data = hdata;
			this->hstate = true;

			this->active_size = total_size;
			this->total_size = total_size;

			this->active_bytes = total_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setHost: Cannot set after either Host or Device data has been initialized");
		}
	}

	void setHost(T* hdata, size_t active_size, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->host_data = hdata;
			this->hstate = true;

			this->active_size = active_size;
			this->total_size = total_size;

			this->active_bytes = active_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setHost: Cannot set after either Host or Device data has been initialized");
		}
	}

	void createHost() {
		if (!this->hstate) {
			this->host_data = static_cast<T*>(malloc(this->total_bytes));
			this->hstate = true;
			std::cout << name << " has been allocated." << std::endl;
		}
		else {
			std::cout << name << " has already been set!" << std::endl;
		}

	}

	void createDevice() {
		if (!this->dstate) {
			CUDA_CHECK(cudaMalloc((void**)&this->device_data, this->total_bytes));
			this->dstate = true;
			std::cout << name << " has been allocated." << std::endl;
		}
		else {
			std::cout << name << " has already been set!" << std::endl;
		}
	}

	void pull() {
		if (this->dstate) {
			if (!this->hstate) {
				this->createHost();
			}
			CUDA_CHECK(cudaMemcpy(this->host_data, this->device_data, this->total_bytes, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "WARNING" << std::endl;
		}
	}

	void push() {
		if (this->hstate) {
			if (!this->dstate) {
				this->createDevice();
			}
			CUDA_CHECK(cudaMemcpy(this->device_data, this->host_data, this->total_bytes, cudaMemcpyHostToDevice));
		}
		else {
			std::cout << "WARNING" << std::endl;
		}
}

	template<typename N>
	DualData<N> convertGPU() {
		size_t old_unit_size = sizeof(T);
		size_t new_unit_size = sizeof(N);
		if (new_unit_size * this->active_size > this->total_bytes) {
			throw std::runtime_error(Formatter() << "Failed to convert. New type requires " << new_unit_size * this->active_size*new_unit_size << " bytes. Only  " << this->total_bytes << " bytes have been allocated");
		}

		N* host_data = reinterpret_cast<N*>(this->host_data);
		N* device_data = reinterpret_cast<N*>(this->host_data);

		//PUT A CUDA KERNEL HERE TO CONVERT

		return DualData(host_data, device_data, this->active_size, this->total_size);
	}

	void destroyDevice() {
		if (this->dstate) {
			this->dstate = false;
			CUDA_CHECK( cudaFree(this->device_data) );
			this->device_data = NULL;
		}
	}

	void destroyHost() {
		if (this->hstate) {
			this->hstate = false;
			free(this->host_data);
			this->host_data = NULL;
		}
	}

	void clean() {
		this->destroyDevice();
		this->destroyHost();
	}

	~DualData() {
		//this->clean();
	}
};

class Parameter {
public:
	Npp64f start = 0;
	Npp64f end = 0;
	unsigned int N = 0;
	Npp64f delta = 0;
	unsigned int initial = 0;
	Npp64f result = -1.0;

	Parameter(Npp64f start, Npp64f end, unsigned int N) {
		this->start = start;
		this->end = end;
		this->N = N;
		this->delta = (end - start) / N;
		this->initial = 0;
	}
	Parameter() = default;

	~Parameter() {};

	void increment(unsigned int k) {
		this->start = this->start + k * this->delta;
	}
};

class SearchContext {
public:
	Parameter theta = Parameter(0, 0, 0);
	Parameter scale_w = Parameter(0, 0, 0);
	Parameter scale_h = Parameter(0, 0, 0);
	float value = 0;

	SearchContext(Parameter theta, Parameter scale_w, Parameter scale_h) {
		this->theta = theta;
		this->scale_w = scale_w;
		this->scale_h = scale_h;
	}

	SearchContext() = default;
	~SearchContext() {};
};

template<typename T>
class WarpContext {
public:
	DualData<T> image;
	DualData<T> batch;
	cv::Size image_shape;
	unsigned int nBatch = 0;

	DualData<Npp64f> transformBuffer;
	DualData<NppiWarpAffineBatchCXR> batchList;
	DualData<cufftComplex> workspace;

	WarpContext() {};

	WarpContext(DualData<T>& image, cv::Size shape, unsigned int nBatch) {
		this->image = image;
		this->image_shape = shape;
		this->nBatch = nBatch;

		size_t transform_buffer_size = 6 * nBatch;
		this->transformBuffer = DualData<Npp64f>("Warp Context: transformBuffer", transform_buffer_size);

		size_t list_buffer_size = nBatch;
		this->batchList = DualData<NppiWarpAffineBatchCXR>("Warp Context: batchList", list_buffer_size);

		size_t batch_buffer_size = 2 * shape.area() * nBatch;
		this->batch = DualData<T>("Warp Context: batch", batch_buffer_size);

		size_t ws_buffer_size = nBatch * shape.area();
		this->workspace = DualData<cufftComplex>("Warp Context: Complex Workspace", ws_buffer_size);
	};

	~WarpContext() {};

};

template<typename T>
class RegistrationContext {
public:
	DualData<T> ReferenceImage;
	DualData<T> MovingImage;

	DualData<T> batchMoving;

	RegistrationContext() {};
	~RegistrationContext() {};

};


void showImage(cv::Mat &Image, std::string title) {
	cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
	cv::imshow(title, Image);
	cv::waitKey(0);
}

void print(std::string name, Npp8u* v, unsigned int n) {
	std::cout << name << std::endl;
	for (unsigned int i = 0; i < n; ++i) {
		std::cout << v[i] << std::endl;
	}
}


void print(std::string name, float* v, unsigned int n) {
	std::cout << name << std::endl;
	for (unsigned int i = 0; i < n; ++i) {
		std::cout << v[i] << std::endl;
	}
}

void print(std::string name, cufftComplex* v, unsigned int n) {
	std::cout << name << std::endl;
	for (unsigned int i = 0; i < n; ++i) {
		std::cout << "(" << v[i].x << "," << v[i].y << ")" << std::endl;
	}
}

Npp8u* toInt(const cv::Mat& Image) {
	return reinterpret_cast<Npp8u*>(Image.data);
}

cv::Mat toMat(Npp8u* image, const cv::Size& shape) {
Npp8u* cimage = static_cast<Npp8u*>(malloc(sizeof(Npp8u) * shape.area()));
std::memcpy(cimage, image, sizeof(Npp8u)* shape.area());
cv::Mat Image_mat = cv::Mat(shape, CV_8UC1, cimage);
return Image_mat;
}

cv::Mat toMat(float* image, const cv::Size& shape) {
	float* cimage = static_cast<float*>(malloc(sizeof(float) * shape.area()));
	std::memcpy(cimage, image, sizeof(float) * shape.area());
	cv::Mat Image_mat = cv::Mat(shape, CV_32FC1, cimage);
	cv::Mat Image_out;
	Image_mat.convertTo(Image_out, CV_8UC1);
	return Image_out;
}

cv::Mat toMat(cufftComplex* image, const cv::Size& shape, std::string part = "MAG") {
	float* cimage = static_cast<float*>(malloc(sizeof(float) * shape.area()));

	if (part == "REAL") {
		for (unsigned int i = 0; i < shape.area(); ++i) {
			cimage[i] = image[i].x;
		}
	}
	else if (part == "IMAG") {
		for (unsigned int i = 0; i < shape.area(); ++i) {
			cimage[i] = image[i].x;
		}
	}
	else if (part == "MAG") {
		for (unsigned int i = 0; i < shape.area(); ++i) {
			float r = image[i].x;
			float c = image[i].y;
			cimage[i] = sqrtf((r) * (r)+(c) * (c));
		}
	}

	cv::Mat Image_mat = cv::Mat(shape, CV_32FC1, cimage);

	cv::Mat Image_out;
	Image_mat.convertTo(Image_out, CV_8UC1);
	return Image_out;
}



void setGradientKernel() {
	float* h_Kernel;
	h_Kernel = (float*)malloc(KERNEL_LENGTH * sizeof(float));
	if (h_Kernel == NULL) {
		throw  std::runtime_error("Failed to allocate memory for gradient kernel.");
	}

	h_Kernel[0] = float(1.0) / 12;
	h_Kernel[1] = -float(2.0) / 2;
	h_Kernel[2] = 0.0;
	h_Kernel[3] = float(2.0) / 3;
	h_Kernel[4] = float(1.0) / 12;

	setConvolutionKernel(h_Kernel);
}

template<typename T, typename N>
__global__ void copy_convert_kernel(T* __restrict__ output, N* __restrict__ input, const unsigned int n) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		output[i] = (T)input[i];
	}
}

__global__ void convert_complex(cufftComplex* __restrict__ output, float* __restrict__ gx, float* __restrict__ gy, const unsigned int n) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		output[i].x = gx[i];
		output[i].y = gy[i];
	}
}

__global__ void pc_kernel(cufftComplex* __restrict__ output, cufftComplex* __restrict__ ref, const unsigned int n, const unsigned int chunk) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		cufftComplex elref = ref[i % chunk];
		cufftComplex elmov = cong(output[i]);
		float magref = mag(elref);
		float magmov = mag(elmov);

		cufftComplex out;
		float thres = 0.1;
		out = ((elref / magref) * (elmov / magmov));
		//out  = out/ (magref * magmov);
		out = magref > thres ? out : zero();
		out = magmov > thres ? out : zero();

		output[i] = out;
	}
}

std::pair<size_t, cufftComplex> argmax(cufftComplex* a, const unsigned int n) {
	thrust::device_ptr<cufftComplex> d_ptr = thrust::device_pointer_cast(a);
	thrust::device_vector<cufftComplex>::iterator iter = thrust::max_element(d_ptr, d_ptr + n, mag_comp());
	size_t position = thrust::device_pointer_cast( &(iter[0]) ) - d_ptr;
	cufftComplex corr = *iter;
	return std::pair<size_t, cufftComplex>(position, corr);
}

void testArgMax() {
	unsigned int n = 512 * 512;
	DualData<cufftComplex> testData = DualData<cufftComplex>("Test", n);
	testData.createHost();

	for (int i = 0; i < n; ++i) {
		testData.host_data[i].x = 0;
		testData.host_data[i].y = 0;
		if (i == n/2) {
			testData.host_data[i].x = 100;
			testData.host_data[i].y = 100;
		}
	}
	testData.push();

	auto location_pair = argmax(testData.device_data, n);

	std::cout << n / 2 << " != " << location_pair.first << std::endl;
	testData.clean();
}

void gradient(Npp8u* image, cv::Size &shape, DualData<float> &buffer) {
	float* drows = buffer.device_data;
	float* dcols = buffer.device_data + ( shape.area() );

	assert(image != NULL);

	convolutionColumnsGPU(dcols, image, shape.width, shape.height);
	convolutionRowsGPU(drows, image, shape.width, shape.height);

	
	//buffer.pull();

	//cv::Size dsize = cv::Size(shape.width, shape.height * 2);
	//auto test = toMat(buffer.host_data, dsize);
	//print("host_data", buffer.host_data, dsize.area());

	//showImage(test, "Gradient Images");
	
}



void gradient(WarpContext<Npp8u> &image_data) {
	const unsigned int nBatch = image_data.nBatch;
	cv::Size shape = image_data.image_shape;

	assert(image_data.batch.dstate);

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK( cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, convert_complex, 0) );

	DualData<float> buffer = DualData<float>("Gradient Buffer", 2 * shape.area());
	buffer.createDevice();

	for (size_t i = 0; i < nBatch; ++i) {
		Npp8u* image = image_data.batch.device_data + i * shape.area();
		cufftComplex* output = image_data.workspace.device_data + i * shape.area();
		gradient(image, shape, buffer);

		float* gx = buffer.device_data;
		float* gy = buffer.device_data + ( shape.area() );
		convert_complex<<<gridsize, blocksize>>>(output, gx, gy, shape.area());
	}

	buffer.destroyDevice();

}


void prepareReference(DualData<Npp8u> &input, DualData<cufftComplex> &output, cv::Size& shape) {
	output.createDevice();

	Npp8u* image = input.device_data;
	cufftComplex* out = output.device_data;
	
	DualData<float> buffer = DualData<float>("Reference Gradient Buffer", 2 * shape.area());
	buffer.createDevice();

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK( cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, convert_complex, 0) );

	//Take Gradient
	gradient(image, shape, buffer);
	float* gx = buffer.device_data;
	float* gy = buffer.device_data + shape.area();
	convert_complex<<<gridsize, blocksize>>>(out, gx, gy, shape.area());

	//Take FFT
	size_t w = 0;
	cufftHandle plan;
	cufftResult r;
	//output.pull();
	//writeBinary("RGTest.bin", output.host_data, shape.area());

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	r = cufftMakePlan2d(plan, shape.width, shape.height, CUFFT_C2C, &w);
	assert(r == CUFFT_SUCCESS);
	
	r = cufftXtExec(plan, out, out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);


	buffer.destroyDevice();
}


std::vector<float> linspace(float start, float end, const unsigned int num) {
	std::vector<float> linspace_v;

	if (num == 0) {
		// Do Nothing
	}
	else if (num == 1) {
		linspace_v.push_back(start);
	}
	else {
		double delta = (end - start) / (num - 1);

		for (unsigned int i = 0; i < num - 1; ++i) {
			linspace_v.push_back(start + delta * i);
		}
		linspace_v.push_back(end);
	}

	return linspace_v;
}

unsigned int nextPowerof2(unsigned int n) {
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}

cv::Size power2pad(const cv::Size &image_shape) {
	unsigned int width  = unsigned(image_shape.width);
	unsigned int height = unsigned(image_shape.height);
	unsigned int pad_width = nextPowerof2(width);
	unsigned int pad_height = nextPowerof2(height);

	return cv::Size(pad_width, pad_height);
}


cv::Mat readImage(std::string filename) {
	cv::Mat Image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	if (Image.empty()) {
		throw std::runtime_error(Formatter() << "Failed readImage(): " << filename << " not found. ");
	}
	return Image;
}

cv::Mat padImage(const cv::Mat &Input, const cv::Size &new_shape) {
	cv::Size old_shape = Input.size();
	cv::Scalar color(0, 0, 0);
	unsigned int width_diff = unsigned(new_shape.width - old_shape.width);
	unsigned int height_diff = unsigned(new_shape.height - old_shape.height);

	unsigned int left = (width_diff % 2) ? unsigned(width_diff) / 2 : unsigned(std::floor(float(width_diff) / 2));
	unsigned int right = (width_diff % 2) ? unsigned(width_diff) / 2 : unsigned(std::ceil(float(width_diff) / 2));

	unsigned int top = (height_diff % 2) ? height_diff / 2 : unsigned(std::floor(float(height_diff) / 2));
	unsigned int bot = (height_diff % 2) ? height_diff / 2 : unsigned(std::ceil(float(height_diff) / 2));

	cv::Mat Output;
	cv::copyMakeBorder(Input, Output, top, bot, left+1, right, cv::BORDER_CONSTANT, 0);
	cv::Size test = Output.size();
	return Output;
}

void padImages(const cv::Mat &Moving, const cv::Mat &Reference, cv::Mat &paddedMoving, cv::Mat &paddedReference) {

	cv::Size m_shape = Moving.size();
	cv::Size r_shape = Reference.size();

	unsigned int largest_width = std::max(m_shape.width, r_shape.width);
	unsigned int largest_height = std::max(m_shape.height, r_shape.height);
	
	unsigned int largest = std::max(largest_width, largest_height);

	cv::Size max_shape = cv::Size(largest, largest);
	cv::Size padded_shape = power2pad(max_shape);

	paddedMoving = padImage(Moving, padded_shape);
	paddedReference = padImage(Reference, padded_shape);

}

cv::Mat downsampleImage(const cv::Mat &Input, const unsigned int resolution = 256) {
	auto shape = Input.size();
	float scale = float(1) / std::min(float(shape.width) / resolution, float(shape.height) / resolution);

	unsigned int ds_width =  unsigned(std::ceil(scale * shape.width));
	unsigned int ds_height = unsigned(std::ceil(scale * shape.height));
	cv::Size scaled_shape = cv::Size(ds_width, ds_height);


	cv::Mat Output;
	cv::resize(Input, Output, cv::Size(ds_width, ds_height), 0, 0, cv::INTER_AREA);
	return Output;
}


Npp8u* transferImageCPUtoGPU(const cv::Mat &Image) {
	cv::Size shape = Image.size();
	Npp8u* d_Image;
	size_t imageMemSize = size_t(shape.width) * size_t(shape.height) * sizeof(Npp8u);
	CUDA_CHECK( cudaMalloc((void**)&d_Image, imageMemSize) );
	CUDA_CHECK( cudaMemcpy(d_Image, reinterpret_cast<Npp8u*>(Image.data), imageMemSize, cudaMemcpyHostToDevice) );
	return d_Image;
}

cv::Mat transferImageGPUtoCPU(Npp8u* d_Image, const cv::Size &shape) {
	size_t imageMemSize = size_t(shape.width) * size_t(shape.height) * sizeof(Npp8u);
	Npp8u* h_pImage = static_cast<Npp8u*>( malloc(imageMemSize) );

	CUDA_CHECK( cudaMemcpy((void*) h_pImage, (void*) d_Image, imageMemSize, cudaMemcpyDeviceToHost) );
	cv::Mat h_Image = cv::Mat(shape, CV_8UC1, h_pImage);
	return h_Image;
}
void populateTransform(SearchContext &sweep, WarpContext<Npp8u> &image_data) {

	unsigned int nBatch = image_data.nBatch;
	Parameter theta = sweep.theta;
	Parameter sw = sweep.scale_w;
	Parameter sh = sweep.scale_h;
	cv::Size shape = image_data.image_shape;

	image_data.image.push();
	image_data.batch.createDevice();
	image_data.transformBuffer.createHost();
	
	const double c_w = double(shape.width) / 2;
	const double c_h = double(shape.height) / 2;

	size_t l = 0;

	for (size_t i = theta.initial; i < theta.N; ++i) {
		for (size_t j = sw.initial; j < sw.N; ++j) {
			for (size_t k = sh.initial; k < sh.N; ++k) {

				if (l >= nBatch) {
					break;
				}

				Npp64f angle = theta.start + theta.delta * i;
				Npp64f scale_w = sw.start + sw.delta * j;
				Npp64f scale_h = sh.start + sh.delta * k;
			
				//TODO: This needs to be changed if multithreading. 
				printf("%i, Queuing parameters (%f, %f, %f) \n", l, angle, scale_w, scale_h);

				double c = cos(angle);
				double s = sin(angle);
				
				const size_t idx = static_cast<size_t>(6) * l;
				image_data.transformBuffer.host_data[idx + 0] = scale_w * c;
				image_data.transformBuffer.host_data[idx + 1] = scale_h * s;
				image_data.transformBuffer.host_data[idx + 2] = -c * scale_w * c_w - scale_h * s * c_h + c_w;

				image_data.transformBuffer.host_data[idx + 3] = -scale_w * s;
				image_data.transformBuffer.host_data[idx + 4] = scale_h * c;
				image_data.transformBuffer.host_data[idx + 5] = -scale_h * c * c_h + scale_w * s * c_w + c_h;
				l++;
			}
			sh.initial = 0;
			if (l >= nBatch) {
				break;
			}
		}
		sw.initial = 0;
		if (l >= nBatch) {
			break;
		}
	}
	theta.initial = 0;

	image_data.transformBuffer.push();
	//image_data.transformBuffer.destroyHost();

	//image_data.transformBuffer.pull();
	//print("transformData", image_data.transformBuffer.host_data, 12);

	image_data.batchList.createHost();
	
	for (size_t l = 0; l < nBatch; ++l) {
		size_t batch_increment = static_cast<size_t>(l) * static_cast<size_t>(shape.area());

		NppiWarpAffineBatchCXR Job;
		Job.pSrc = (void*)(image_data.image.device_data);
		Job.pDst = (void*)(image_data.batch.device_data +batch_increment);
		Job.nSrcStep = sizeof(Npp8u) * shape.width;
		Job.nDstStep = sizeof(Npp8u) * shape.width;
		Job.pCoeffs = &image_data.transformBuffer.device_data[6 * l];
		if (image_data.batchList.host_data != NULL) {
			image_data.batchList.host_data[l] = Job;
		}
		else {
			throw std::runtime_error(Formatter() << "Near SegFault: Host BatchList is NULL");
		}
	}

	image_data.batchList.push();

	NPP_CHECK( nppiWarpAffineBatchInit(image_data.batchList.device_data, nBatch) );

	//NppiRect bbox = { (int)0, (int)0, (int)shape.width, (int)shape.height };
	//NppiSize nppi_shape = { (int) shape.width, (int) shape.height };
	//std::cout << std::endl << "shape: " << shape.width << ":" << shape.height << std::endl;
	//NPP_CHECK( nppiWarpAffineBatch_8u_C1R(nppi_shape, bbox, bbox, NPPI_INTER_LINEAR, image_data.batchList.device_data, nBatch) );


}

struct Location {
	int x;
	int y;
};

Location convertXY(int loc, cv::Size &shape) {
	Location l;
	l.y = (int)(loc / shape.width);
	l.x = loc - (l.y * shape.width);
	return l;
}

void print(Location l) {
	std::cout << "(" << l.x << "," << l.y << ")" << std::endl;
}

std::pair<size_t, cufftComplex> computePhaseCorrelation(DualData<cufftComplex>& reference, DualData<cufftComplex>& batchBuffer, const unsigned int nBatch, cv::Size& shape) {
	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	cv::Size batch_shape(shape.width, shape.height * nBatch);
	cufftComplex* batch_data = batchBuffer.device_data;
	cufftComplex* ref_data = reference.device_data;

	size_t w = 0;
	long long *n = (long long*)malloc(sizeof(long long) * 2);
	n[0] = shape.width;
	n[1] = shape.height;

	r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, (long long) nBatch, &w, CUDA_C_32F);
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	//batchBuffer.pull();
	//writeBinary("FFTTest.bin", batchBuffer.host_data, shape.area());

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, pc_kernel, 0));

	pc_kernel<<<gridsize, blocksize>>>(batch_data, ref_data, nBatch * shape.area(), shape.area());

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_INVERSE);
	assert(r == CUFFT_SUCCESS);

	//batchBuffer.pull();
	//writeBinary("PCTest.bin", batchBuffer.host_data, shape.area());
	//auto testi = toMat(batchBuffer.host_data, batch_shape);
	//print("host_data", buffer.host_data, dsize.area());
	//showImage(testi, "Correlation");


	auto location_pair = argmax(batch_data, batch_shape.area());
	free(n);
	cufftDestroy(plan);
	return location_pair;
}

void batchFFT(DualData<cufftComplex> &fftbatch, cv::Size& shape, unsigned int nBatch) {
	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	cv::Size batch_shape(shape.width, shape.height * nBatch);
	cufftComplex* batch_data = fftbatch.device_data;

	//fftbatch.pull();
	//auto test = toMat(fftbatch.host_data, batch_shape);
	//print("host_data", buffer.host_data, dsize.area());
	//showImage(test, "complexGradient");


	size_t w = 0;
	long long *n = (long long*)malloc(sizeof(long long) * 2);
	n[0] = shape.width;
	n[1] = shape.height;
	//Try default?

	long long width = (long long)shape.width;
	long long height = (long long)shape.height;

	r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, (long long) nBatch, &w, CUDA_C_32F);
	std::cout << r << std::endl;
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_FORWARD);

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_INVERSE);

	free(n);
	cufftDestroy(plan);

	//fftbatch.pull();
	//auto testi = toMat(fftbatch.host_data, batch_shape);
	//print("host_data", buffer.host_data, dsize.area());
	//showImage(testi, "iFFT");
	
	//Forward FFT

	//Backward FFT
	//PULL and Visualized. 



}

void warp(WarpContext<Npp8u> &image_data) {
	unsigned int nBatch = image_data.nBatch;
	cv::Size shape = image_data.image_shape;

	NppiRect bbox = { (int)0, (int)0, (int)shape.width, (int)shape.height };
	NppiSize nppi_shape = { (int) shape.width, (int) shape.height };
	std::cout << std::endl << "shape: " << shape.width << ":" << shape.height << std::endl;
	NPP_CHECK( nppiWarpAffineBatch_8u_C1R(nppi_shape, bbox, bbox, NPPI_INTER_LINEAR, image_data.batchList.device_data, nBatch) );
}

void warp1(WarpContext<Npp8u>& image_data) {
	cv::Size shape = image_data.image_shape;
	NppiRect bbox = { (int)0, (int)0, (int)shape.width, (int)shape.height };
	NppiSize nppi_shape = { shape.width, shape.height };

	double c_w = double(shape.width) / 2;
	double c_h = double(shape.height) / 2;

	double angle = 3.1415/2;
	double c = cos(angle);
	double s = sin(angle);

	const double aCoeff[2][3] = {
								{c, s, -c*c_w - s*c_h + c_w},
								{-s, c, -c*c_h + s*c_w + c_h}
	};

	int step = shape.width * sizeof(Npp8u);

	NPP_CHECK( nppiWarpAffine_8u_C1R(image_data.image.device_data, nppi_shape, step, bbox,
						  image_data.batch.device_data, step, bbox, aCoeff, NPPI_INTER_LINEAR) );
}
void print(int* soln, unsigned int n){
	for (size_t i = 0; i < n; ++i) {
		std::cout << soln[i] << " ";
	}
}

void print(double* soln, unsigned int n){
	for (size_t i = 0; i < n; ++i) {
		std::cout << soln[i] << " ";
	}
}
void print(cufftComplex a) {
	std::cout << "(" << a.x << "," << a.y << "i)" << std::endl;
}

void adjustSweep(size_t index, SearchContext& sweep) {
	unsigned int angle_id = (int)(index / (sweep.scale_h.N * sweep.scale_w.N));
	unsigned int remain = index % (sweep.scale_h.N * sweep.scale_w.N);
	unsigned int sw_id = (int)(remain / sweep.scale_h.N);
	unsigned int sh_id = (int)(remain % sweep.scale_h.N);

	sweep.theta.initial = angle_id;
	sweep.scale_h.initial = sh_id;
	sweep.scale_w.initial = sw_id;

	printf("Starting search at (%i, %i, %i) \n", angle_id, sw_id, sh_id);
}

void interpretLocation(std::pair<size_t, cufftComplex> gloc, SearchContext &sweep, WarpContext<Npp8u> &context) {
	unsigned int nBatch = context.nBatch;
	cv::Size shape = context.image_shape;
	size_t loc = (int)(gloc.first / shape.area());
	std::cout << "Location: " << loc << std::endl;
	assert(loc < sweep.theta.N * sweep.scale_w.N * sweep.scale_h.N);
	loc = loc + (sweep.scale_h.N * sweep.scale_w.N) * (sweep.theta.initial) + (sweep.scale_h.N) * (sweep.scale_w.initial) + (sweep.scale_h.initial);
	std::cout << "Location (after shift): " << loc << std::endl;
	if ( mag(gloc.second) > sweep.value) {
		unsigned int angle_id = (int)(loc / (sweep.scale_h.N * sweep.scale_w.N));
		unsigned int remain = loc % (sweep.scale_h.N * sweep.scale_w.N);
		unsigned int sw_id = (int)(remain / sweep.scale_h.N);
		unsigned int sh_id = (int)(remain % sweep.scale_h.N);

		sweep.theta.result = (angle_id) * sweep.theta.delta + sweep.theta.start;
		sweep.scale_w.result = (sw_id) * sweep.scale_w.delta + sweep.scale_w.start;
		sweep.scale_h.result = (sh_id) * sweep.scale_h.delta + sweep.scale_h.start;

		sweep.value = mag(gloc.second);
	}

}

void printResult(SearchContext& sweep) {
	printf("The Result of the search is (%f, %f, %f) \n", sweep.theta.result, sweep.scale_w.result, sweep.scale_h.result);
}

void correlationSearch(DualData<cufftComplex>& reference, SearchContext& sweep, WarpContext<Npp8u> &batch) {
	cv::Size shape = batch.image_shape;

	size_t permutations = sweep.theta.N * sweep.scale_w.N * sweep.scale_h.N;
	size_t batch_size = batch.nBatch;

	size_t iters = (size_t)ceil((double)permutations / (double)batch_size);

	size_t starting_idx = 0;

	batch.workspace.createDevice();

	for (size_t l = 0; l < iters; l++) {
		starting_idx = l * batch_size;
		adjustSweep(starting_idx, sweep);

		if ((l == iters - 1) && (permutations % batch_size)) {
			batch_size = permutations % batch_size;
			batch.nBatch = batch_size;
		}

		populateTransform(sweep, batch);
		warp(batch);
		gradient(batch);
		auto location_pair = computePhaseCorrelation(reference, batch.workspace, batch_size, shape);
		location_pair.first = location_pair.first;
		interpretLocation(location_pair, sweep, batch);
		printResult(sweep);
		
	}

	batch.workspace.destroyDevice();
}

void performSearch(unsigned int batch_size, unsigned char* reference, unsigned char* moving, int* shape, double* params, double* soln) {
	cv::Size cv_shape(shape[0], shape[1]);

	//Parameter(start, stop, steps)
	Parameter theta = Parameter(params[0], params[1], params[2]);
	Parameter sw = Parameter(params[3], params[4], params[5]);
	Parameter sh = Parameter(params[6], params[7], params[8]);

	batch_size = (unsigned int) std::min(theta.N * sw.N * sh.N, batch_size);

	SearchContext sweep = SearchContext(theta, sw, sh);

	cv::Size output_shape(cv_shape.width, cv_shape.height * batch_size);

	DualData<Npp8u> referenceDD = DualData<Npp8u>("Reference", (Npp8u*) reference, cv_shape.area(), "HOST");
	DualData<Npp8u> movingDD = DualData<Npp8u>("Moving", (Npp8u*) moving, cv_shape.area(), "HOST");

	movingDD.push();
	referenceDD.push();
	
	DualData<cufftComplex> RFFT = DualData<cufftComplex>("Reference FFT", cv_shape.area());
	prepareReference(referenceDD, RFFT, cv_shape);

	//reference_spectrum.pull();
	//cv::Mat Ref = toMat(padded_reference.host_data, template_shape);
	//showImage(Ref, "Reference Image");
	//cv::Mat Refft = toMat(reference_spectrum.host_data, template_shape);
	//showImage(Refft, "Reference Spectrum");

	//writeBinary("RTest.bin", padded_reference.host_data, template_shape.area());
	//writeBinary("RFTTest.bin", reference_spectrum.host_data, template_shape.area());

	//cv::Mat Mov = toMat(padded_moving.host_data, template_shape);
	//showImage(Mov, "Moving Image");

	WarpContext<Npp8u> warp_context = WarpContext<Npp8u>(movingDD, cv_shape, batch_size);
	correlationSearch(RFFT, sweep, warp_context);

	soln[0] = sweep.theta.result;
	soln[1] = sweep.scale_w.result;
	soln[2] = sweep.scale_h.result;

	referenceDD.destroyDevice();
	RFFT.clean();

	warp_context.batchList.clean();
	warp_context.transformBuffer.clean();
	warp_context.batch.clean();
	warp_context.image.destroyDevice();
}

int main()
{
	static_assert(std::is_same_v<std::uint8_t, unsigned char>,
    "This library requires std::uint8_t to be implemented as char or unsigned char.");

	static_assert(std::is_same_v<Npp8u, unsigned char>,
    "This library requires Nppu8 to be implemented as char or unsigned char.");

	CUDA_CHECK(cudaSetDevice(0));

	cv::Mat Moving = readImage("images/rome_image.png");
	cv::Mat Reference = readImage("images/rome_image.png");

	unsigned int resolution = 256;
	cv::Mat ds_Moving = downsampleImage(Moving, resolution);
	cv::Mat ds_Reference = downsampleImage(Reference, resolution);

	cv::Mat pad_Moving, pad_Reference;
	padImages(ds_Moving, ds_Reference, pad_Moving, pad_Reference);
	showImage(pad_Reference, "Reference Image");
	showImage(pad_Moving, "Moving Image");

	cv::Size cv_shape = pad_Moving.size();
	unsigned int nBatch = 1000;
	int shape[2] = {cv_shape.width, cv_shape.height };
	double soln[3] = { 0, 0, 0 };
	double params[9] = { -1, 1, 50, 0.5, 2, 50, 0.5, 2, 50 };

	print((int*)pad_Reference.data, 3);

	performSearch(nBatch, pad_Reference.data, pad_Moving.data, &shape[0], &params[0], &soln[0]);

	print(soln, 3);

 	return 0;
}

