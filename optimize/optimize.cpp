#define _CRT_SECURE_NO_WARNINGS		// stop warning on fopen
#include <stdio.h>
#include <vector>
#include <chrono>
#include <assert.h>
#include <immintrin.h>

std::vector<unsigned char> LoadFile(const char *name)
{
	FILE * f = fopen(name, "rb");
	fseek(f, 0, SEEK_END);
	size_t len = ftell(f);
	fseek(f, 0, SEEK_SET);
	std::vector<unsigned char> buf;
	buf.resize(len);
	size_t r = fread((void *)&buf[0], 1, len, f);
	if (r != len)
	{
		printf("Read failed for %s", name);
		exit(1);
	}
	fclose(f);

	return buf;
}

int BigLong(const std::vector<unsigned char> &vec, int ofs)
{
	return (vec[ofs] << 24) + (vec[ofs + 1] << 16) + (vec[ofs + 2] << 8) + vec[ofs + 3];
}

const int NumDimensions = 28 * 28;

void TestBestMatch()
{
	std::vector<unsigned char>	testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char>	testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char>	trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char>	trainLabels = LoadFile("train-labels.idx1-ubyte");

	int trainCount = BigLong(trainImages, 4);
	int testCount = BigLong(testImages, 4) >> 5;	// shrink for faster tests

	int	miss = 0;
	for (int i = 0; i < testCount; i++)
	{
		unsigned char label = testLabels[8 + i];
		assert(label >= 0 && label <= 9);
		const unsigned char * testData = &testImages[16 + i*NumDimensions];

		int bestLabel = -1;
		int bestError = INT_MAX;
		for (int j = 0; j < trainCount; j++)
		{
			const unsigned char * trainData = &trainImages[16 + j*NumDimensions];
			int error = 0;
			for (int k = 0; k < NumDimensions; k++)
			{
				const int v = (int)testData[k] - (int)trainData[k];
				error += v * v;
			}
			//5739837, 7025841, 5795133, 6536253, 5500006
			//6093159, 5652340, 6993727, 4700000, 5134448
			if (error < bestError)
			{
				bestError = error;
				bestLabel = trainLabels[8 + j];
			}
		}
		if (bestLabel != label)
		{
			miss++;
		}
	}
	printf("%i misses for accuracy: %f\n", miss, (float)(testCount - miss) / (float)testCount);
}

void TestBestMatchOpt1()
{
	std::vector<unsigned char>	testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char>	testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char>	trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char>	trainLabels = LoadFile("train-labels.idx1-ubyte");

	int trainCount = BigLong(trainImages, 4);
	int testCount = BigLong(testImages, 4) >> 5;	// shrink for faster tests

	__m128i shuffle = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);

	int	miss = 0;
	for (int i = 0; i < testCount; i++)
	{
		unsigned char label = testLabels[8 + i];
		assert(label >= 0 && label <= 9);
		const unsigned char * testData = &testImages[16 + i*NumDimensions];

		int bestLabel = -1;
		int bestError = INT_MAX;
		__m256i zero = _mm256_setzero_si256();
		for (int j = 0; j < trainCount; j++)
		{
			const unsigned char * trainData = &trainImages[16 + j*NumDimensions];
			__m256i error8 = _mm256_setzero_si256();
			for (int k = 0; k < NumDimensions; k+=16)
			{
				__m128i test8 = _mm_load_si128((const __m128i*)(testData+k));
				__m128i train8 = _mm_load_si128((const __m128i*)(trainData+k));
				__m256i test32l = _mm256_cvtepu8_epi32(test8);
				__m256i train32l = _mm256_cvtepu8_epi32(train8);

				test8 = _mm_shuffle_epi8(test8, shuffle);
				train8 = _mm_shuffle_epi8(train8, shuffle);
				__m256i test32h = _mm256_cvtepu8_epi32(test8);
				__m256i train32h = _mm256_cvtepu8_epi32(train8);

				__m256i diffl = _mm256_sub_epi32(test32l, train32l);
				__m256i diffh = _mm256_sub_epi32(test32h, train32h);

				__m256i squaredl = _mm256_mullo_epi32(diffl, diffl);
				__m256i squaredh = _mm256_mullo_epi32(diffh, diffh);

				error8 = _mm256_add_epi32(error8, squaredl);
				error8 = _mm256_add_epi32(error8, squaredh);
			}
			__m256i error4 = _mm256_hadd_epi32(error8, zero);
			__m256i error2 = _mm256_hadd_epi32(error4, zero); //error2 has 2 partial sums in 0 and 4
			__m128i error2l = _mm256_extracti128_si256(error2, 0);
			__m128i error2h = _mm256_extracti128_si256(error2, 1);
			__m128i error1 = _mm_add_epi32(error2l, error2h);

			int error = _mm_cvtsi128_si32(error1);
			//5739837, 7025841, 5795133, 6536253, 5500006
			//6093159, 5652340, 6993727, 4700000, 5134448
			if (error < bestError)
			{
				bestError = error;
				bestLabel = trainLabels[8 + j];
			}
		}
		if (bestLabel != label)
		{
			//115, 195, 241, 268, 300
			miss++;
		}
	}
	printf("%i misses for accuracy: %f\n", miss, (float)(testCount - miss) / (float)testCount);
}

void TestBestMatchOpt2()
{
	std::vector<unsigned char>	testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char>	testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char>	trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char>	trainLabels = LoadFile("train-labels.idx1-ubyte");

	int trainCount = BigLong(trainImages, 4);
	int testCount = BigLong(testImages, 4) >> 5;	// shrink for faster tests

	__m128i shuffle = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);

	int	miss = 0;
	for (int i = 0; i < testCount; i++)
	{
		unsigned char label = testLabels[8 + i];
		assert(label >= 0 && label <= 9);
		const unsigned char * testData = &testImages[16 + i*NumDimensions];

		int bestLabel = -1;
		int bestError = INT_MAX;
		__m256i zero = _mm256_setzero_si256();
		for (int j = 0; j < trainCount; j++)
		{
			const unsigned char * trainData = &trainImages[16 + j*NumDimensions];
			__m256i error8 = _mm256_setzero_si256();
			for (int k = 0; k < NumDimensions; k += 16)
			{
				__m128i test8 = _mm_load_si128((const __m128i*)(testData + k));
				__m128i train8 = _mm_load_si128((const __m128i*)(trainData + k));
				__m256i test32l = _mm256_cvtepu8_epi32(test8);
				__m256i train32l = _mm256_cvtepu8_epi32(train8);

				test8 = _mm_shuffle_epi8(test8, shuffle);
				train8 = _mm_shuffle_epi8(train8, shuffle);
				__m256i test32h = _mm256_cvtepu8_epi32(test8);
				__m256i train32h = _mm256_cvtepu8_epi32(train8);

				__m256i diffl = _mm256_sub_epi32(test32l, train32l);
				__m256i diffh = _mm256_sub_epi32(test32h, train32h);

				__m256i squaredl = _mm256_mullo_epi32(diffl, diffl);
				__m256i squaredh = _mm256_mullo_epi32(diffh, diffh);

				error8 = _mm256_add_epi32(error8, squaredl);
				error8 = _mm256_add_epi32(error8, squaredh);
			}
			__m256i error4 = _mm256_hadd_epi32(error8, zero);
			__m256i error2 = _mm256_hadd_epi32(error4, zero); //error2 has 2 partial sums in 0 and 4
			__m128i error2l = _mm256_extracti128_si256(error2, 0);
			__m128i error2h = _mm256_extracti128_si256(error2, 1);
			__m128i error1 = _mm_add_epi32(error2l, error2h);

			int error = _mm_cvtsi128_si32(error1);
			//5739837, 7025841, 5795133, 6536253, 5500006
			//6093159, 5652340, 6993727, 4700000, 5134448
			if (error < bestError)
			{
				bestError = error;
				bestLabel = trainLabels[8 + j];
			}
		}
		if (bestLabel != label)
		{
			//115, 195, 241, 268, 300
			miss++;
		}
	}
	printf("%i misses for accuracy: %f\n", miss, (float)(testCount - miss) / (float)testCount);
}

int main(int argc, char ** argv)
{
	std::chrono::time_point<std::chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();
	//TestBestMatch();
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());

	start = std::chrono::system_clock::now();
	//TestBestMatchOpt1();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());

	start = std::chrono::system_clock::now();
	TestBestMatchOpt2();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());
}