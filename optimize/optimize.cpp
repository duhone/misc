#define _CRT_SECURE_NO_WARNINGS		// stop warning on fopen
#include <stdio.h>
#include <vector>
#include <chrono>
#include <assert.h>
#include <immintrin.h>
#include <ppl.h>
#include <amp.h> 
#include <numeric>
#include <algorithm>

using namespace concurrency;

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

const int testdivider = 0;

void TestBestMatch()
{
	std::vector<unsigned char>	testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char>	testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char>	trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char>	trainLabels = LoadFile("train-labels.idx1-ubyte");

	int trainCount = BigLong(trainImages, 4);
	int testCount = BigLong(testImages, 4) >> testdivider;	// shrink for faster tests

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
	int testCount = BigLong(testImages, 4) >> testdivider;	// shrink for faster tests

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
			++miss;
		}
	}
	printf("%i misses for accuracy: %f\n", miss, (float)(testCount - miss) / (float)testCount);
}

//Compared to Opt1 his version is about 2x faster. About 6X faster than c version
//Mosty the improvement is keeping the ints 16 bits as long as possible, then using
//a mad to save on some instruction count.
void TestBestMatchOpt2()
{
	std::vector<unsigned char>	testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char>	testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char>	trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char>	trainLabels = LoadFile("train-labels.idx1-ubyte");

	uint32_t trainCount = BigLong(trainImages, 4);
	uint32_t testCount = BigLong(testImages, 4) >> testdivider;	// shrink for faster tests

	int	miss = 0;
	for (uint32_t i = 0; i < testCount; i++)
	{
		unsigned char label = testLabels[8 + i];
		assert(label >= 0 && label <= 9);
		const unsigned char * testData_0 = &testImages[16 + i*NumDimensions];
		const unsigned char * testData_1 = &testImages[16 + i*NumDimensions+16];

		int bestLabel = -1;
		int bestError = INT_MAX;
		__m256i zero = _mm256_setzero_si256();
		for (uint32_t j = 0; j < trainCount; j++)
		{
			const unsigned char * trainData_0 = &trainImages[16 + j*NumDimensions];
			const unsigned char * trainData_1 = &trainImages[16 + j*NumDimensions+16];
			__m256i error8_0 = _mm256_setzero_si256();
			__m256i error8_1 = _mm256_setzero_si256();

			for (uint32_t k = 0; k < NumDimensions; k += 32)
			{
				__m128i test8_0 = _mm_load_si128((const __m128i*)(testData_0 + k));
				__m128i train8_0 = _mm_load_si128((const __m128i*)(trainData_0 + k));
				__m128i test8_1 = _mm_load_si128((const __m128i*)(testData_1 + k));
				__m128i train8_1 = _mm_load_si128((const __m128i*)(trainData_1 + k));

				__m256i test32_0 = _mm256_cvtepu8_epi16(test8_0);
				__m256i train32_0 = _mm256_cvtepu8_epi16(train8_0);
				__m256i test32_1 = _mm256_cvtepu8_epi16(test8_1);
				__m256i train32_1 = _mm256_cvtepu8_epi16(train8_1);

				__m256i diff_0 = _mm256_sub_epi16(test32_0, train32_0);
				__m256i diff_1 = _mm256_sub_epi16(test32_1, train32_1);

				__m256i newerr_0 = _mm256_madd_epi16(diff_0, diff_0);
				__m256i newerr_1 = _mm256_madd_epi16(diff_1, diff_1);

				error8_0 = _mm256_add_epi32(error8_0, newerr_0);
				error8_1 = _mm256_add_epi32(error8_1, newerr_1);
			}
			__m256i error8 = _mm256_add_epi32(error8_0, error8_1);

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
			++miss;
		}
	}
	printf("%i misses for accuracy: %f\n", miss, (float)(testCount - miss) / (float)testCount);
}

//use ppl to run on more cores, need visual studio, or tbb on other platforms.
//about 2x faster than opt2 on my dual core haswell. about 3.5x faster on my
//quad core skylake
void TestBestMatchOpt3()
{
	std::vector<unsigned char>	testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char>	testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char>	trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char>	trainLabels = LoadFile("train-labels.idx1-ubyte");

	uint32_t trainCount = BigLong(trainImages, 4);
	uint32_t testCount = BigLong(testImages, 4) >> testdivider;	// shrink for faster tests

	std::atomic_int	miss = 0;
	//we only read from the vectors, so should be safe to use my multiple threads
	parallel_for(size_t(0), size_t(testCount), [&testLabels = std::as_const(testLabels), 
		&testImages = std::as_const(testImages), &trainImages = std::as_const(trainImages),
		&trainLabels = std::as_const(trainLabels), trainCount, &miss](size_t i)
	{
		unsigned char label = testLabels[8 + i];
		assert(label >= 0 && label <= 9);
		const unsigned char * testData_0 = &testImages[16 + i*NumDimensions];
		const unsigned char * testData_1 = &testImages[16 + i*NumDimensions + 16];

		int bestLabel = -1;
		int bestError = INT_MAX;
		__m256i zero = _mm256_setzero_si256();
		for (uint32_t j = 0; j < trainCount; j++)
		{
			const unsigned char * trainData_0 = &trainImages[16 + j*NumDimensions];
			const unsigned char * trainData_1 = &trainImages[16 + j*NumDimensions + 16];
			__m256i error8_0 = _mm256_setzero_si256();
			__m256i error8_1 = _mm256_setzero_si256();

			for (uint32_t k = 0; k < NumDimensions; k += 32)
			{
				__m128i test8_0 = _mm_load_si128((const __m128i*)(testData_0 + k));
				__m128i train8_0 = _mm_load_si128((const __m128i*)(trainData_0 + k));
				__m128i test8_1 = _mm_load_si128((const __m128i*)(testData_1 + k));
				__m128i train8_1 = _mm_load_si128((const __m128i*)(trainData_1 + k));

				__m256i test32_0 = _mm256_cvtepu8_epi16(test8_0);
				__m256i train32_0 = _mm256_cvtepu8_epi16(train8_0);
				__m256i test32_1 = _mm256_cvtepu8_epi16(test8_1);
				__m256i train32_1 = _mm256_cvtepu8_epi16(train8_1);

				__m256i diff_0 = _mm256_sub_epi16(test32_0, train32_0);
				__m256i diff_1 = _mm256_sub_epi16(test32_1, train32_1);

				__m256i newerr_0 = _mm256_madd_epi16(diff_0, diff_0);
				__m256i newerr_1 = _mm256_madd_epi16(diff_1, diff_1);

				error8_0 = _mm256_add_epi32(error8_0, newerr_0);
				error8_1 = _mm256_add_epi32(error8_1, newerr_1);
			}
			__m256i error8 = _mm256_add_epi32(error8_0, error8_1);

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
			++miss;
		}
	});
	printf("%i misses for accuracy: %f\n", miss.load(), (float)(testCount - miss.load()) / (float)testCount);
}

void Avx2()
{
	std::vector<unsigned char> testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char> testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char> trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char> trainLabels = LoadFile("train-labels.idx1-ubyte");

	// align the images to 64 byte boundaries
	// pad testImages to TEST_BLOCK images

	int trainCount = BigLong(trainImages, 4);
	int testCount = BigLong(testImages, 4) >> testdivider;// shrink for faster tests

	static const int TRAIN_COUNT = 60000;
	assert(TRAIN_COUNT == trainCount);

	static const int TEST_BLOCK = 8;
	int miss = 0;
	for (int i = 0; i < testCount; i += TEST_BLOCK)
	{
		// Expand a block of the 8 bit test images to 16 bit integers
		// and interleave them by 256 bit blocks.
		__m256i expandedTest[NumDimensions / 16][TEST_BLOCK];
		__m256i * expanded_p = &expandedTest[0][0];
		const unsigned char * testSrc = &testImages[16 + i*NumDimensions];
		for (int j = 0; j < NumDimensions / 16; j++)
		{
			for (int k = 0; k < TEST_BLOCK; k++)
			{
				__m128i bytes8 = *(__m128i *)&testSrc[k * NumDimensions];
				expanded_p[k] = _mm256_cvtepu8_epi16(bytes8);
			}
			testSrc += 16;
			expanded_p += TEST_BLOCK;
		}

		int bestError[TEST_BLOCK];
		int bestLabel[TEST_BLOCK] = {};
		for (int j = 0; j < TEST_BLOCK; j++)
		{
			bestError[j] = INT_MAX;
		}
		for (int j = 0; j < TRAIN_COUNT; j++)
		{
			// Get the error for this block of test vectors against the entire training set
			__m256i sums[TEST_BLOCK] = {};

			static __m256i m256zero;// there is probably a better way to do this...
			__m256i sums0 = m256zero;
			__m256i sums1 = m256zero;
			__m256i sums2 = m256zero;
			__m256i sums3 = m256zero;
			__m256i sums4 = m256zero;
			__m256i sums5 = m256zero;
			__m256i sums6 = m256zero;
			__m256i sums7 = m256zero;

			int error = 0;
			const unsigned char * testData = &trainImages[16 + j*NumDimensions];
			for (int k = 0; k < NumDimensions / 16; k++)
			{
				__m128i bytes8 = *(__m128i *)&testData[k * 16];
				// 16 x 16 bit ints to compare against our block of test vectors
				__m256i expandedTrain = _mm256_cvtepu8_epi16(bytes8);

				const __m256i * testVec = &expandedTest[k][0];

				// Use the 16 bit multiply-and-add-pairs instruction
				__m256i delta0 = _mm256_sub_epi16(expandedTrain, testVec[0]);
				__m256i delta1 = _mm256_sub_epi16(expandedTrain, testVec[1]);
				__m256i delta2 = _mm256_sub_epi16(expandedTrain, testVec[2]);
				__m256i delta3 = _mm256_sub_epi16(expandedTrain, testVec[3]);
				__m256i delta4 = _mm256_sub_epi16(expandedTrain, testVec[4]);
				__m256i delta5 = _mm256_sub_epi16(expandedTrain, testVec[5]);
				__m256i delta6 = _mm256_sub_epi16(expandedTrain, testVec[6]);
				__m256i delta7 = _mm256_sub_epi16(expandedTrain, testVec[7]);
				__m256i sqr0 = _mm256_madd_epi16(delta0, delta0);
				__m256i sqr1 = _mm256_madd_epi16(delta1, delta1);
				__m256i sqr2 = _mm256_madd_epi16(delta2, delta2);
				__m256i sqr3 = _mm256_madd_epi16(delta3, delta3);
				__m256i sqr4 = _mm256_madd_epi16(delta4, delta4);
				__m256i sqr5 = _mm256_madd_epi16(delta5, delta5);
				__m256i sqr6 = _mm256_madd_epi16(delta6, delta6);
				__m256i sqr7 = _mm256_madd_epi16(delta7, delta7);
				sums0 = _mm256_add_epi32(sums0, sqr0);
				sums1 = _mm256_add_epi32(sums1, sqr1);
				sums2 = _mm256_add_epi32(sums2, sqr2);
				sums3 = _mm256_add_epi32(sums3, sqr3);
				sums4 = _mm256_add_epi32(sums4, sqr4);
				sums5 = _mm256_add_epi32(sums5, sqr5);
				sums6 = _mm256_add_epi32(sums6, sqr6);
				sums7 = _mm256_add_epi32(sums7, sqr7);
			}
			// The sum registers now have 8 32 bit values that need to be horizontally added
			// to get the total error, which can be compared against the best error so far
			// to see if we have a new best match.  This isn't very performance critical,
			// so do it with a loop.
			sums[0] = sums0;
			sums[1] = sums1;
			sums[2] = sums2;
			sums[3] = sums3;
			sums[4] = sums4;
			sums[5] = sums5;
			sums[6] = sums6;
			sums[7] = sums7;
			for (int k = 0; k < TEST_BLOCK; k++)
			{
				const int * partials = (int *)&sums[k];
				const int sum = partials[0] + partials[1] + partials[2] + partials[3] + partials[4] + partials[5] + partials[6] + partials[7];
				if (sum < bestError[k])
				{
					bestError[k] = sum;
					bestLabel[k] = trainLabels[8 + j];
				}
			}
		}
		// Check the final bestLabel against the real label
		for (int k = 0; k < TEST_BLOCK; k++)
		{
			if (bestLabel[k] != testLabels[8 + i + k])
			{
				miss++;
			}
		}
	}
	printf("%i misses for accuracy: %f\n", miss, (float)(testCount - miss) / (float)testCount);
}

void Avx2ppl()
{
	std::vector<unsigned char> testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char> testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char> trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char> trainLabels = LoadFile("train-labels.idx1-ubyte");

	// align the images to 64 byte boundaries
	// pad testImages to TEST_BLOCK images

	int trainCount = BigLong(trainImages, 4);
	int testCount = BigLong(testImages, 4) >> testdivider;// shrink for faster tests

	static const int TRAIN_COUNT = 60000;
	assert(TRAIN_COUNT == trainCount);

	static const int TEST_BLOCK = 8;
	std::atomic_int	miss = 0;
	//we only read from the vectors, so should be safe to use my multiple threads
	parallel_for(size_t(0), size_t(testCount/TEST_BLOCK), [&testLabels = std::as_const(testLabels),
		&testImages = std::as_const(testImages), &trainImages = std::as_const(trainImages),
		&trainLabels = std::as_const(trainLabels), trainCount, &miss](size_t i)
	{
		i *= TEST_BLOCK;
		// Expand a block of the 8 bit test images to 16 bit integers
		// and interleave them by 256 bit blocks.
		__m256i expandedTest[NumDimensions / 16][TEST_BLOCK];
		__m256i * expanded_p = &expandedTest[0][0];
		const unsigned char * testSrc = &testImages[16 + i*NumDimensions];
		for (int j = 0; j < NumDimensions / 16; j++)
		{
			for (int k = 0; k < TEST_BLOCK; k++)
			{
				__m128i bytes8 = *(__m128i *)&testSrc[k * NumDimensions];
				expanded_p[k] = _mm256_cvtepu8_epi16(bytes8);
			}
			testSrc += 16;
			expanded_p += TEST_BLOCK;
		}

		int bestError[TEST_BLOCK];
		int bestLabel[TEST_BLOCK] = {};
		for (int j = 0; j < TEST_BLOCK; j++)
		{
			bestError[j] = INT_MAX;
		}
		for (int j = 0; j < TRAIN_COUNT; j++)
		{
			// Get the error for this block of test vectors against the entire training set
			__m256i sums[TEST_BLOCK] = {};

			static __m256i m256zero;// there is probably a better way to do this...
			__m256i sums0 = m256zero;
			__m256i sums1 = m256zero;
			__m256i sums2 = m256zero;
			__m256i sums3 = m256zero;
			__m256i sums4 = m256zero;
			__m256i sums5 = m256zero;
			__m256i sums6 = m256zero;
			__m256i sums7 = m256zero;

			int error = 0;
			const unsigned char * testData = &trainImages[16 + j*NumDimensions];
			for (int k = 0; k < NumDimensions / 16; k++)
			{
				__m128i bytes8 = *(__m128i *)&testData[k * 16];
				// 16 x 16 bit ints to compare against our block of test vectors
				__m256i expandedTrain = _mm256_cvtepu8_epi16(bytes8);

				const __m256i * testVec = &expandedTest[k][0];

				// Use the 16 bit multiply-and-add-pairs instruction
				__m256i delta0 = _mm256_sub_epi16(expandedTrain, testVec[0]);
				__m256i delta1 = _mm256_sub_epi16(expandedTrain, testVec[1]);
				__m256i delta2 = _mm256_sub_epi16(expandedTrain, testVec[2]);
				__m256i delta3 = _mm256_sub_epi16(expandedTrain, testVec[3]);
				__m256i delta4 = _mm256_sub_epi16(expandedTrain, testVec[4]);
				__m256i delta5 = _mm256_sub_epi16(expandedTrain, testVec[5]);
				__m256i delta6 = _mm256_sub_epi16(expandedTrain, testVec[6]);
				__m256i delta7 = _mm256_sub_epi16(expandedTrain, testVec[7]);
				__m256i sqr0 = _mm256_madd_epi16(delta0, delta0);
				__m256i sqr1 = _mm256_madd_epi16(delta1, delta1);
				__m256i sqr2 = _mm256_madd_epi16(delta2, delta2);
				__m256i sqr3 = _mm256_madd_epi16(delta3, delta3);
				__m256i sqr4 = _mm256_madd_epi16(delta4, delta4);
				__m256i sqr5 = _mm256_madd_epi16(delta5, delta5);
				__m256i sqr6 = _mm256_madd_epi16(delta6, delta6);
				__m256i sqr7 = _mm256_madd_epi16(delta7, delta7);
				sums0 = _mm256_add_epi32(sums0, sqr0);
				sums1 = _mm256_add_epi32(sums1, sqr1);
				sums2 = _mm256_add_epi32(sums2, sqr2);
				sums3 = _mm256_add_epi32(sums3, sqr3);
				sums4 = _mm256_add_epi32(sums4, sqr4);
				sums5 = _mm256_add_epi32(sums5, sqr5);
				sums6 = _mm256_add_epi32(sums6, sqr6);
				sums7 = _mm256_add_epi32(sums7, sqr7);
			}
			// The sum registers now have 8 32 bit values that need to be horizontally added
			// to get the total error, which can be compared against the best error so far
			// to see if we have a new best match.  This isn't very performance critical,
			// so do it with a loop.
			sums[0] = sums0;
			sums[1] = sums1;
			sums[2] = sums2;
			sums[3] = sums3;
			sums[4] = sums4;
			sums[5] = sums5;
			sums[6] = sums6;
			sums[7] = sums7;
			for (int k = 0; k < TEST_BLOCK; k++)
			{
				const int * partials = (int *)&sums[k];
				const int sum = partials[0] + partials[1] + partials[2] + partials[3] + partials[4] + partials[5] + partials[6] + partials[7];
				if (sum < bestError[k])
				{
					bestError[k] = sum;
					bestLabel[k] = trainLabels[8 + j];
				}
			}
		}
		// Check the final bestLabel against the real label
		for (int k = 0; k < TEST_BLOCK; k++)
		{
			if (bestLabel[k] != testLabels[8 + i + k])
			{
				miss++;
			}
		}
	});
	printf("%i misses for accuracy: %f\n", miss.load(), (float)(testCount - miss) / (float)testCount);
}

void Amp()
{
	std::vector<unsigned char>	testImages = LoadFile("t10k-images.idx3-ubyte");
	std::vector<unsigned char>	testLabels = LoadFile("t10k-labels.idx1-ubyte");
	std::vector<unsigned char>	trainImages = LoadFile("train-images.idx3-ubyte");
	std::vector<unsigned char>	trainLabels = LoadFile("train-labels.idx1-ubyte");

	std::vector<int> testImageInts(testImages.begin(), testImages.end());
	std::vector<int> testLabelsInts(testLabels.begin(), testLabels.end());
	std::vector<int> trainImagesInts(trainImages.begin(), trainImages.end());
	std::vector<int> trainLabelsInts(trainLabels.begin(), trainLabels.end());

	// Create C++ AMP objects.  
	array<int, 1> testImagesAmp((int)testImageInts.size(), testImageInts.begin(), testImageInts.end());
	array<int, 1> testLabelsAmp((int)testLabelsInts.size(), testLabelsInts.begin(), testLabelsInts.end());
	array<int, 1> trainImagesAmp((int)trainImagesInts.size(), trainImagesInts.begin(), trainImagesInts.end());
	array<int, 1> trainLabelsAmp((int)trainLabelsInts.size(), trainLabelsInts.begin(), trainLabelsInts.end());
	
	int trainCount = BigLong(trainImages, 4);
	int testCount = BigLong(testImages, 4) >> testdivider;	// shrink for faster tests

	std::vector<int> miss(testCount, 0);
	//array<int, 1> missAmp(testCount, miss.begin(), miss.end());

	//extent<1> testCountAmp(testCount);

	/*parallel_for_each(
		testCountAmp,
		[=, &missAmp, &testImagesAmp, &testLabelsAmp, &trainImagesAmp, &trainLabelsAmp]
	(index<1> idx) restrict(amp)
	{
		//int label = testLabelsAmp[8 + idx];
		const int * testData = &testImagesAmp[16 + idx*NumDimensions];

		int bestLabel = -1;
		int bestError = INT_MAX;
		for (int j = 0; j < trainCount; j++)
		{
			//const int * trainData = &trainImagesAmp[16 + j*NumDimensions];
			int error = 0;
			for (int k = 0; k < NumDimensions; k++)
			{
				//const int v = k;// trainData[k];// -(int)trainData[k];
				//error += v * v;
			}
			if (error < bestError)
			{
				bestError = error;
				//bestLabel = trainLabelsAmp[8 + j];
			}
		}
		//if (bestLabel != label)
		{
			missAmp[idx]++;
		}
	});*/

	//miss = missAmp;

	int finalMiss = std::accumulate(miss.begin(), miss.end(), 0);
	printf("%i misses for accuracy: %f\n", finalMiss, (float)(testCount - finalMiss) / (float)testCount);
}

int main(int argc, char ** argv)
{
	auto exec = [](const char * label, auto func) {
		std::chrono::time_point<std::chrono::system_clock> start, end;
		printf("\n%s\n", label);
		start = std::chrono::system_clock::now();
		func();
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		printf("%f seconds\n", elapsed_seconds.count());
	};

	//exec("original c version", TestBestMatch);
	//exec("avx", TestBestMatchOpt1);
	//exec("avx v2 use 16 bit ints when possible, use mad", TestBestMatchOpt2);
	//exec("use ppl and avx v2", TestBestMatchOpt3);
	//exec("john's", Avx2);
	exec("john's with ppl", Avx2ppl);
	exec("c++ Amp", Amp);

	/*printf("\noriginal c version\n");
	start = std::chrono::system_clock::now();
	TestBestMatch();
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());

	printf("\navx\n");
	start = std::chrono::system_clock::now();
	TestBestMatchOpt1();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());

	printf("\navx v2 use 16 bit ints when possible, use mad\n");
	start = std::chrono::system_clock::now();
	TestBestMatchOpt2();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());

	printf("\nuse ppl and avx v2\n");
	start = std::chrono::system_clock::now();
	TestBestMatchOpt3();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());


	printf("\nJohns\n");
	start = std::chrono::system_clock::now();
	Avx2();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	printf("%f seconds\n", elapsed_seconds.count());
	*/
}