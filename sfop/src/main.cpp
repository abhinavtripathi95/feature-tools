/*
This is a modification of the main.cpp file which comes with the 
original SFOP-0.9 code. We use this modified version to interface Python 
with C using ctypes to get the SFOP keypoints.
The original code can be downloaded from
http://www.ipb.uni-bonn.de/data-software/sfop-keypoint-detector/ 

Apart from this, the other files which have been modified are: CSfop.cpp
and CSfop.h because we have added the function return_kp()
*/

#include "CSfop.h"
#include "CImg.h"
#include "CImageFactory.h"
#include "CImageCpu.h"
#ifdef GPU
#include "CImageCl.h"
#endif

using namespace SFOP;
extern "C" float* mymain(const char* inFile);
extern "C" void free_mem(float* a);
/**
 * @brief This program demonstrates the use of the SFOP detector class CSfop.
 *
 * Run \code ./main --help \endcode to see a list of possible arguments, while
 * SFOP detection is performed on a standard test image with default parameters.
 *
 * @param[in] argc
 * @param[in] argv[]
 *
 * @return 0 if succeeded
 */
float* mymain(const char* inFile)
{
    // parse options
    const char* outFile = "sfop.dat";
    const bool display = false; // Display result
    const float imageNoise = 0.02; // Image noise [0.02]
    const float precisionThreshold = 0; // Threshold on precision [0]
    const float lambdaWeight = 2; // Weighting for lambda [2]
    const unsigned int numOctaves = 3; // Number of octaves [3]
    const unsigned int numLayers = 4; // Number of layers per octave [4]
#ifdef GPU
    const int device = 0; //cimg_option("--device", 0, "Device, 0 for CPU, 1 for OpenCL on GPU [0]");
#endif
    const int type = -1; //cimg_option("--type", -1, "Angle in degrees, or -1 for optimal features [-1]");

    // create image factory depending on device
    CImageFactoryAbstract* l_factory_p;
#ifdef GPU
    if (device == 0) {
        l_factory_p = new CImageFactory<CImageCpu>();
    }
    else {
        l_factory_p = new CImageFactory<CImageCl>();
    }
#else
    l_factory_p = new CImageFactory<CImageCpu>();
#endif

    // process input image
    CSfop detector(l_factory_p, inFile);
    detector.detect(numOctaves, numLayers, imageNoise, lambdaWeight, precisionThreshold, type);
    detector.writeFile(outFile);
    if (display) {
        CImageCpu image;
        image.load(inFile);
        image.displayFeatures(detector.features_p());
    }

    float* kp = detector.return_kp();

    return kp;	
}

void free_mem(float* a)
{
    delete[] a;
}


int main(){
	return 0;
}
