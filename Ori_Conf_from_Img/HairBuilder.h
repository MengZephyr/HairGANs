#pragma once
#ifndef HAIR_BUILDER_H
#define HAIR_BUILDER_H

#include<opencv/cv.h>

class HairBuilder
{
private:
	cv::Mat             m_maskData;
	cv::Mat             m_orientData;
	cv::Mat				m_varianceData;
	cv::Mat				m_confidenceData;
	cv::Mat				m_maxRespData;

	//////////////////////////////////////////////////////////////////////////
	// parameters
	//////////////////////////////////////////////////////////////////////////
	// Orientation detection
	int para_kernelWidth, para_kernelHeight;
	float para_sigmaX, para_sigmaY;
	float para_lambdar, para_phasee;
	int   para_numKernels, para_numPhases;
	// Confidence value
	float para_clampConfidenceHigh, para_clamConfidenceLow;

public:
	HairBuilder(cv::Mat colorImg, cv::Mat maskImg);
	~HairBuilder() {}

	void toRunConfidenceBuider(int iter = 3);

	void toRunParisFielter(cv::Mat oriImg, int iter = 3);

	cv::Mat getConfMap() const { return m_confidenceData.clone(); }
	cv::Mat getImgVecMap(cv::Mat disAmb);

	cv::Mat getParisVecMap(cv::Mat disAmb);

private:
	cv::Mat prepData(cv::Mat Img);
	void calcOrientation(cv::Mat& inputGray);
	void calcConfidence();
	void filterOri(cv::Mat& rstMap);
};
#endif // !HAIR_BUILDER_H
