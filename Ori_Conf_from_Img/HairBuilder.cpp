#include "HairBuilder.h"
#include "fftw3.h"
#include <omp.h>

HairBuilder::HairBuilder(cv::Mat colorImg, cv::Mat maskImg)
{
	para_kernelWidth = 17;
	para_kernelHeight = 17;
	para_sigmaX = 1.8;
	para_sigmaY = 2.4;
	para_lambdar = 4;
	para_phasee = 0.;
	para_numKernels = 32;
	para_numPhases = 1;
	para_clamConfidenceLow = 0.01;
	para_clampConfidenceHigh = 0.4;

	m_maskData = maskImg.clone();
	m_confidenceData = prepData(colorImg);
}

void HairBuilder::toRunConfidenceBuider(int iter)
{
	m_orientData = cv::Mat::zeros(m_confidenceData.size(), CV_32F);
	m_varianceData = cv::Mat::zeros(m_confidenceData.size(), CV_32F);
	m_maxRespData = cv::Mat::zeros(m_confidenceData.size(), CV_32F);

	while (iter > 0)
	{
		cv::Mat next_input = m_confidenceData.clone();
		calcOrientation(next_input);
		calcConfidence();
		iter--;
	}

	for (int yI = 0; yI < m_confidenceData.rows; yI++)
	{
		for (int xI = 0; xI < m_confidenceData.cols; xI++)
		{
			if (m_maskData.at<uchar>(yI, xI) < 100)
				m_confidenceData.at<float>(yI, xI) = 0;
			//m_confidenceData.at<float>(yI, xI) = m_confidenceData.at<float>(yI, xI) > 0.01 ? m_confidenceData.at<float>(yI, xI) : 0.;
		} // end for xI
	} // end for yI
}

void HairBuilder::toRunParisFielter(cv::Mat colorImg, int npass)
{
	int w = colorImg.cols;
	int h = colorImg.rows;
	cv::Mat filterImg;
	cv::cvtColor(colorImg, filterImg, CV_BGRA2GRAY);
	filterImg.convertTo(filterImg, CV_64FC1);
	filterImg /= 255.;

	int ndegree = 32;
	double sigma_h = 1.;
	double sigma_l = 2.;
	double sigma_y = 2.;

	double *ffttmp = (double *)fftw_malloc(sizeof(double) * w * h);
	fftw_complex *imfft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (w / 2 + 1) * h);
	fftw_plan imdft = fftw_plan_dft_r2c_2d(h, w, ffttmp, imfft, FFTW_ESTIMATE);
	fftw_complex *filtfft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (w / 2 + 1) * h);
	double *filtered = (double *)fftw_malloc(sizeof(double) * w * h);
	fftw_plan idft = fftw_plan_dft_c2r_2d(h, w, filtfft, filtered, FFTW_ESTIMATE);

	for (int iterI = 0; iterI < npass; iterI++)
	{
		double fftscale = 1. / (w * h);
		for (int j = 0; j < w * h; j++)
			ffttmp[j] = fftscale * filterImg.at<double>(j / w, j % w);
		fftw_execute(imdft);

		cv::Mat m_hairConf = cv::Mat::zeros(colorImg.size(), CV_64FC1);
		cv::Mat m_hairOrient = cv::Mat::zeros(colorImg.size(), CV_64FC1);

		for (int i = 0; i < ndegree; i++)
		{
			double angle = CV_PI * i / ndegree;
			double s = -sin(angle);
			double c = cos(angle);
			double xhmult = -2. * CV_PI * CV_PI * sigma_h * sigma_h;
			double xlmult = -2. * CV_PI * CV_PI * sigma_l * sigma_l;
			double ymult = -2. * CV_PI * CV_PI * sigma_y * sigma_y;

#pragma omp parallel for
			for (int y = 0; y < h; y++)
			{
				double ynorm = (double)((y >= h / 2) ? y - h : y) / h;
				for (int x = 0; x <= w / 2; x++)
				{
					double xnorm = (double)x / w;
					double xrot2 = (s * xnorm - c * ynorm) * (s * xnorm - c * ynorm);
					double yrot2 = (c * xnorm + s * ynorm) * (c * xnorm + s * ynorm);
					int i = x + y * (w / 2 + 1);
					double g = exp(xhmult * xrot2 + ymult * yrot2) - exp(xlmult * xrot2 + ymult * yrot2);
					filtfft[i][0] = imfft[i][0] * g;
					filtfft[i][1] = imfft[i][1] * g;
				}
			}

			fftw_execute(idft);

#pragma omp parallel for
			for (int j = 0; j < w * h; j++)
			{
				double res = filtered[j];
				if (abs(m_hairConf.at<double>(j / w, j % w)) < abs(res))
				{
					m_hairOrient.at<double>(j / w, j % w) = angle;
					m_hairConf.at<double>(j / w, j % w) = res;
				}
			}
		}

		double maxConf = 0.;
		for (int hI = 0; hI < h; hI++)
		{
			for (int wI = 0; wI < w; wI++)
			{
				double conf = m_hairConf.at<double>(hI, wI);
				filterImg.at<double>(hI, wI) = MAX(conf, 0.);
				m_hairConf.at<double>(hI, wI) = MAX(conf, 0.);
				maxConf = MAX(maxConf, MAX(conf, 0.));
			}
		}
		for (int hI = 0; hI < h; hI++) {
			for (int wI = 0; wI < w; wI++)
				filterImg.at<double>(hI, wI) /= maxConf;
		}

		if (iterI == npass - 1)
		{
			m_hairConf.convertTo(m_confidenceData, CV_32FC1);
			m_hairOrient.convertTo(m_orientData, CV_32FC1);
		}
	}

	fftw_destroy_plan(imdft);
	fftw_free(imfft);
	fftw_free(ffttmp);
	fftw_destroy_plan(idft);
	fftw_free(filtered);
	fftw_free(filtfft);

	cv::Mat filterHairConf;
	cv::GaussianBlur(m_confidenceData, filterHairConf, cv::Size(21, 21), 0.);
	double maxConf = 0.;
	for (int hI = 0; hI < h; hI++)
	{
		for (int wI = 0; wI < w; wI++)
		{
			m_confidenceData.at<float>(hI, wI) /= MAX(filterHairConf.at<float>(hI, wI), DBL_MIN);
			if (m_maskData.at<uchar>(hI, wI) > 127)
				maxConf = MAX(maxConf, m_confidenceData.at<float>(hI, wI));
		}
	}
	for (int hI = 0; hI < h; hI++)
	{
		for (int wI = 0; wI < w; wI++)
		{
			m_confidenceData.at<float>(hI, wI) /= maxConf;
			if (m_maskData.at<uchar>(hI, wI) <= 127)
				m_confidenceData.at<float>(hI, wI) = 0.;
		}
	}

}

void HairBuilder::filterOri(cv::Mat& rstMap)
{
	int w = m_confidenceData.cols;
	int h = m_confidenceData.rows;

	cv::Mat flow = cv::Mat::zeros(m_confidenceData.size(), CV_64FC2);
#pragma omp parallel for
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			cv::Vec3f vec = rstMap.at<cv::Vec3f>(y, x);
			cv::Vec2f cc(vec[2], vec[1]);
			cc = normalize(cc * 2 - cv::Vec2f(1., 1.));
			flow.at<cv::Vec2d>(y, x) = cv::Vec2d(cc[0], cc[1]) * m_confidenceData.at<float>(y, x);
		}
	}
	cv::GaussianBlur(flow, flow, cv::Size(), 4.);
#pragma omp parallel for
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			if (m_maskData.at<uchar>(y, x) >= 100)
			{
				double orientX = flow.at<cv::Vec2d>(y, x)[0];
				double orientY = flow.at<cv::Vec2d>(y, x)[1];
				if (orientX == 0. || orientY == 0.)
				{
					rstMap.at<cv::Vec3f>(y, x) = cv::Vec3f(0., 0., 0.);
					m_confidenceData.at<float>(y, x) = 0.;
				}
				else
				{
					cv::Vec2f cc(orientX, orientY);
					cc = normalize(cc);
					cc = (cc + cv::Vec2f(1., 1.)) * 0.5;
					rstMap.at<cv::Vec3f>(y, x) = cv::Vec3f(0., cc[1], cc[0]);
				}
			}
		}
	}
}

cv::Mat HairBuilder::getImgVecMap(cv::Mat disAmb)
{
	cv::Mat rstMap = cv::Mat::zeros(m_confidenceData.size(), CV_32FC3);
	for (int yi = 0; yi < rstMap.rows; yi++)
	{
		for (int xi = 0; xi < rstMap.cols; xi++)
		{
			if (m_maskData.at<uchar>(yi, xi) >= 100)
			{
				cv::Vec3f dVV = disAmb.at<cv::Vec3f>(yi, xi);
				cv::Vec2f refV(dVV[0], dVV[1]);
				refV = refV * 2 - cv::Vec2f(1., 1.);
				refV = normalize(refV);
				float ori = m_orientData.at<float>(yi, xi);
				cv::Vec2f vec(sinf(ori), -cosf(ori));
				vec = vec.dot(refV) < 0. ? -vec : vec;
				vec = (vec + cv::Vec2f(1., 1.)) * 0.5;
				rstMap.at<cv::Vec3f>(yi, xi) = cv::Vec3f(0., vec[1], vec[0]);
			}

		} // end for xi
	} // end for yi
	filterOri(rstMap);
	return rstMap;
}

cv::Mat HairBuilder::getParisVecMap(cv::Mat disAmb)
{
	cv::Mat rstMap = cv::Mat::zeros(m_confidenceData.size(), CV_32FC3);
	for (int yi = 0; yi < rstMap.rows; yi++)
	{
		for (int xi = 0; xi < rstMap.cols; xi++)
		{
			if (m_maskData.at<uchar>(yi, xi) >= 100)
			{
				cv::Vec3f dVV = disAmb.at<cv::Vec3f>(yi, xi);
				cv::Vec2f refV(dVV[0], dVV[1]);
				refV = refV * 2 - cv::Vec2f(1., 1.);
				refV = normalize(refV);
				float ori = m_orientData.at<float>(yi, xi);
				cv::Vec2f vec(sinf(ori*2), -cosf(ori*2));
				vec = vec.dot(refV) < 0. ? -vec : vec;
				vec = (vec + cv::Vec2f(1., 1.)) * 0.5;
				rstMap.at<cv::Vec3f>(yi, xi) = cv::Vec3f(0., vec[1], vec[0]);
			}

		} // end for xi
	} // end for yi
	filterOri(rstMap);
	return rstMap;
}

cv::Mat HairBuilder::prepData(cv::Mat Img)
{
	// DoG
	double sigmaHigh = 0.4;
	double sigmaLow = 10.;
	cv::Mat gray;
	cv::cvtColor(Img, gray, CV_BGRA2GRAY);
	gray.convertTo(gray, CV_32FC1);
	cv::Mat allFreq, lowFreq;
	cv::GaussianBlur(gray, allFreq, cv::Size(0, 0), sigmaHigh);
	cv::GaussianBlur(gray, lowFreq, cv::Size(0, 0), sigmaLow);
	cv::Mat rst = allFreq - lowFreq;
	rst.convertTo(rst, CV_8U, 1, 127.5);
	return rst;
}

void HairBuilder::calcOrientation(cv::Mat & inputGray)
{
	int kernelWidth = para_kernelWidth;
	int kernelHeight = para_kernelHeight;
	double sigmaX = para_sigmaX;
	double sigmaY = para_sigmaY;
	double lambdar = para_lambdar;
	double phasee = para_phasee;
	int numKernels = para_numKernels;
	int numPhases = para_numPhases;

	double phaseStep = CV_PI * 2. / (double)numPhases;
	// Try Gabor kernel of each phase
	for (int iPhase = 0; iPhase < numPhases; iPhase++)
	{
		// Array of responses to each orientation
		std::vector<cv::Mat> respArray(numKernels);
		// Try Gabor kernel of each orientation
#pragma omp parallel for
		for (int iOrient = 0; iOrient < numKernels; iOrient++)
		{
			//Create Gabor kernel
			cv::Mat kernel;
			const  double theta = CV_PI * (double)iOrient / (double)numKernels;

			kernel.create(kernelHeight, kernelWidth, CV_32F);
			const double sigmaXSq = sigmaX * sigmaX;
			const double sigmaYSq = sigmaY * sigmaY;
			const double sinTheta = sinf(theta);
			const double cosTheta = cosf(theta);
			for (int row = 0; row < kernelWidth; row++)
			{
				double tempY = row - kernelHeight / 2.;
				float* pValues = kernel.ptr<float>(row);
				for (int col = 0; col < kernelWidth; col++)
				{
					double tempX = col - kernelWidth / 2.;
					double x = tempX*cosTheta - tempY*sinTheta;
					double y = tempX*sinTheta + tempY*cosTheta;
					double compWave = cosf(2.*CV_PI*x / lambdar + phasee);
					double compGauss = expf(-0.5*(x*x / sigmaXSq + y*y / sigmaYSq));
					pValues[col] = compGauss * compWave;
				}
			}

			// Filter and store response
			cv::Mat response;
			filter2D(inputGray, response, CV_32F, kernel);
			respArray[iOrient] = response.clone();
		} // end for iOrient

		// Statistics: calc per-pixel variance...
		for (int y = 0; y < inputGray.rows; y++)
		{
			for (int x = 0; x < inputGray.cols; x++)
			{
				// Find max response at the pixel
				double maxResp = 0.;
				double bestOrient = 0.;
				for (int iOrient = 0; iOrient < numKernels; iOrient++)
				{
					float& resp = respArray[iOrient].ptr<float>(y)[x];
					if (resp < 0.)
						resp = 0.;
					else if (resp > maxResp) {
						maxResp = resp;
						bestOrient = CV_PI * (double)iOrient / (double)numKernels;
					}
				}
				// Calculate variance
				double variance = 0.;
				for (int iOrient = 0; iOrient < numKernels; iOrient++)
				{
					double orient = CV_PI * (double)iOrient / (double)numKernels;
					double orientDiff = MIN(abs(orient - bestOrient), MIN(abs(orient - bestOrient - CV_PI), abs(orient - bestOrient + CV_PI)));
					double respDiff = respArray[iOrient].ptr<float>(y)[x] - maxResp;
					variance += orientDiff * respDiff * respDiff;
				}
				// Standard variance
				variance = sqrt(variance);
				// Update overall variance/orientation if necessary
				if (variance > m_varianceData.ptr<float>(y)[x])
				{
					m_orientData.ptr<float>(y)[x] = bestOrient;
					m_varianceData.ptr<float>(y)[x] = variance;
					m_maxRespData.ptr<float>(y)[x] = maxResp;
				}
			}
		}

		phasee += phaseStep;
	}

	// Normalize variance and max response
	double maxAllResponse = 0.;
	double maxAllVariance = 0.;
	for (int y = 0; y < inputGray.rows; y++)
	{
		for (int x = 0; x < inputGray.cols; x++)
		{
			if (m_varianceData.ptr<float>(y)[x] > maxAllVariance)
				maxAllVariance = m_varianceData.ptr<float>(y)[x];
			if (m_maxRespData.ptr<float>(y)[x] > maxAllResponse)
				maxAllResponse = m_maxRespData.ptr<float>(y)[x];
		}
	}
	m_maxRespData /= maxAllResponse;
	m_varianceData /= maxAllVariance;
}

void HairBuilder::calcConfidence()
{
	// Variance
	double clampConfidLow = para_clamConfidenceLow;
	double clampConfidHigh = para_clampConfidenceHigh;

	m_confidenceData = m_varianceData.clone();
	for (int y = 0; y < m_confidenceData.rows; y++)
	{
		for (int x = 0; x < m_confidenceData.cols; x++)
		{
			float& c = m_confidenceData.at<float>(y, x);
			c = (c - clampConfidLow) / (clampConfidHigh - clampConfidLow);
			if (c != c)
				c = 0.;
			else if (c < 0.)
				c = 0.;
			else if (c > 1.)
				c = 1.;
		}
	}
}

