//
//  main.cpp
//  firstOpenCV
//
//  Created by Youngwook Paul Kwon on 7/12/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <vector>

#include "Eigen/Dense"

CvPoint start, end;
bool bDrawing_s = false;
bool bDrawing_d = false;

int nth_color = 0;
CvScalar colors[4];

using namespace std;
using namespace Eigen;

typedef Vector3d line;

class View
{
public:
	char sName[100];
	
	IplImage* image;
	IplImage* tempDrawing;
	bool bDrawing;
	
	vector<line> lines;
	int nthColor;
	
	View()
	{
		nthColor = 0;
		bDrawing = false;
	}
	
	bool Load(const char* str)
	{
		image = cvLoadImage(str,1);
		if (!image) return false;
		
		tempDrawing = cvCloneImage(image);
		cvNamedWindow(sName);
		ShowImage();
		return true;
	}
	
	void ShowImage()
	{
		cvShowImage(sName, image);
	}
	
	void ShowTempImage()
	{
		cvCopyImage(image, tempDrawing);
		if(bDrawing) cvLine(tempDrawing, start, end, CV_RGB(0,0,250), 2);
		cvShowImage(sName, tempDrawing);
	}
	
	void AddLine(Vector3d l)
	{
		lines.push_back(l);
	}
};

void my_mouse_callback(int event, int x, int y, int flags, void* param)
{	
	View* target = (View*) param;
	
	switch(event)
	{
		case CV_EVENT_MOUSEMOVE:
			if (target->bDrawing)
			{
				end.x=x;
				end.y=y;
			}
			break;
		case CV_EVENT_LBUTTONDOWN:
			target->bDrawing = true;
			start = cvPoint(x,y);
			end = cvPoint(x,y);
			break;
		case CV_EVENT_RBUTTONDOWN:
			target->bDrawing = false;
			break;
		case CV_EVENT_LBUTTONUP:
			
			if (target->bDrawing) {
				end = cvPoint(x, y);
				
				if (start.x==end.x && start.y==end.y) 
				{
					target->bDrawing = false;
					break;
				}
				
				cvLine(target->image, start, end, colors[(target->nthColor)%4],2);
				
				// line vector
				Vector3d p1 = Vector3d(start.x, start.y, 1);
				Vector3d p2 = Vector3d(end.x, end.y, 1);
				Vector3d l = p1.cross(p2);
				l = l/l[2];
				
				assert(abs(l[2]-1)<1.0e-6);
				
				target->AddLine(l);
				target->nthColor++;
					
				///
				double a = l[0];
				double b = l[1];
				
				if (b!=0)
				{
					double y1 = -1/b;
					double y2 = -(a*1000+1)/b;
					cvLine(target->image, cvPoint(0,y1), cvPoint(1000,y2), CV_RGB(0,0,0), 1, 4);
				}
				else
				{
					double x1 = -1/a;
					cvLine(target->image, cvPoint(x1,0), cvPoint(x1,1000), CV_RGB(0,0,0), 1);
					
					cout << "l" << l << endl;
					cout << "x,y: " << x << ", " << y << endl;
					cout << "x1,0: " << x1 << ", " << 0 << endl;
					cout << "x1,1000: " << x1 << ", " << 1000 << endl;
				}
			}
			
			target->bDrawing = false;
			break;
	}
}

cv::Mat H;

void GetHomography(const vector<line> &s, const vector<line> &d)
{	
	if (s.size()<4 || d.size()<4) return;

	cv::Mat A;
	cv::Mat x_(9,1,CV_32F,0);
	
	for (int i=0; i<4; i++) {
		
//		cout << "l" << i << " = [ " << s.at(i)[0] <<";"<< s.at(i)[1] << ";" << s.at(i)[2] << "];" << endl;
//		cout << "l" << i << "_ = [ " << d.at(i)[0] <<";"<< d.at(i)[1] << ";" << d.at(i)[2] << "];" << endl;
		
		double u, v, x, y;
		x = (s.at(i))[0];
		y = (s.at(i))[1];
		assert((s.at(0))[2]-1<1.0e-6);
		
		u = (d.at(i))[0];
		v = (d.at(i))[1];
		assert((d.at(0))[2]-1<1.0e-6);
		
		cv::Mat a = (cv::Mat_<float>(2,9) << -u, 0, u*x, -v, 0, v*x, -1, 0, x,
											  0, -u, u*y, 0, -v, v*y, 0, -1, y);
		A.push_back(a);
	}
	
	cv::SVD::solveZ(A, x_);	
	
	H = x_.reshape(0,3);
	H = H/H.at<float>(2,2);
	
	//cout << "A" << endl << A << endl;
	//cout << "H" << endl << H << endl;

	return;
}


char biliearinterpolation(IplImage *src, float fx, float fy, char *result_pixel)
{
	int ix = (int) fx;
	int iy = (int) fy;
	
	float tx = fx - ix;
	float ty = fy - iy;
	
	tx=0; ty=0;
	assert(tx>=0 && tx<=1);
	assert(ty>=0 && ty<=1);
	
	char* data = src->imageData;
	int nchannels = src->nChannels;
	int step = src->widthStep;

	char interp_val;
	for (int ch=0; ch < nchannels; ch++)
	{
		char x_y = data[iy*step + ix*nchannels + ch];
		char xinc_y = data[iy*step + (ix+1)*nchannels + ch];
		char x_yinc = data[(iy+1)*step + ix*nchannels + ch];
		char xinc_yinc = data[(iy+1)*step + (ix+1)*nchannels + ch];
		
		char interp_x1 = x_y*(1-tx) + xinc_y*tx;
		char interp_x2 = x_yinc*(1-tx) + xinc_yinc*tx;
		
		result_pixel[ch] = interp_x1*(1-ty) + interp_x2*ty;
	}
}

IplImage* Remap(IplImage *srcImage, IplImage *dstImage, const cv::Mat& H)
{
	IplImage *resultImage = cvCloneImage(dstImage);
	cvZero(resultImage);
	
	cv::Mat H_inv = H.inv();
	
	int width = resultImage->width;
	int height = resultImage->height;
	int nchannels = resultImage->nChannels;
	int step = resultImage->widthStep;
	
	for (int x=0; x < width; x++) {
		for (int y=0; y < height; y++) {
			float original_x = x*H_inv.at<float>(0,0) + y*H_inv.at<float>(0,1) + H_inv.at<float>(0,2);
			float original_y = x*H_inv.at<float>(1,0) + y*H_inv.at<float>(1,1) + H_inv.at<float>(1,2);
			float original_z = x*H_inv.at<float>(2,0) + y*H_inv.at<float>(2,1) + H_inv.at<float>(2,2);
			
			if (fabs(original_z)<1e-6) continue;
			
			original_x /= original_z;
			original_y /= original_z;
			
			if (original_x < 0 || original_x > srcImage->width-2) continue;
			if (original_y < 0 || original_y > srcImage->height-2) continue;
			
			char result[5] = {0, 0, 0, 0, 0};
			biliearinterpolation(srcImage, original_x, original_y, result);
			memcpy(resultImage->imageData + y*step + x*nchannels, result, nchannels);
		}
	}
	
	return resultImage;
}
				 
int main (int argc, const char * argv[])
{
	colors[0] = CV_RGB(250, 0, 0);
	colors[1] = CV_RGB(250, 125, 0);
	colors[2] = CV_RGB(250, 250, 0);
	colors[3] = CV_RGB(0, 250, 0);
	
	H = cv::Mat::eye(3,3,CV_32F);
	H.at<float>(1,2) = 30.3;
	
	cout << H << endl;
	
	View source, destination;
	IplImage *remapImage = NULL;
	
	strcpy(source.sName, "source");
	strcpy(destination.sName, "destination");
	
	if (!source.Load("/Users/garlicbread/Desktop/liver_now.jpg") ||
		!destination.Load("/Users/garlicbread/Desktop/liver	_old.jpg"))
	{
		cout << "no pic" << endl;
		return 0;
	}
	
	cvSetMouseCallback(source.sName, my_mouse_callback, (void*) &source);
	cvSetMouseCallback(destination.sName, my_mouse_callback, (void*) &destination);
	
	while (1) {
		source.ShowTempImage();
		destination.ShowTempImage();
		
		int key = cvWaitKey(15);
		
		switch (key) {
			case 27:
				return 0;
				break;
			case 'h':
				GetHomography(source.lines, destination.lines);
				remapImage = Remap(source.image, destination.image, H);
				cvNamedWindow("result",CV_WINDOW_AUTOSIZE);
				cvShowImage("result", remapImage);
				break;
			case 'r':
				remapImage = Remap(source.image, destination.image, H);
				cvNamedWindow("result",CV_WINDOW_AUTOSIZE);
				cvShowImage("result", remapImage);
				break;
			case 's':
				if (source.image) cvSaveImage("source.jpg", source.image);
				if (destination.image) cvSaveImage("destination.jpg", destination.image);
				if (remapImage) cvSaveImage("remap.jpg", remapImage);
				break;
				
		}
	}

    return 0;
}