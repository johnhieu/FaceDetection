#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name, eyes_cascade_name, lefteye_cascade_name, righteye_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier lefteye_cascade;
CascadeClassifier righteye_cascade;
String window_name = "Capture - Face detection";

/** @function main */
int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
		"{eyes_cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
		"{lefteye_cascade|../../data/haarcascades/haarcascade_lefteye_2splits.xml}"
		"{righteye_cascade|../../data/haarcascades/haarcascade_righteye_2splits.xml}");

	cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n";
	parser.printMessage();

	face_cascade_name = parser.get<string>("face_cascade");
	eyes_cascade_name = parser.get<string>("eyes_cascade");
	lefteye_cascade_name = parser.get<string>("lefteye_cascade");
	righteye_cascade_name = parser.get<string>("righteye_cascade");
	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	if (!lefteye_cascade.load(lefteye_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	if (!righteye_cascade.load(righteye_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	//-- 2. Read the video stream
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);

		char c = (char)waitKey(10);
		if (c == 27) { break; } // escape
	}
	return 0;
}


/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.2, 2, 0 | CASCADE_SCALE_IMAGE, Size(20,20));

	for (size_t i = 0; i < faces.size(); i++)
	{
		//
		Rect face = faces[i];
	
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		rectangle(frame, face, Scalar(142, 0, 123), 4, 8, 0);
		
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		
		/*-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.05, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
		*/


		//-- Detect left eye each face
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.2, 10, 0 |CASCADE_SCALE_IMAGE, Size(20,20), Size(100,100));
		for (size_t j = 0; j < eyes.size(); j++)
		{

			Rect eye = eyes[j];
			eye.x += faces[i].x;
			eye.y += faces[i].y;
			/*
			eye.y += eye.height / 4;
			eye.height /= 2;
			*/
			rectangle(frame, eye, Scalar(255, 0, 0), 4, 8, 0);
			
			Mat eyesROI = frame_gray(eye);
			
			/*
			Canny(eyesROI, eyesROI, 50, 150 , 3);
			eyesROI.convertTo(eyesROI, CV_8U);
			*/
			threshold(eyesROI, eyesROI, 20, 255, 0);
			threshold(eyesROI, eyesROI, 70, 255, 2);
			
			Canny(eyesROI, eyesROI, 20, 30, 3);
			//GaussianBlur(eyesROI, eyesROI, Size(9, 9), 4, 4);
			std::vector<Vec3f> circles;
			
			if (j == 0)
			{
				imshow("Eyes", eyesROI);
				namedWindow("Eyes", WINDOW_NORMAL);
			}
			
			HoughCircles(eyesROI, circles, CV_HOUGH_GRADIENT, 10, 16, 150, 80,3,15);
			/*
			Mat canny_output;
			vector<vector<Point>> contours;
			vector<Vec4i> hierachy;
			Canny(eyesROI, canny_output, 100, 200, 3);
			findContours(canny_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
			Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

			for (size_t k = 0; k < contours.size(); k++)
			{
				drawContours(drawing, contours, (int)k, Scalar(0, 0, 255), 2, 8, hierachy, 0, Point());
			}
			imshow("Eyes", drawing);
			namedWindow("Eyes", WINDOW_AUTOSIZE);
			*/
			for (size_t k = 0; k < circles.size(); k++)
			{ 
				Point center(cvRound(circles[k][0] + faces[i].x + eyes[j].x), cvRound(circles[k][1] + faces[i].y + eyes[j].y ));
				int radius = cvRound(circles[k][2]);
				circle(frame, center, 3, Scalar(0, 255, 0), -1, 8, 0);
				circle(frame, center, radius, Scalar(0, 0, 255), 2, 8, 0);
			}
			
		}
		
		}
	//-- Show what you got
	imshow(window_name, frame);
	
}