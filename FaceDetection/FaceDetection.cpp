
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <chrono>
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
chrono::system_clock::time_point startTime;
chrono::system_clock::time_point endTime;
bool eyeClosed = false;
int FPS = 0;

/** @function main */
int main(int argc, const char** argv)
{
 	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|../../data/haarcascades_cuda/haarcascade_frontalface_alt.xml|}"
		//"{face_cascade|../../data/lbpcascades/lbpcascade_frontalface.xml}"
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
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(200, 200));

	for (size_t i = 0; i < faces.size(); i++)
	{

		Rect face = faces[i];

		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		rectangle(frame, face, Scalar(142, 0, 123), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		GaussianBlur(faceROI, faceROI, Size(5, 5), 3, 3);
		//-- Detect left eye each face
		righteye_cascade.detectMultiScale(faceROI, eyes, 1.1, 10, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(80, 80));
		for (size_t j = 0; j < eyes.size(); j++)
		{

			Rect eye = eyes[j];
			eye.x += faces[i].x;
			eye.height /= 2;
			eye.y += faces[i].y + (3 * eye.height / 4);
			//eye.y += faces[i].y;
			rectangle(frame, eye, Scalar(255, 0, 0), 4, 8, 0);


			Mat eyesROI = frame_gray(eye);
			/*
			Canny(eyesROI, eyesROI, 50, 150 , 3);
			eyesROI.convertTo(eyesROI, CV_8U);
			*/

			// For dark frame
			//threshold(eyesROI, eyesROI, 80, 255, 2);
			GaussianBlur(eyesROI, eyesROI, Size(5, 5), 3);
			threshold(eyesROI, eyesROI, 20, 255, 0);
			
			adaptiveThreshold(eyesROI, eyesROI, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 3);
			
		
			Canny(eyesROI, eyesROI, 30 ,90, 5);

			std::vector<Vec3f> circles;


			HoughCircles(eyesROI, circles, CV_HOUGH_GRADIENT, 8, 12, 220, 90, 2, 13);
			/*
			for (size_t k = 0; k < circles.size(); k++)
			{
				Point center(cvRound(circles[k][0] + eye.x), cvRound(circles[k][1] + eye.y));
				int radius = cvRound(circles[k][2]);
				circle(frame, center, 3, Scalar(0, 255, 0), -1, 8, 0);
				circle(frame, center, radius, Scalar(0, 0, 255), 2, 8, 0);
			}
			*/
			if (circles.size() > 0)
			{			
				putText(frame, "Eye Open", cvPoint(50, 50), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 255, 255), 3, 8);
				if (!eyeClosed)
				{
					
					eyeClosed = true;
					endTime = chrono::system_clock::now();				
					FPS = 0;
				}		
			}
			else
			{

				
				//+
				FPS++;
				if (FPS > 3)
				{
					if (eyeClosed)
					{
						startTime = chrono::system_clock::now();
					}				
					string interval = to_string(chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - startTime).count());
					putText(frame, "Eye closed: " + interval, cvPoint(50, 50), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 255, 255), 3, 8);
					eyeClosed = false;
				}
				else
				{
					putText(frame, "Eye Open", cvPoint(50, 50), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 255, 255), 3, 8);
				}
				
			}
			imshow("Eyes", eyesROI);
			namedWindow("Eyes", WINDOW_NORMAL);

		}
	}
	//-- Show what you got
	imshow(window_name, frame);
	
}
