/**
 * @brief Sample code showing how to segment overlapping objects using Laplacian filtering, in addition to Watershed and Distance Transformation
 * @author OpenCV Team
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    double total_start = (double)getTickCount();

    //! [load_image]
    // Load the image
    double t1 = (double)getTickCount();

    CommandLineParser parser(argc, argv, "{@input | cards.png | input image}");
    Mat src = imread(samples::findFile(parser.get<String>("@input")));
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    double t1_elapsed = ((double)getTickCount() - t1) / getTickFrequency();
    cout << "Time taken to load image: " << t1_elapsed << " seconds." << endl;

    // Show the source image
    imshow("Source Image", src);
    //! [load_image]

    //! [black_bg]
    // Change the background from white to black
    double t2 = (double)getTickCount();

    Mat mask;
    inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(0, 0, 0), mask);

    double t2_elapsed = ((double)getTickCount() - t2) / getTickFrequency();
    cout << "Time taken to change background: " << t2_elapsed << " seconds." << endl;

    // Show output image
    imshow("Black Background Image", src);
    //! [black_bg]

    //! [sharp]
    // Sharpen the image
    double t3 = (double)getTickCount();

    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1); // an approximation of second derivative, a quite strong kernel

    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    double t3_elapsed = ((double)getTickCount() - t3) / getTickFrequency();
    cout << "Time taken to sharpen image: " << t3_elapsed << " seconds." << endl;

    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow("New Sharpened Image", imgResult);
    //! [sharp]

    //! [bin]
    // Create binary image from source image
    double t4 = (double)getTickCount();

    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);

    double t4_elapsed = ((double)getTickCount() - t4) / getTickFrequency();
    cout << "Time taken to create binary image: " << t4_elapsed << " seconds." << endl;

    imshow("Binary Image", bw);
    //! [bin]

    //! [dist]
    // Perform the distance transform algorithm
    double t5 = (double)getTickCount();

    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);

    double t5_elapsed = ((double)getTickCount() - t5) / getTickFrequency();
    cout << "Time taken for distance transform: " << t5_elapsed << " seconds." << endl;

    imshow("Distance Transform Image", dist);
    //! [dist]

    //! [peaks]
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    double t6 = (double)getTickCount();

    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);

    double t6_elapsed = ((double)getTickCount() - t6) / getTickFrequency();
    cout << "Time taken to obtain peaks: " << t6_elapsed << " seconds." << endl;

    imshow("Peaks", dist);
    //! [peaks]

    //! [seeds]
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    double t7 = (double)getTickCount();

    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);

    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }

    // Draw the background marker
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);

    double t7_elapsed = ((double)getTickCount() - t7) / getTickFrequency();
    cout << "Time taken to create markers: " << t7_elapsed << " seconds." << endl;

    imshow("Markers", markers8u);
    //! [seeds]

    //! [watershed]
    // Perform the watershed algorithm
    double t8 = (double)getTickCount();

    watershed(imgResult, markers);

    double t8_elapsed = ((double)getTickCount() - t8) / getTickFrequency();
    cout << "Time taken for watershed: " << t8_elapsed << " seconds." << endl;

    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point

    // Generate random colors
    double t9 = (double)getTickCount();

    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);

        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }

    double t9_elapsed = ((double)getTickCount() - t9) / getTickFrequency();
    cout << "Time taken to generate result image: " << t9_elapsed << " seconds." << endl;

    // Visualize the final image
    imshow("Final Result", dst);
    //! [watershed]

    double total_elapsed = ((double)getTickCount() - total_start) / getTickFrequency();
    cout << "Total time taken: " << total_elapsed << " seconds." << endl;

    waitKey();
    return 0;
}
