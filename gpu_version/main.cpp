/**
 * @brief Sample code showing how to segment overlapping objects using Laplacian filtering,
 * with CUDA acceleration where possible, using the GPU version of the watershed algorithm.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// Include CUDA headers
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

// Declare the cudaWatershed function
void cudaWatershed(const Mat& image, Mat& markers);

int main(int argc, char* argv[])
{
    double total_start = (double)getTickCount();

    // Load the image on CPU
    double t1 = (double)getTickCount();

    CommandLineParser parser(argc, argv, "{@input | cards.png | input image}");
    string inputImagePath = parser.get<string>("@input");
    Mat src = imread(samples::findFile(inputImagePath));
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    double t1_elapsed = ((double)getTickCount() - t1) / getTickFrequency();
    cout << "Time taken to load image: " << t1_elapsed << " seconds." << endl;

    // Upload image to GPU
    double t_upload = (double)getTickCount();

    cv::cuda::GpuMat d_src;
    d_src.upload(src);

    double t_upload_elapsed = ((double)getTickCount() - t_upload) / getTickFrequency();
    cout << "Time taken to upload image to GPU: " << t_upload_elapsed << " seconds." << endl;

    // Change the background from white to black using CUDA
    double t2 = (double)getTickCount();

    cv::cuda::GpuMat d_mask;
    cv::cuda::inRange(d_src, Scalar(255, 255, 255), Scalar(255, 255, 255), d_mask);
    d_src.setTo(Scalar(0, 0, 0), d_mask);

    double t2_elapsed = ((double)getTickCount() - t2) / getTickFrequency();
    cout << "Time taken to change background: " << t2_elapsed << " seconds." << endl;

    // Sharpen the image using CUDA
    double t3 = (double)getTickCount();

    // Create a kernel for sharpening
    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1); // An approximation of second derivative, a quite strong kernel

    // Convert d_src to BGRA (4 channels)
    cv::cuda::GpuMat d_src_rgba;
    cv::cuda::cvtColor(d_src, d_src_rgba, COLOR_BGR2BGRA);

    // Convert to float
    cv::cuda::GpuMat d_src_rgba_float;
    d_src_rgba.convertTo(d_src_rgba_float, CV_32F);

    // Create filter
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLinearFilter(
        d_src_rgba_float.type(), d_src_rgba_float.type(), kernel);

    // Apply the filter
    cv::cuda::GpuMat d_imgLaplacian;
    filter->apply(d_src_rgba_float, d_imgLaplacian);

    // Subtract Laplacian from the original image
    cv::cuda::GpuMat d_imgResult;
    cv::cuda::subtract(d_src_rgba_float, d_imgLaplacian, d_imgResult);

    // Convert back to 8-bit
    d_imgResult.convertTo(d_imgResult, CV_8UC4);

    // Convert back to BGR (3 channels)
    cv::cuda::GpuMat d_imgResult_bgr;
    cv::cuda::cvtColor(d_imgResult, d_imgResult_bgr, COLOR_BGRA2BGR);

    double t3_elapsed = ((double)getTickCount() - t3) / getTickFrequency();
    cout << "Time taken to sharpen image: " << t3_elapsed << " seconds." << endl;

    // Download the result back to CPU for further processing
    double t_download = (double)getTickCount();

    Mat imgResult;
    d_imgResult_bgr.download(imgResult);

    double t_download_elapsed = ((double)getTickCount() - t_download) / getTickFrequency();
    cout << "Time taken to download image from GPU: " << t_download_elapsed << " seconds." << endl;

    // Continue processing on CPU
    // Create binary image from source image
    double t4 = (double)getTickCount();

    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);

    double t4_elapsed = ((double)getTickCount() - t4) / getTickFrequency();
    cout << "Time taken to create binary image: " << t4_elapsed << " seconds." << endl;

    // Perform the distance transform algorithm
    double t5 = (double)getTickCount();

    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);

    double t5_elapsed = ((double)getTickCount() - t5) / getTickFrequency();
    cout << "Time taken for distance transform: " << t5_elapsed << " seconds." << endl;

    // Threshold to obtain the peaks
    double t6 = (double)getTickCount();

    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);

    double t6_elapsed = ((double)getTickCount() - t6) / getTickFrequency();
    cout << "Time taken to obtain peaks: " << t6_elapsed << " seconds." << endl;

    // Create markers for watershed
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

    // Ensure that unlabeled pixels are 0 and markers are positive integers

    double t7_elapsed = ((double)getTickCount() - t7) / getTickFrequency();
    cout << "Time taken to create markers: " << t7_elapsed << " seconds." << endl;

    // Perform the watershed algorithm using CUDA
    // double t8 = (double)getTickCount();

    // cudaWatershed(imgResult, markers);

    // double t8_elapsed = ((double)getTickCount() - t8) / getTickFrequency();
    // cout << "Time taken for CUDA watershed: " << t8_elapsed << " seconds." << endl;

    // Uncomment the following lines to use the CPU version of watershed
    
    // Perform the watershed algorithm on CPU
    double t8 = (double)getTickCount();

    watershed(imgResult, markers);

    double t8_elapsed = ((double)getTickCount() - t8) / getTickFrequency();
    cout << "Time taken for watershed: " << t8_elapsed << " seconds." << endl;
    

    // Generate random colors and create the result image
    double t9 = (double)getTickCount();

    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    // Generate random colors
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
            else if (index == -1)
            {
                dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255); // Mark watershed boundaries in white
            }
        }
    }

    double t9_elapsed = ((double)getTickCount() - t9) / getTickFrequency();
    cout << "Time taken to generate result image: " << t9_elapsed << " seconds." << endl;

    double total_elapsed = ((double)getTickCount() - total_start) / getTickFrequency();
    cout << "Total time taken: " << total_elapsed << " seconds." << endl;

    // Save the final result image
    // Extract the original image filename without extension
    string filename = inputImagePath;
    size_t lastSlash = filename.find_last_of("/\\");
    if (lastSlash != string::npos)
    {
        filename = filename.substr(lastSlash + 1);
    }
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != string::npos)
    {
        filename = filename.substr(0, lastDot);
    }
    string outputFilename = "final_result_" + filename + ".png";
    imwrite(outputFilename, dst);
    cout << "Final result image saved as: " << outputFilename << endl;

    // Uncomment the following lines to display images
    // imshow("Source Image", src);
    // imshow("Sharpened Image", imgResult);
    // imshow("Binary Image", bw);
    // imshow("Distance Transform Image", dist);
    // imshow("Markers", mark);
    // imshow("Final Result", dst);

    // waitKey();
    return 0;
}
