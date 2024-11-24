/**
 * @brief CUDA-Accelerated Watershed Algorithm Implementation Using OpenCV
 * @details This program performs image segmentation using the Watershed algorithm,
 *          leveraging CUDA for accelerated image processing tasks.
 * @author 
 */

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

/**
 * @brief Extracts the base name from a file path.
 * @param path The full path to the file.
 * @return The base name of the file (e.g., "image.jpg" from "/path/to/image.jpg").
 */
string getBaseName(const string& path) {
    // Find the last occurrence of '/' or '\\'
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos)
        return path; // No path separators found
    else
        return path.substr(pos + 1);
}

int main(int argc, char* argv[])
{
    // Start total execution timer
    double total_start = (double)getTickCount();

    //! [load_image]
    // Load the image
    double t1 = (double)getTickCount();

    // Parse command line arguments
    CommandLineParser parser(argc, argv, "{@input | cards.png | input image}");
    string input_image_path = parser.get<String>("@input");
    Mat src = imread(samples::findFile(input_image_path));
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    double t1_elapsed = ((double)getTickCount() - t1) / getTickFrequency();
    cout << "Time taken to load image: " << t1_elapsed << " seconds." << endl;
    //! [load_image]

    //! [upload_image]
    // Upload the image to GPU
    double t1_gpu = (double)getTickCount();

    cv::cuda::GpuMat d_src;
    d_src.upload(src);

    double t1_gpu_elapsed = ((double)getTickCount() - t1_gpu) / getTickFrequency();
    cout << "Time taken to upload image to GPU: " << t1_gpu_elapsed << " seconds." << endl;
    //! [upload_image]

    //! [black_bg]
    // Change the background from white to black
    double t2 = (double)getTickCount();

    cv::cuda::GpuMat d_mask;
    cv::cuda::inRange(d_src, Scalar(255, 255, 255), Scalar(255, 255, 255), d_mask);
    d_src.setTo(Scalar(0, 0, 0), d_mask);

    double t2_elapsed = ((double)getTickCount() - t2) / getTickFrequency();
    cout << "Time taken to change background: " << t2_elapsed << " seconds." << endl;
    //! [black_bg]

    //! [sharp]
    // Sharpen the image using Laplacian filtering
    double t3 = (double)getTickCount();

    // **Convert to grayscale**
    cv::cuda::GpuMat d_gray;
    cv::cuda::cvtColor(d_src, d_gray, COLOR_BGR2GRAY);

    // Convert to float
    cv::cuda::GpuMat d_gray_float;
    d_gray.convertTo(d_gray_float, CV_32F);

    // Create Laplacian kernel
    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    // Apply Laplacian filter
    cv::Ptr<cv::cuda::Filter> laplacian_filter = cv::cuda::createLinearFilter(
        d_gray_float.type(), d_gray_float.type(), kernel);

    cv::cuda::GpuMat d_imgLaplacian;
    laplacian_filter->apply(d_gray_float, d_imgLaplacian);

    // Sharpen the image
    cv::cuda::GpuMat d_imgResult;
    cv::cuda::subtract(d_gray_float, d_imgLaplacian, d_imgResult);

    // Convert back to 8-bit
    d_imgResult.convertTo(d_imgResult, CV_8U);

    double t3_elapsed = ((double)getTickCount() - t3) / getTickFrequency();
    cout << "Time taken to sharpen image: " << t3_elapsed << " seconds." << endl;
    //! [sharp]

    //! [bin]
    // Create binary image from sharpened grayscale image
    double t4 = (double)getTickCount();

    // Since the image is already grayscale after sharpening, no need to convert
    // Threshold to create binary image
    double thresh_value = 40.0;
    cv::cuda::GpuMat d_bw_thresh;
    cv::cuda::threshold(d_imgResult, d_bw_thresh, thresh_value, 255, THRESH_BINARY);

    double t4_elapsed = ((double)getTickCount() - t4) / getTickFrequency();
    cout << "Time taken to create binary image: " << t4_elapsed << " seconds." << endl;
    //! [bin]

    //! [dist_cpu]
    // Perform the distance transform algorithm on CPU
    double t5 = (double)getTickCount();

    // Download the binary image from GPU to CPU
    Mat bw_thresh;
    d_bw_thresh.download(bw_thresh);

    // Perform distance transform on CPU
    Mat dist;
    distanceTransform(bw_thresh, dist, DIST_L2, 3);

    double t5_elapsed = ((double)getTickCount() - t5) / getTickFrequency();
    cout << "Time taken for distance transform (CPU): " << t5_elapsed << " seconds." << endl;
    //! [dist_cpu]

    //! [peaks]
    // Normalize and threshold to obtain peaks
    double t6 = (double)getTickCount();

    Mat dist_norm;
    // Normalize to range 0 to 255
    normalize(dist, dist_norm, 0.0, 255.0, NORM_MINMAX);

    // Convert to 8-bit image
    dist_norm.convertTo(dist_norm, CV_8U);

    // Threshold to obtain peaks
    Mat peaks;
    threshold(dist_norm, peaks, 0.4 * 255.0, 255, THRESH_BINARY);

    // Dilate the peaks
    dilate(peaks, peaks, Mat::ones(3, 3, CV_8U));

    double t6_elapsed = ((double)getTickCount() - t6) / getTickFrequency();
    cout << "Time taken to obtain peaks (CPU): " << t6_elapsed << " seconds." << endl;
    //! [peaks]

    //! [seeds]
    // Create markers for watershed
    double t7 = (double)getTickCount();

    // Find contours
    vector<vector<Point>> contours;
    findContours(peaks, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create marker image
    Mat markers = Mat::zeros(peaks.size(), CV_32S);

    // Draw foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }

    // Draw background marker
    circle(markers, Point(5, 5), 3, Scalar(255), -1);

    double t7_elapsed = ((double)getTickCount() - t7) / getTickFrequency();
    cout << "Time taken to create markers: " << t7_elapsed << " seconds." << endl;
    //! [seeds]

    //! [watershed]
    // Perform the watershed algorithm
    double t8 = (double)getTickCount();

    // Download the sharpened grayscale image to CPU
    Mat imgResult_host_gray;
    d_imgResult.download(imgResult_host_gray);

    // **Convert grayscale image to BGR**
    Mat imgResult_host;
    cvtColor(imgResult_host_gray, imgResult_host, COLOR_GRAY2BGR);

    // Apply watershed
    watershed(imgResult_host, markers);

    double t8_elapsed = ((double)getTickCount() - t8) / getTickFrequency();
    cout << "Time taken for watershed: " << t8_elapsed << " seconds." << endl;
    //! [watershed]

    //! [result]
    // Generate the result image
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
        }
    }

    double t9_elapsed = ((double)getTickCount() - t9) / getTickFrequency();
    cout << "Time taken to generate result image: " << t9_elapsed << " seconds." << endl;
    //! [result]

    // **Save the final result image**
    // Extract the base name from the input image path
    string base_name = getBaseName(input_image_path);

    // Construct the output file name: "final_result_<base_name>"
    string output_file_name = "final_result_" + base_name;

    // Save the result image
    bool isSaved = imwrite(output_file_name, dst);
    if (isSaved)
    {
        cout << "Final segmented image saved as: " << output_file_name << endl;
    }
    else
    {
        cout << "Failed to save the final segmented image." << endl;
    }

    double total_elapsed = ((double)getTickCount() - total_start) / getTickFrequency() - t1_gpu_elapsed;
    cout << "Total time taken: " << total_elapsed << " seconds." << endl;

    // Optionally, display the final result
    // imshow("Final Result", dst);
    // waitKey();

    return 0;
}
