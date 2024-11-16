// cuda_watershed.cu

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda_runtime.h>

#include <iostream>

// Using the cv namespace
using namespace cv;

// CUDA kernel for the watershed algorithm
__global__ void watershedKernel(
    const float* grad, // Gradient magnitude image
    int* labels,       // Marker labels
    int width,
    int height,
    bool* changed)     // Flag to indicate if any label has changed
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    if (labels[idx] <= 0) // Process only labeled pixels
        return;

    // Offsets for 4-connected neighbors
    int dx[4] = { -1, 1, 0, 0 };
    int dy[4] = { 0, 0, -1, 1 };

    int current_label = labels[idx];

    for (int i = 0; i < 4; ++i)
    {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            continue;

        int nidx = ny * width + nx;

        int neighbor_label = labels[nidx];

        if (neighbor_label == 0)
        {
            // Assign label to neighbor
            labels[nidx] = current_label;
            *changed = true;
        }
        else if (neighbor_label > 0 && neighbor_label != current_label)
        {
            // Conflict detected, mark as watershed
            labels[idx] = -1; // WSHED
        }
    }
}

// Function to perform the watershed algorithm using CUDA
void cudaWatershed(const Mat& image, Mat& markers)
{
    // Convert image to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Upload images to GPU
    cv::cuda::GpuMat d_gray, d_markers;
    d_gray.upload(gray);
    d_markers.upload(markers);

    // Compute gradient magnitude using Sobel operator in CUDA
    cv::cuda::GpuMat d_grad_x, d_grad_y;
    cv::Ptr<cv::cuda::Filter> sobel_x = cv::cuda::createSobelFilter(CV_8U, CV_32F, 1, 0, 3);
    cv::Ptr<cv::cuda::Filter> sobel_y = cv::cuda::createSobelFilter(CV_8U, CV_32F, 0, 1, 3);

    sobel_x->apply(d_gray, d_grad_x);
    sobel_y->apply(d_gray, d_grad_y);

    // Compute gradient magnitude
    cv::cuda::GpuMat d_grad;
    cv::cuda::magnitude(d_grad_x, d_grad_y, d_grad);

    // Normalize gradient
    cv::cuda::GpuMat d_grad_norm;
    cv::cuda::normalize(d_grad, d_grad_norm, 0.0, 1.0, NORM_MINMAX, -1, cv::noArray(), cv::cuda::Stream::Null());

    // Perform the watershed algorithm on GPU
    int width = d_grad_norm.cols;
    int height = d_grad_norm.rows;
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Allocate memory for the flag indicating if changes have occurred
    bool h_changed;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    // Maximum number of iterations
    int maxIterations = 1000;

    // Iterative flooding process
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // Call the CUDA kernel
        watershedKernel<<<gridSize, blockSize>>>(
            d_grad_norm.ptr<float>(),
            d_markers.ptr<int>(),
            width,
            height,
            d_changed);

        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        if (!h_changed)
            break;
    }
    cudaFree(d_changed);

    // Download the result back to CPU
    d_markers.download(markers);
}
