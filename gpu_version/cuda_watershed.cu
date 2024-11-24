// cuda_watershed.cu

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

using namespace cv;

// Constants for labels
#define WATERSHED -1
#define UNVISITED 0

// CUDA kernel to perform flooding
__global__ void floodingKernel(
    const float* grad,    // Gradient magnitude image
    const int* labels_in, // Input labels from previous iteration
    int* labels_out,      // Output labels for current iteration
    int width,
    int height,
    bool* changed)        // Flag to indicate if any label has changed
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    int current_label = labels_in[idx];
    float current_grad = grad[idx];

    // If the pixel is already labeled (marker or watershed), copy the label
    if (current_label > 0 || current_label == WATERSHED)
    {
        labels_out[idx] = current_label;
        return;
    }

    // Offsets for 8-connected neighbors
    int dx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
    int dy[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };

    int neighbor_labels[8];
    float neighbor_grads[8];
    int num_neighbors = 0;

    bool has_labeled_neighbor = false;
    int neighbor_label = 0;
    bool multiple_labels = false;

    // Examine all neighbors
    for (int i = 0; i < 8; ++i)
    {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            continue;

        int nidx = ny * width + nx;
        int n_label = labels_in[nidx];
        float n_grad = grad[nidx];

        if (n_label > 0 || n_label == WATERSHED)
        {
            if (!has_labeled_neighbor)
            {
                neighbor_label = n_label;
                has_labeled_neighbor = true;
            }
            else
            {
                if (n_label != neighbor_label)
                {
                    multiple_labels = true;
                }
            }
        }
    }

    if (has_labeled_neighbor)
    {
        if (multiple_labels)
        {
            labels_out[idx] = WATERSHED;
        }
        else
        {
            labels_out[idx] = neighbor_label;
        }

        if (labels_out[idx] != labels_in[idx])
            *changed = true;
    }
    else
    {
        labels_out[idx] = labels_in[idx];
    }
}

// Function to perform the watershed algorithm using CUDA
void cudaWatershed(const Mat& image, Mat& markers)
{
    // Convert image to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Compute gradient magnitude using Sobel operator in CUDA
    cv::cuda::GpuMat d_gray;
    d_gray.upload(gray);

    cv::cuda::GpuMat d_grad_x, d_grad_y;
    cv::Ptr<cv::cuda::Filter> sobel_x = cv::cuda::createSobelFilter(CV_8U, CV_32F, 1, 0, 3);
    cv::Ptr<cv::cuda::Filter> sobel_y = cv::cuda::createSobelFilter(CV_8U, CV_32F, 0, 1, 3);

    sobel_x->apply(d_gray, d_grad_x);
    sobel_y->apply(d_gray, d_grad_y);

    // Compute gradient magnitude
    cv::cuda::GpuMat d_grad;
    cv::cuda::magnitude(d_grad_x, d_grad_y, d_grad);

    // Normalize gradient to range [0, 1]
    cv::cuda::GpuMat d_grad_norm;
    cv::cuda::normalize(d_grad, d_grad_norm, 0.0, 1.0, NORM_MINMAX, -1);

    int width = image.cols;
    int height = image.rows;
    size_t size = width * height;

    // Prepare labels
    cv::cuda::GpuMat d_labels_in, d_labels_out;
    d_labels_in.upload(markers);
    d_labels_out.create(markers.size(), markers.type());

    // Gradient data
    cv::cuda::GpuMat d_grad_data;
    d_grad_norm.convertTo(d_grad_data, CV_32F);

    // Allocate device memory for 'changed' flag
    bool h_changed;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    int maxIterations = 10000;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // Call the CUDA kernel
        floodingKernel<<<gridSize, blockSize>>>(
            d_grad_data.ptr<float>(),
            d_labels_in.ptr<int>(),
            d_labels_out.ptr<int>(),
            width,
            height,
            d_changed);

        cudaDeviceSynchronize();

        // Swap labels_in and labels_out
        d_labels_in.swap(d_labels_out);

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

        if (!h_changed)
            break;
    }

    cudaFree(d_changed);

    // Download the result back to CPU
    d_labels_in.download(markers);
}
