# watershed_algorithm

### CPU
CPU-version `cv::watershed` source code: [here](https://github.com/opencv/opencv/blob/ff639d11d4a1533078e84c3514c9bb0cfe98defe/modules/imgproc/src/segmentation.cpp#L88-L326)

`coins_2.jpg`
```shell
Time taken to load image: 0.0158874 seconds.
Time taken to change background: 0.000278746 seconds.
Time taken to sharpen image: 0.00373372 seconds.
Time taken to create binary image: 0.0271546 seconds.
Time taken for distance transform: 0.000934162 seconds.
Time taken to obtain peaks: 0.000151577 seconds.
Time taken to create markers: 0.000424884 seconds.
Time taken for watershed: 0.00741594 seconds.
Time taken to generate result image: 0.00134051 seconds.
Total time taken: 0.0574317 seconds.
```
### GPU
`coins_2.jpg`
```shell
Time taken to load image: 0.00780219 seconds.
Time taken to upload image to GPU: 0.231262 seconds.
Time taken to change background: 0.00256987 seconds.
Time taken to sharpen image: 0.00492542 seconds.
Time taken to download image from GPU: 0.000336792 seconds.
Time taken to create binary image: 0.0263622 seconds.
Time taken for distance transform: 0.000907847 seconds.
Time taken to obtain peaks: 0.000184721 seconds.
Time taken to create markers: 0.000380115 seconds.
Time taken for CUDA watershed: 0.0177236 seconds.
Time taken to generate result image: 0.00133401 seconds.
Total time taken: 0.2939 seconds.
```

### Usage
`like23@ug58.eecg.toronto.edu`
