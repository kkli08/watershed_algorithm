# watershed_algorithm

### CPU
CPU-version `cv::watershed` source code: [here](https://github.com/opencv/opencv/blob/ff639d11d4a1533078e84c3514c9bb0cfe98defe/modules/imgproc/src/segmentation.cpp#L88-L326)

`cards_1.jpg`
```shell
Time taken to load image: 0.00356564 seconds.
Time taken to change background: 0.000589504 seconds.
Time taken to sharpen image: 0.00625338 seconds.
Time taken to create binary image: 0.0064271 seconds.
Time taken for distance transform: 0.000841224 seconds.
Time taken to obtain peaks: 0.000150624 seconds.
Time taken to create markers: 0.000510975 seconds.
Time taken for watershed: 0.00597983 seconds.
Time taken to generate result image: 0.00123451 seconds.
Total time taken: 0.0257004 seconds.
```
### GPU
`cards_1.jpg`
```shell
Time taken to load image: 0.0100761 seconds.
Time taken to upload image to GPU: 0.252296 seconds.
Time taken to change background: 0.0476375 seconds.
Time taken to sharpen image: 0.0901306 seconds.
Time taken to download image from GPU: 0.000141773 seconds.
Time taken to create binary image: 0.0198654 seconds.
Time taken for distance transform: 0.0109282 seconds.
Time taken to obtain peaks: 0.0143061 seconds.
Time taken to create markers: 0.000206545 seconds.
Time taken for CUDA watershed: 0.116677 seconds.
Time taken to generate result image: 0.00251092 seconds.
Total time taken: 0.564951 seconds.
```

### Usage
`like23@ug58.eecg.toronto.edu`
