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
Time taken to load image: 0.00217921 seconds.
Time taken to change background: 0.000238478 seconds.
Time taken to sharpen image: 0.00352492 seconds.
Time taken to create binary image: 0.0345925 seconds.
Time taken for distance transform: 0.000812833 seconds.
Time taken to obtain peaks: 0.000149282 seconds.
Time taken to create markers: 0.000611991 seconds.
Time taken for watershed: 0.00601303 seconds.
Time taken to generate result image: 0.00123459 seconds.
Total time taken: 0.0494591 seconds.
```

### Usage
`like23@ug58.eecg.toronto.edu`
