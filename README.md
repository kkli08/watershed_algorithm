# watershed_algorithm

---

## **Performance Summary**

| **Image**       | **CPU Total Time (s)** | **GPU Total Time (s)** | **Speed Up (CPU/GPU)** |
|-----------------|------------------------|------------------------|------------------------|
| **coins_1.jpg** | 0.01198                | 0.01454                | **0.82x**              |
| **coins_2.jpg** | 0.06708                | 0.02599                | **2.58x**              |
| **coins_3.jpg** | 0.04663                | 0.01923                | **2.42x**              |
| **earth_map.jpg** | 1.73138               | 1.34069                | **1.29x**              |

---

### CPU
CPU-version `cv::watershed` source code: [here](https://github.com/opencv/opencv/blob/ff639d11d4a1533078e84c3514c9bb0cfe98defe/modules/imgproc/src/segmentation.cpp#L88-L326)

### `coins_1.jpg`
#### CPU
```shell
Time taken to load image: 0.00237813 seconds.
Time taken to change background: 8.0726e-05 seconds.
Time taken to sharpen image: 0.00199139 seconds.
Time taken to create binary image: 0.000287062 seconds.
Time taken for distance transform: 0.000867324 seconds.
Time taken to obtain peaks: 0.000144872 seconds.
Time taken to create markers: 0.000510288 seconds.
Time taken for watershed: 0.00477043 seconds.
Time taken to generate result image: 0.000839796 seconds.
Total time taken: 0.0119787 seconds.
```
#### GPU
```shell
Time taken to load image: 0.00137635 seconds.
Time taken to upload image to GPU: 0.235225 seconds.
Time taken to change background: 0.00306485 seconds.
Time taken to sharpen image: 0.00465648 seconds.
Time taken to create binary image: 0.000337388 seconds.
Time taken for distance transform (CPU): 0.000364986 seconds.
Time taken to obtain peaks (CPU): 0.000191674 seconds.
Time taken to create markers: 0.0001341 seconds.
Time taken for watershed: 0.00219978 seconds.
Time taken to generate result image: 0.000171412 seconds.
Final segmented image saved as: final_result_coins_1.jpeg
Total time taken: 0.0145434 seconds.
```

### `coins_2.jpg`
#### CPU
```shell
Time taken to load image: 0.0217088 seconds.
Time taken to change background: 0.00067064 seconds.
Time taken to sharpen image: 0.005839 seconds.
Time taken to create binary image: 0.028573 seconds.
Time taken for distance transform: 0.000888695 seconds.
Time taken to obtain peaks: 0.000167132 seconds.
Time taken to create markers: 0.000428886 seconds.
Time taken for watershed: 0.00730217 seconds.
Time taken to generate result image: 0.00136695 seconds.
Total time taken: 0.0670797 seconds.
```
#### GPU
```shell
Time taken to load image: 0.0035518 seconds.
Time taken to upload image to GPU: 0.233816 seconds.
Time taken to change background: 0.00254603 seconds.
Time taken to sharpen image: 0.00483674 seconds.
Time taken to create binary image: 0.000403177 seconds.
Time taken for distance transform (CPU): 0.000972277 seconds.
Time taken to obtain peaks (CPU): 0.00341521 seconds.
Time taken to create markers: 0.000329267 seconds.
Time taken for watershed: 0.00696109 seconds.
Time taken to generate result image: 0.000490142 seconds.
Final segmented image saved as: final_result_coins_2.jpeg
Total time taken: 0.0259871 seconds.
```

### `coins_3.jpg`
#### CPU
```shell
Time taken to load image: 0.00834892 seconds.
Time taken to change background: 0.000244169 seconds.
Time taken to sharpen image: 0.00340136 seconds.
Time taken to create binary image: 0.0302169 seconds.
Time taken for distance transform: 0.000348438 seconds.
Time taken to obtain peaks: 7.2124e-05 seconds.
Time taken to create markers: 0.000382008 seconds.
Time taken for watershed: 0.00309053 seconds.
Time taken to generate result image: 0.000406288 seconds.
Total time taken: 0.0466261 seconds.
```
#### GPU
```shell
Time taken to load image: 0.00274296 seconds.
Time taken to upload image to GPU: 0.235419 seconds.
Time taken to change background: 0.00260739 seconds.
Time taken to sharpen image: 0.00467214 seconds.
Time taken to create binary image: 0.000345426 seconds.
Time taken for distance transform (CPU): 0.000397065 seconds.
Time taken to obtain peaks (CPU): 0.00302618 seconds.
Time taken to create markers: 0.00023493 seconds.
Time taken for watershed: 0.00317864 seconds.
Time taken to generate result image: 0.000152382 seconds.
Final segmented image saved as: final_result_coins_3.jpeg
Total time taken: 0.0192252 seconds.
```

### `earth_map.jpg`
#### CPU
```shell
Time taken to load image: 0.108538 seconds.
Time taken to change background: 0.0137902 seconds.
Time taken to sharpen image: 0.235727 seconds.
Time taken to create binary image: 0.045146 seconds.
Time taken for distance transform: 0.0699876 seconds.
Time taken to obtain peaks: 0.011464 seconds.
Time taken to create markers: 0.0227247 seconds.
Time taken for watershed: 1.1113 seconds.
Time taken to generate result image: 0.107246 seconds.
Total time taken: 1.73138 seconds.
```
#### GPU
```shell
Time taken to load image: 0.103691 seconds.
Time taken to upload image to GPU: 0.239608 seconds.
Time taken to change background: 0.00319743 seconds.
Time taken to sharpen image: 0.00743555 seconds.
Time taken to create binary image: 0.00227563 seconds.
Time taken for distance transform (CPU): 0.0735741 seconds.
Time taken to obtain peaks (CPU): 0.0395348 seconds.
Time taken to create markers: 0.013975 seconds.
Time taken for watershed: 1.02962 seconds.
Time taken to generate result image: 0.0376807 seconds.
Final segmented image saved as: final_result_earth_map.jpg
Total time taken: 1.34069 seconds.
```

### Usage
`like23@ug58.eecg.toronto.edu`
