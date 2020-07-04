# Hough-Circle-Detection
Implementation of Simple Hough Circle Detection Algorithm in Python.\
This is based on paper [Use of the Hough Transformation To Detect Lines and Curves in Pictures](/Paper/HoughTransformPaper.pdf) by Richard O. Duda and Peter E. Hart.\
This is an extension of the Hough Transform to detect circles using the equation,\
&nbsp; &nbsp; &nbsp; &nbsp; r^2 = ( x - a )^2 + ( y - b )^2 &nbsp; &nbsp; &nbsp; &nbsp; in parameter space rho = ( a, b, r) 

Please refer to [Hough Line Detection] (https://github.com/adityaintwala/Hough-Line-Detection#hough-line-detection) python implementation at the following git repository for more information.

## Usage
''' python find_hough_circles.py ./images/ex1.png --r_min 10 --r_max 200 --delta_r 1 --num_thetas 100 --bin_threshold 0.4 --min_edge_threshold 100 --max_edge_threshold 200 '''

### Input
The script requires one positional argument and few optional parameters:
* image_path - Complete path to the image file for circle detection.
* r_min - Min radius circle to detect. Default is 10.
* r_max - Max radius circle to detect. Default is 200.
* delta_r - Delta change in radius from r_min to r_max. Default is 1.
* num_thetas - Number of steps for theta from 0 to 2PI. Default is 100.
* bin_threshold - Thresholding value in percentage to shortlist candidate for circle. Default is 0.4 i.e. 40%.
* min_edge_threshold - Minimum threshold value for edge detection. Default 100.
* max_edge_threshold - Maximum threshold value for edge detection. Default 200.

### Output
The output of the script would be two files:
* circles.txt - List of circles in format (x,y,r,votes)
* circle_img.png - Image with the Circles drawn in Green color.

## Samples
Sample Input Image  |  Sample Output Image
:------------------:|:--------------------:
![Sample Input Image](/images/ex1.png)  |  ![Sample Output Image](/images/output_ex1.png)
![Sample Input Image](/images/ex2.png)  |  ![Sample Output Image](/images/output_ex2.png)
![Sample Input Image](/images/ex3.png)  |  ![Sample Output Image](/images/output_ex3.png)
![Sample Input Image](/images/standard.png)  |  ![Sample Output Image](/images/output_standard.png)


