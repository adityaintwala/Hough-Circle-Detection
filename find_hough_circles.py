# -*- coding: utf-8 -*-
"""
Aditya Intwala

This is a script to demonstrate the implementation of Hough transform function to
detect circles from the image.

Input:
    img - Full path of the input image.
    r_min - Min radius circle to detect. Default is 10.
    r_max - Max radius circle to detect. Default is 200.
    delta_r - Delta change in radius from r_min to r_max. Default is 1.
    num_thetas - Number of steps for theta from 0 to 2PI. Default is 100.
    bin_threshold - Thresholding value in percentage to shortlist candidate for circle. Default is 0.4 i.e. 40%.
    min_edge_threshold - Minimum threshold value for edge detection. Default 100.
    max_edge_threshold - Maximum threshold value for edge detection. Default 200.

Note:
    Playing with the input parameters is required to obtain desired circles in different images.
    
Returns:
    circle_img - Image with the Circles drawn
    circles - List of circles in format (rho,theta,x1,y1,x2,y2)

"""

import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

def find_hough_circles(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold):
  #image size
  img_height, img_width = edge_image.shape[:2]
  
  # R and Theta ranges
  dtheta = int(360 / num_thetas)
  
  ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
  thetas = np.arange(0, 360, step=dtheta)
  
  ## Radius ranges from r_min to r_max 
  rs = np.arange(r_min, r_max, step=delta_r)
  
  # Calculate Cos(theta) and Sin(theta) it will be required later
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  
  # Evaluate and keep ready the candidate circles dx and dy for different delta radius
  # based on the the parametric equation of circle.
  # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
  # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)
  circle_candidates = []
  for r in rs:
    for t in range(num_thetas):
      #instead of using pre-calculated cos and sin theta values you can calculate here itself by following
      #circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
      #but its better to pre-calculate and use it here.
      circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
  
  # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
  # aready present in the dictionary instead of throwing exception.
  accumulator = defaultdict(int)
  
  for y in range(img_height):
    for x in range(img_width):
      if edge_image[y][x] != 0: #white pixel
        # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
        for r, rcos_t, rsin_t in circle_candidates:
          x_center = x - rcos_t
          y_center = y - rsin_t
          accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
  
  # Output image with detected lines drawn
  output_img = image.copy()
  # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
  out_circles = []
  
  for k, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
    x, y, r = k
    current_vote_percentage = votes / num_thetas
    if current_vote_percentage >= bin_threshold: 
      # Shortlist the circle for final result
      out_circles.append((x, y, r, current_vote_percentage))
      print(x, y, r, current_vote_percentage)
    
  # Draw shortlisted circles on the output image
  for x, y, r, v in out_circles:
    output_img = cv2.circle(output_img, (x,y), r, (0,255,0), 2)
  
  return output_img, out_circles

def main():
    
    parser = argparse.ArgumentParser(description='Find Hough circles from the image.')
    parser.add_argument('image_path', type=str, help='Full path of the input image.')
    parser.add_argument('--r_min', type=float, help='Min radius circle to detect. Default is 5.')
    parser.add_argument('--r_max', type=float, help='Max radius circle to detect.')
    parser.add_argument('--delta_r', type=float, help='Delta change in radius from r_min to r_max. Default is 1.')
    parser.add_argument('--num_thetas', type=float, help='Number of steps for theta from 0 to 2PI. Default is 100.')
    parser.add_argument('--bin_threshold', type=int, help='Thresholding value to shortlist candidate for circle. Default is 0.4 i.e. 40%.')
    parser.add_argument('--min_edge_threshold', type=int, help='Minimum threshold value for edge detection. Default 100.')
    parser.add_argument('--max_edge_threshold', type=int, help='Maximum threshold value for edge detection. Default 200.')
    
    args = parser.parse_args()
    
    img_path = args.image_path
    r_min = 10
    r_max = 200
    delta_r = 1
    num_thetas = 100
    bin_threshold = 0.4
    min_edge_threshold = 100
    max_edge_threshold = 200
    
    if args.r_min:
        r_min = args.r_min
    
    if args.r_max:
        r_max = args.r_max
        
    if args.delta_r:
        delta_r = args.delta_r
    
    if args.num_thetas:
        num_thetas = args.num_thetas
    
    if args.bin_threshold:
        bin_threshold = args.bin_threshold
        
    if args.min_edge_threshold:
        min_edge_threshold = args.min_edge_threshold
        
    if args.max_edge_threshold:
        max_edge_threshold = args.max_edge_threshold
    
    input_img = cv2.imread(img_path)
    
    #Edge detection on the input image
    edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    #ret, edge_image = cv2.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
    edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)
    
    cv2.imshow('Edge Image', edge_image)
    cv2.waitKey(0)
    
    if edge_image is not None:
        
        print ("Detecting Hough Circles Started!")
        circle_img, circles = find_hough_circles(input_img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold)
        
        cv2.imshow('Detected Circles', circle_img)
        cv2.waitKey(0)
        
        circle_file = open('circles_list.txt', 'w')
        circle_file.write('x ,\t y,\t Radius,\t Threshold \n')
        for i in range(len(circles)):
            circle_file.write(str(circles[i][0]) + ' , ' + str(circles[i][1]) + ' , ' + str(circles[i][2]) + ' , ' + str(circles[i][3]) + '\n')
        circle_file.close()
        
        if circle_img is not None:
            cv2.imwrite("circles_img.png", circle_img)
    else:
        print ("Error in input image!")
            
    print ("Detecting Hough Circles Complete!")



if __name__ == "__main__":
    main()
