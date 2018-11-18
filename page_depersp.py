#!/usr/bin/env python

import os
import sys
import argparse
import cv2
import numpy as np

def dist(x,y):
    return numpy.sqrt(numpy.sum((x-y)**2))

def rect_of_points_gen(points, k=1.0):
    x = points[:,0]
    y = points[:,1]
    xm = np.mean(x)
    ym = np.mean(y)
    m = (xm, ym)
    xd = np.std(x) * k
    yd = np.std(y) * k
    d = (xd, yd)
    return np.array([[xm-xd, ym-yd], [xm+xd, ym-yd], [xm+xd, ym+yd], [xm-xd, ym+yd]], dtype=float)


def mouse_handler(event, x, y, flags, data) :
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):
    
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    
    return points

def main(imgfile, prev):

    outfiles = []

    im_src = cv2.imread(imgfile)
    basename = os.path.basename(imgfile)
    name, _ = os.path.splitext(basename)

    im_size = im_src.shape
    im_height = im_size[0]
    im_width = im_size[1]
    im_col = im_size[2]
    im_size = (im_width, im_height, im_col)

    print 'loaded', basename, 'with size', im_src.shape

    k = im_height / float(prev)
    kw = im_width / float(prev)
    if (kw > k):
        k = kw
    print str(k)

    prev_height = int(im_height / k)
    prev_width = int(im_width / k)
    prev_col = im_col
    prev_size =(prev_width, prev_height, prev_col)
    print 'resize to', prev_size
    im_prev = cv2.resize(im_src, (prev_width, prev_height))

    im_dst = np.zeros(im_size, np.uint8)
    
    print '''
        Click on the four corners of the book -- top left first and
        bottom left last -- and then hit ENTER
        '''
    
    # Show image and wait for 4 clicks.
    cv2.imshow("Image", im_prev)
    pts_src = get_four_points(im_prev)
    pts_src = pts_src * k
    print "area:", str(pts_src)
    pts_dst = rect_of_points_gen(pts_src)
    print "rect:", str(pts_dst)
    

    # Calculate the homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination
    im_dst = cv2.warpPerspective(im_src, h, im_size[0:2], borderMode = 1)
    im_prev = cv2.resize(im_dst, (prev_width, prev_height))

    depersp_file = name + '_depersp.png'
    cv2.imwrite(depersp_file, im_dst)
    # Show output
    cv2.imshow("Image", im_prev)
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Page DeWarp")
    parser.add_argument('-p', '--prev', metavar='prev', type=int, default=768, help='Preview size')
    parser.add_argument("imgfile", help="Name dewarp image")
    args = parser.parse_args()
    main(args.imgfile, args.prev)


