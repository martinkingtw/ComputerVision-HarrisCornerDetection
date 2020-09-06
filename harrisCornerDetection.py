# Harris Corner Detection

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

rgbToYiqY = np.array([0.299, 0.587, 0.114])

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image
    img_gray = img_color @ rgbToYiqY
    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    size = 2147483647
    for i in range(2147483647):
        if np.exp((i ** 2) / -2 / (sigma ** 2)) < 1 / 1000:
            size = i-1
            break
    x = np.arange(-size, size+1)
    filterSmooth = np.exp((x ** 2) / -2 / (sigma ** 2))
    img_filtered = convolve1d(img, filterSmooth, mode='constant')
    img_weighted = convolve1d(np.ones(img.shape), filterSmooth, mode='constant')
    img_smoothed = img_filtered / img_weighted
    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result
    img = smooth1D(img, sigma)
    img = smooth1D(img.T, sigma)
    img_smoothed = img.T
    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    ix, iy = np.gradient(img)
    ix2 = ix * ix
    iy2 = iy * iy
    ixIy = ix * iy
    ix2 = smooth2D(ix2, sigma)
    iy2 = smooth2D(iy2, sigma)
    ixIy = smooth2D(ixIy, sigma)
    detA = ix2 * iy2 - ixIy * ixIy
    traceA = ix2 + iy2
    r = detA - 0.04 * (traceA ** 2)
    cornerCandidates = []
    corners = []
    for i in range(1, r.shape[0] - 1):
        for j in range(1, r.shape[1] - 1):
            if r[i][j] >= max(r[i-1][j-1], r[i][j-1], r[i+1][j-1], r[i-1][j+1], r[i][j+1], r[i+1][j+1], r[i-1][j], r[i+1][j]):
                cornerCandidates.append((i, j, r[i][j]))
    for i in cornerCandidates:
        a = (r[i[0]][i[1]-1] + r[i[0]][i[1]+1] - 2 * r[i[0]][i[1]]) / 2
        b = (r[i[0]-1][i[1]] + r[i[0]+1][i[1]] - 2 * r[i[0]][i[1]]) / 2
        c = (r[i[0]][i[1]+1] - r[i[0]][i[1]-1]) / 2
        d = (r[i[0]+1][i[1]] - r[i[0]-1][i[1]]) / 2
        e = r[i[0]][i[1]]
        x = -c / 2 / a
        y = -d / 2 / b
        f = a * (x ** 2) + b * (y ** 2) + c * x + d * y + e
        if i[2] >= threshold:
            corners.append((i[1] + x, i[0] + y, f))
    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'Harris Corner Detection')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
