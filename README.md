# Harris Corner Detection

## Installation

In order to use the program, you need to have numpy, matplotlib and scipy.

```
pip install numpy; pip install matplotlib; pip install scipy;
```

## Usage

```
py harrisCornerDetection.py
```

* -i INPUTFILE, --inputfile INPUTFILE:    filename of input image
* -s SIGMA, --sigma SIGMA:                sigma value for Gaussian filter
  * A higher sigma *blurs* the image more, making less corners.
* -t THRESHOLD, --threshold THRESHOLD:    threshold value for corner detection
  * A higher threshold rejects more potential corners, making less corners.
* -o OUTPUTFILE, --outputfile OUTPUTFILE: filename of output results

## Implementation

* Use the formula for the Y-channel of the YIQ model in performing the color-to-grayscale image conversion.
* Compute Ix and Iy by finite differences.
* Construct images of Ix2, Iy2 and IxIy.
* Compute a proper filter size for a Gaussian filter based on its sigma value.
* Construct a 1D Gaussian filter.
* Smooth a 2D image by convolving it with two 1D Gaussian filters.
* Handle the image border using partial filters in smoothing.
* Construct an image of the cornerness function R.
* Identify potential corners at local maxima in the image of the cornerness function R.
* Compute the cornerness value and coordinates of the potential corners up to sub-pixel accuract by quadratic approximation.
* Use the threshold value to identify strong corners for output.
