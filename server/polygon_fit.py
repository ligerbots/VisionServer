#!/usr/bin/python3

# Create a few routine to do better polygon fits

# minAreaRect() is great, but only is sufficient if the shape is actually a rectangle.
# approxPolyDP() is OK, but seems to always get 1 corner wrong.
# When you are trying to get exact corners to use in pose calculations (solvePnP),
#  getting the corners as close a possible is important.

import cv2

approx_poly_max_error = 0.05


def approxPolyDP_adaptive(contour, nsides, max_dp_error=0.1):
    '''Fit a polygon, using with approxPolyDP() as the starting point'''

    step = 0.01
    peri = cv2.arcLength(contour, True)
    dp_err = step
    while dp_err < max_dp_error:
        res = cv2.approxPolyDP(contour, dp_err * peri, True)
        if len(res) <= nsides:
            return res
    return None


def convex_polygon_fit(contour, nsides):
    '''Fit a polygon and then refine the sides.
    Refining only works if this a convex shape'''

    start_contour = approxPolyDP_adaptive(contour, nsides)
    if start_contour is None:
        # failed. Not much to do
        return None

    return refine_convex_fit(contour, start_contour)


def rectangle_fit(contour):
    '''Fit a quadrilater that is close to true rectangle'''

    start_contour = cv2.minAreaRect(contour)
    return refine_convex_fit(contour, start_contour)


def refine_convex_fit(contour, start_contour):
    '''Refine a fit to a convex contour. Look for an approximation of the
    minimum bounding polygon.

    Each side is refined, without changing the other sides.
    The number of sides is not changed.'''

    # TODO
    return start_contour
