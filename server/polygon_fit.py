#!/usr/bin/python3

# Create a few routine to do better polygon fits

# minAreaRect() is great, but only is sufficient if the shape is actually a rectangle.
# approxPolyDP() is OK, but seems to always get 1 corner wrong.
# When you are trying to get exact corners to use in pose calculations (solvePnP),
#  getting the corners as close a possible is important.

import copy
import math
import cv2
import numpy

from codetimer import CodeTimer


def convex_polygon_fit(contour, nsides):
    '''Fit a polygon and then refine the sides.
    Refining only works if this a convex shape'''

    with CodeTimer("approxPolyDP_adaptive"):
        start_contour = approxPolyDP_adaptive(contour, nsides)
    if start_contour is None:
        # failed. Not much to do
        return None

    return refine_convex_fit(contour, start_contour)


def rectangle_fit(contour):
    '''Fit a quadrilateral that is close to true rectangle'''

    start_contour = cv2.minAreaRect(contour)
    return refine_convex_fit(contour, start_contour)


def approxPolyDP_adaptive(contour, nsides, max_dp_error=0.1):
    '''Fit a polygon, using with approxPolyDP() as the starting point'''

    step = 0.01
    peri = cv2.arcLength(contour, True)
    dp_err = step
    while dp_err < max_dp_error:
        res = cv2.approxPolyDP(contour, dp_err * peri, True)
        if len(res) <= nsides:
            return res
        dp_err += step
    return None


def refine_convex_fit(contour, contour_fit):
    '''Refine a fit to a convex contour. Look for an approximation of the
    minimum bounding polygon.

    Each side is refined, without changing the other sides.
    The number of sides is not changed.'''

    angle_step = math.pi / 180.0

    nsides = len(contour_fit)
    for iside1 in range(nsides):
        iside2 = (iside1 + 1) % nsides
        iside3 = (iside2 + 1) % nsides
        iside0 = (iside1 - 1) % nsides

        hesse_vec = _hesse_form(contour_fit[iside1][0], contour_fit[iside2][0])
        midpt = (contour_fit[iside1][0] + contour_fit[iside2][0]) / 2.0

        best_area = None
        best_pts = None
        start_angle = math.atan2(hesse_vec[1], hesse_vec[0])

        test_contour = copy.copy(contour_fit)

        for iangle in range(-5, 6):   # step angle by -5 to 5 inclusive
            angle = start_angle + iangle * angle_step
            perp_unit_vec = numpy.array([math.cos(angle), math.sin(angle)])
            midpt_dist = numpy.dot(midpt, perp_unit_vec)

            min_dist = 1.0e8
            max_dist = -1.0e8
            for pt in contour:
                dist_to_line = numpy.dot(pt[0], perp_unit_vec) - midpt_dist
                min_dist = min(min_dist, dist_to_line)
                max_dist = max(max_dist, dist_to_line)

            offset = min_dist if abs(min_dist) < abs(max_dist) else max_dist
            if abs(offset) > 20:
                # print("offset too big", offset)
                continue

            # print('dist:', counts, min_dist, max_dist, offset)
            new_hesse = (numpy.dot(midpt, perp_unit_vec) + offset) * perp_unit_vec

            # compute the intersections with the curve
            intersection1 = _intersection(new_hesse, contour_fit[iside0][0], contour_fit[iside1][0])
            intersection2 = _intersection(new_hesse, contour_fit[iside2][0], contour_fit[iside3][0])

            test_contour[iside1][0] = intersection1
            test_contour[iside2][0] = intersection2
            with CodeTimer("contourArea"):
                area = cv2.contourArea(test_contour)

            if best_area is None or area < best_area:
                best_area = area
                best_pts = (intersection1, intersection2)
                # print('new best', iside1, iangle, offset, area)

        if best_pts is not None:
            contour_fit[iside1][0] = best_pts[0]
            contour_fit[iside2][0] = best_pts[1]

    return contour_fit


def _hesse_form(pt1, pt2):
    '''Compute the Hesse form vector for the line through the points'''

    delta = pt2 - pt1
    parallel = delta / numpy.linalg.norm(delta)
    return pt2 - numpy.dot(pt2, parallel) * parallel


def _intersection(hesse_vec, pt1, pt2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """

    with CodeTimer("_intersection"):
        # Compute the Hesse form for the lines
        rho1 = numpy.linalg.norm(hesse_vec)
        norm1 = hesse_vec / rho1

        hesse2 = _hesse_form(pt1, pt2)
        rho2 = numpy.linalg.norm(hesse2)
        norm2 = hesse2 / rho2

        A = numpy.array([[norm1[0], norm1[1]],
                         [norm2[0], norm2[1]]])
        b = numpy.array([[rho1], [rho2]])
        res = numpy.transpose(numpy.linalg.solve(A, b))

        res = numpy.around(res).astype(int)
    return res
