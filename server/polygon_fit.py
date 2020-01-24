#!/usr/bin/python3

# Create a few routine to do better polygon fits

# minAreaRect() is great, but only is sufficient if the shape is actually a rectangle.
# approxPolyDP() is OK, but seems to always get 1 corner wrong.
# When you are trying to get exact corners to use in pose calculations (solvePnP),
#  getting the corners as close a possible is important.

import copy
import cv2
from math import atan2, sin, cos, tan
from numpy import array, amin, amax, sqrt, pi, round, dot
from numpy.linalg import solve

from codetimer import CodeTimer


def convex_polygon_fit(contour, nsides):
    '''Fit a polygon and then refine the sides.
    Refining only works if this a convex shape'''

    with CodeTimer("approxPolyDP_adaptive"):
        start_contour = approxPolyDP_adaptive(contour, nsides)
    if start_contour is None:
        # failed. Not much to do
        print('adapt failed')
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

    angle_step = pi / 180.0

    nsides = len(contour_fit)
    for iside1 in range(nsides):
        iside2 = (iside1 + 1) % nsides
        iside3 = (iside2 + 1) % nsides
        iside0 = (iside1 - 1) % nsides

        hesse_vec = _hesse_form(contour_fit[iside1][0], contour_fit[iside2][0])
        midpt = (contour_fit[iside1][0] + contour_fit[iside2][0]) / 2.0

        delta = contour_fit[iside1][0] - contour_fit[iside2][0]
        line_length = sqrt(delta.dot(delta))
        
        start_area = cv2.contourArea(contour_fit)

        best_area = None
        best_pts = None
        best_offset = None
        start_angle = atan2(hesse_vec[1], hesse_vec[0])

        test_contour = copy.copy(contour_fit)

        for iangle in range(-5, 6):   # step angle by -5 to 5 inclusive
            # for iangle in range(-3, 3, 3):   # step angle by -5 to 5 inclusive
            angle = start_angle + iangle * angle_step
            perp_unit_vec = array([cos(angle), sin(angle)])
            midpt_dist = dot(midpt, perp_unit_vec)

            perp_dists = contour.dot(perp_unit_vec)
            min_dist = amin(perp_dists) - midpt_dist
            max_dist = amax(perp_dists) - midpt_dist

            offset = min_dist if abs(min_dist) < abs(max_dist) else max_dist
            if abs(offset) > 20:
                print("offset too big", offset)
                continue

            new_hesse = (midpt.dot(perp_unit_vec) + offset) * perp_unit_vec

            # compute the intersections with the curve
            intersection1 = _intersection(new_hesse, contour_fit[iside0][0], contour_fit[iside1][0])
            intersection2 = _intersection(new_hesse, contour_fit[iside2][0], contour_fit[iside3][0])

            test_contour[iside1][0] = intersection1
            test_contour[iside2][0] = intersection2
            with CodeTimer("contourArea"):
                area = cv2.contourArea(test_contour) - start_area
            # print("area", iside1, iangle, offset, area, _delta_area(line_length, offset, iangle * angle_step))

            # intersection1 = contour_fit[iside1][0]
            # intersection2 = contour_fit[iside2][0]
            # area = iangle**2

            if best_area is None or area < best_area[0]:
                best_area = (area, iangle)
                best_pts = (intersection1, intersection2)
                # print('new best', iside1, iangle, offset, area)

            if best_offset is None or abs(offset) < best_offset[0]:
                best_offset = (abs(offset), iangle)
                best_pts = (intersection1, intersection2)
                # print('new best', iside1, iangle, offset, area)

        if best_offset[1] != best_area[1]:
            print("Best offset and best area disagree")

        if best_pts is not None:
            contour_fit[iside1][0] = best_pts[0]
            contour_fit[iside2][0] = best_pts[1]

    return contour_fit


def _hesse_form(pt1, pt2):
    '''Compute the Hesse form vector for the line through the points'''

    delta = pt2 - pt1
    mag2 = delta.dot(delta)
    return pt2 - pt2.dot(delta) * delta / mag2


def _intersection(hesse_vec, pt1, pt2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """

    with CodeTimer("_intersection"):
        # Compute the Hesse form for the lines
        rho1 = sqrt(hesse_vec.dot(hesse_vec))
        norm1 = hesse_vec / rho1

        hesse2 = _hesse_form(pt1, pt2)
        rho2 = sqrt(hesse2.dot(hesse2))
        norm2 = hesse2 / rho2

        A = array([[norm1[0], norm1[1]],
                   [norm2[0], norm2[1]]])
        b = array([[rho1], [rho2]])
        res = solve(A, b).transpose()

        res = round(res).astype(int)
    return res


def _delta_area(line_length, offset, angle):
    '''Compute an approx. change in area of the shape'''

    l_half = line_length / 2.0

    cos_a = cos(angle)
    if abs(angle) < 1e-6:
        # parallel to the original line
        return line_length * offset / cos_a

    sin_a = abs(angle)          # small angle approx. - faster
    y0 = offset / sin_a
    if y0 > l_half:
        # does not intersect the original line segment
        return line_length * offset / cos_a

    return 0.5 * sin_a / cos_a * ((y0 + l_half)**2 - (l_half - y0)**2)
