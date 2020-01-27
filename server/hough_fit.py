#!/usr/bin/python3

# Use the Hough line algorithm to fix a contour

# See bottom of this file for some timing results

import cv2

# it is faster if you import the needed functions directly
# these are used a lot, so it helps, a little
from math import sin, cos, atan2  # , degrees
from numpy import array, zeros, pi, sqrt, uint8

from codetimer import CodeTimer
two_pi = 2.0 * pi
pi_by_2 = pi / 2.0

def hough_fit(contour, nsides=None, approx_fit=None):
    '''Use the Hough line finding algorithm to find a polygon for contour.
    It is faster if you can provide an decent initial fit - see approxPolyDP_adaptive().'''

    if approx_fit is not None:
        nsides = len(approx_fit)
    if not nsides:
        raise Exception("You need to set nsides or pass approx_fit")

    x, y, w, h = cv2.boundingRect(contour)
    offset_vec = array((x, y))

    shifted_con = contour - offset_vec

    # the binning does affect the speed, so tune it....
    with CodeTimer("HoughLines"):
        contour_plot = zeros(shape=(h, w), dtype=uint8)
        cv2.drawContours(contour_plot, [shifted_con, ], -1, 255, 1)
        lines = cv2.HoughLines(contour_plot, 1, pi / 180, threshold=10)

    if lines is None or len(lines) < nsides:
        # print("HoughLines found too few lines")
        return None

    if approx_fit is not None:
        res = _match_lines_to_fit(approx_fit - offset_vec, lines, w, h)
    else:
        res = _find_sides(nsides, lines, w, h)

    if res is None:
        return None
    return array(res) + offset_vec


def approxPolyDP_adaptive(contour, nsides, max_dp_error=0.1):
    '''Use approxPolyDP to fit a polygon to a contour.
    Find the smallest dp_error that gets the correct number of sides.
    The results seem to often be a little wrong, but they are a quick starting point.'''

    step = 0.01
    peri = cv2.arcLength(contour, True)
    dp_err = step
    while dp_err <= max_dp_error:
        res = cv2.approxPolyDP(contour, dp_err * peri, True)
        if len(res) <= nsides:
            return res
        dp_err += step
    return None


def plot_hough_line(frame, rho, theta, color, thickness=1):
    '''Given (rho, theta) of a line in Hesse form, plot it on a frame.
    Useful for debugging, mostly.'''

    a = cos(theta)
    b = sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(frame, pt1, pt2, color, thickness)
    return


# --------------------------------------------------------------------------------
# Private routines


def _find_sides(nsides, hough_lines, w, h):
    # The returned lines from HoughLines() are ordered by confidence, but there may/will be
    #  many variants of the best lines. Loop through the lines and pick the best from
    #  each cluster.

    contour_center = (w / 2, h / 2)
    boundaries = (-5, w+5, -5, h+5)

    dist_thres = 10
    theta_thres = pi / 36  # 5 degrees
    best_lines = []
    for linelist in hough_lines:
        line = linelist[0]
        if line[0] < 0:
            line[0] *= -1
            line[1] -= pi

        coord_near_ref = _compute_line_near_reference(line, contour_center)

        if not best_lines or not _is_close(best_lines, line, coord_near_ref, dist_thres, theta_thres):
            # print('best line:', line[0], math.degrees(line[1]))
            best_lines.append((line, coord_near_ref))

        if len(best_lines) == nsides:
            break

    if len(best_lines) != nsides:
        # print("hough_fit: found %s lines" % len(best_lines))
        return None

    # print('best')
    # for l in best_lines:
    #     print('   ', l[0][0], degrees(l[0][1]))

    # Find the nsides vertices which are inside the bounding box (with a little slop).
    # There will be extra intersections. Assume the right ones (and only those) are within the bounding box.
    vertices = []
    iline1 = 0
    used = set()
    used.add(iline1)
    while len(used) < nsides:
        found = False
        for iline2 in range(nsides):
            if iline2 in used:
                continue

            inter = _intersection(best_lines[iline1][0], best_lines[iline2][0])
            if inter is not None and \
               inter[0] >= boundaries[0] and inter[0] <= boundaries[1] and \
               inter[1] >= boundaries[2] and inter[1] <= boundaries[3]:
                vertices.append(inter)
                used.add(iline2)
                iline1 = iline2
                found = True
                break
        if not found:
            # print("No intersection with %s and available lines" % iline1)
            return None

    # add in the last pair
    inter = _intersection(best_lines[0][0], best_lines[iline1][0])
    if inter is not None and \
       inter[0] >= boundaries[0] and inter[0] <= boundaries[1] and \
       inter[1] >= boundaries[2] and inter[1] <= boundaries[3]:
        vertices.append(inter)

    if len(vertices) != nsides:
        # print('Not correct number of vertices:', len(vertices))
        return None

    # remember to unshift the resulting contour
    return vertices


def _delta_angle(a, b):
    d = a - b
    return (d + pi) % two_pi - pi


def _match_lines_to_fit(approx_fit, hough_lines, w, h):
    '''Given the approximate shape and a set of lines from the Hough algorithm
    find the matching lines and rebuild the fit'''

    theta_thres = pi / 36  # 5 degrees
    nsides = len(approx_fit)
    fit_sides = []
    hough_used = set()
    for ivrtx in range(nsides):
        ivrtx2 = (ivrtx + 1) % nsides
        pt1 = approx_fit[ivrtx][0]
        pt2 = approx_fit[ivrtx2][0]

        rho, theta = _hesse_form(pt1, pt2)

        # Hough lines are in order of confidence, so look for the first unused one
        #  which matches the line
        for ih, linelist in enumerate(hough_lines):
            if ih in hough_used:
                continue
            line = linelist[0]

            # There is an ambiguity of -rho and adding 180deg to theta
            # So test them both.

            if (abs(rho - line[0]) < 10 and abs(_delta_angle(theta, line[1])) < theta_thres) or \
               (abs(rho + line[0]) < 10 and abs(_delta_angle(theta, line[1] - pi)) < theta_thres):
                fit_sides.append(line)
                hough_used.add(ih)
                break

    if len(fit_sides) != nsides:
        # print("did not match enough lines")
        return None

    vertices = []
    for ivrtx in range(nsides):
        ivrtx2 = (ivrtx + 1) % nsides
        inter = _intersection(fit_sides[ivrtx], fit_sides[ivrtx2])
        if inter is None:
            # print("No intersection between lines")
            return None
        vertices.append(inter)

    return vertices


def _compute_line_near_reference(line, ref_point):
    with CodeTimer("compute_line_near_reference"):
        rho, theta = line

        # remember: theta is actually perpendicular to the line, so there is a sign flip
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho

        if abs(cos_theta) < 1e-6:
            x_near_ref = None
            y_near_ref = y0
        elif abs(sin_theta) < 1e-6:
            x_near_ref = x0
            y_near_ref = None
        else:
            x_near_ref = x0 + (y0 - ref_point[1]) * sin_theta / cos_theta
            y_near_ref = y0 + (x0 - ref_point[0]) * cos_theta / sin_theta

    return x_near_ref, y_near_ref


def _is_close(best_lines, candidate, coord_near_ref, dist_thres, theta_thres):
    cand_rho, cand_theta = candidate

    # print('cand:', cand_rho, math.degrees(cand_theta))
    for line in best_lines:
        line, best_near_ref = line
        # print('best', line, best_near_ref)

        delta_dists = []
        if coord_near_ref[0] is not None and best_near_ref[0] is not None:
            delta_dists.append(abs(coord_near_ref[0] - best_near_ref[0]))
        if coord_near_ref[1] is not None and best_near_ref[1] is not None:
            delta_dists.append(abs(coord_near_ref[1] - best_near_ref[1]))
        if not delta_dists:
            return True
        delta_dist = min(delta_dists)

        # angle differences greater than 180deg are not real
        delta_theta = cand_theta - line[1]
        while delta_theta >= pi_by_2:
            delta_theta -= pi
        while delta_theta <= -pi_by_2:
            delta_theta += pi
        delta_theta = abs(delta_theta)

        # print('test:', line[0], math.degrees(line[1]), delta_dist, delta_theta)
        if delta_dist <= dist_thres and delta_theta <= theta_thres:
            return True
    return False


def _intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """

    with CodeTimer("intersection"):
        rho1, theta1 = line1
        rho2, theta2 = line2
        if abs(theta1 - theta2) < 1e-6:
            # parallel
            return None

        cos1 = cos(theta1)
        sin1 = sin(theta1)
        cos2 = cos(theta2)
        sin2 = sin(theta2)

        denom = cos1*sin2 - sin1*cos2
        x = (sin2*rho1 - sin1*rho2) / denom
        y = (cos1*rho2 - cos2*rho1) / denom
        res = array((x, y))
    return res


def _hesse_form(pt1, pt2):
    '''Compute the Hesse form for the line through the points'''

    delta = pt2 - pt1
    mag2 = delta.dot(delta)
    vec = pt2 - pt2.dot(delta) * delta / mag2

    rho = sqrt(vec.dot(vec))
    if abs(rho) < 1e-6:
        # through 0. Need to compute theta differently
        theta = atan2(delta[1], delta[0]) + pi_by_2
        if theta > two_pi:
            theta -= two_pi
    else:
        theta = atan2(vec[1], vec[0])

    return rho, theta


# 2020-01-26: hough fit with/without initial fit from approxPolyDP_adaptive
# - using 640x480 2020 WPI images
#
#                                    Lenovo X1 Extreme       ODROID-XU4
# hough fit, no initial              0.853 msec/call         4.43 ms/call
# hough fit with initial fit         0.744 msec/call         3.82 ms/call
#
# - using WPI scaled to 320x240
#                                    Lenovo X1 Extreme       ODROID-XU4
# hough fit, no initial              0.368 ms/call           2.31 ms/call
# hough fit with initial fit         0.314 ms/call           1.95 ms/call
#
# - enable OpenCL
# hough fit with initial fit                                 5.17 ms/call
# (openCL disabled)                                          2.02 ms/call
