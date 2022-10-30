#!/usr/bin/python

# Small class to hold an OpenCV contour
# It provides access methods to the common attributes of contours which we want
# Most of the values are cached so that they are only computed once. This helps (a little) with speed.

import cv2


class Contour:
    '''Small class to cache info about a contour, to save repeating the calcs'''

    def __init__(self, contour):
        self._contour = contour

        # this is used so much, just do it. Saves all the tests
        self._bb_rect = cv2.boundingRect(self._contour)
                
        # variables to use for caching.
        # Initialize to None to indicate they have not been filled yet
        self._bb_area = None    # used often, so cache it directly
        self._moments = None
        self._convex_hull = None
        return

    @property
    def contour(self):
        '''The plain OpenCV contour'''
        return self._contour

    @property
    def bb_rectangle(self):
        '''The bounding box rectangle of the contour. See cv2.boundingRect'''
        return self._bb_rect

    @property
    def bb_center(self):
        '''The (x, y) center of the bounding box rectangle'''
        return (self._bb_rect[0] + self._bb_rect[1] // 2, self._bb_rect[1] + self._bb_rect[3] // 2)    # // is integer division

    @property
    def bb_size(self):
        '''The (width, height) of the bounding box rectangle'''
        return self._bb_rect[2:4]

    @property
    def bb_width(self):
        '''The width of the bounding box rectangle'''
        return self._bb_rect[2]

    @property
    def bb_height(self):
        '''The height of the bounding box rectangle'''
        return self._bb_rect[3]

    @property
    def bb_area(self):
        '''The area of the bounding box rectangle'''

        # this is used a lot, so cache it directly
        if self._bb_area is None:
            self._bb_area = self._bb_rect[2] * self._bb_rect[3]
        return self._bb_area

    @property
    def moments(self):
        '''The moments of the contour. See cv2.moments'''

        if self._moments is None:
            self._moments = cv2.moments(self._contour)
        return self._moments

    @property
    def contour_area(self):
        '''The area of the contour. See moments['m00'] or cv2.contourArea'''
        return self.moments['m00']

    @property
    def centroid(self):
        '''The centroid (center of mass) of the contour'''

        m = self.moments
        m00 = m['m00']
        return (m['m10'] / m00, m['m01'] / m00)

    @property
    def convex_hull(self):
        '''The convex hull of the contour, as a Contour instance'''

        if self._convex_hull is None:
            hull = cv2.convexHull(self._contour)
            self._convex_hull = Contour(hull)
        return self._convex_hull
