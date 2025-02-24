#!/usr/bin/env python3

# Copyright (c) 2024 Leigh C. Smith - lsmith@ast.cam.ac.uk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import unittest
import numpy as np
from cetra import LightCurve, TransitDetector

seconds_per_day = 60. * 60. * 24.


class TransitDetectorTestCase(unittest.TestCase):
    def setUp(self):
        # load the data package
        self.test_data = np.load('test_package.npz')

    def test_lightcurve(self):
        # initialise the lightcurve
        lc = LightCurve(
            times=self.test_data["times"],
            fluxes=self.test_data["fluxes"],
            flux_errors=self.test_data["errors"],
            verbose=True
        )
        # make sure the cadence were determined correctly
        self.assertEqual(lc.cadence, self.test_data["cadence"])

    def test_linear_search(self):
        # initialise the lightcurve
        lc = LightCurve(
            times=self.test_data["times"],
            fluxes=self.test_data["fluxes"],
            flux_errors=self.test_data["errors"],
            verbose=False
        )
        # initialise the transit detector
        tced = TransitDetector(lc)
        # run the linear search
        mtr = tced.linear_search(verbose=True)
        # extract the max-likelihood transit parameters
        transit = mtr.get_max_likelihood_parameters()

        # verify that at least one of the real transits was found, find which one
        t0s = np.arange(self.test_data["t0"], np.max(self.test_data["times"]), self.test_data["period"])
        idx_found = np.argmin(np.abs(transit.t0 - t0s))

        # check the transit time is consistent to within 1 t0 stride length
        # the error on the test LC flux points is tiny (1 ppm) so it should find it fine
        time_diff = np.abs(t0s[idx_found] - transit.t0)
        self.assertLess(time_diff, tced.t0_stride_length)

        # check the transit duration is the best it could have found in the grid
        idx_best_true_dur = np.argmin(np.abs(tced.durations - self.test_data["duration"]))
        best_duration = tced.durations[idx_best_true_dur]
        self.assertEqual(best_duration, transit.duration)

        # check the depth is consistent
        self.assertAlmostEqual(transit.depth, self.test_data["depth"], places=3)

    def test_periodic_search(self):
        # initialise the lightcurve
        lc = LightCurve(
            times=self.test_data["times"],
            fluxes=self.test_data["fluxes"],
            flux_errors=self.test_data["errors"],
            verbose=False
        )
        # initialise the transit detector
        tced = TransitDetector(lc)
        # run the linear search
        _ = tced.linear_search(verbose=False)
        # run the periodic search
        ptr = tced.period_search(verbose=True)
        # extract the max-likelihood transit parameters
        transit = ptr.get_max_likelihood_parameters()

        # check that the period is the best it could have found in the grid
        idx_best_true_per = np.argmin(np.abs(tced.periods - self.test_data["period"]))
        best_period = tced.periods[idx_best_true_per]
        self.assertEqual(best_period, transit.period)

        # we have to allow some error in the remaining parameters to take into account
        # the limited period grid resolution
        self.assertAlmostEqual(transit.t0, self.test_data["t0"], places=2)
        self.assertAlmostEqual(transit.duration, self.test_data["duration"], places=2)
        self.assertAlmostEqual(transit.depth, self.test_data["depth"], places=3)


if __name__ == '__main__':
    unittest.main()
