import numpy as np
import warnings
from math import exp


class Membership:
    """
    This class instantiates membership function objects that can then be used to evaluate inputs
    The purpose of this class is to calculate the output of a single value against a membership function

    The user needs to specify params in the form of a list of tuples
    :param params : Membership function values, [(MF11, MF12, MF13), (MF21, MF22, MF23, MF24), ...]
    """
    def __init__(self, params: list):
        self.params = params

    def triangle(self, x):
        le = self.params[0]
        ce = self.params[1]
        re = self.params[2]
        mu = 0

        if not le <= ce <= re:
            warnings.warn('Triangular MF parameters were out of order, they were reordered')
            self.params.sort()
            le = self.params[0]
            ce = self.params[1]
            re = self.params[2]

        if le != ce:
            if le < x < ce:
                mu = (x - le) * (1/(ce - le))
        if re != ce:
            if ce < x < re:
                mu = (re - x) * (1/(re - ce))
        if x == ce:
            mu = 1
        return mu

    def trapezoid(self, x):
        le = self.params[0]
        ce1 = self.params[1]
        ce2 = self.params[2]
        re = self.params[3]
        mu = 0

        if not le <= ce1 <= ce2 <= re:
            warnings.warn('Trapezoidal MF parameters were out of order, they were reordered')
            self.params.sort()
            le = self.params[0]
            ce1 = self.params[1]
            ce2 = self.params[2]
            re = self.params[3]

        if le != ce1:
            if le < x < ce1:
                mu = (x - le) * (1/(ce1 - le))
        if re != ce2:
            if ce2 < x < re:
                mu = (re - x) * (1/(re - ce2))
        if ce1 <= x <= ce2:
            mu = 1
        return mu

    def gauss(self, x):
        c = self.params[0]
        sigma = self.params[1]

        mu = exp(-(x - c)**2 / (2*sigma**2))
        return mu


class MembershipArray:
    """
    This class instantiates membership function objects that can then be used to evaluate inputs
    The purpose of this class is to calculate an array of outputs between the bounds of the system

    The user needs to specify params in the form of a list of tuples
    : param params : Membership function values, [(MF11, MF12, MF13), (MF21, MF22, MF23, MF24)]
    : param xvals : an array of values between the upper and lower bound of the system
    """
    def __init__(self, params: list, xvals):
        self.params = params
        self.xvals = xvals

    def triangle(self, x):
        le = self.params[0]
        ce = self.params[1]
        re = self.params[2]
        mu = np.zeros_like(self.xvals)

        if not le <= ce <= re:
            warnings.warn('Triangular MF parameters were out of order, they were reordered')
            self.params.sort()
            le = self.params[0]
            ce = self.params[1]
            re = self.params[2]

        if le != ce:
            index = np.argwhere(np.logical_and(le < x, x < ce))
            mu[index] = (x[index] - le) * (1/(ce - le))

        if re != ce:
            index = np.argwhere(np.logical_and(ce < x, x < re))
            mu[index] = (re - x[index]) * (1/(re - ce))
        index = np.argwhere(x == ce)

        if np.any(index):
            mu[index] = 1

        return mu

    def trapezoid(self, x):
        le = self.params[0]
        ce1 = self.params[1]
        ce2 = self.params[2]
        re = self.params[3]
        mu = np.zeros_like(self.xvals)

        if not le <= ce1 <= ce2 <= re:
            warnings.warn('Trapezoidal MF parameters were out of order, they were reordered')
            self.params.sort()
            le = self.params[0]
            ce1 = self.params[1]
            ce2 = self.params[2]
            re = self.params[3]

        if le != ce1:
            index = np.argwhere(np.logical_and(le < x, x < ce1))
            if np.any(index):
                mu[index] = (x[index] - le) * (1/(ce1 - le))

        if re != ce2:
            index = np.argwhere(np.logical_and(ce2 < x, x < re))
            if np.any(index):
                mu[index] = (re - x[index]) * (1/(re - ce2))

        index = np.argwhere(np.logical_and(ce1 <= x, x <= ce2))
        if np.any(index):
            mu[index] = 1

        return mu

    def gauss(self, x):
        c = self.params[0]
        sigma = self.params[1]

        mu = np.exp(np.divice(-np.square(x - c),  (2*np.square(sigma))))
        return mu


class Rulebase:
    def __init__(self):
        self.output = None

    def AND_rule(self, x):
        self.output = np.amin(x)
        return self.output

    def OR_rule(self, x):
        self.output = np.amax(x)
        return self.output


class Defuzz:
    def __init__(self, mu, output, bounds):
        outMFarrays = []
        xvals = np.linspace(bounds[0], bounds[1], num=100)

        for num, val in zip(mu, output):
            values = val.params
            if len(values) == 2:
                temp_out = val.gauss(xvals)
            elif len(values) == 3:
                temp_out = val.triangle(xvals)
            else:
                temp_out = val.trapezoid(xvals)

            index = np.argwhere(temp_out >= num)
            temp_out[index] = num
            outMFarrays.append(temp_out)

        outStacked = np.vstack(outMFarrays)
        maxOuts = np.max(outStacked, axis=0)

        self.max = maxOuts
        total_area = np.sum(maxOuts)

        if total_area == 0:
            crisp_out = 0
        else:
            crisp_out = np.divide(np.sum(np.multiply(maxOuts, xvals)), total_area)

        self.out = crisp_out
