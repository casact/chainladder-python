# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import copy
from chainladder.utils.cupy import cp

def _get_full_expectation(cdf_, ultimate_, process_variance_):
    """ Private method that builds full expectation"""
    xp = cp.get_array_module(ultimate_.values)
    o, d = ultimate_.shape[-2:]
    cdf = copy.deepcopy(cdf_)
    cdf.values = xp.repeat(cdf.values[..., 0:1, :], o, axis=2)
    cdf.odims = ultimate_.odims
    cdf.valuation_date = ultimate_.valuation_date
    full = copy.deepcopy(ultimate_)
    full.values = xp.concatenate((full.values / cdf.values, full.values), -1)
    full.ddims = xp.append(cdf_.ddims, '9999-Ult')
    full.ddims = xp.array([int(item.split('-')[0]) for item in full.ddims])
    full.valuation = full._valuation_triangle()
    full.vdim = ultimate_.vdims
    if process_variance_:
        full.values = (xp.nan_to_num(full.values) +
                       xp.nan_to_num(process_variance_.values))
    return full

def _get_full_triangle(full_expectation_, triangle_):
    """ Private method that builds full triangle"""
    full = full_expectation_
    full = (triangle_ + full - full[full.valuation<=triangle_.valuation_date])
    full.vdims = triangle_.vdims
    return full

def _get_cdf(ldf_):
    """ Private method that computes CDFs"""
    cdf = copy.deepcopy(ldf_)
    xp = cp.get_array_module(cdf.values)
    cdf.values = xp.flip(xp.cumprod(xp.flip(cdf.values, -1), -1), -1)
    cdf.ddims = xp.array(
        [item.replace(item[item.find("-")+1:], '9999')
         for item in cdf.ddims])
    return cdf

class Common:
    """ Class that contains common properties of a "fitted" Triangle. """
    @property
    def cdf_(self):
        if not hasattr(self, 'ldf_'):
            raise AttributeError("'" + self.__class__.__name__ + "'" +
                                 " object has no attribute 'cdf_'")
        return _get_cdf(self.ldf_)

    @property
    def ibnr_(self):
        if not hasattr(self, 'ultimate_'):
            raise AttributeError("'" + self.__class__.__name__ + "'" +
                                 " object has no attribute 'ibnr_'")
        ibnr =  self.ultimate_ - self.latest_diagonal
        ibnr.vdims = self.ultimate_.vdims
        return ibnr

    @property
    def full_expectation_(self):
        if not hasattr(self, 'ultimate_'):
            raise AttributeError("'" + self.__class__.__name__ + "'" +
                                 " object has no attribute 'full_expectation_'")
        if hasattr(self, 'process_variance_'):
            return _get_full_expectation(self.cdf_, self.ultimate_, self.process_variance_)
        else:
            return _get_full_expectation(self.cdf_, self.ultimate_, None)

    @property
    def full_triangle_(self):
        if not hasattr(self, 'ultimate_'):
            raise AttributeError("'" + self.__class__.__name__ + "'" +
                                 " object has no attribute 'full_triangle_'")
        if hasattr(self, 'X_'):
            return _get_full_triangle(self.full_expectation_, self.X_)
        else:
            return _get_full_triangle(self.full_expectation_, self)
