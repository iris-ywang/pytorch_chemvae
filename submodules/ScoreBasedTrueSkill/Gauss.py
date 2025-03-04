#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:12:51 2022

@author: dangoo
"""
from math import sqrt, pi, log, exp, erfc, erf
import numpy as np


#trueskill/lib/saulabs/gauss/distribution.rb
class Distribution:
    
    sqrt2 = sqrt(2)
    inv_sqrt_2pi = (1 / sqrt(2 * pi))
    log_sqrt_2pi = log( sqrt(2 * pi))
    
    def __init__(self, mean = 0.0, deviation = 0.0):
        
        if np.isinf(mean): mean = 0 
        if np.isinf(deviation): deviation = 0 
        self.mean = mean
        self.deviation = deviation
        
        self.variance = deviation ** 2
        
        self.precision = 0.0 if deviation == 0.0 else 1 / float(self.variance)
        self.precision_mean = self.precision * mean #(?)

    
    @staticmethod
    def standard():
        return Distribution(0.0, 1.0)
    
    @staticmethod
    def with_deviation(mean, deviation):
        return Distribution(mean, deviation)
    
    @staticmethod
    def with_variance(mean, deviation):
        return Distribution(mean, sqrt(deviation))
    
    @staticmethod
    def with_precision(mean, precision):
        
        if precision == 0:
            mean, deviation = 0,0
        else:
            mean = float(mean) / precision
            try:
                deviation = sqrt( float(1) / precision)
            except:
                print(precision)
                deviation = sqrt( float(1) / precision)
        return Distribution(mean, deviation)
    
    @staticmethod
    def absolute_difference(x,y):
        return max([abs((x.precision_mean - y.precision_mean)), 
                    sqrt(abs(x.precision - y.precision))])
    
    @classmethod
    def log_product_normalization(cls, x, y):
        if x.precision == 0.0 or y.precision == 0.0: return 0.0
        variance_sum = x.variance + y.variance
        mean_diff = x.mean - y.mean
        n = -cls.log_sqrt_2pi - (log(variance_sum) / 2.0) - (mean_diff**2 / (2.0 * variance_sum))
        return n
    
     
    # Computes the cummulative Gaussian distribution at a specified point of interest
    @classmethod
    def cumulative_distribution_function(cls, x):
        return 0.5 * (1 + erf(x / cls.sqrt2))
    cdf = cumulative_distribution_function

    @classmethod
    def probability_density_function(cls, x):
        return cls.inv_sqrt_2pi * exp(-0.5 * (x**2))
    pdf = probability_density_function
        
    @classmethod
    def quantile_function(cls, x):
        return -cls.sqrt2 * cls.inv_erf(2.0 * x)
    inv_cdf = quantile_function
    
    @staticmethod
    def inv_erf(p):
        
        if p >= 2.0: return -100 
        if p <= 0.0: return 100

        pp = p if p < 1.0 else 2-p
        
        t = sqrt(-2*log(pp/2.0)) # Initial guess
        
        x = -0.70711*((2.30753 + t*0.27061)/(1.0 + t*(0.99229 + t*0.04481)) - t)
        
        for _ in [0,1]:
            err = erfc(x) - pp
            x += err / (1.12837916709551257 * exp(-(x*x)) - x*err) # Halley
            
        return x if p < 1.0 else -x
     
    def value_at(self, x):
       expo = -(x - self.mean)**2.0 / (2.0 * self.variance)
       return (1.0/self.deviation) * self.inv_sqrt_2pi * exp(expo)
   
    # copy values from other distribution
    def replace(self, other):
        self.precision = other.precision
        self.precision_mean = other.precision_mean
        self.mean = other.mean
        self.deviation = other.deviation
        self.variance = other.variance
      
    def __mul__(self,other):
        return self.with_precision(self.precision_mean + other.precision_mean, 
                                           self.precision + other.precision)
    
    def __truediv__(self, other):
        return self.with_precision(self.precision_mean - other.precision_mean, self.precision - other.precision)
      
        
    def __sub__(self, other):
        return self.absolute_difference(self, other)
    
    def __eq__(self, other):
        return self.mean == other.mean and self.variance == other.variance
      
    def equals(self, other):
        return self == other
    
    def __str__(self):
        return "[μ= %.4g, σ=%.4g]" % (self.mean, self.deviation)
      
    
# end(checked)


#trueskill/lib/saulabs/gauss/truncated_correction.rb
class TruncatedCorrection:
    def __init__(self):
        pass
     
    @classmethod
    def w_within_margin(cls, perf_diff, draw_margin):
        abs_diff = abs(perf_diff)
        denom = Distribution.cdf(draw_margin - abs_diff) - \
                Distribution.cdf(-draw_margin - abs_diff)
        if denom < 2.2e-162: return 1.0
        vt = cls.v_within_margin(abs_diff, draw_margin)
        
        return vt**2 + (
                          (draw_margin - abs_diff) * 
                          Distribution.standard().value_at(draw_margin - abs_diff) -
                          (-draw_margin - abs_diff) *
                          Distribution.standard().value_at(-draw_margin - abs_diff) 
                         ) / denom
    
    @staticmethod
    def v_within_margin(perf_diff, draw_margin):
        abs_diff = abs(perf_diff)
        denom = Distribution.cdf(draw_margin - abs_diff) - Distribution.cdf(-draw_margin - abs_diff)
        if denom < 2.2e-162:
            return (-perf_diff - draw_margin) if perf_diff < 0 else (-perf_diff + draw_margin)
      
        
        num = Distribution.standard().value_at(-draw_margin - abs_diff) - \
              Distribution.standard().value_at(draw_margin - abs_diff)
        
        return ( -num/denom) if perf_diff < 0 else num/denom
    
    
    @classmethod
    def w_exceeds_margin(cls, perf_diff, draw_margin):
        denom = Distribution.cdf(perf_diff - draw_margin)
        if denom < 2.2e-162:
            return 1.0 if perf_diff < 0.0 else 0.0
        else:
            v = cls.v_exceeds_margin(perf_diff, draw_margin)
            return v * (v + perf_diff - draw_margin)


    @staticmethod
    def v_exceeds_margin( perf_diff, draw_margin):
        denom = Distribution.cdf(perf_diff - draw_margin)
        res =(-perf_diff + draw_margin) if denom < 2.2e-162 else Distribution.standard().value_at(perf_diff - draw_margin)/denom 

        return res
    
    @classmethod
    def exceeds_margin(cls, perf_diff, draw_margin):
          abs_diff = abs(perf_diff)
          denom = Distribution.cdf(draw_margin - abs_diff) - Distribution.cdf(-draw_margin - abs_diff)
          if denom < 2.2e-162:
              return 1.0
          else:
              v = cls.v_exceeds_margin(abs_diff, draw_margin)
              return v**2 + \
                   ((draw_margin - abs_diff) * Distribution.standard().value_at(draw_margin - abs_diff) - \
                   (-draw_margin - abs_diff) * Distribution.standard().value_at(-draw_margin - abs_diff)) / denom
          


#end(checked)






