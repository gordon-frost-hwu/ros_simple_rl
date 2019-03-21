#! /usr/bin/env python

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """ rel_tol: relative tolerance allows a greater allowed difference between values while still considering them equal,
    as the values get larger
    abs_tol: absolute tolerance which is the same no matter what magnitude the values have"""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)