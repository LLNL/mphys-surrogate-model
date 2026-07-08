"""
Global constants used across the surrogate model package.
"""

# Numerical tolerance to prevent division by zero
DIV_TOLERANCE = 1e-12

# Tolerance for logarithm operations to avoid log(0)
LOG_TOLERANCE = 1e-8

# Tolerance for clipping very small values before log operations
LOG_CLIP_TOLERANCE = 1e-10
