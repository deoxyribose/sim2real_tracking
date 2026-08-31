"""Discrete Inverse Rendering (DIR) for flagella tracking.

Given many candidate flagellum skeletons from the energy-UNet (several noise
draws × grid cells × suggestions × frames), an integer program picks a subset
that (a) reconstructs the median-subtracted residual well, (b) forms smooth
tracks across frames, and (c) doesn't select overlapping copies at the same
frame. Adapted (simplified) from `/home/frans/discrete_linking_opt`.
"""
