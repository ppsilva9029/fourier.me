import numpy as np
import matplotlib.pyplot as plt
from time import time
from svgpathtools import svg2paths
import Christofides as christofides

# Parameters
max_coords_length = 10
threshold_amp = 0.001

debug = __name__ == "__main__"

def get_vectors_from_svg(svg_file):

    paths, _ = svg2paths(svg_file)

    points = []
    for path in paths:
        for segment in path:
            for t in np.linspace(0, 1, 100):
                point = segment.point(t)
                points.append([point.real, -point.imag])

    print("Number of points=", len(points))

    # Calculate the dimensions and center of SVG
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    width = max_x - min_x
    height = max_y - min_y

    x_shift = (max_x + min_x) / 2
    y_shift = (max_y + min_y) / 2
    sf = max_coords_length / max(width, height)

    # Translate and scale points
    translated_points = []
    for p in points:
        translated_points.append([(p[0] - x_shift) * sf, (p[1] - y_shift) * sf])

    if debug:
        plt.scatter(*zip(*translated_points), s=1)
        plt.show()

    print("Running Christofides...")
    start = time()
    path_idx = christofides.tsp(translated_points)
    print("Time taken by Christofides=", time() - start)

    x = []
    y = []

    path = []
    for i in path_idx:
        path.append(complex(translated_points[i][0], translated_points[i][1]))
        if debug:
            x.append(translated_points[i][0])
            y.append(translated_points[i][1])

    if debug:
        
        plt.scatter(x, y, s=1)
        plt.show()
        x.clear()
        y.clear()

    del path[-1]

    N = len(path)
    freqs = np.fft.fftfreq(N) * 20 * np.pi
    fft = np.fft.fft(path) / N

    mask = abs(fft) >= threshold_amp
    fft = fft[mask]
    freqs = freqs[mask]
    N = len(freqs)

    dtype = [("amp", float), ("freq", float), ("real", float), ("imag", float)]
    arrow_dat = []
    for i in range(N):
        c = fft[i]
        amp = round(abs(c), 4)
        freq = round(freqs[i], 4)
        arrow_dat.append((amp, freq, round(c.real, 4), round(c.imag, 4)))

    if debug:
        plt.scatter(freqs, abs(fft), s=3)
        plt.show()

    print("Number of vectors=", N)

    arrow_dat = np.array(arrow_dat, dtype=dtype)
    arrow_dat = np.sort(arrow_dat, order="amp")
    arrow_dat = arrow_dat[::-1]

    if debug:
        np.savetxt("../arrow_data/arrow_dat_" + svg_file.split('/')[-1] + ".csv", arrow_dat, delimiter=",")
    else:
        np.savetxt("../arrow_data/arrow_dat_last.csv", arrow_dat, delimiter=",")

    return arrow_dat


if __name__ == "__main__":
    svgFile = "cloud.svg"  # Replace with SVG file
    get_vectors_from_svg(svgFile)
