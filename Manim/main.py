import sys
import cv2

import getLineArt
import getVectors
import epicycle_manim
import getVectorsSvg

if __name__ == "__main__":
    img_path = sys.argv[1]
    image = cv2.imread(img_path)

    if img_path.endswith("svg"):
        arrow_dat = getVectorsSvg.get_vectors_from_svg(img_path)
    else:
        lineart = getLineArt.get_lineart(image)
        arrow_dat = getVectors.get_vectors(lineart)
        
    epicycle_manim.Fourier_Epicycles(arrow_dat).render(preview=True)