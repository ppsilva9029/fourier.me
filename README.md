# fourier.me

In this project I am building an WebApp that takes your photos and turns them into Fourier Epicycle Drawings.

Not getting it ? Here's the inspiration: https://youtu.be/r6sGWTCMz2k

## How it works ?

Here's a TLDR of how it works:

1. Takes any image(jpg/jpeg/png...etc) as input.
2. Resizes this image to feasible proportions.
3. Uses Canny Edge Detector(of the OpenCV library) to find the edges or outline of the image.
4. From this image, it gets the coordinates of the outline pixels.
5. Using the Christofides' algorithm, it finds the shortest closed tour of the points.(doing this is important for the epicycles to work as intended)
6. Converts these points into complex numbers.
7. Takes the Complex Discrete Fourier Transform of these points(using numpy's fft(fast fourier transform) function).
8. Saves the fft data along with other details like frequency, amplitude in a csv file.
9. Now comes the animation part, the manim code reads this csv file and makes the corresponding rotating arrows and circles and animates them.
* I have also added some optimizations in the manim code for faster rendering which is irrelevant for this TLDR.

I am planning to make a WebApp out of this. Let's see how it goes!!!

## How to run locally

```
git clone https://github.com/shlok-007/fourier.me.git
cd fourier.me
mkdir images
mkdir arrow_data
cd Manim
pip install requirements.txt
```

If you don't have chocolatey installed, follow [this](https://chocolatey.org/install) to install it.

Now,
```
choco install manimce
```
Now, you can run the main.py file by giving it the path to your desired image as input.
```
python main.py path/to/image
```
For example:
```
python main.py ../sample_images/robot.png
```
You can checkout and try varying the parameters declared at the top of the files `getLineArt.py`, `getVectors.py`, and `epicycle_manim.py` to control various aspects of the process. Also, you can checkout the `manim.cfg` file for varying the parameters related to animation rendering.