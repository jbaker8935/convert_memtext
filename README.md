# Wildbits K2 Memory Text based Animation Converter

Simple python tools for converting an image sequence to a binary used by the anim_memtext program.

## Installation

Setup a python environment and load the dependencies

```python -m pip install -r requirements.txt```

For GPU acceleration of clustering you can install a rapids environment to get cuml.
See install instructions at the [RAPIDS website](https://docs.rapids.ai/install/)

## Creating an image sequence

Images should be named similarly with sequence numbers so conversion can do proper ordering.  The tool will consume all images in the directory provided as an argument and order them lexicographically.  So, one directory per video.

Easiest way is to use ffmpeg:

```ffmpeg -i videofile.mp4 -r 10 output_dir/output_%04d.png```

This will convert 'videofile.mp4' to a series of numbered png files in the 'output_dir'.  Images will be output for 10 fps playback.

## Converting

```python convert_memtext.py --animation input_dir --output-bin filename.bin```

Hybrid encoding (global palette + per-frame 512 medoids, with alternating font IDs for double buffering):

```python convert_memtext.py --animation input_dir --output-bin filename.bin --encoding-mode hybrid```

The Hybrid mode is experimental and will increase file size.  Generally the default global mode is the best encoding-mode, but hybrid may be perceptually better for some content. 

Note: frame encoding does not currently work on the hardware.  Do not use that encoding-mode.

If you want to see the reconstructed images on your host the render_memtext.py tool can take a bin file and show you what each frame will look like when output by the anim_memtext.pgz.

```python render_memtext.py animation.bin  output_dir```

animation.bin will be converted to a reconstructed png sequence in output_dir.

## Playing the video

save filename.bin to the K2 in the same directory as anim_memtext.pgz and execute.

```/- anim_memtext filename.bin```

## Converter
Images are converted to a 640x480 canvas which is divided into 80x60 cells, which will be reconstructed using memtext.   Since each reconstructed cell is a glyph and 2 colors, the trick is to find an optimal set of colors and limited glyph patterns that will give the best reconstruction with minimal perceptual error.  All cells throughout the entire animation are weighted to find the best 1024 representative glyphs and color sets.
Then each frame is reconstructed by finding the best glyph,color combination for each cell and writing the resulting font sets, color palettes and frame information to the binary file.

