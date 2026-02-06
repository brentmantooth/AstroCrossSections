# AstroCrossSections

AstroCrossSections is a small Tkinter desktop app for loading two images and extracting matched line
cross sections. It supports common image formats and astronomical data, and plots both profiles
on the same graph for quick comparison.

See the YouTube video for a demo: https://youtu.be/OwB1C-wKpCU

![AstroCrossSections screenshot](AstroCrossScreen.png)

## Features

- Load two images (PNG/JPEG/TIF/BMP, FITS, XISF)
- For RGB images, choose luminance or keep RGB channels
- RGB mode plots separate red/green/blue cross sections
- Click two points on Image 1 to draw a line and sample both images
- Live preview while moving the mouse before the second click
- Plot both cross sections together (linear or log scale)
- Optional histogram mode with log-spaced bins (combined Image 1 + Image 2)
- Registration check when Image 2 loads (dx/dy/error shown in status bar)
- One-click image registration (align Image 2 to Image 1)
- Automatic center-cropping to a common size for pixel-accurate mapping
- Export cross section data to CSV (supports RGB channels)
- Export histogram data to CSV (supports RGB channels)
- Synchronized zoom and pan across both images
- Optional auto-stretch display toggle (does not affect measurements)
- PixInsight-style STF auto-stretch for display
- Clear images button to reset both panels

## Executable

Download the Windows 64-bit executable: [AstroCross.exe](dist/AstroCross.exe) (right click and "save as")


## Requirements

- Python 3
- Packages: `numpy`, `pillow`, `matplotlib`, `scikit-image`, `astroalign`
- Optional: `astropy` for FITS, `xisf` for XISF

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pillow matplotlib astropy xisf scikit-image astroalign
```

If you do not need FITS/XISF support, you can omit `astropy` or `xisf`.

## Run

```powershell
python AstroCross.py
```

## Usage

1. Click **Load Image 1** (this enables **Load Image 2**).
2. If the image is RGB, choose luminance or keep RGB channels.
3. Click **Load Image 2** (forced to the same mode as Image 1).
4. Review the registration info in the status bar; if |dx| or |dy| > 1.5, consider **Register Images**.
5. Click two points on Image 1 to define the cross section.
6. Move the mouse after the first click to preview the line, then click again to lock it.
7. Click a third time to reset and start a new selection.
8. Use **Export CSV** to save the distance/intensity table.

### Controls

- Mouse wheel over an image: zoom (synchronized)
- Right-click + drag: pan (synchronized)
- **Reset View**: return to fit-to-window view
- **Auto-stretch display**: toggle between raw and stretched display
- **Log scale plot**: toggle linear vs log y-axis
- **Histograms**: switch plot to histograms (log-spaced bins; includes median markers)
- **Register Images**: align Image 2 to Image 1 (requires `astroalign`)
- **Clear Images**: clears both images and disables Image 2 until Image 1 is loaded

## Notes

- The plot and CSV always use the original image data, not the display stretch.
- When image sizes differ, both are center-cropped to a common size for 1:1 pixel mapping.
- The status bar shows registration shift and error (requires `scikit-image`).
- In RGB mode, plots and CSV export keep channels separate for both images.
- Histogram mode ignores non-positive values (log-spaced bins) and shows per-image medians.

## License

See `LICENSE`.
