# AstroCrossSections

AstroCrossSections is a small Tkinter desktop app for loading two images and extracting matched line
cross sections. It supports common image formats and astronomical data, and plots both profiles
on the same graph for quick comparison.

![AstroCrossSections screenshot](AstroCrossScreen.png)

## Features

- Load two images (PNG/JPEG/TIF/BMP, FITS, XISF)
- For RGB images, choose luminance or keep RGB channels
- RGB mode plots separate red/green/blue cross sections
- Click two points on Image 1 to draw a line and sample both images
- Live preview while moving the mouse before the second click
- Plot both cross sections together (linear or log scale)
- Export cross section data to CSV (supports RGB channels)
- Synchronized zoom and pan across both images
- Optional auto-stretch display toggle (does not affect measurements)
- Clear images button to reset both panels

## Executable

Download the Windows 64-bit executable: [AstroCross.exe](dist/AstroCross.exe) (right click and "save as")


## Requirements

- Python 3
- Packages: `numpy`, `pillow`, `matplotlib`
- Optional: `astropy` for FITS, `xisf` for XISF

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pillow matplotlib astropy xisf
```

If you do not need FITS or XISF support, you can omit `astropy` or `xisf`.

## Run

```powershell
python AstroCross.py
```

## Usage

1. Click **Load Image 1** (this enables **Load Image 2**).
2. If the image is RGB, choose luminance or keep RGB channels.
3. Click **Load Image 2** (forced to the same mode as Image 1).
4. Click two points on Image 1 to define the cross section.
3. Move the mouse after the first click to preview the line, then click again to lock it.
4. Click a third time to reset and start a new selection.
5. Use **Export CSV** to save the distance/intensity table.

### Controls

- Mouse wheel over an image: zoom (synchronized)
- Right-click + drag: pan (synchronized)
- **Reset View**: return to fit-to-window view
- **Auto-stretch display**: toggle between raw and stretched display
- **Log scale plot**: toggle linear vs log y-axis
- **Clear Images**: clears both images and disables Image 2 until Image 1 is loaded

## Notes

- The plot and CSV always use the original image data, not the display stretch.
- Image 2 is sampled using a proportional mapping of Image 1 coordinates when sizes differ.
- In RGB mode, plots and CSV export keep channels separate for both images.

## License

See `LICENSE`.
