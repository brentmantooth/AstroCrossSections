# AstroCross.py
# Original author: Brent A. Mantooth
# To run from python: & python AstroCross.py
# to compile to exe: pyinstaller --onefile --windowed AstroCross.py

import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

try:
    from astropy.io import fits
except Exception:  # pragma: no cover - optional dependency
    fits = None

try:
    from skimage.registration import phase_cross_correlation
except Exception:  # pragma: no cover - optional dependency
    phase_cross_correlation = None

try:
    import astroalign as aa
except Exception:  # pragma: no cover - optional dependency
    aa = None

try:
    from PIL import Image, ImageTk
except Exception as exc:  # pragma: no cover - required for GUI
    raise SystemExit("Pillow is required: pip install pillow") from exc

try:
    from xisf import XISF
except Exception:  # pragma: no cover - optional dependency
    XISF = None

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

COLOR_LUMINANCE = "luminance"
COLOR_RGB = "rgb"
REG_SHIFT_WARN = 1.5
AXIS_LABEL_SIZE = 12
AXIS_TICK_SIZE = 10
AXIS_TITLE_SIZE = 13
LEGEND_FONT_SIZE = 9


def is_fits_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".fits", ".fit", ".fts"}


def is_xisf_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".xisf", ".xifs"}


def rgb_to_luminance(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def image_hw(data: np.ndarray) -> tuple[int, int]:
    if data.ndim == 2:
        return data.shape
    if data.ndim == 3:
        return data.shape[0], data.shape[1]
    raise RuntimeError("Expected 2D or RGB image data")


def normalize_image_array(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    while data.ndim > 3:
        data = data[0]
    if data.ndim == 3:
        if data.shape[0] in (3, 4) and (data.shape[1] > 4 or data.shape[2] > 4):
            return np.transpose(data[:3, ...], (1, 2, 0))
        if data.shape[2] in (3, 4):
            return data[..., :3]
        if data.shape[0] in (3, 4):
            return np.transpose(data[:3, ...], (1, 2, 0))
        if data.shape[2] == 1:
            return data[..., 0]
        if data.shape[0] == 1:
            return data[0, ...]
        return data[..., 0]
    return data


def apply_color_mode(data: np.ndarray, mode: str) -> np.ndarray:
    data = normalize_image_array(data)
    if mode == COLOR_RGB:
        if data.ndim == 2:
            return np.repeat(data[..., None], 3, axis=2)
        return data
    if mode == COLOR_LUMINANCE:
        if data.ndim == 3:
            return rgb_to_luminance(data[..., :3])
        return data
    raise RuntimeError(f"Unknown color mode: {mode}")


def data_kind_from_array(data: np.ndarray) -> str:
    if np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.bool_):
        return "int"
    return "float"


def normalize_unit_interval(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    dtype = data.dtype
    data = data.astype(np.float32, copy=False)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    if min_val >= 0.0 and max_val <= 1.0:
        return data
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scale = float(info.max) if info.max > 0 else None
        if scale:
            return data / scale
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return np.zeros_like(data, dtype=np.float32)
    return (data - min_val) / (max_val - min_val)


def mtf(x: np.ndarray | float, m: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.clip(m, 1e-6, 1.0 - 1e-6))
    y = np.zeros_like(x)
    mask = (x > 0.0) & (x < 1.0)
    if np.any(mask):
        num = (m - 1.0) * x[mask]
        den = (2.0 * m - 1.0) * x[mask] - m
        y[mask] = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    y = np.where(x >= 1.0, 1.0, y)
    return np.clip(y, 0.0, 1.0)


def stf_stretch(data: np.ndarray, shadow_clip: float = -2.8, target_bkg: float = 0.2) -> np.ndarray:
    norm = normalize_unit_interval(data)
    finite = norm[np.isfinite(norm)]
    if finite.size == 0:
        return np.zeros_like(norm, dtype=np.float32)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    c = med + shadow_clip * 1.4826 * mad
    c = float(np.clip(c, 0.0, 1.0))
    if not np.isfinite(c):
        return np.zeros_like(norm, dtype=np.float32)
    if c >= 1.0:
        c = 1.0 - 1e-6
    denom = 1.0 - c
    if denom <= 0.0:
        return np.zeros_like(norm, dtype=np.float32)
    scaled = (norm - c) / denom
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = np.clip(scaled, 0.0, 1.0)
    med_c = float(np.clip(med - c, 0.0, 1.0))
    midtones = float(mtf(med_c, target_bkg))
    if not np.isfinite(midtones) or midtones <= 0.0 or midtones >= 1.0:
        return scaled
    return mtf(scaled, midtones)


def normalize_for_display(data: np.ndarray, stretch: bool = True) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    if stretch:
        scaled = stf_stretch(data)
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if hi <= lo:
                hi = lo + 1.0
        scaled = (data - lo) / (hi - lo)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def clamp_value(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bilinear_sample(data: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = data.shape
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    Ia = data[y0, x0]
    Ib = data[y1, x0]
    Ic = data[y0, x1]
    Id = data[y1, x1]

    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def sample_line(data: np.ndarray, p0: tuple[float, float], p1: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    num = max(2, int(math.ceil(length)) + 1)
    xs = np.linspace(x0, x1, num)
    ys = np.linspace(y0, y1, num)
    if data.ndim == 2:
        profile = bilinear_sample(data, xs, ys)
    else:
        profiles = [bilinear_sample(data[..., idx], xs, ys) for idx in range(3)]
        profile = np.stack(profiles, axis=1)
    dist = np.linspace(0.0, length, num)
    return dist, profile


def center_crop(data: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = image_hw(data)
    if new_h > h or new_w > w:
        raise ValueError("Crop size must be <= original size")
    if new_h == h and new_w == w:
        return data
    y0 = max(0, (h - new_h) // 2)
    x0 = max(0, (w - new_w) // 2)
    if data.ndim == 2:
        return data[y0 : y0 + new_h, x0 : x0 + new_w]
    return data[y0 : y0 + new_h, x0 : x0 + new_w, ...]


def crop_pair_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    h1, w1 = image_hw(a)
    h2, w2 = image_hw(b)
    new_h = min(h1, h2)
    new_w = min(w1, w2)
    changed = (h1 != new_h or w1 != new_w or h2 != new_h or w2 != new_w)
    if not changed:
        return a, b, False
    return center_crop(a, new_h, new_w), center_crop(b, new_h, new_w), True


class ImageState:
    def __init__(self, canvas: tk.Canvas, name: str):
        self.canvas = canvas
        self.name = name
        self.path: str | None = None
        self.data: np.ndarray | None = None
        self.data_kind: str | None = None
        self.data_dtype: np.dtype | None = None
        self.display_image: ImageTk.PhotoImage | None = None
        self.scale: float = 1.0
        self.base_scale: float = 1.0
        self.offset: tuple[int, int] = (0, 0)
        self.image_id: int | None = None

    @property
    def shape(self) -> tuple[int, int] | None:
        if self.data is None:
            return None
        return image_hw(self.data)

    def clear(self) -> None:
        self.path = None
        self.data = None
        self.data_kind = None
        self.data_dtype = None
        self.display_image = None
        if self.image_id is not None:
            self.canvas.delete(self.image_id)
            self.image_id = None

    def set_data(
        self,
        path: str,
        data: np.ndarray,
        stretch: bool = True,
        zoom: float = 1.0,
        center_norm: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self.path = path
        self.data = data
        self.redraw(stretch=stretch, zoom=zoom, center_norm=center_norm)

    def redraw(
        self,
        stretch: bool = True,
        zoom: float = 1.0,
        center_norm: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        if self.data is None:
            return
        h, w = image_hw(self.data)
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        base_scale = min(canvas_w / w, canvas_h / h)
        scale = base_scale * zoom
        self.base_scale = base_scale
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))

        if w > 1:
            cx = clamp_value(center_norm[0], 0.0, 1.0) * (w - 1)
        else:
            cx = 0.0
        if h > 1:
            cy = clamp_value(center_norm[1], 0.0, 1.0) * (h - 1)
        else:
            cy = 0.0
        x0 = int(round(canvas_w / 2 - cx * scale))
        y0 = int(round(canvas_h / 2 - cy * scale))
        self.scale = scale
        self.offset = (x0, y0)

        if self.data.ndim == 2:
            disp = normalize_for_display(self.data, stretch=stretch)
            img = Image.fromarray(disp, mode="L")
        else:
            channels = [normalize_for_display(self.data[..., idx], stretch=stretch) for idx in range(3)]
            disp = np.stack(channels, axis=2)
            img = Image.fromarray(disp, mode="RGB")
        img = img.resize((disp_w, disp_h), Image.Resampling.NEAREST)
        self.display_image = ImageTk.PhotoImage(img)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(x0, y0, anchor="nw", image=self.display_image)
        else:
            self.canvas.itemconfigure(self.image_id, image=self.display_image)
            self.canvas.coords(self.image_id, x0, y0)

    def to_image_coords(self, x: float, y: float) -> tuple[float, float] | None:
        if self.data is None:
            return None
        x0, y0 = self.offset
        h, w = image_hw(self.data)
        disp_w = w * self.scale
        disp_h = h * self.scale
        if x < x0 or y < y0 or x > x0 + disp_w or y > y0 + disp_h:
            return None
        ix = (x - x0) / self.scale
        iy = (y - y0) / self.scale
        return float(ix), float(iy)


class AstroCrossApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AstroCross Sections")
        self.root.geometry("1200x800")

        self.main = ttk.Frame(root, padding=8)
        self.main.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        self.top = ttk.Frame(self.main)
        self.top.grid(row=0, column=0, sticky="nsew")
        self.bottom = ttk.Frame(self.main)
        self.bottom.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        self.main.rowconfigure(0, weight=3)
        self.main.rowconfigure(1, weight=2)
        self.main.columnconfigure(0, weight=1)

        self.left_frame = ttk.Frame(self.top)
        self.right_frame = ttk.Frame(self.top)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.top.columnconfigure(0, weight=1)
        self.top.columnconfigure(1, weight=1)
        self.top.rowconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(self.left_frame, background="#111111", highlightthickness=1, highlightbackground="#444")
        self.right_canvas = tk.Canvas(self.right_frame, background="#111111", highlightthickness=1, highlightbackground="#444")
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        self.left_frame.rowconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)
        self.left_frame.columnconfigure(0, weight=1)
        self.right_frame.columnconfigure(0, weight=1)

        self.left_controls = ttk.Frame(self.left_frame)
        self.right_controls = ttk.Frame(self.right_frame)
        self.left_controls.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.right_controls.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.left_frame.rowconfigure(1, weight=0)
        self.right_frame.rowconfigure(1, weight=0)

        self.btn_load_left = ttk.Button(self.left_controls, text="Load Image 1", command=lambda: self.load_image(1))
        self.btn_load_right = ttk.Button(self.right_controls, text="Load Image 2", command=lambda: self.load_image(2))
        self.btn_load_left.grid(row=0, column=0, sticky="w")
        self.btn_load_right.grid(row=0, column=0, sticky="w")
        self.btn_load_right.configure(state="disabled")

        self.lbl_left = ttk.Label(self.left_controls, text="No file loaded", width=100)
        self.lbl_right = ttk.Label(self.right_controls, text="No file loaded", width=100)
        self.lbl_left.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.lbl_right.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.plot_frame = ttk.Frame(self.bottom)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.bottom.rowconfigure(0, weight=1)
        self.bottom.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        self.plot_mode: str | None = None
        self.axes: list = []
        self.lines: list = []
        self.configure_plot_axes(COLOR_LUMINANCE)

        self.controls = ttk.Frame(self.bottom)
        self.controls.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.btn_export = ttk.Button(self.controls, text="Export CSV", command=self.export_csv)
        self.btn_clear = ttk.Button(self.controls, text="Clear Selection", command=self.reset_selection)
        self.btn_clear_images = ttk.Button(self.controls, text="Clear Images", command=self.clear_images)
        self.btn_register = ttk.Button(self.controls, text="Register Images", command=self.register_images)
        self.autostretch_var = tk.BooleanVar(value=True)
        self.chk_autostretch = ttk.Checkbutton(
            self.controls, text="Auto-stretch display", variable=self.autostretch_var, command=self.update_display_mode
        )
        self.logscale_var = tk.BooleanVar(value=False)
        self.chk_logscale = ttk.Checkbutton(
            self.controls, text="Log scale plot", variable=self.logscale_var, command=self.update_plot_scale
        )
        self.btn_reset_view = ttk.Button(self.controls, text="Reset View", command=self.reset_view)
        self.status_base = "Click two points on Image 1 to sample a cross section."
        self.registration_info = ""
        self.lbl_status = ttk.Label(self.controls, text=self.status_base)
        self.btn_export.grid(row=0, column=0, sticky="w")
        self.btn_clear.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.btn_clear_images.grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.btn_register.grid(row=0, column=3, sticky="w", padx=(8, 0))
        self.btn_register.configure(state="disabled")
        self.chk_autostretch.grid(row=0, column=4, sticky="w", padx=(8, 0))
        self.chk_logscale.grid(row=0, column=5, sticky="w", padx=(8, 0))
        self.btn_reset_view.grid(row=0, column=6, sticky="w", padx=(8, 0))
        self.lbl_status.grid(row=0, column=7, sticky="w", padx=(16, 0))
        self.controls.columnconfigure(7, weight=1)

        self.image1 = ImageState(self.left_canvas, "Image 1")
        self.image2 = ImageState(self.right_canvas, "Image 2")
        self.color_mode: str | None = None
        self.update_plot_title()

        self.zoom = 1.0
        self.center_norm = (0.5, 0.5)
        self.pan_anchor: tuple[int, int, tuple[float, float]] | None = None
        self.pan_source: ImageState | None = None

        self.selection_state = "idle"  # idle -> drawing -> fixed
        self.start_pt: tuple[float, float] | None = None
        self.end_pt: tuple[float, float] | None = None
        self.line_id: int | None = None
        self.line_id_right: int | None = None
        self.last_profile: dict[str, np.ndarray | str] | None = None

        self.left_canvas.bind("<Button-1>", self.on_left_click)
        self.left_canvas.bind("<Motion>", self.on_left_move)
        self.left_canvas.bind("<MouseWheel>", lambda event: self.on_zoom(event, self.image1))
        self.right_canvas.bind("<MouseWheel>", lambda event: self.on_zoom(event, self.image2))
        self.left_canvas.bind("<Button-3>", lambda event: self.on_pan_start(event, self.image1))
        self.left_canvas.bind("<B3-Motion>", lambda event: self.on_pan_move(event, self.image1))
        self.right_canvas.bind("<Button-3>", lambda event: self.on_pan_start(event, self.image2))
        self.right_canvas.bind("<B3-Motion>", lambda event: self.on_pan_move(event, self.image2))
        self.left_canvas.bind("<ButtonRelease-3>", self.on_pan_end)
        self.right_canvas.bind("<ButtonRelease-3>", self.on_pan_end)
        self.left_canvas.bind("<Configure>", lambda _evt: self.apply_view())
        self.right_canvas.bind("<Configure>", lambda _evt: self.apply_view())

    def update_status_label(self) -> None:
        if self.registration_info:
            text = f"{self.status_base} | {self.registration_info}"
        else:
            text = self.status_base
        self.lbl_status.configure(text=text)

    def set_status(self, text: str) -> None:
        self.status_base = text
        self.update_status_label()

    def set_registration_info(self, text: str) -> None:
        self.registration_info = text
        self.update_status_label()

    def update_register_button_state(self) -> None:
        if not hasattr(self, "btn_register"):
            return
        can_register = self.image1.data is not None and self.image2.data is not None and aa is not None
        self.btn_register.configure(state="normal" if can_register else "disabled")

    def configure_plot_axes(self, mode: str) -> None:
        if mode == self.plot_mode and self.axes:
            return
        self.fig.clear()
        self.axes = []
        self.lines = []
        if mode == COLOR_RGB:
            axes = self.fig.subplots(3, 1, sharex=True)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            channel_info = [("Red", "tab:red"), ("Green", "tab:green"), ("Blue", "tab:blue")]
            for ax, (label, color) in zip(axes, channel_info):
                ax.set_ylabel(label, fontsize=AXIS_LABEL_SIZE)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=AXIS_TICK_SIZE)
                line1, = ax.plot([], [], label="Image 1", color=color)
                line2, = ax.plot([], [], label="Image 2", color=color, linestyle="--")
                ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
                self.axes.append(ax)
                self.lines.append((line1, line2))
            axes[-1].set_xlabel("Distance (pixels)", fontsize=AXIS_LABEL_SIZE)
        else:
            ax = self.fig.add_subplot(111)
            ax.set_xlabel("Distance (pixels)", fontsize=AXIS_LABEL_SIZE)
            ax.set_ylabel("Intensity", fontsize=AXIS_LABEL_SIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=AXIS_TICK_SIZE)
            line1, = ax.plot([], [], label="Image 1", color="tab:blue")
            line2, = ax.plot([], [], label="Image 2", color="tab:orange")
            ax.legend(fontsize=LEGEND_FONT_SIZE)
            self.axes = [ax]
            self.lines = [(line1, line2)]
        self.plot_mode = mode
        if hasattr(self, "image1"):
            self.update_plot_title()
        self.canvas_plot.draw_idle()

    def clear_plot_lines(self) -> None:
        for line1, line2 in self.lines:
            line1.set_data([], [])
            line2.set_data([], [])

    def log_offset_for(self, image_state: ImageState) -> float:
        if image_state.data is None:
            return 0.0
        if image_state.data_kind == "int":
            return 1.0
        min_val = float(np.nanmin(image_state.data))
        if not np.isfinite(min_val) or min_val <= 0:
            return 1e-6
        return 0.1 * min_val

    def apply_log_offset(self, profile: np.ndarray | None, image_state: ImageState) -> np.ndarray | None:
        if profile is None:
            return None
        if not self.logscale_var.get():
            return profile
        offset = self.log_offset_for(image_state)
        adjusted = profile + offset
        return np.maximum(adjusted, 1e-12)

    def describe_dtype(self, image_state: ImageState) -> str:
        dtype = image_state.data_dtype
        if dtype is None:
            return "unknown"
        if dtype.kind in {"i", "u"}:
            return f"{dtype.itemsize * 8}-bit int"
        if dtype.kind == "f":
            return f"{dtype.itemsize * 8}-bit float"
        if dtype.kind == "b":
            return "bool"
        return dtype.name

    def image_label(self, image_state: ImageState, fallback: str) -> str:
        if image_state.path:
            return os.path.basename(image_state.path)
        return fallback

    def update_plot_title(self) -> None:
        if self.image1.data is None and self.image2.data is None:
            title = "No image loaded"
        elif self.image2.data is None:
            title = f"Data type: {self.describe_dtype(self.image1)}"
        elif self.image1.data is None:
            title = f"Data type: {self.describe_dtype(self.image2)}"
        else:
            dtype1 = self.describe_dtype(self.image1)
            dtype2 = self.describe_dtype(self.image2)
            if dtype1 == dtype2:
                title = f"Data type: {dtype1}"
            else:
                title = f"Data types: Image 1 {dtype1}, Image 2 {dtype2}"
        self.fig.suptitle(title, fontsize=AXIS_TITLE_SIZE)
        self.canvas_plot.draw_idle()

    def prompt_color_mode(self, is_rgb: bool) -> str:
        if not is_rgb:
            return COLOR_LUMINANCE
        convert = messagebox.askyesno(
            "RGB Image Detected",
            "Convert this image to luminance?\nYes = Luminance, No = Keep RGB channels.",
        )
        return COLOR_LUMINANCE if convert else COLOR_RGB

    def read_image_data(self, path: str) -> tuple[np.ndarray, bool, str, np.dtype]:
        if is_fits_path(path):
            if fits is None:
                raise RuntimeError("astropy is required to open FITS files (pip install astropy)")
            data = fits.getdata(path)
            if data is None:
                raise RuntimeError("FITS file has no data")
            data = normalize_image_array(np.asarray(data))
            is_rgb = data.ndim == 3 and data.shape[2] >= 3
            return data, is_rgb, data_kind_from_array(data), data.dtype
        if is_xisf_path(path):
            if XISF is None:
                raise RuntimeError("xisf is required to open XISF files (pip install xisf)")
            data = normalize_image_array(np.asarray(XISF.read(path)))
            is_rgb = data.ndim == 3 and data.shape[2] >= 3
            return data, is_rgb, data_kind_from_array(data), data.dtype
        with Image.open(path) as img:
            bands = img.getbands()
            is_rgb = len(bands) >= 3
            img = img.convert("RGB")
            data = normalize_image_array(np.asarray(img))
        return data, is_rgb, data_kind_from_array(data), data.dtype

    def prepare_image_data(self, data: np.ndarray, mode: str) -> np.ndarray:
        data = apply_color_mode(data, mode)
        if data.ndim == 3 and data.shape[2] != 3:
            raise RuntimeError("Expected a 3-channel RGB image after conversion")
        if data.ndim not in (2, 3):
            raise RuntimeError("Expected a 2D image or RGB image after conversion")
        data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def registration_view(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            return rgb_to_luminance(data[..., :3])
        return data

    def prepare_registration_arrays(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.image1.data is None or self.image2.data is None:
            return None
        ref = self.registration_view(self.image1.data).astype(np.float32, copy=False)
        mov = self.registration_view(self.image2.data).astype(np.float32, copy=False)
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
        mov = np.nan_to_num(mov, nan=0.0, posinf=0.0, neginf=0.0)
        ref, mov, _changed = crop_pair_to_common(ref, mov)
        return ref, mov

    def crop_images_to_common(self) -> bool:
        if self.image1.data is None or self.image2.data is None:
            return False
        data1, data2, changed = crop_pair_to_common(self.image1.data, self.image2.data)
        if not changed:
            return False
        self.image1.data = data1
        self.image2.data = data2
        return True

    def registration_needs_warning(self, dx: float, dy: float, error: float) -> bool:
        return abs(dx) > REG_SHIFT_WARN or abs(dy) > REG_SHIFT_WARN

    def check_registration(self, warn: bool = False) -> None:
        if self.image1.data is None or self.image2.data is None:
            self.set_registration_info("")
            return
        if phase_cross_correlation is None:
            self.set_registration_info("Registration check unavailable (install scikit-image)")
            return
        prepared = self.prepare_registration_arrays()
        if prepared is None:
            self.set_registration_info("")
            return
        ref, mov = prepared
        try:
            shift, error, _phasediff = phase_cross_correlation(ref, mov, upsample_factor=50)
        except Exception as exc:
            self.set_registration_info(f"Registration check failed: {exc}")
            return
        dy, dx = shift
        self.set_registration_info(f"Reg shift dx={dx:.2f}, dy={dy:.2f}, err={error:.3f}")
        if warn and self.registration_needs_warning(dx, dy, error):
            messagebox.showwarning(
                "Registration Check",
                "Potential registration issues detected.\nConsider using the Register Images button.",
            )

    def sync_image_pair(self, warn_on_misalignment: bool = False) -> None:
        if self.image1.data is None or self.image2.data is None:
            self.set_registration_info("")
            self.update_register_button_state()
            return
        cropped = self.crop_images_to_common()
        if cropped:
            self.reset_selection()
            self.apply_view()
        self.update_register_button_state()
        self.check_registration(warn=warn_on_misalignment)

    def reload_image2_for_mode(self) -> None:
        if self.image2.path is None or self.color_mode is None:
            return
        try:
            raw, _is_rgb, kind, dtype = self.read_image_data(self.image2.path)
            data = self.prepare_image_data(raw, self.color_mode)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to reload Image 2:\n{exc}")
            return
        self.image2.set_data(self.image2.path, data, stretch=self.autostretch_var.get(), zoom=self.zoom, center_norm=self.center_norm)
        self.image2.data_kind = kind
        self.image2.data_dtype = dtype
        self.update_plot_title()

    def load_image(self, which: int) -> None:
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.fits *.fit *.fts *.xisf *.xifs"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Open image", filetypes=filetypes)
        if not path:
            return
        try:
            raw, is_rgb, kind, dtype = self.read_image_data(path)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load image:\n{exc}")
            return
        if which == 1:
            mode = self.prompt_color_mode(is_rgb)
            self.color_mode = mode
            self.configure_plot_axes(mode)
        else:
            if self.color_mode is None:
                mode = self.prompt_color_mode(is_rgb) if is_rgb else COLOR_LUMINANCE
                self.color_mode = mode
                self.configure_plot_axes(mode)
            else:
                mode = self.color_mode
        try:
            data = self.prepare_image_data(raw, mode)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to prepare image:\n{exc}")
            return
        target = self.image1 if which == 1 else self.image2
        target.set_data(path, data, stretch=self.autostretch_var.get(), zoom=self.zoom, center_norm=self.center_norm)
        target.data_kind = kind
        target.data_dtype = dtype
        label = os.path.basename(path)
        if which == 1:
            self.lbl_left.configure(text=label)
            self.btn_load_right.configure(state="normal")
            self.reload_image2_for_mode()
        else:
            self.lbl_right.configure(text=label)
        self.sync_image_pair(warn_on_misalignment=(which == 2))
        self.update_plot()
        self.update_plot_title()
        self.apply_view()

    def _apply_transform(self, transform, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        try:
            result = aa.apply_transform(transform, source, target, fill_value=np.nan)
        except TypeError:
            result = aa.apply_transform(transform, source, target)
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None

    def register_images(self) -> None:
        if self.image1.data is None or self.image2.data is None:
            messagebox.showinfo("Register Images", "Load both images before registering.")
            return
        if aa is None:
            messagebox.showerror("Register Images", "astroalign is required (pip install astroalign).")
            return
        try:
            cropped = self.crop_images_to_common()
            if cropped:
                self.reset_selection()
            ref = self.registration_view(self.image1.data).astype(np.float32, copy=False)
            mov = self.registration_view(self.image2.data).astype(np.float32, copy=False)
            ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
            mov = np.nan_to_num(mov, nan=0.0, posinf=0.0, neginf=0.0)
            if hasattr(aa, "find_transform") and hasattr(aa, "apply_transform"):
                transform, _matched = aa.find_transform(mov, ref)
                if self.image2.data.ndim == 2:
                    aligned, _foot = self._apply_transform(transform, mov, ref)
                    aligned_data = aligned
                else:
                    aligned_channels = []
                    for idx in range(3):
                        aligned_ch, _foot = self._apply_transform(transform, self.image2.data[..., idx], ref)
                        aligned_channels.append(aligned_ch)
                    aligned_data = np.stack(aligned_channels, axis=2)
            else:
                if self.image2.data.ndim == 2:
                    aligned_data, _foot = aa.register(mov, ref)
                else:
                    aligned_channels = []
                    for idx in range(3):
                        target_ch = self.image1.data[..., idx] if self.image1.data.ndim == 3 else ref
                        aligned_ch, _foot = aa.register(self.image2.data[..., idx], target_ch)
                        aligned_channels.append(aligned_ch)
                    aligned_data = np.stack(aligned_channels, axis=2)
            self.image2.data = aligned_data.astype(np.float32, copy=False)
            self.apply_view()
            self.update_plot()
            self.check_registration(warn=True)
        except Exception as exc:
            messagebox.showerror("Register Images", f"Registration failed:\n{exc}")

    def on_left_click(self, event: tk.Event) -> None:
        pt = self.image1.to_image_coords(event.x, event.y)
        if pt is None:
            return
        if self.selection_state == "fixed":
            self.reset_selection()
        if self.selection_state == "idle":
            self.start_pt = pt
            self.end_pt = pt
            self.selection_state = "drawing"
            self.set_status("Move mouse to adjust line, click second point to fix.")
            self.update_line_overlay()
            self.update_plot()
        elif self.selection_state == "drawing":
            self.end_pt = pt
            self.selection_state = "fixed"
            self.set_status("Cross section locked. Click again to start a new one.")
            self.update_line_overlay()
            self.update_plot()

    def on_left_move(self, event: tk.Event) -> None:
        if self.selection_state != "drawing":
            return
        pt = self.image1.to_image_coords(event.x, event.y)
        if pt is None:
            return
        self.end_pt = pt
        self.update_line_overlay()
        self.update_plot()

    def update_line_overlay(self) -> None:
        if self.start_pt is None or self.end_pt is None:
            return
        x0, y0 = self.start_pt
        x1, y1 = self.end_pt
        x0d, y0d = self.image_to_canvas(self.image1, x0, y0)
        x1d, y1d = self.image_to_canvas(self.image1, x1, y1)
        if self.line_id is None:
            self.line_id = self.left_canvas.create_line(
                x0d, y0d, x1d, y1d, fill="#17becf", width=3
            )
        else:
            self.left_canvas.coords(self.line_id, x0d, y0d, x1d, y1d)
            self.left_canvas.itemconfigure(self.line_id, fill="#17becf", width=3)

        if self.image2.data is None:
            if self.line_id_right is not None:
                self.right_canvas.delete(self.line_id_right)
                self.line_id_right = None
            return
        p0 = self.map_point_to_image2(x0, y0)
        p1 = self.map_point_to_image2(x1, y1)
        x0r, y0r = self.image_to_canvas(self.image2, p0[0], p0[1])
        x1r, y1r = self.image_to_canvas(self.image2, p1[0], p1[1])
        if self.line_id_right is None:
            self.line_id_right = self.right_canvas.create_line(
                x0r, y0r, x1r, y1r, fill="#ff7f0e", width=3
            )
        else:
            self.right_canvas.coords(self.line_id_right, x0r, y0r, x1r, y1r)
            self.right_canvas.itemconfigure(self.line_id_right, fill="#ff7f0e", width=3)

    def image_to_canvas(self, image_state: ImageState, x: float, y: float) -> tuple[float, float]:
        x0, y0 = image_state.offset
        return x0 + x * image_state.scale, y0 + y * image_state.scale

    def map_point_to_image2(self, x: float, y: float) -> tuple[float, float]:
        if self.image1.data is None or self.image2.data is None:
            return x, y
        h1, w1 = image_hw(self.image1.data)
        h2, w2 = image_hw(self.image2.data)
        if w1 <= 1 or h1 <= 1:
            return x, y
        sx = (w2 - 1) / (w1 - 1) if w1 > 1 else 1.0
        sy = (h2 - 1) / (h1 - 1) if h1 > 1 else 1.0
        return x * sx, y * sy

    def update_plot(self) -> None:
        if self.start_pt is None or self.end_pt is None:
            self.clear_plot_lines()
            self.canvas_plot.draw_idle()
            self.last_profile = None
            return
        if self.image1.data is None and self.image2.data is None:
            return

        label1 = self.image_label(self.image1, "Image 1")
        label2 = self.image_label(self.image2, "Image 2")

        dist = None
        prof1 = None
        prof2 = None
        plot_prof1 = None
        plot_prof2 = None

        if self.image1.data is not None:
            dist, prof1 = sample_line(self.image1.data, self.start_pt, self.end_pt)
            plot_prof1 = self.apply_log_offset(prof1, self.image1)
            if self.plot_mode == COLOR_RGB:
                for idx, (line1, _line2) in enumerate(self.lines):
                    line1.set_data(dist, plot_prof1[:, idx])
            else:
                self.lines[0][0].set_data(dist, plot_prof1)
        else:
            for line1, _line2 in self.lines:
                line1.set_data([], [])

        if self.image2.data is not None:
            p0 = self.map_point_to_image2(*self.start_pt)
            p1 = self.map_point_to_image2(*self.end_pt)
            dist2, prof2 = sample_line(self.image2.data, p0, p1)
            if dist is None:
                dist = dist2
            plot_prof2 = self.apply_log_offset(prof2, self.image2)
            if self.plot_mode == COLOR_RGB:
                for idx, (_line1, line2) in enumerate(self.lines):
                    line2.set_data(dist2, plot_prof2[:, idx])
            else:
                self.lines[0][1].set_data(dist2, plot_prof2)
        else:
            for _line1, line2 in self.lines:
                line2.set_data([], [])

        if dist is not None:
            for ax in self.axes:
                ax.set_yscale("log" if self.logscale_var.get() else "linear")
                ax.relim()
                ax.autoscale_view()
        legend_loc = "upper right" if self.plot_mode == COLOR_RGB else "best"
        for ax, (line1, line2) in zip(self.axes, self.lines):
            line1.set_label(label1)
            line2.set_label(label2)
            ax.legend(loc=legend_loc, fontsize=LEGEND_FONT_SIZE)
        self.canvas_plot.draw_idle()
        self.last_profile = {
            "distance": dist if dist is not None else np.array([]),
            "image1": prof1 if prof1 is not None else np.array([]),
            "image2": prof2 if prof2 is not None else np.array([]),
            "mode": self.plot_mode,
        }

    def apply_view(self) -> None:
        stretch = self.autostretch_var.get()
        self.image1.redraw(stretch=stretch, zoom=self.zoom, center_norm=self.center_norm)
        self.image2.redraw(stretch=stretch, zoom=self.zoom, center_norm=self.center_norm)
        self.update_line_overlay()

    def update_display_mode(self) -> None:
        self.apply_view()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.center_norm = (0.5, 0.5)
        self.apply_view()

    def on_zoom(self, event: tk.Event, image_state: ImageState) -> None:
        if image_state.data is None:
            return
        pt = image_state.to_image_coords(event.x, event.y)
        if pt is None:
            return
        if event.delta == 0:
            return
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        new_zoom = clamp_value(self.zoom * factor, 0.2, 10.0)
        if new_zoom == self.zoom:
            return
        h, w = image_hw(image_state.data)
        if w < 2 or h < 2:
            return
        canvas_w = max(1, image_state.canvas.winfo_width())
        canvas_h = max(1, image_state.canvas.winfo_height())
        base_scale = min(canvas_w / w, canvas_h / h)
        scale_new = base_scale * new_zoom
        ix, iy = pt
        cx = ix + (canvas_w / 2 - event.x) / scale_new
        cy = iy + (canvas_h / 2 - event.y) / scale_new
        cx = clamp_value(cx, 0.0, w - 1)
        cy = clamp_value(cy, 0.0, h - 1)
        self.zoom = new_zoom
        self.center_norm = (cx / (w - 1), cy / (h - 1))
        self.apply_view()

    def on_pan_start(self, event: tk.Event, image_state: ImageState) -> None:
        if image_state.data is None:
            return
        self.pan_anchor = (event.x, event.y, self.center_norm)
        self.pan_source = image_state

    def on_pan_move(self, event: tk.Event, image_state: ImageState) -> None:
        if self.pan_anchor is None or self.pan_source is not image_state:
            return
        if image_state.data is None:
            return
        h, w = image_hw(image_state.data)
        if w < 2 or h < 2:
            return
        start_x, start_y, start_center = self.pan_anchor
        dx = event.x - start_x
        dy = event.y - start_y
        canvas_w = max(1, image_state.canvas.winfo_width())
        canvas_h = max(1, image_state.canvas.winfo_height())
        base_scale = min(canvas_w / w, canvas_h / h)
        scale = base_scale * self.zoom
        cx0 = start_center[0] * (w - 1)
        cy0 = start_center[1] * (h - 1)
        cx = cx0 - dx / scale
        cy = cy0 - dy / scale
        cx = clamp_value(cx, 0.0, w - 1)
        cy = clamp_value(cy, 0.0, h - 1)
        self.center_norm = (cx / (w - 1), cy / (h - 1))
        self.apply_view()

    def on_pan_end(self, _event: tk.Event) -> None:
        self.pan_anchor = None
        self.pan_source = None

    def update_plot_scale(self) -> None:
        for ax in self.axes:
            ax.set_yscale("log" if self.logscale_var.get() else "linear")
            ax.relim()
            ax.autoscale_view()
        if self.start_pt is None or self.end_pt is None:
            self.canvas_plot.draw_idle()
            return
        self.update_plot()

    def reset_selection(self) -> None:
        self.selection_state = "idle"
        self.start_pt = None
        self.end_pt = None
        self.last_profile = None
        if self.line_id is not None:
            self.left_canvas.delete(self.line_id)
            self.line_id = None
        if self.line_id_right is not None:
            self.right_canvas.delete(self.line_id_right)
            self.line_id_right = None
        self.clear_plot_lines()
        self.canvas_plot.draw_idle()
        self.set_status("Click two points on Image 1 to sample a cross section.")

    def clear_images(self) -> None:
        self.image1.clear()
        self.image2.clear()
        self.lbl_left.configure(text="No file loaded")
        self.lbl_right.configure(text="No file loaded")
        self.color_mode = None
        self.configure_plot_axes(COLOR_LUMINANCE)
        self.reset_selection()
        self.btn_load_right.configure(state="disabled")
        self.set_registration_info("")
        self.update_register_button_state()
        self.update_plot_title()

    def export_csv(self) -> None:
        if not self.last_profile or self.last_profile["distance"].size == 0:
            messagebox.showinfo("Export CSV", "No cross section available yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        dist = self.last_profile["distance"]
        img1 = self.last_profile["image1"]
        img2 = self.last_profile["image2"]
        mode = self.last_profile.get("mode", self.plot_mode)
        label1 = self.image_label(self.image1, "image1").replace(",", "_")
        label2 = self.image_label(self.image2, "image2").replace(",", "_")
        if img1.size == 0 and img2.size == 0:
            messagebox.showinfo("Export CSV", "No cross section available yet.")
            return
        if mode == COLOR_RGB:
            lengths = [dist.size]
            if img1.size:
                lengths.append(img1.shape[0])
            if img2.size:
                lengths.append(img2.shape[0])
            min_len = min(lengths) if lengths else 0
            if min_len == 0:
                messagebox.showinfo("Export CSV", "No cross section available yet.")
                return
            columns = [dist[:min_len]]
            headers = ["distance"]
            if img1.size:
                columns.extend([img1[:min_len, 0], img1[:min_len, 1], img1[:min_len, 2]])
                headers.extend([f"{label1}_r", f"{label1}_g", f"{label1}_b"])
            if img2.size:
                columns.extend([img2[:min_len, 0], img2[:min_len, 1], img2[:min_len, 2]])
                headers.extend([f"{label2}_r", f"{label2}_g", f"{label2}_b"])
            data = np.column_stack(columns)
            header = ",".join(headers)
        else:
            if img2.size == 0:
                data = np.column_stack([dist, img1])
                header = f"distance,{label1}"
            elif img1.size == 0:
                data = np.column_stack([dist, img2])
                header = f"distance,{label2}"
            else:
                min_len = min(dist.size, img1.size, img2.size)
                data = np.column_stack([dist[:min_len], img1[:min_len], img2[:min_len]])
                header = f"distance,{label1},{label2}"
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        messagebox.showinfo("Export CSV", f"Saved cross section to:\n{path}")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = AstroCrossApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
