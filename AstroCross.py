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


def normalize_for_display(data: np.ndarray, stretch: bool = True) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    if stretch:
        lo, hi = np.percentile(finite, (1, 99))
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
    scaled = (data - lo) / (hi - lo)
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
    profile = bilinear_sample(data, xs, ys)
    dist = np.linspace(0.0, length, num)
    return dist, profile


class ImageState:
    def __init__(self, canvas: tk.Canvas, name: str):
        self.canvas = canvas
        self.name = name
        self.path: str | None = None
        self.data: np.ndarray | None = None
        self.display_image: ImageTk.PhotoImage | None = None
        self.scale: float = 1.0
        self.base_scale: float = 1.0
        self.offset: tuple[int, int] = (0, 0)
        self.image_id: int | None = None

    @property
    def shape(self) -> tuple[int, int] | None:
        if self.data is None:
            return None
        return self.data.shape

    def clear(self) -> None:
        self.path = None
        self.data = None
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
        h, w = self.data.shape
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

        disp = normalize_for_display(self.data, stretch=stretch)
        img = Image.fromarray(disp, mode="L").resize((disp_w, disp_h), Image.Resampling.NEAREST)
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
        h, w = self.data.shape
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

        self.lbl_left = ttk.Label(self.left_controls, text="No file loaded", width=100)
        self.lbl_right = ttk.Label(self.right_controls, text="No file loaded", width=100)
        self.lbl_left.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.lbl_right.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.plot_frame = ttk.Frame(self.bottom)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.bottom.rowconfigure(0, weight=1)
        self.bottom.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Distance (pixels)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, alpha=0.3)
        self.line1, = self.ax.plot([], [], label="Image 1", color="tab:blue")
        self.line2, = self.ax.plot([], [], label="Image 2", color="tab:orange")
        self.ax.legend()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        self.controls = ttk.Frame(self.bottom)
        self.controls.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.btn_export = ttk.Button(self.controls, text="Export CSV", command=self.export_csv)
        self.btn_clear = ttk.Button(self.controls, text="Clear Selection", command=self.reset_selection)
        self.autostretch_var = tk.BooleanVar(value=True)
        self.chk_autostretch = ttk.Checkbutton(
            self.controls, text="Auto-stretch display", variable=self.autostretch_var, command=self.update_display_mode
        )
        self.logscale_var = tk.BooleanVar(value=False)
        self.chk_logscale = ttk.Checkbutton(
            self.controls, text="Log scale plot", variable=self.logscale_var, command=self.update_plot_scale
        )
        self.btn_reset_view = ttk.Button(self.controls, text="Reset View", command=self.reset_view)
        self.lbl_status = ttk.Label(self.controls, text="Click two points on Image 1 to sample a cross section.")
        self.btn_export.grid(row=0, column=0, sticky="w")
        self.btn_clear.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.chk_autostretch.grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.chk_logscale.grid(row=0, column=3, sticky="w", padx=(8, 0))
        self.btn_reset_view.grid(row=0, column=4, sticky="w", padx=(8, 0))
        self.lbl_status.grid(row=0, column=5, sticky="w", padx=(16, 0))
        self.controls.columnconfigure(5, weight=1)

        self.image1 = ImageState(self.left_canvas, "Image 1")
        self.image2 = ImageState(self.right_canvas, "Image 2")

        self.zoom = 1.0
        self.center_norm = (0.5, 0.5)
        self.pan_anchor: tuple[int, int, tuple[float, float]] | None = None
        self.pan_source: ImageState | None = None

        self.selection_state = "idle"  # idle -> drawing -> fixed
        self.start_pt: tuple[float, float] | None = None
        self.end_pt: tuple[float, float] | None = None
        self.line_id: int | None = None
        self.line_id_right: int | None = None
        self.last_profile: dict[str, np.ndarray] | None = None

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

    def load_image(self, which: int) -> None:
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.fits *.fit *.fts *.xisf *.xifs"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Open image", filetypes=filetypes)
        if not path:
            return
        try:
            data = self.read_image(path)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load image:\n{exc}")
            return
        target = self.image1 if which == 1 else self.image2
        target.set_data(path, data, stretch=self.autostretch_var.get(), zoom=self.zoom, center_norm=self.center_norm)
        label = os.path.basename(path)
        if which == 1:
            self.lbl_left.configure(text=label)
        else:
            self.lbl_right.configure(text=label)
        self.update_plot()
        self.apply_view()

    def read_image(self, path: str) -> np.ndarray:
        if is_fits_path(path):
            if fits is None:
                raise RuntimeError("astropy is required to open FITS files (pip install astropy)")
            data = fits.getdata(path)
            if data is None:
                raise RuntimeError("FITS file has no data")
            data = np.asarray(data)
            while data.ndim > 2:
                data = data[0]
        elif is_xisf_path(path):
            if XISF is None:
                raise RuntimeError("xisf is required to open XISF files (pip install xisf)")
            data = np.asarray(XISF.read(path))
            if data.ndim == 3 and data.shape[2] >= 3:
                data = rgb_to_luminance(data[..., :3])
            elif data.ndim == 3 and data.shape[2] == 1:
                data = data[..., 0]
            while data.ndim > 2:
                data = data[0]
        else:
            with Image.open(path) as img:
                img = img.convert("RGB")
                arr = np.asarray(img)
                data = rgb_to_luminance(arr)
        if data.ndim != 2:
            raise RuntimeError("Expected a 2D image after conversion")
        data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

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
            self.lbl_status.configure(text="Move mouse to adjust line, click second point to fix.")
            self.update_line_overlay()
            self.update_plot()
        elif self.selection_state == "drawing":
            self.end_pt = pt
            self.selection_state = "fixed"
            self.lbl_status.configure(text="Cross section locked. Click again to start a new one.")
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
        h1, w1 = self.image1.data.shape
        h2, w2 = self.image2.data.shape
        if w1 <= 1 or h1 <= 1:
            return x, y
        sx = (w2 - 1) / (w1 - 1) if w1 > 1 else 1.0
        sy = (h2 - 1) / (h1 - 1) if h1 > 1 else 1.0
        return x * sx, y * sy

    def update_plot(self) -> None:
        if self.start_pt is None or self.end_pt is None:
            self.line1.set_data([], [])
            self.line2.set_data([], [])
            self.canvas_plot.draw_idle()
            self.last_profile = None
            return
        if self.image1.data is None and self.image2.data is None:
            return

        dist = None
        prof1 = None
        prof2 = None

        if self.image1.data is not None:
            dist, prof1 = sample_line(self.image1.data, self.start_pt, self.end_pt)
            self.line1.set_data(dist, prof1)
        else:
            self.line1.set_data([], [])

        if self.image2.data is not None:
            p0 = self.map_point_to_image2(*self.start_pt)
            p1 = self.map_point_to_image2(*self.end_pt)
            dist2, prof2 = sample_line(self.image2.data, p0, p1)
            if dist is None:
                dist = dist2
            self.line2.set_data(dist2, prof2)
        else:
            self.line2.set_data([], [])

        if dist is not None:
            self.ax.set_yscale("log" if self.logscale_var.get() else "linear")
            self.ax.relim()
            self.ax.autoscale_view()
        self.canvas_plot.draw_idle()
        self.last_profile = {
            "distance": dist if dist is not None else np.array([]),
            "image1": prof1 if prof1 is not None else np.array([]),
            "image2": prof2 if prof2 is not None else np.array([]),
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
        h, w = image_state.data.shape
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
        h, w = image_state.data.shape
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
        self.ax.set_yscale("log" if self.logscale_var.get() else "linear")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_plot.draw_idle()

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
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.canvas_plot.draw_idle()
        self.lbl_status.configure(text="Click two points on Image 1 to sample a cross section.")

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
        if img1.size == 0 and img2.size == 0:
            messagebox.showinfo("Export CSV", "No cross section available yet.")
            return
        if img2.size == 0:
            data = np.column_stack([dist, img1])
            header = "distance,image1"
        elif img1.size == 0:
            data = np.column_stack([dist, img2])
            header = "distance,image2"
        else:
            min_len = min(dist.size, img1.size, img2.size)
            data = np.column_stack([dist[:min_len], img1[:min_len], img2[:min_len]])
            header = "distance,image1,image2"
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
