import csv
import io
import math
import os
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class CrossSectionViewer:

    def __init__(self, root: tk.Tk, enable_csv: bool = True) -> None:
        self.root = root
        self.root.title("Cross Section Analysis")
        self.root.geometry("1200x720")
        self.enable_csv = enable_csv
        self.distance: np.ndarray | None = None
        self.series_names: list[str] = []
        self.series_values: list[np.ndarray] = []
        self.data_is_integer = False
        self.table_decimals = 3
        self.bg_index_var = tk.IntVar(value=0)
        self.bg_width_var = tk.IntVar(value=15)
        self.bright_index_var = tk.IntVar(value=0)
        self.bright_width_var = tk.IntVar(value=15)
        self.bg_value_text = tk.StringVar(value="X: -")
        self.bright_value_text = tk.StringVar(value="X: -")
        self.file_label_text = tk.StringVar(value="No data loaded")
        self.y_log_var = tk.BooleanVar(value=False)
        self.axis_log_override: dict[object, bool] = {}
        self._build_layout()
        self._set_controls_state("disabled")
        self._bind_canvas_interactions()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_pane = ttk.Panedwindow(self.root, orient="vertical")
        self.main_pane.grid(row=0, column=0, sticky="nsew")
        top_section = ttk.Frame(self.main_pane)
        top_section.columnconfigure(0, weight=1)
        top_section.rowconfigure(0, weight=1)
        top_section.rowconfigure(1, weight=0)
        top_frame = ttk.Frame(top_section, padding=6)
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.columnconfigure(0, weight=1)
        top_frame.rowconfigure(0, weight=1)
        self.top_fig = Figure(figsize=(6, 3), dpi=100)
        self.top_ax = self.top_fig.add_subplot(111)
        self.top_ax.set_title("Cross Section")
        self.top_ax.set_xlabel("Distance (pixels)")
        self.top_ax.set_ylabel("Value")
        self.top_canvas = FigureCanvasTkAgg(self.top_fig, master=top_frame)
        self.top_canvas.draw()
        self.top_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        controls = ttk.Frame(top_section, padding=(6, 4))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=0)
        col = 0
        if self.enable_csv:
            load_button = ttk.Button(controls, text="Load CSV", command=self.load_csv)
            load_button.grid(row=0, column=col, sticky="w")
            col += 1
        file_label = ttk.Label(controls, textvariable=self.file_label_text)
        file_label.grid(row=0, column=col, sticky="w", padx=(8, 8))
        col += 1
        y_log_check = ttk.Checkbutton(
            controls,
            text="Y log scale",
            variable=self.y_log_var,
            command=self._on_log_toggle,
        )
        y_log_check.grid(row=0, column=col, sticky="e")
        bg_label = ttk.Label(controls, text="Background")
        bg_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.bg_scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            resolution=1,
            showvalue=0,
            variable=self.bg_index_var,
            command=self._on_bg_slider,
        )
        self.bg_scale.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.bg_value_label = ttk.Label(controls, textvariable=self.bg_value_text, width=12)
        self.bg_value_label.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=(6, 0))
        bg_width_label = ttk.Label(controls, text="Width")
        bg_width_label.grid(row=1, column=3, sticky="e", padx=(8, 0), pady=(6, 0))
        self.bg_width_spin = ttk.Spinbox(
            controls,
            from_=1,
            to=1,
            textvariable=self.bg_width_var,
            width=6,
            command=self._on_width_change,
        )
        self.bg_width_spin.grid(row=1, column=4, sticky="e", pady=(6, 0))
        bright_label = ttk.Label(controls, text="Bright Region")
        bright_label.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.bright_scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            resolution=1,
            showvalue=0,
            variable=self.bright_index_var,
            command=self._on_bright_slider,
        )
        self.bright_scale.grid(row=2, column=1, sticky="ew", pady=(4, 0))
        self.bright_value_label = ttk.Label(controls, textvariable=self.bright_value_text, width=12)
        self.bright_value_label.grid(row=2, column=2, sticky="e", padx=(6, 0), pady=(4, 0))
        bright_width_label = ttk.Label(controls, text="Width")
        bright_width_label.grid(row=2, column=3, sticky="e", padx=(8, 0), pady=(4, 0))
        self.bright_width_spin = ttk.Spinbox(
            controls,
            from_=1,
            to=1,
            textvariable=self.bright_width_var,
            width=6,
            command=self._on_width_change,
        )
        self.bright_width_spin.grid(row=2, column=4, sticky="e", pady=(4, 0))
        self.bg_width_var.trace_add("write", self._on_width_trace)
        self.bright_width_var.trace_add("write", self._on_width_trace)
        bottom_section = ttk.Frame(self.main_pane)
        bottom_section.columnconfigure(0, weight=1)
        bottom_section.rowconfigure(0, weight=1)
        self.bottom_pane = ttk.Panedwindow(bottom_section, orient="horizontal")
        self.bottom_pane.grid(row=0, column=0, sticky="nsew")
        self.left_pane = ttk.Panedwindow(self.bottom_pane, orient="vertical")
        self.top_row_pane = ttk.Panedwindow(self.left_pane, orient="horizontal")
        self.bottom_row_pane = ttk.Panedwindow(self.left_pane, orient="horizontal")
        bg_sub_frame = ttk.Frame(self.top_row_pane)
        bg_sub_frame.columnconfigure(0, weight=1)
        bg_sub_frame.rowconfigure(0, weight=1)
        self.bottom_fig = Figure(figsize=(5.2, 2.5), dpi=100)
        self.bottom_fig.subplots_adjust(left=0.2)
        self.bottom_ax = self.bottom_fig.add_subplot(111)
        self.bottom_ax.set_title("Background Subtracted")
        self.bottom_ax.set_xlabel("Distance (pixels)")
        self.bottom_ax.set_ylabel("Value - Background")
        self.bottom_canvas = FigureCanvasTkAgg(self.bottom_fig, master=bg_sub_frame)
        self.bottom_canvas.draw()
        self.bottom_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ratio_frame = ttk.Frame(self.top_row_pane)
        ratio_frame.columnconfigure(0, weight=1)
        ratio_frame.rowconfigure(0, weight=1)
        self.ratio_fig = Figure(figsize=(5.2, 2.5), dpi=100)
        self.ratio_fig.subplots_adjust(left=0.2)
        self.ratio_ax = self.ratio_fig.add_subplot(111)
        self.ratio_ax.set_title("Abs Ratio (BG Subtracted)")
        self.ratio_ax.set_xlabel("Distance (pixels)")
        self.ratio_ax.set_ylabel("Abs Ratio")
        self.ratio_canvas = FigureCanvasTkAgg(self.ratio_fig, master=ratio_frame)
        self.ratio_canvas.draw()
        self.ratio_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        diff_orig_frame = ttk.Frame(self.bottom_row_pane)
        diff_orig_frame.columnconfigure(0, weight=1)
        diff_orig_frame.rowconfigure(0, weight=1)
        self.diff_orig_fig = Figure(figsize=(5.2, 2.5), dpi=100)
        self.diff_orig_fig.subplots_adjust(left=0.2)
        self.diff_orig_ax = self.diff_orig_fig.add_subplot(111)
        self.diff_orig_ax.set_title("Original Differences")
        self.diff_orig_ax.set_xlabel("Distance (pixels)")
        self.diff_orig_ax.set_ylabel("Value Difference")
        self.diff_orig_canvas = FigureCanvasTkAgg(self.diff_orig_fig, master=diff_orig_frame)
        self.diff_orig_canvas.draw()
        self.diff_orig_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        diff_bg_frame = ttk.Frame(self.bottom_row_pane)
        diff_bg_frame.columnconfigure(0, weight=1)
        diff_bg_frame.rowconfigure(0, weight=1)
        self.diff_bg_fig = Figure(figsize=(5.2, 2.5), dpi=100)
        self.diff_bg_fig.subplots_adjust(left=0.2)
        self.diff_bg_ax = self.diff_bg_fig.add_subplot(111)
        self.diff_bg_ax.set_title("BG-Subtracted Differences")
        self.diff_bg_ax.set_xlabel("Distance (pixels)")
        self.diff_bg_ax.set_ylabel("Value Difference")
        self.diff_bg_canvas = FigureCanvasTkAgg(self.diff_bg_fig, master=diff_bg_frame)
        self.diff_bg_canvas.draw()
        self.diff_bg_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.top_row_pane.add(bg_sub_frame, weight=1)
        self.top_row_pane.add(ratio_frame, weight=1)
        self.bottom_row_pane.add(diff_bg_frame, weight=1)
        self.bottom_row_pane.add(diff_orig_frame, weight=1)
        self.left_pane.add(self.top_row_pane, weight=1)
        self.left_pane.add(self.bottom_row_pane, weight=1)
        table_frame = ttk.Frame(self.bottom_pane)
        table_frame.rowconfigure(1, weight=1)
        table_frame.rowconfigure(4, weight=1)
        table_frame.columnconfigure(0, weight=1)
        table_label = ttk.Label(table_frame, text="Sample Values")
        table_label.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        self.summary_table = ttk.Treeview(table_frame, columns=(), show="headings", height=6)
        self.summary_table.grid(row=1, column=0, sticky="nsew")
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.summary_table.yview)
        table_scroll.grid(row=1, column=1, sticky="ns")
        table_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.summary_table.xview)
        table_scroll_x.grid(row=2, column=0, sticky="ew")
        self.summary_table.configure(
            yscrollcommand=table_scroll.set,
            xscrollcommand=table_scroll_x.set,
        )
        summary_text_label = ttk.Label(table_frame, text="Analysis")
        summary_text_label.grid(row=3, column=0, sticky="w", padx=4, pady=(6, 2))
        self.summary_text = tk.Text(table_frame, height=5, wrap="none")
        self.summary_text.grid(row=4, column=0, sticky="nsew")
        summary_text_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.summary_text.yview)
        summary_text_scroll.grid(row=4, column=1, sticky="ns")
        summary_text_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.summary_text.xview)
        summary_text_scroll_x.grid(row=5, column=0, sticky="ew")
        self.summary_text.configure(
            yscrollcommand=summary_text_scroll.set,
            xscrollcommand=summary_text_scroll_x.set,
        )
        self.summary_text.configure(state="disabled")
        self.bottom_pane.add(self.left_pane, weight=3)
        self.bottom_pane.add(table_frame, weight=1)
        self.main_pane.add(top_section, weight=3)
        self.main_pane.add(bottom_section, weight=2)
        self.root.after(100, self._set_pane_defaults)

    def _set_controls_state(self, state: str) -> None:
        for widget in (
            self.bg_scale,
            self.bright_scale,
            self.bg_width_spin,
            self.bright_width_spin,
            self.bg_value_label,
            self.bright_value_label,
        ):
            widget.configure(state=state)

    def _bind_canvas_interactions(self) -> None:
        self._drag_state: dict[str, object] | None = None
        canvases = [
            self.top_canvas,
            self.bottom_canvas,
            self.ratio_canvas,
            self.diff_orig_canvas,
            self.diff_bg_canvas,
        ]
        for canvas in canvases:
            canvas.mpl_connect("button_press_event", self._on_plot_press)
            canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
            canvas.mpl_connect("button_release_event", self._on_plot_release)
            canvas.get_tk_widget().bind(
                "<Button-3>",
                lambda event, c=canvas: self._show_canvas_menu(event, c),
            )

    def _show_canvas_menu(self, event: tk.Event, canvas: FigureCanvasTkAgg) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Copy Graph",
            command=lambda: self._copy_graph_to_clipboard(canvas.figure),
        )
        menu.add_command(
            label="Reset Axes",
            command=lambda: self._reset_axes(canvas.figure),
        )
        menu.add_command(
            label="Synchronize X Axis",
            command=lambda: self._sync_x_axes(canvas.figure),
        )
        menu.add_command(
            label="Toggle Y Log Scale",
            command=lambda: self._toggle_y_log_scale(canvas.figure),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _reset_axes(self, figure: Figure) -> None:
        for ax in figure.axes:
            ax.relim()
            ax.autoscale()
        figure.canvas.draw_idle()

    def _toggle_y_log_scale(self, figure: Figure) -> None:
        if not figure.axes:
            return
        for ax in figure.axes:
            current = self._axis_use_log(ax)
            self.axis_log_override[ax] = not current
        self._refresh_plots()
        for ax in figure.axes:
            ax.relim()
            ax.autoscale()
            ax.figure.canvas.draw_idle()

    def _axis_use_log(self, ax) -> bool:
        if ax in self.axis_log_override:
            return bool(self.axis_log_override[ax])
        return bool(self.y_log_var.get())

    def _sync_x_axes(self, figure: Figure) -> None:
        if not figure.axes:
            return
        ref_ax = figure.axes[0]
        xlim = ref_ax.get_xlim()
        for ax in self._all_axes():
            ax.set_xlim(xlim)
            ax.figure.canvas.draw_idle()

    def _all_axes(self) -> list:
        return [
            self.top_ax,
            self.bottom_ax,
            self.ratio_ax,
            self.diff_orig_ax,
            self.diff_bg_ax,
        ]

    def _copy_graph_to_clipboard(self, figure: Figure) -> None:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        png_bytes = buffer.getvalue()
        if self._copy_image_windows(png_bytes):
            return
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as handle:
                handle.write(png_bytes)
                temp_path = handle.name
            self.root.clipboard_clear()
            self.root.clipboard_append(temp_path)
            messagebox.showinfo(
                "Copy Graph",
                "Image saved and path copied to clipboard:\n"
                f"{temp_path}\n\n"
                "Install pywin32 to copy images directly to the clipboard.",
            )
        except Exception as exc:
            messagebox.showerror("Copy Graph", f"Unable to copy graph: {exc}")

    def _copy_image_windows(self, png_bytes: bytes) -> bool:
        try:
            import win32clipboard
            from PIL import Image
        except Exception:
            return False
        try:
            image = Image.open(io.BytesIO(png_bytes))
            output = io.BytesIO()
            image.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
            return True
        except Exception:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
            return False

    def _on_plot_press(self, event) -> None:
        if event.inaxes is None or event.button != 1:
            return
        ax = event.inaxes
        bbox = ax.bbox
        pad = 25
        axis_mode = None
        zone = None
        if event.x < bbox.x0 + pad:
            axis_mode = "y"
            rel = (event.y - bbox.y0) / max(bbox.height, 1.0)
            if rel > 0.66:
                zone = "top"
            elif rel < 0.33:
                zone = "bottom"
            else:
                zone = "middle"
        elif event.y < bbox.y0 + pad:
            axis_mode = "x"
            rel = (event.x - bbox.x0) / max(bbox.width, 1.0)
            if rel > 0.66:
                zone = "right"
            elif rel < 0.33:
                zone = "left"
            else:
                zone = "middle"
        if axis_mode is None:
            return
        self._drag_state = {
            "ax": ax,
            "mode": axis_mode,
            "zone": zone,
            "start_xlim": ax.get_xlim(),
            "start_ylim": ax.get_ylim(),
            "start_xdata": event.xdata,
            "start_ydata": event.ydata,
        }

    def _on_plot_motion(self, event) -> None:
        if not self._drag_state or event.inaxes is None:
            return
        ax = self._drag_state["ax"]
        if ax is not event.inaxes:
            return
        mode = self._drag_state["mode"]
        zone = self._drag_state["zone"]
        start_xlim = self._drag_state["start_xlim"]
        start_ylim = self._drag_state["start_ylim"]
        start_x = self._drag_state["start_xdata"]
        start_y = self._drag_state["start_ydata"]
        if mode == "y":
            if start_y is None or event.ydata is None:
                return
            dy = event.ydata - start_y
            ymin, ymax = start_ylim
            if zone == "top":
                ymax = ymax + dy
            elif zone == "bottom":
                ymin = ymin + dy
            else:
                ymin = ymin + dy
                ymax = ymax + dy
            ymin, ymax = self._sanitize_limits(ax, ymin, ymax, axis="y")
            ax.set_ylim(ymin, ymax)
        elif mode == "x":
            if start_x is None or event.xdata is None:
                return
            dx = event.xdata - start_x
            xmin, xmax = start_xlim
            if zone == "right":
                xmax = xmax + dx
            elif zone == "left":
                xmin = xmin + dx
            else:
                xmin = xmin + dx
                xmax = xmax + dx
            xmin, xmax = self._sanitize_limits(ax, xmin, xmax, axis="x")
            ax.set_xlim(xmin, xmax)
        event.canvas.draw_idle()

    def _on_plot_release(self, _event) -> None:
        self._drag_state = None

    def _sanitize_limits(self, ax, lo: float, hi: float, axis: str) -> tuple[float, float]:
        if lo == hi:
            hi = lo + 1.0
        if lo > hi:
            lo, hi = hi, lo
        if axis == "y" and ax.get_yscale() == "log":
            min_pos = 1e-12
            if hi <= min_pos:
                hi = min_pos * 10.0
            if lo <= min_pos:
                lo = min_pos
            if lo >= hi:
                lo = hi / 10.0
        if axis == "x" and ax.get_xscale() == "log":
            min_pos = 1e-12
            if hi <= min_pos:
                hi = min_pos * 10.0
            if lo <= min_pos:
                lo = min_pos
            if lo >= hi:
                lo = hi / 10.0
        return lo, hi

    def _set_pane_defaults(self) -> None:
        try:
            self.root.update_idletasks()
            total_width = self.bottom_pane.winfo_width()
            if total_width > 0:
                self.bottom_pane.sashpos(0, int(total_width * 0.67))
                top_row_width = self.top_row_pane.winfo_width()
                bottom_row_width = self.bottom_row_pane.winfo_width()
                if top_row_width > 0:
                    self.top_row_pane.sashpos(0, int(top_row_width * 0.5))
                if bottom_row_width > 0:
                    self.bottom_row_pane.sashpos(0, int(bottom_row_width * 0.5))
            left_height = self.left_pane.winfo_height()
            if left_height > 0:
                self.left_pane.sashpos(0, int(left_height * 0.5))
            total_height = self.main_pane.winfo_height()
            if total_height > 0:
                self.main_pane.sashpos(0, int(total_height * 0.33))
        except Exception:
            pass

    def _on_bg_slider(self, value: str) -> None:
        if self.distance is None:
            return
        idx = int(round(float(value)))
        idx = max(0, min(idx, self.distance.size - 1))
        self.bg_index_var.set(idx)
        self._refresh_plots()

    def _on_bright_slider(self, value: str) -> None:
        if self.distance is None:
            return
        idx = int(round(float(value)))
        idx = max(0, min(idx, self.distance.size - 1))
        self.bright_index_var.set(idx)
        self._refresh_plots()

    def _on_width_change(self) -> None:
        self._refresh_plots()

    def _on_width_trace(self, *_: object) -> None:
        self._refresh_plots()

    def _on_log_toggle(self) -> None:
        self.axis_log_override.clear()
        self._refresh_plots()

    def _set_default_indices(self) -> None:
        if self.distance is None or not self.series_values:
            self.bg_index_var.set(0)
            self.bright_index_var.set(0)
            return
        values = np.vstack(self.series_values)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            self.bg_index_var.set(0)
            self.bright_index_var.set(0)
            return
        safe_values = np.where(finite_mask, values, np.nan)
        flat_min = int(np.nanargmin(safe_values))
        flat_max = int(np.nanargmax(safe_values))
        count = self.distance.size
        bg_index = flat_min % count
        bright_index = flat_max % count
        self.bg_index_var.set(bg_index)
        self.bright_index_var.set(bright_index)
        self.bg_scale.set(bg_index)
        self.bright_scale.set(bright_index)

    def load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Open cross section CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            distance, names, values = self._read_cross_section_csv(path)
        except Exception as exc:
            messagebox.showerror("CSV Load Error", str(exc))
            return
        if distance.size == 0 or not values:
            messagebox.showinfo("CSV Load Error", "No data found in CSV.")
            return
        self.distance = distance
        self.series_names = names
        self.series_values = values
        self.file_label_text.set(os.path.basename(path))
        self._configure_table()
        self._compute_data_type()
        max_index = max(self.distance.size - 1, 0)
        for scale in (self.bg_scale, self.bright_scale):
            scale.configure(from_=0, to=max_index)
            scale.set(0)
        max_width = max(self.distance.size, 1)
        self.bg_width_spin.configure(from_=1, to=max_width)
        self.bright_width_spin.configure(from_=1, to=max_width)
        self.bg_width_var.set(min(self.bg_width_var.get(), max_width))
        self.bright_width_var.set(min(self.bright_width_var.get(), max_width))
        self._set_controls_state("normal")
        self._set_default_indices()
        self._refresh_plots()

    def _read_cross_section_csv(
        self, path: str
    ) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty.")
            if not header:
                raise ValueError("CSV header is empty.")
            rows: list[list[float]] = []
            for row in reader:
                if not row:
                    continue
                if len(row) < len(header):
                    continue
                try:
                    rows.append([float(val) for val in row[: len(header)]])
                except ValueError:
                    continue
        if not rows:
            raise ValueError("CSV file does not contain numeric data.")
        data = np.array(rows, dtype=np.float64)
        header_lower = [name.strip().lower() for name in header]
        if "distance" in header_lower:
            dist_idx = header_lower.index("distance")
        else:
            dist_idx = 0
        distance = data[:, dist_idx]
        names: list[str] = []
        values: list[np.ndarray] = []
        for idx, name in enumerate(header):
            if idx == dist_idx:
                continue
            names.append(name.strip() or f"series_{idx}")
            values.append(data[:, idx])
        if not names:
            raise ValueError("No cross section columns found in CSV.")
        return distance, names, values

    def load_data(
        self,
        distance: np.ndarray,
        names: list[str],
        values: list[np.ndarray],
        source_label: str | None = None,
    ) -> None:
        if distance is None or len(values) == 0:
            messagebox.showinfo("Analyze", "No cross section data available.")
            return
        dist = np.asarray(distance, dtype=np.float64)
        series_values = [np.asarray(v, dtype=np.float64) for v in values]
        lengths = [dist.size] + [v.size for v in series_values if v.size]
        min_len = min(lengths) if lengths else 0
        if min_len <= 0:
            messagebox.showinfo("Analyze", "No cross section data available.")
            return
        dist = dist[:min_len]
        series_values = [v[:min_len] for v in series_values]
        self.distance = dist
        self.series_names = list(names)
        self.series_values = series_values
        if source_label:
            self.file_label_text.set(source_label)
        else:
            self.file_label_text.set("AstroCross selection")
        self._configure_table()
        self._compute_data_type()
        max_index = max(self.distance.size - 1, 0)
        for scale in (self.bg_scale, self.bright_scale):
            scale.configure(from_=0, to=max_index)
            scale.set(0)
        max_width = max(self.distance.size, 1)
        self.bg_width_spin.configure(from_=1, to=max_width)
        self.bright_width_spin.configure(from_=1, to=max_width)
        self.bg_width_var.set(min(self.bg_width_var.get(), max_width))
        self.bright_width_var.set(min(self.bright_width_var.get(), max_width))
        self._set_controls_state("normal")
        self._set_default_indices()
        self._refresh_plots()

    def _current_width(self, var: tk.IntVar, count: int) -> int:
        try:
            width = int(var.get())
        except (TypeError, ValueError):
            width = 1
        width = max(1, width)
        return min(width, max(count, 1))

    def _selection_bounds(self, index: int, width: int, count: int) -> tuple[int, int]:
        if count <= 0:
            return 0, 0
        width = max(1, min(width, count))
        left = width // 2
        right = width - left
        start = index - left
        end = index + right
        if start < 0:
            end = min(count, end - start)
            start = 0
        if end > count:
            start = max(0, start - (end - count))
            end = count
        return start, end

    def _configure_table(self) -> None:
        columns = ["Output", "Signal Type", "Mean", "Std Dev", "CV"]
        self.summary_table.configure(columns=columns, show="headings")
        for col in columns:
            self.summary_table.heading(col, text=col)
            if col == "Output":
                width = 210
            elif col == "Signal Type":
                width = 95
            else:
                width = 110
            self.summary_table.column(col, width=width, anchor="center")
        for item in self.summary_table.get_children():
            self.summary_table.delete(item)

    def _compute_data_type(self) -> None:
        if self.distance is None or not self.series_values:
            self.data_is_integer = False
            self.table_decimals = 3
            return
        values = np.column_stack(self.series_values)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            self.data_is_integer = False
            self.table_decimals = 3
            return
        is_integer = np.all(np.isclose(finite, np.round(finite)))
        self.data_is_integer = bool(is_integer)
        self.table_decimals = 0 if self.data_is_integer else 6

    def _refresh_plots(self) -> None:
        if self.distance is None or not self.series_values:
            return
        count = self.distance.size
        bg_index = max(0, min(self.bg_index_var.get(), count - 1))
        bright_index = max(0, min(self.bright_index_var.get(), count - 1))
        bg_width = self._current_width(self.bg_width_var, count)
        bright_width = self._current_width(self.bright_width_var, count)
        self.bg_value_text.set(f"X: {self.distance[bg_index]:.2f}")
        self.bright_value_text.set(f"X: {self.distance[bright_index]:.2f}")
        bg_start, bg_end = self._selection_bounds(bg_index, bg_width, count)
        bright_start, bright_end = self._selection_bounds(bright_index, bright_width, count)
        use_log_top = self._axis_use_log(self.top_ax)
        use_log_bottom = self._axis_use_log(self.bottom_ax)
        use_log_ratio = self._axis_use_log(self.ratio_ax)
        use_log_diff_orig = self._axis_use_log(self.diff_orig_ax)
        use_log_diff_bg = self._axis_use_log(self.diff_bg_ax)

        def safe_mean(sample: np.ndarray) -> float:
            if sample.size == 0:
                return float("nan")
            return float(np.nanmean(sample))
        self.top_ax.clear()
        top_has_positive = False
        for name, values in zip(self.series_names, self.series_values):
            plot_values = np.asarray(values, dtype=np.float64)
            if use_log_top:
                valid = np.isfinite(plot_values) & (plot_values > 0)
                top_has_positive = top_has_positive or bool(np.any(valid))
                plot_values = np.where(valid, plot_values, np.nan)
            self.top_ax.plot(self.distance, plot_values, label=name, linewidth=1.2, alpha=0.6)
        if bg_end > bg_start:
            left = self.distance[bg_start]
            right = self.distance[bg_end - 1]
            self.top_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
            self.top_ax.axvspan(left, right, color="black", alpha=0.2)
        if bright_end > bright_start:
            left = self.distance[bright_start]
            right = self.distance[bright_end - 1]
            self.top_ax.axvline(self.distance[bright_index], color="magenta", linewidth=1.2)
            self.top_ax.axvspan(left, right, color="#ffb84d", alpha=0.25)
        self.top_ax.set_title("Cross Section")
        self.top_ax.set_xlabel("Distance (pixels)")
        self.top_ax.set_ylabel("Value")
        self.top_ax.legend(loc="best", fontsize=9)
        self.top_ax.grid(True, alpha=0.3)
        if use_log_top and top_has_positive:
            self.top_ax.set_yscale("log")
        else:
            self.top_ax.set_yscale("linear")
        background_offsets: list[float] = []
        raw_bg_means: list[float] = []
        raw_bright_means: list[float] = []
        for values in self.series_values:
            bg_sample = values[bg_start:bg_end]
            bright_sample = values[bright_start:bright_end]
            bg_mean = safe_mean(bg_sample)
            bright_mean = safe_mean(bright_sample)
            raw_bg_means.append(bg_mean)
            raw_bright_means.append(bright_mean)
            background_offsets.append(bg_mean if np.isfinite(bg_mean) else 0.0)
        bg_sub_series: list[np.ndarray] = []
        for values, offset in zip(self.series_values, background_offsets):
            bg_sub_series.append(values - offset)
        self.bottom_ax.clear()
        bottom_has_positive = False
        for name, values in zip(self.series_names, bg_sub_series):
            plot_values = np.asarray(values, dtype=np.float64)
            if use_log_bottom:
                valid = np.isfinite(plot_values) & (plot_values > 0)
                bottom_has_positive = bottom_has_positive or bool(np.any(valid))
                plot_values = np.where(valid, plot_values, np.nan)
            self.bottom_ax.plot(
                self.distance,
                plot_values,
                label=name,
                linewidth=1.2,
                alpha=0.6,
            )
        self.bottom_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
        self.bottom_ax.axvline(self.distance[bright_index], color="magenta", linewidth=1.2)
        self.bottom_ax.set_title("Background Subtracted")
        self.bottom_ax.set_xlabel("Distance (pixels)")
        self.bottom_ax.set_ylabel("Value - Background")
        self.bottom_ax.legend(loc="best", fontsize=9)
        self.bottom_ax.grid(True, alpha=0.3)
        if use_log_bottom and bottom_has_positive:
            self.bottom_ax.set_yscale("log")
        else:
            self.bottom_ax.set_yscale("linear")
        self.ratio_ax.clear()
        ratio_series: list[np.ndarray] = []
        ratio_has_positive = False
        if len(bg_sub_series) > 1:
            baseline = bg_sub_series[0]
            ratio = np.abs(
                np.divide(
                    bg_sub_series[1],
                    baseline,
                    out=np.full_like(bg_sub_series[1], np.nan),
                    where=baseline != 0,
                )
            )
            ratio_series.append(ratio)
            plot_values = ratio
            if use_log_ratio:
                valid = np.isfinite(plot_values) & (plot_values > 0)
                ratio_has_positive = ratio_has_positive or bool(np.any(valid))
                plot_values = np.where(valid, plot_values, np.nan)
            self.ratio_ax.plot(
                self.distance,
                plot_values,
                label="Abs ratio (BG subtracted)",
                linewidth=1.2,
            )
        self.ratio_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
        self.ratio_ax.axvline(self.distance[bright_index], color="magenta", linewidth=1.2)
        self.ratio_ax.set_title("")
        self.ratio_ax.set_xlabel("Distance (pixels)")
        self.ratio_ax.set_ylabel("Abs Ratio")
        if ratio_series:
            self.ratio_ax.legend(loc="best", fontsize=9)
        self.ratio_ax.grid(True, alpha=0.3)
        if use_log_ratio and ratio_has_positive:
            self.ratio_ax.set_yscale("log")
        else:
            self.ratio_ax.set_yscale("linear")
        self.diff_orig_ax.clear()
        diff_orig_has_positive = False
        if len(self.series_values) > 1:
            baseline = self.series_values[0]
            diff = self.series_values[1] - baseline
            plot_values = np.asarray(diff, dtype=np.float64)
            if use_log_diff_orig:
                valid = np.isfinite(plot_values) & (plot_values > 0)
                diff_orig_has_positive = diff_orig_has_positive or bool(np.any(valid))
                plot_values = np.where(valid, plot_values, np.nan)
            self.diff_orig_ax.plot(
                self.distance,
                plot_values,
                label="Original Differences",
                linewidth=1.2,
            )
        self.diff_orig_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
        self.diff_orig_ax.axvline(self.distance[bright_index], color="magenta", linewidth=1.2)
        self.diff_orig_ax.set_title("")
        self.diff_orig_ax.set_xlabel("Distance (pixels)")
        self.diff_orig_ax.set_ylabel("Value Difference")
        if len(self.series_values) > 1:
            self.diff_orig_ax.legend(loc="best", fontsize=9)
        self.diff_orig_ax.grid(True, alpha=0.3)
        if use_log_diff_orig and diff_orig_has_positive:
            self.diff_orig_ax.set_yscale("log")
        else:
            self.diff_orig_ax.set_yscale("linear")
        self.diff_bg_ax.clear()
        diff_bg_has_positive = False
        if len(bg_sub_series) > 1:
            baseline = bg_sub_series[0]
            diff = bg_sub_series[1] - baseline
            plot_values = np.asarray(diff, dtype=np.float64)
            if use_log_diff_bg:
                valid = np.isfinite(plot_values) & (plot_values > 0)
                diff_bg_has_positive = diff_bg_has_positive or bool(np.any(valid))
                plot_values = np.where(valid, plot_values, np.nan)
            self.diff_bg_ax.plot(
                self.distance,
                plot_values,
                label="BG-subtracted differences",
                linewidth=1.2,
            )
        self.diff_bg_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
        self.diff_bg_ax.axvline(self.distance[bright_index], color="magenta", linewidth=1.2)
        self.diff_bg_ax.set_title("")
        self.diff_bg_ax.set_xlabel("Distance (pixels)")
        self.diff_bg_ax.set_ylabel("Value Difference")
        if len(bg_sub_series) > 1:
            self.diff_bg_ax.legend(loc="best", fontsize=9)
        self.diff_bg_ax.grid(True, alpha=0.3)
        if use_log_diff_bg and diff_bg_has_positive:
            self.diff_bg_ax.set_yscale("log")
        else:
            self.diff_bg_ax.set_yscale("linear")
        self._update_table(
            raw_bg_means,
            raw_bright_means,
            background_offsets,
            bg_sub_series,
            ratio_series,
            bg_start,
            bg_end,
            bright_start,
            bright_end,
        )
        self.top_canvas.draw_idle()
        self.bottom_canvas.draw_idle()
        self.ratio_canvas.draw_idle()
        self.diff_orig_canvas.draw_idle()
        self.diff_bg_canvas.draw_idle()

    def _update_table(
        self,
        raw_bg_means: list[float],
        raw_bright_means: list[float],
        background_offsets: list[float],
        bg_sub_series: list[np.ndarray],
        ratio_series: list[np.ndarray],
        bg_start: int,
        bg_end: int,
        bright_start: int,
        bright_end: int,
    ) -> None:
        if not self.series_names:
            return

        def fmt(value: float) -> str:
            if value is None or not np.isfinite(value):
                return "-"
            if self.table_decimals <= 0:
                return f"{value:.0f}"
            return f"{value:.{self.table_decimals}f}"

        def mean_std(series: np.ndarray, start: int, end: int) -> tuple[float, float]:
            if series.size == 0:
                return float("nan"), float("nan")
            sample = series[start:end]
            if sample.size == 0:
                return float("nan"), float("nan")
            return float(np.nanmean(sample)), float(np.nanstd(sample))
        for item in self.summary_table.get_children():
            self.summary_table.delete(item)
        rows: list[tuple[str, str, float, float]] = []
        name1 = self.series_names[0]
        name2 = self.series_names[1] if len(self.series_names) > 1 else None
        cs1 = self.series_values[0]
        cs2 = self.series_values[1] if len(self.series_values) > 1 else None
        cs1_bg_mean, cs1_bg_std = mean_std(cs1, bg_start, bg_end)
        cs1_br_mean, cs1_br_std = mean_std(cs1, bright_start, bright_end)
        rows.append((f"{name1} signal", "Background", cs1_bg_mean, cs1_bg_std))
        rows.append((f"{name1} signal", "Bright", cs1_br_mean, cs1_br_std))
        if cs2 is not None and name2 is not None:
            cs2_bg_mean, cs2_bg_std = mean_std(cs2, bg_start, bg_end)
            cs2_br_mean, cs2_br_std = mean_std(cs2, bright_start, bright_end)
            rows.append((f"{name2} signal", "Background", cs2_bg_mean, cs2_bg_std))
            rows.append((f"{name2} signal", "Bright", cs2_br_mean, cs2_br_std))
        else:
            cs2_bg_mean = cs2_bg_std = cs2_br_mean = cs2_br_std = float("nan")
        bright_width = max(bright_end - bright_start, 1)
        dark_width = max(bg_end - bg_start, 1)

        def snr_value(bright_mean: float, dark_mean: float, width_bright: int, width_dark: int) -> float:
            if width_bright <= 0 or width_dark <= 0:
                return float("nan")
            if bright_mean is None or dark_mean is None:
                return float("nan")
            if not np.isfinite(bright_mean) or not np.isfinite(dark_mean):
                return float("nan")
            denom_term = (bright_mean / width_bright) + (dark_mean / width_dark)
            if denom_term <= 0:
                return float("nan")
            return (bright_mean - dark_mean) / math.sqrt(denom_term)

        def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int, int] | None:
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            n1 = int(a.size)
            n2 = int(b.size)
            if n1 < 2 or n2 < 2:
                return None
            mean1 = float(np.mean(a))
            mean2 = float(np.mean(b))
            var1 = float(np.var(a, ddof=1))
            var2 = float(np.var(b, ddof=1))
            denom = (var1 / n1) + (var2 / n2)
            if denom <= 0:
                return None
            t_stat = (mean1 - mean2) / math.sqrt(denom)
            df_num = denom ** 2
            df_den = (var1 ** 2) / (n1 ** 2 * (n1 - 1)) + (var2 ** 2) / (n2 ** 2 * (n2 - 1))
            if df_den <= 0:
                return None
            df = df_num / df_den
            p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
            return t_stat, df, p_val, n1, n2
        snr_cs1 = snr_value(cs1_br_mean, cs1_bg_mean, bright_width, dark_width)
        snr_cs2 = (
            snr_value(cs2_br_mean, cs2_bg_mean, bright_width, dark_width)
            if cs2 is not None
            else float("nan")
        )
        snr_factor_method1 = None
        if cs2 is not None and np.isfinite(snr_cs1) and np.isfinite(snr_cs2):
            abs1 = abs(snr_cs1)
            abs2 = abs(snr_cs2)
            lo = min(abs1, abs2)
            hi = max(abs1, abs2)
            if lo > 0:
                snr_factor_method1 = hi / lo

        snr_factor_method2 = None
        if cs2 is not None and np.isfinite(cs1_br_mean) and np.isfinite(cs1_bg_mean) and np.isfinite(cs2_bg_mean):
            numerator = cs1_br_mean + cs1_bg_mean
            denominator = cs1_br_mean + cs2_bg_mean
            if numerator > 0 and denominator > 0:
                snr_factor_method2 = math.sqrt(numerator / denominator)
        ttest_result = (
            welch_ttest(
                bg_sub_series[0][bright_start:bright_end],
                bg_sub_series[1][bright_start:bright_end],
            )
            if len(bg_sub_series) > 1
            else None
        )
        attenuation_pct = None
        if len(bg_sub_series) > 1:
            cs1_bgsub_br_mean, _ = mean_std(bg_sub_series[0], bright_start, bright_end)
            cs2_bgsub_br_mean, _ = mean_std(bg_sub_series[1], bright_start, bright_end)
            if (
                np.isfinite(cs1_bgsub_br_mean)
                and cs1_bgsub_br_mean != 0
                and np.isfinite(cs2_bgsub_br_mean)
            ):
                attenuation_pct = (1.0 - (cs2_bgsub_br_mean / cs1_bgsub_br_mean)) * 100.0
        cs1_bgsub = bg_sub_series[0] if bg_sub_series else np.array([])
        cs2_bgsub = bg_sub_series[1] if len(bg_sub_series) > 1 else np.array([])
        cs1_bgsub_br_mean, cs1_bgsub_br_std = mean_std(cs1_bgsub, bright_start, bright_end)
        rows.append((f"{name1} minus BG", "Bright", cs1_bgsub_br_mean, cs1_bgsub_br_std))
        if cs2 is not None and name2 is not None:
            cs2_bgsub_br_mean, cs2_bgsub_br_std = mean_std(cs2_bgsub, bright_start, bright_end)
            rows.append((f"{name2} minus BG", "Bright", cs2_bgsub_br_mean, cs2_bgsub_br_std))
        else:
            cs2_bgsub_br_mean = cs2_bgsub_br_std = float("nan")
        if cs2 is not None:
            ratio_values = np.abs(
                np.divide(
                    cs1_bgsub,
                    cs2_bgsub,
                    out=np.full_like(cs1_bgsub, np.nan),
                    where=cs2_bgsub != 0,
                )
            )
            ratio_mean, ratio_std = mean_std(ratio_values, bright_start, bright_end)
            rows.append(("Abs Ratio (cs1/cs2)", "Bright", ratio_mean, ratio_std))
        else:
            rows.append(("Abs Ratio (cs1/cs2)", "Bright", float("nan"), float("nan")))
        if cs2 is not None:
            diff_bg = cs1[bg_start:bg_end] - cs2[bg_start:bg_end]
            diff_br = cs1[bright_start:bright_end] - cs2[bright_start:bright_end]
            diff_bg_mean = float(np.nanmean(diff_bg)) if diff_bg.size else float("nan")
            diff_bg_std = float(np.nanstd(diff_bg)) if diff_bg.size else float("nan")
            diff_br_mean = float(np.nanmean(diff_br)) if diff_br.size else float("nan")
            diff_br_std = float(np.nanstd(diff_br)) if diff_br.size else float("nan")
            rows.append(("cs1 - cs2", "Background", diff_bg_mean, diff_bg_std))
            rows.append(("cs1 - cs2", "Bright", diff_br_mean, diff_br_std))
        else:
            rows.append(("cs1 - cs2", "Background", float("nan"), float("nan")))
            rows.append(("cs1 - cs2", "Bright", float("nan"), float("nan")))
        if cs2 is not None and cs1_bgsub.size and cs2_bgsub.size:
            diff_bgsub_bg = cs1_bgsub[bg_start:bg_end] - cs2_bgsub[bg_start:bg_end]
            diff_bgsub_br = cs1_bgsub[bright_start:bright_end] - cs2_bgsub[bright_start:bright_end]
            diff_bgsub_bg_mean = float(np.nanmean(diff_bgsub_bg)) if diff_bgsub_bg.size else float("nan")
            diff_bgsub_bg_std = float(np.nanstd(diff_bgsub_bg)) if diff_bgsub_bg.size else float("nan")
            diff_bgsub_br_mean = float(np.nanmean(diff_bgsub_br)) if diff_bgsub_br.size else float("nan")
            diff_bgsub_br_std = float(np.nanstd(diff_bgsub_br)) if diff_bgsub_br.size else float("nan")
            rows.append(("BG-sub diff", "Background", diff_bgsub_bg_mean, diff_bgsub_bg_std))
            rows.append(("BG-sub diff", "Bright", diff_bgsub_br_mean, diff_bgsub_br_std))
        else:
            rows.append(("BG-sub diff", "Background", float("nan"), float("nan")))
            rows.append(("BG-sub diff", "Bright", float("nan"), float("nan")))

        def fmt_snr(value: float) -> str:
            if value is None or not np.isfinite(value):
                return "-"
            return f"{value:.3f}"
        text_lines = ["Analysis"]
        text_lines.append("Assumptions:")
        text_lines.append("- Same camera, settings, and integration time.")
        text_lines.append("- Only difference is the filter.")
        text_lines.append("- Signal is not read-noise dominated.")
        for label, signal_type, mean_val, std_val in rows:
            if mean_val is None or not np.isfinite(mean_val) or mean_val == 0:
                cv_val = float("nan")
            else:
                cv_val = std_val / mean_val
            mean_text = fmt(mean_val)
            std_text = fmt(std_val)
            cv_text = fmt(cv_val)
            self.summary_table.insert(
                "",
                "end",
                values=[label, signal_type, mean_text, std_text, cv_text],
            )
        text_lines.append("")
        text_lines.append("Bright Region Similarity (BG-subtracted, CS1 vs CS2)")
        if ttest_result is None:
            text_lines.append("Not enough data to run Welch t-test on BG-subtracted bright regions.")
        else:
            t_stat, df, p_val, n1, n2 = ttest_result
            text_lines.append(
                f"Welch t-test (normal approx): t={t_stat:.3f}, df={df:.1f}, p~{p_val:.3g}, n1={n1}, n2={n2}"
            )
            if p_val >= 0.05:
                text_lines.append(
                    "Result: Bright regions are statistically similar; signal attenuation is unlikely."
                )
            else:
                text_lines.append(
                    "Result: Bright regions are statistically different; filters may have induced some signal attenuation."
                )
            if attenuation_pct is not None and np.isfinite(attenuation_pct):
                text_lines.append(
                    f"Attenuation (BG-sub bright, CS2 vs CS1) approx {attenuation_pct:.2f}%"
                )
        text_lines.append("")
        text_lines.append("SNR (bright vs dark)")
        text_lines.append(f"Widths: bright={bright_width}, dark={dark_width}")
        cs1_label = name1 or "CS1"
        text_lines.append(
            f"{cs1_label}: B={fmt(cs1_br_mean)}, D={fmt(cs1_bg_mean)}, SNR={fmt_snr(snr_cs1)}"
        )
        if cs2 is not None:
            cs2_label = name2 or "CS2"
            text_lines.append(
                f"{cs2_label}: B={fmt(cs2_br_mean)}, D={fmt(cs2_bg_mean)}, SNR={fmt_snr(snr_cs2)}"
            )
        else:
            text_lines.append("CS2: -")
        text_lines.append("")
        text_lines.append("SNR Factor")
        if snr_factor_method1 is not None and np.isfinite(snr_factor_method1):
            text_lines.append(f"Method 1 (max/min SNR): {snr_factor_method1:.3f}")
        else:
            text_lines.append("Method 1 (max/min SNR): -")
        if snr_factor_method2 is not None and np.isfinite(snr_factor_method2):
            text_lines.append(
                f"Method 2: sqrt((B1 + D1) / (B1 + D2)) = {snr_factor_method2:.3f}"
            )
        else:
            text_lines.append("Method 2: sqrt((B1 + D1) / (B1 + D2)) = -")
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "\n".join(text_lines))
        self.summary_text.configure(state="disabled")
def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    CrossSectionViewer(root)
    root.mainloop()
if __name__ == "__main__":
    main()
