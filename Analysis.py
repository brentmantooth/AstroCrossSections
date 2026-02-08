import csv
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class CrossSectionViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cross Section Analysis")
        self.root.geometry("1200x720")

        self.distance: np.ndarray | None = None
        self.series_names: list[str] = []
        self.series_values: list[np.ndarray] = []

        self.bg_index_var = tk.IntVar(value=0)
        self.bg_width_var = tk.IntVar(value=5)
        self.bright_index_var = tk.IntVar(value=0)
        self.bright_width_var = tk.IntVar(value=5)

        self.bg_value_text = tk.StringVar(value="X: -")
        self.bright_value_text = tk.StringVar(value="X: -")
        self.file_label_text = tk.StringVar(value="No CSV loaded")
        self.y_log_var = tk.BooleanVar(value=False)

        self._build_layout()
        self._set_controls_state("disabled")

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=2)

        top_frame = ttk.Frame(self.root, padding=6)
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

        controls = ttk.Frame(self.root, padding=(6, 4))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=0)

        load_button = ttk.Button(controls, text="Load CSV", command=self.load_csv)
        load_button.grid(row=0, column=0, sticky="w")
        file_label = ttk.Label(controls, textvariable=self.file_label_text)
        file_label.grid(row=0, column=1, sticky="w", padx=(8, 8))
        y_log_check = ttk.Checkbutton(
            controls,
            text="Y log scale",
            variable=self.y_log_var,
            command=self._on_log_toggle,
        )
        y_log_check.grid(row=0, column=2, sticky="e")

        bg_label = ttk.Label(controls, text="Background")
        bg_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.bg_scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            resolution=1,
            showvalue=0,
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

        bottom_frame = ttk.Frame(self.root, padding=6)
        bottom_frame.grid(row=2, column=0, sticky="nsew")
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=0)
        bottom_frame.rowconfigure(0, weight=1)

        self.bottom_fig = Figure(figsize=(5.2, 2.5), dpi=100)
        self.bottom_ax = self.bottom_fig.add_subplot(111)
        self.bottom_ax.set_title("Background Subtracted")
        self.bottom_ax.set_xlabel("Distance (pixels)")
        self.bottom_ax.set_ylabel("Value - Background")
        self.bottom_canvas = FigureCanvasTkAgg(self.bottom_fig, master=bottom_frame)
        self.bottom_canvas.draw()
        self.bottom_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.ratio_fig = Figure(figsize=(5.2, 2.5), dpi=100)
        self.ratio_ax = self.ratio_fig.add_subplot(111)
        self.ratio_ax.set_title("Abs Ratio (BG Subtracted)")
        self.ratio_ax.set_xlabel("Distance (pixels)")
        self.ratio_ax.set_ylabel("Abs Ratio")
        self.ratio_canvas = FigureCanvasTkAgg(self.ratio_fig, master=bottom_frame)
        self.ratio_canvas.draw()
        self.ratio_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=(0, 6))

        table_frame = ttk.Frame(bottom_frame)
        table_frame.grid(row=0, column=2, sticky="nsew")
        table_frame.rowconfigure(1, weight=1)
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
        self._refresh_plots()

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

        max_index = max(self.distance.size - 1, 0)
        for scale in (self.bg_scale, self.bright_scale):
            scale.configure(from_=0, to=max_index)
            scale.set(0)

        max_width = max(self.distance.size, 1)
        self.bg_width_spin.configure(from_=1, to=max_width)
        self.bright_width_spin.configure(from_=1, to=max_width)
        self.bg_width_var.set(min(self.bg_width_var.get(), max_width))
        self.bright_width_var.set(min(self.bright_width_var.get(), max_width))

        self.bg_index_var.set(0)
        self.bright_index_var.set(0)
        self._set_controls_state("normal")
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
        columns = ["Region"]
        for name in self.series_names:
            columns.extend([f"{name} raw", f"{name} -bg", f"{name} ratio"])
        self.summary_table.configure(columns=columns)
        for col in columns:
            self.summary_table.heading(col, text=col)
            width = 90 if col == "Region" else 110
            self.summary_table.column(col, width=width, anchor="center")
        for item in self.summary_table.get_children():
            self.summary_table.delete(item)

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

        use_log = bool(self.y_log_var.get())

        def safe_mean(sample: np.ndarray) -> float:
            if sample.size == 0:
                return float("nan")
            return float(np.nanmean(sample))

        self.top_ax.clear()
        top_has_positive = False
        for name, values in zip(self.series_names, self.series_values):
            plot_values = np.asarray(values, dtype=np.float64)
            if use_log:
                valid = np.isfinite(plot_values) & (plot_values > 0)
                top_has_positive = top_has_positive or bool(np.any(valid))
                plot_values = np.where(valid, plot_values, np.nan)
            self.top_ax.plot(self.distance, plot_values, label=name, linewidth=1.2)

        if bg_end > bg_start:
            left = self.distance[bg_start]
            right = self.distance[bg_end - 1]
            self.top_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
            self.top_ax.axvspan(left, right, color="black", alpha=0.2)

        if bright_end > bright_start:
            left = self.distance[bright_start]
            right = self.distance[bright_end - 1]
            self.top_ax.axvline(self.distance[bright_index], color="#cc7a00", linewidth=1.2)
            self.top_ax.axvspan(left, right, color="#ffb84d", alpha=0.25)

        self.top_ax.set_title("Cross Section")
        self.top_ax.set_xlabel("Distance (pixels)")
        self.top_ax.set_ylabel("Value")
        self.top_ax.legend(loc="best", fontsize=9)
        self.top_ax.grid(True, alpha=0.3)
        if use_log and top_has_positive:
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
            if use_log:
                valid = np.isfinite(plot_values) & (plot_values > 0)
                bottom_has_positive = bottom_has_positive or bool(np.any(valid))
                plot_values = np.where(valid, plot_values, np.nan)
            self.bottom_ax.plot(self.distance, plot_values, label=name, linewidth=1.2)
        self.bottom_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
        self.bottom_ax.axvline(self.distance[bright_index], color="#cc7a00", linewidth=1.2)
        self.bottom_ax.set_title("Background Subtracted")
        self.bottom_ax.set_xlabel("Distance (pixels)")
        self.bottom_ax.set_ylabel("Value - Background")
        self.bottom_ax.legend(loc="best", fontsize=9)
        self.bottom_ax.grid(True, alpha=0.3)
        if use_log and bottom_has_positive:
            self.bottom_ax.set_yscale("log")
        else:
            self.bottom_ax.set_yscale("linear")

        self.ratio_ax.clear()
        ratio_series: list[np.ndarray] = []
        ratio_has_positive = False
        if bg_sub_series:
            baseline = bg_sub_series[0]
            base_name = self.series_names[0]
            for name, values in zip(self.series_names, bg_sub_series):
                ratio = np.abs(
                    np.divide(
                        values,
                        baseline,
                        out=np.full_like(values, np.nan),
                        where=baseline != 0,
                    )
                )
                ratio_series.append(ratio)
                plot_values = ratio
                if use_log:
                    valid = np.isfinite(plot_values) & (plot_values > 0)
                    ratio_has_positive = ratio_has_positive or bool(np.any(valid))
                    plot_values = np.where(valid, plot_values, np.nan)
                label = f"{name}/{base_name}"
                self.ratio_ax.plot(self.distance, plot_values, label=label, linewidth=1.2)
        self.ratio_ax.axvline(self.distance[bg_index], color="black", linewidth=1.2)
        self.ratio_ax.axvline(self.distance[bright_index], color="#cc7a00", linewidth=1.2)
        self.ratio_ax.set_title("Abs Ratio (BG Subtracted)")
        self.ratio_ax.set_xlabel("Distance (pixels)")
        self.ratio_ax.set_ylabel("Abs Ratio")
        if ratio_series:
            self.ratio_ax.legend(loc="best", fontsize=9)
        self.ratio_ax.grid(True, alpha=0.3)
        if use_log and ratio_has_positive:
            self.ratio_ax.set_yscale("log")
        else:
            self.ratio_ax.set_yscale("linear")

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
            return f"{value:.3f}"

        def ratio_mean(series: np.ndarray, start: int, end: int) -> float:
            if series.size == 0:
                return float("nan")
            sample = series[start:end]
            if sample.size == 0:
                return float("nan")
            return float(np.nanmean(sample))

        for item in self.summary_table.get_children():
            self.summary_table.delete(item)

        bg_row = ["Background"]
        bright_row = ["Bright"]
        for idx, name in enumerate(self.series_names):
            raw_bg = raw_bg_means[idx] if idx < len(raw_bg_means) else float("nan")
            raw_bright = raw_bright_means[idx] if idx < len(raw_bright_means) else float("nan")
            offset = background_offsets[idx] if idx < len(background_offsets) else 0.0
            sub_bg = raw_bg - offset
            sub_bright = raw_bright - offset

            if idx < len(ratio_series):
                ratio_bg = ratio_mean(ratio_series[idx], bg_start, bg_end)
                ratio_bright = ratio_mean(ratio_series[idx], bright_start, bright_end)
            else:
                ratio_bg = float("nan")
                ratio_bright = float("nan")

            bg_row.extend([fmt(raw_bg), fmt(sub_bg), fmt(ratio_bg)])
            bright_row.extend([fmt(raw_bright), fmt(sub_bright), fmt(ratio_bright)])

        self.summary_table.insert("", "end", values=bg_row)
        self.summary_table.insert("", "end", values=bright_row)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    CrossSectionViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
