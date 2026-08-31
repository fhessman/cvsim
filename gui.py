"""
Generic PyQt/PySide GUI wrapper for a command-line Python script.
Fields (and how they're grouped into panes) are defined in an external
YAML config file rather than in code.

HOW TO USE
----------
1. Install dependencies if you don't have them:
       pip install PySide6 pyyaml

2. Edit config.yaml (created alongside this script) to describe:
   - script_path: the python script you want to run
   - python_executable: (optional) interpreter to use, defaults to the
     one running this GUI
   - layout: (optional) "tabs" or "splitter" — how panes are arranged
   - panes: a list of panes, each with a "title" and its own "arguments"
     list. Each argument becomes one row in that pane's form, and its
     "flag" is used to build the actual command line when you press Run.

3. Run this file:
       python gui_runner.py
   or point it at a specific config file:
       python gui_runner.py path/to/other_config.yaml
   If no config file is found, a file-picker dialog opens so you can
   choose one.

The GUI builds one form per pane from the YAML's "panes" list, arranges
the panes as tabs or side-by-side (per "layout"), and when you click
"Run", it assembles:

    python script_path <flag1> <value1> <flag2> <value2> ...

using arguments from every pane, then streams stdout/stderr live into
the output console below.

YAML schema
-----------
script_path: your_script.py    # resolved relative to THIS config file's own
                                # directory if not absolute (so a config and
                                # the script it runs can live in different
                                # directories, and it doesn't matter where
                                # you launch the GUI from). The script is
                                # then also run with its own directory as the
                                # working directory, so any relative paths
                                # *inside* the script's own arguments/defaults
                                # resolve the same way regardless of launch
                                # location too.
python_executable: null        # optional; defaults to the running interpreter
layout: tabs                   # optional: "tabs" (default) or "splitter"

panes:
  - title: Input/Output
    arguments:
      - flag: --input           # command-line flag ("" for positional args)
        label: Input file        # text shown next to the field
        type: file               # str | int | float | bool | choice | file | dir
        required: true           # optional, default false
        default: ""              # optional starting value
        choices: [a, b, c]       # required when type is "choice"
        tip: "Shown as a tooltip when hovering over this field"  # optional
  - title: Model Parameters
    arguments:
      - flag: --rate
        label: Learning rate
        type: float
        default: "0.01"

Backward compatibility: a flat top-level "arguments" list (no "panes")
is still accepted and is shown as a single pane named "Parameters".

Supported argument "type" values:
    "str"     -> single-line text box
    "int"     -> text box restricted to integers
    "float"   -> text box restricted to floating point numbers
    "bool"    -> checkbox; when checked, only the flag is added (no value)
    "choice"  -> dropdown; requires a "choices": [...] list
    "file"    -> text box + "Browse..." button (choose existing file)
    "dir"     -> text box + "Browse..." button (choose a folder)
    "csv"     -> text box + "Browse..." button (choose a *.csv file) plus
                 a table underneath that displays the file's contents
                 once one is chosen
    "text"    -> not an input field; renders a read-only comment/help box.
                 Needs a "text" key instead of "flag"/"label"/"default":
                     - type: text
                       text: |
                         Free-form help or notes shown to the user.
                 Contributes nothing to the command line.
    "multi"   -> one or more side-by-side text boxes plus a "+" button to
                 add more. Each non-empty box becomes its own token after
                 the flag, e.g. "--many" "a" "b" "c" for nargs='+' style
                 arguments:
                     - flag: --many
                       label: Many values
                       type: multi
                       count: 3            # optional; initial number of boxes (default: 1)
                       default: [a, b, c]  # optional; one value per box

Example command line this can build:
    python myscript.py --many a b c -single d
"""

import sys
import os
import csv
import copy
import shlex
import yaml
from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout,
    QHBoxLayout, QLineEdit, QCheckBox, QComboBox, QPushButton,
    QTextEdit, QFileDialog, QLabel, QMessageBox, QTabWidget,
    QSplitter, QGroupBox, QScrollArea, QTableWidget, QTableWidgetItem
)
from PySide6.QtGui import QIntValidator, QDoubleValidator, QTextCursor, QFontDatabase

DEFAULT_CONFIG_NAME = "config.yaml"
VALID_LAYOUTS = ("tabs", "splitter")


def validate_config(config, config_path):
    """
    Validate (and fill in defaults for) a parsed YAML config dict.
    Shared by startup loading and the in-app "Load Config..." button.
    """
    config = config or {}

    if "script_path" not in config:
        raise ValueError(f"'script_path' missing from config file: {config_path}")

    # Backward compatibility: wrap a flat "arguments" list into one pane.
    if "panes" not in config:
        if "arguments" not in config:
            raise ValueError(
                f"Config must define either 'panes' or 'arguments': {config_path}"
            )
        config["panes"] = [{"title": "Parameters", "arguments": config["arguments"]}]

    if not config["panes"]:
        raise ValueError(f"'panes' is empty in config file: {config_path}")

    layout = config.get("layout", "tabs")
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"'layout' must be one of {VALID_LAYOUTS}, got: {layout!r}")
    config["layout"] = layout

    seen_flags = set()
    for pane in config["panes"]:
        if "title" not in pane or "arguments" not in pane:
            raise ValueError(f"Each pane needs 'title' and 'arguments'. Bad entry: {pane}")
        for arg in pane["arguments"]:
            if "type" not in arg:
                raise ValueError(f"Each argument needs a 'type'. Bad entry: {arg}")

            if arg["type"] == "text":
                # Comment/help item: no flag/label, just display text.
                if "text" not in arg:
                    raise ValueError(f"Argument with type 'text' needs a 'text' field. Bad entry: {arg}")
                arg.setdefault("flag", "")
                arg.setdefault("label", "")
                continue

            if "flag" not in arg or "label" not in arg:
                raise ValueError(
                    f"Each argument needs 'flag' and 'label'. Bad entry: {arg}"
                )
            if arg["type"] == "choice" and not arg.get("choices"):
                raise ValueError(f"Argument '{arg['label']}' has type 'choice' but no 'choices' list.")
            if arg["type"] == "multi" and "count" in arg:
                if not isinstance(arg["count"], int) or arg["count"] < 1:
                    raise ValueError(
                        f"Argument '{arg['label']}' has type 'multi' but 'count' is not a positive integer."
                    )
            if arg["flag"] and arg["flag"] in seen_flags:
                raise ValueError(f"Duplicate flag '{arg['flag']}' used across panes.")
            seen_flags.add(arg["flag"])

    return config


def read_and_validate_config(config_path):
    """Read a YAML file from disk and validate it."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return validate_config(config, config_path)


def load_config():
    """
    Resolve and load the YAML config file at startup.
    Priority: command-line argument > ./config.yaml > file-picker dialog.
    """
    config_path = None

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        config_path = sys.argv[1]
    elif os.path.isfile(DEFAULT_CONFIG_NAME):
        config_path = DEFAULT_CONFIG_NAME
    else:
        app = QApplication.instance() or QApplication(sys.argv)
        path, _ = QFileDialog.getOpenFileName(
            None, "Select GUI config file", "", "YAML files (*.yaml *.yml)"
        )
        if not path:
            print("No config file selected. Exiting.")
            sys.exit(1)
        config_path = path

    config = read_and_validate_config(config_path)
    return config, config_path


class ScriptRunnerGUI(QMainWindow):
    def __init__(self, config, config_path):
        super().__init__()
        self.resize(760, 640)

        self.fields = {}  # flag -> widget
        self.process = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- multi-pane input area: a host widget whose contents get
        # replaced whenever a new config is loaded via "Load Config..." ---
        self.panes_host = QWidget()
        self.panes_host_layout = QVBoxLayout(self.panes_host)
        self.panes_host_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.panes_host, stretch=1)

        # --- buttons ---
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_script)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_script)
        self.stop_btn.setEnabled(False)
        self.load_btn = QPushButton("Load Config...")
        self.load_btn.clicked.connect(self.load_config_dialog)
        self.save_btn = QPushButton("Save Config...")
        self.save_btn.clicked.connect(self.save_config)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        # Use Qt's own fixed-width system font instead of the CSS "monospace"
        # keyword, which can trigger "missing font family" warnings/delays
        # on systems where no family is literally named "Monospace".
        mono_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

        # --- command preview (single line, but scrollable when it overflows) ---
        self.cmd_preview = QTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.cmd_preview.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cmd_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cmd_preview.setFont(mono_font)
        # Leave enough room for the text line AND the horizontal scrollbar
        # that appears once the command overflows the field's width.
        scrollbar_height = self.cmd_preview.horizontalScrollBar().sizeHint().height()
        self.cmd_preview.setFixedHeight(
            self.cmd_preview.fontMetrics().height() + scrollbar_height + 18
        )
        layout.addWidget(QLabel("Command:"))
        layout.addWidget(self.cmd_preview)

        # --- output console ---
        output_header = QHBoxLayout()
        output_header.addWidget(QLabel("Output:"))
        output_header.addStretch()
        self.save_output_btn = QPushButton("Save Output...")
        self.save_output_btn.clicked.connect(self.save_output)
        output_header.addWidget(self.save_output_btn)
        layout.addLayout(output_header)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(mono_font)
        self.output.setStyleSheet("background-color:#111;color:#eee;")
        layout.addWidget(self.output, stretch=1)

        self._apply_config(config, config_path)

    # -----------------------------------------------------------------
    def _apply_config(self, config, config_path):
        """(Re)build the pane form and internal state from a validated config dict."""
        self.config = config
        self.panes = config["panes"]
        self.pane_layout_mode = config["layout"]
        # Flattened list of every argument across all panes, in order.
        # Used for building the command line and validating required fields.
        self.arguments = [arg for pane in self.panes for arg in pane["arguments"]]
        # A relative script_path is resolved against this config file's own
        # directory, not whatever directory the GUI happens to be launched
        # from -- so a config can point at a script next to it (or a few
        # directories up/over) and keep working no matter where the GUI is
        # started. An absolute script_path is left untouched.
        self.script_path = config["script_path"]
        if not os.path.isabs(self.script_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            self.script_path = os.path.normpath(os.path.join(config_dir, self.script_path))
        self.python_executable = config.get("python_executable") or sys.executable
        self.config_path = config_path

        self.setWindowTitle(f"Run: {self.script_path}  (config: {os.path.basename(config_path)})")

        # Clear any previously built pane widgets/fields, then rebuild.
        self.fields = {}
        while self.panes_host_layout.count():
            item = self.panes_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.panes_host_layout.addWidget(self._build_panes())

        # Reset run state so stale output/commands from a previous config don't linger.
        if hasattr(self, "cmd_preview"):
            self.cmd_preview.clear()
        if hasattr(self, "output"):
            self.output.clear()
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(True)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(False)

    # -----------------------------------------------------------------
    def load_config_dialog(self):
        """Prompt for a YAML file and rebuild the GUI from it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load configuration", "", "YAML files (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            config = read_and_validate_config(path)
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Could not load configuration:\n{e}")
            return
        self._apply_config(config, path)

    # -----------------------------------------------------------------
    def save_output(self):
        """Save the contents of the output console to a text file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output as", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(self.output.toPlainText())
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Could not save output:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Output saved to:\n{path}")

    # -----------------------------------------------------------------
    def _build_panes(self):
        """Build the multi-pane input area as tabs or a side-by-side splitter."""
        if self.pane_layout_mode == "splitter":
            container = QSplitter(Qt.Horizontal)
            for pane in self.panes:
                box = QGroupBox(pane["title"])
                box_layout = QVBoxLayout(box)
                box_layout.addWidget(self._build_pane_form(pane))
                container.addWidget(box)
            return container

        # default: tabs
        tabs = QTabWidget()
        for pane in self.panes:
            tabs.addTab(self._build_pane_form(pane), pane["title"])
        return tabs

    @staticmethod
    def _apply_tooltip(widget, tip):
        """Set a tooltip on a widget and all its children (covers composite
        file/dir/csv rows, so hovering the browse button or table also works)."""
        if not tip:
            return
        widget.setToolTip(tip)
        for child in widget.findChildren(QWidget):
            child.setToolTip(tip)

    def _build_pane_form(self, pane):
        """Build a scrollable form widget for a single pane's arguments."""
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        for arg in pane["arguments"]:
            widget = self._build_widget(arg)
            self._apply_tooltip(widget, arg.get("tip"))

            if arg["type"] == "text":
                # Comment/help item: full-width row, not tracked as a field.
                form.addRow(widget)
                continue

            self.fields[arg["flag"]] = widget
            label_text = arg["label"] + (" *" if arg.get("required") else "")
            label_widget = QLabel(label_text)
            self._apply_tooltip(label_widget, arg.get("tip"))
            form.addRow(label_widget, widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        return scroll

    # -----------------------------------------------------------------
    def _build_widget(self, arg):
        t = arg["type"]
        default = arg.get("default", "")

        if t == "text":
            label = QLabel(str(arg.get("text", "")))
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setStyleSheet(
                "background-color: rgba(255, 244, 200, 0.6);"
                "border: 1px solid #d8c48c;"
                "border-radius: 4px;"
                "padding: 6px;"
                "color: #555;"
            )
            return label

        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w

        if t == "choice":
            w = QComboBox()
            w.addItems(arg.get("choices", []))
            if default in arg.get("choices", []):
                w.setCurrentText(default)
            return w

        if t in ("file", "dir"):
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            line = QLineEdit(str(default))
            browse = QPushButton("Browse...")

            def do_browse():
                if t == "file":
                    path, _ = QFileDialog.getOpenFileName(self, "Select file")
                else:
                    path = QFileDialog.getExistingDirectory(self, "Select folder")
                if path:
                    line.setText(path)

            browse.clicked.connect(do_browse)
            h.addWidget(line)
            h.addWidget(browse)
            container.line_edit = line  # keep a handle for value retrieval
            return container

        if t == "csv":
            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(0, 0, 0, 0)

            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            line = QLineEdit(str(default))
            browse = QPushButton("Browse...")
            h.addWidget(line)
            h.addWidget(browse)

            table = QTableWidget()
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setMinimumHeight(160)
            table.setMaximumHeight(240)

            def do_browse():
                path, _ = QFileDialog.getOpenFileName(
                    self, "Select CSV file", "", "CSV files (*.csv)"
                )
                if path:
                    line.setText(path)
                    self._load_csv_into_table(path, table, base_tip=arg.get("tip"))

            browse.clicked.connect(do_browse)

            v.addWidget(row)
            v.addWidget(table)
            container.line_edit = line  # keep a handle for value retrieval
            container.table = table
            return container

        if t == "multi":
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)

            line_edits = []

            add_btn = QPushButton("+")
            add_btn.setFixedWidth(28)
            add_btn.setToolTip("Add another value box")

            def add_field(initial_value=""):
                le = QLineEdit(str(initial_value))
                if arg.get("tip"):
                    le.setToolTip(arg["tip"])
                line_edits.append(le)
                # Insert right before the "+" button, which always stays last.
                h.insertWidget(h.count() - 1, le)

            add_btn.clicked.connect(lambda: add_field(""))
            h.addWidget(add_btn)

            count = max(1, int(arg.get("count", 1)))
            defaults = arg.get("default", [])
            if not isinstance(defaults, list):
                defaults = [defaults] if defaults not in (None, "") else []
            for i in range(count):
                add_field(defaults[i] if i < len(defaults) else "")

            container.line_edits = line_edits  # keep a handle for value retrieval
            return container

        # str / int / float all use a QLineEdit
        w = QLineEdit(str(default))
        if t == "int":
            w.setValidator(QIntValidator())
        elif t == "float":
            w.setValidator(QDoubleValidator())
        return w

    # -----------------------------------------------------------------
    MAX_CSV_PREVIEW_ROWS = 500

    def _load_csv_into_table(self, path, table, base_tip=None):
        """Read a CSV file and populate the given QTableWidget with its contents."""
        prefix = f"{base_tip}\n\n" if base_tip else ""

        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
        except Exception as e:
            QMessageBox.warning(self, "CSV load error", f"Could not read CSV file:\n{e}")
            table.setRowCount(0)
            table.setColumnCount(0)
            return

        if not rows:
            table.setRowCount(0)
            table.setColumnCount(0)
            return

        header, data = rows[0], rows[1:]
        total_rows = len(data)
        truncated = total_rows > self.MAX_CSV_PREVIEW_ROWS
        if truncated:
            data = data[: self.MAX_CSV_PREVIEW_ROWS]

        table.setColumnCount(len(header))
        table.setHorizontalHeaderLabels(header)
        table.setRowCount(len(data))
        for r, row_data in enumerate(data):
            for c in range(len(header)):
                value = row_data[c] if c < len(row_data) else ""
                table.setItem(r, c, QTableWidgetItem(value))
        table.resizeColumnsToContents()

        if truncated:
            table.setToolTip(
                f"{prefix}Showing first {self.MAX_CSV_PREVIEW_ROWS} of {total_rows} rows."
            )
        else:
            table.setToolTip(f"{prefix}{total_rows} rows.")

    # -----------------------------------------------------------------
    def _widget_value(self, arg, widget):
        t = arg["type"]
        if t == "bool":
            return widget.isChecked()
        if t == "choice":
            return widget.currentText()
        if t in ("file", "dir", "csv"):
            return widget.line_edit.text().strip()
        if t == "multi":
            return [le.text().strip() for le in widget.line_edits]
        return widget.text().strip()

    # -----------------------------------------------------------------
    def build_command(self):
        cmd = [self.python_executable, self.script_path]
        for arg in self.arguments:
            if arg["type"] == "text":
                continue  # comment/help item, not a real argument
            widget = self.fields[arg["flag"]]
            value = self._widget_value(arg, widget)

            if arg["type"] == "bool":
                if value and arg["flag"]:
                    cmd.append(arg["flag"])
                continue

            if arg["type"] == "multi":
                values = [v for v in value if v]  # drop empty boxes
                if arg.get("required") and not values:
                    raise ValueError(f"'{arg['label']}' is required.")
                if not values:
                    continue
                if arg["flag"]:
                    cmd.append(arg["flag"])
                cmd.extend(values)
                continue

            if arg.get("required") and not value:
                raise ValueError(f"'{arg['label']}' is required.")

            if not value:
                continue  # skip empty optional fields

            if arg["flag"]:
                cmd.append(arg["flag"])
            cmd.append(value)
        return cmd

    # -----------------------------------------------------------------
    def run_script(self):
        try:
            cmd = self.build_command()
        except ValueError as e:
            QMessageBox.warning(self, "Missing argument", str(e))
            return

        self.cmd_preview.setText(" ".join(shlex.quote(c) for c in cmd))
        self.output.clear()

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._on_finished)

        self.process.setProgram(cmd[0])
        self.process.setArguments(cmd[1:])
        # Run with the script's own directory as the working directory, so
        # any relative paths in ITS arguments/defaults (e.g. a config's own
        # relative --data-file default) resolve the same way regardless of
        # where the GUI itself was launched from -- see script_path's
        # resolution in _apply_config, above.
        self.process.setWorkingDirectory(os.path.dirname(self.script_path))
        self.process.start()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _read_output(self):
        data = self.process.readAllStandardOutput().data().decode(errors="replace")
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(data)

    def _on_finished(self, exit_code, exit_status):
        self.output.append(f"\n[process finished with exit code {exit_code}]")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop_script(self):
        if self.process:
            self.process.kill()

    # -----------------------------------------------------------------
    def save_config(self):
        """Write the current field values as a new YAML config file (as defaults)."""
        new_config = copy.deepcopy(self.config)
        for pane in new_config["panes"]:
            for arg in pane["arguments"]:
                if arg["type"] == "text":
                    continue  # comment/help item, nothing to save
                widget = self.fields[arg["flag"]]
                value = self._widget_value(arg, widget)
                arg["default"] = bool(value) if arg["type"] == "bool" else value

        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration as", "", "YAML files (*.yaml *.yml)"
        )
        if not path:
            return
        if not path.lower().endswith((".yaml", ".yml")):
            path += ".yaml"

        try:
            with open(path, "w") as f:
                yaml.safe_dump(new_config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Could not save configuration:\n{e}")
            return

        QMessageBox.information(self, "Saved", f"Configuration saved to:\n{path}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    config, config_path = load_config()
    win = ScriptRunnerGUI(config, config_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
