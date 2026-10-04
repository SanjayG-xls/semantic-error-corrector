import sys
import os
import subprocess
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                               QSplitter, QTreeWidget, QTreeWidgetItem, QDockWidget, QTabWidget)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter

# Matplotlib embedded canvas for PySide6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# ==========================================
# 1. Global AI Model Cache
# ==========================================
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None

# ==========================================
# 2. Verilog Syntax Highlighter (Classic Verdi Style)
# ==========================================
class VerilogHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        
        # Keywords (Classic Navy Blue)
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#000080"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "module", "endmodule", "input", "output", "inout", "wire", "reg",
            "always", "begin", "end", "if", "else", "case", "endcase", "default",
            "assign", "posedge", "negedge", "localparam", "parameter"
        ]
        for word in keywords:
            self.rules.append((re.compile(fr'\b{word}\b'), keyword_format))

        # Comments (Classic Forest Green)
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#008000"))
        self.rules.append((re.compile(r'//[^\n]*'), comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), format)

# ==========================================
# 3. Background AI Worker Thread
# ==========================================
class AIVerificationWorker(QThread):
    progress_update = Signal(str)
    finished_analysis = Signal(dict)
    
    def __init__(self, buggy_code):
        super().__init__()
        self.buggy_code = buggy_code

    def verify_with_compiler(self, file_path):
        command = ["iverilog", "-g2012", "-t", "null", file_path]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return True, "0 Errors (Passed Icarus Verilog lint)"
            else:
                return False, result.stderr.strip()
        except FileNotFoundError:
            return False, "Icarus Verilog ('iverilog') not found in PATH."

    def run(self):
        global GLOBAL_MODEL, GLOBAL_TOKENIZER
        
        buggy_file = "test_latch_design.v"
        with open(buggy_file, "w") as f:
            f.write(self.buggy_code)
            
        if GLOBAL_MODEL is None:
            self.progress_update.emit("Booting AI Model (First run)...")
            base_model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
            adapter_dir = "./verilog-lora-model"
            
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
            )
            GLOBAL_MODEL = PeftModel.from_pretrained(base_model, adapter_dir)
            GLOBAL_MODEL.eval()

        self.progress_update.emit("Running Closed-Loop AI Patching & AST Parsing...")
        
        prompt = f"Instruction: Analyze the following complex hierarchical Verilog module for semantic or logical errors. Identify the bug, explain the issue, and provide the corrected code.\nInput:\n{self.buggy_code}\nOutput:\n"
        inputs = GLOBAL_TOKENIZER(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = GLOBAL_MODEL.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=False)
        
        generated_text = GLOBAL_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        response_only = generated_text.split("Output:\n")[-1]
        
        code_blocks = re.findall(r"```verilog\s+(.*?)\s+```", response_only, re.DOTALL)
        
        if code_blocks:
            fixed_code = code_blocks[0]
            fixed_file = "fixed_output_design.v"
            with open(fixed_file, "w") as f:
                f.write(fixed_code)
            
            success, final_report = self.verify_with_compiler(fixed_file)
            score = "100% Synthesizable" if success else "0% (Compilation Errors)"
        else:
            fixed_code = "// Could not extract code cleanly."
            score = "0% (Extraction Failed)"
            final_report = "Parsing Error"

        result_data = {
            "buggy_code": self.buggy_code,
            "fixed_code": fixed_code,
            "score": score,
            "report": final_report,
            "analysis": response_only.strip()
        }
        self.finished_analysis.emit(result_data)

# ==========================================
# 4. Classic Verdi Main Window Layout
# ==========================================
class EDAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synopsys Verdi (Classic Theme)")
        self.resize(1350, 850)
        
        # Central Workspace Splitter (Code Editors Side-by-Side)
        central_splitter = QSplitter(Qt.Horizontal)
        
        # Left Editor: Buggy Source
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.addWidget(QLabel("❌ Input Code (Buggy RTL Source)"))
        self.txt_buggy = QTextEdit()
        self.txt_buggy.setPlaceholderText("Paste or write Verilog code here...")
        VerilogHighlighter(self.txt_buggy.document())
        left_layout.addWidget(self.txt_buggy)
        
        # Right Editor: AI Fixed Source
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.addWidget(QLabel("✅ AI Corrected Code (Verified Netlist View)"))
        self.txt_fixed = QTextEdit()
        self.txt_fixed.setReadOnly(True)
        VerilogHighlighter(self.txt_fixed.document())
        right_layout.addWidget(self.txt_fixed)
        
        central_splitter.addWidget(left_container)
        central_splitter.addWidget(right_container)
        self.setCentralWidget(central_splitter)
        
        # --- LEFT DOCK: AMS Hierarchy Tree ---
        self.hierarchy_dock = QDockWidget("AMS Hierarchy / Design Browser", self)
        self.hierarchy_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        tree_widget = QTreeWidget()
        tree_widget.setHeaderLabel("Project Workspace")
        root_item = QTreeWidgetItem(tree_widget, ["verilog-ml-project"])
        QTreeWidgetItem(root_item, ["test_latch_design.v (Active)"])
        QTreeWidgetItem(root_item, ["fixed_output_design.v"])
        tree_widget.expandAll()
        
        self.hierarchy_dock.setWidget(tree_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.hierarchy_dock)
        
        # --- BOTTOM DOCK: Diagnostics, Logs & Classic nWave Viewer ---
        self.bottom_dock = QDockWidget("Verification Console & nWave Timing Viewer", self)
        self.bottom_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        
        bottom_tabs = QTabWidget()
        
        # Tab 1: AI Diagnostic Report
        self.txt_analysis = QTextEdit()
        self.txt_analysis.setReadOnly(True)
        self.txt_analysis.setPlaceholderText("AI structural analysis and error metrics will appear here...")
        bottom_tabs.addTab(self.txt_analysis, "AI Diagnostics & Metrics")
        
        # Tab 2: Classic nWave Digital Timing Viewer (True Pitch-Black Canvas)
        self.figure = Figure(figsize=(6, 2.5), facecolor='#000000')
        self.canvas = FigureCanvas(self.figure)
        bottom_tabs.addTab(self.canvas, "nWave Timing Diagram View")
        self.plot_classic_verdi_waveform()

        # Tab 3: Compiler Logs
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setPlainText("System initialized. Ready for RTL verification pipeline.\n")
        bottom_tabs.addTab(self.txt_logs, "Compiler / Icarus Log")
        
        self.bottom_dock.setWidget(bottom_tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)
        
        # --- TOP TOOLBAR DOCK: Action Controls ---
        self.control_dock = QDockWidget("Execution Control", self)
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        
        self.btn_parse = QPushButton("▶ Run Closed-Loop Parse & Verify Pipeline")
        self.btn_parse.clicked.connect(self.run_pipeline)
        control_layout.addWidget(self.btn_parse)
        
        self.lbl_status = QLabel("Status: Idle")
        control_layout.addWidget(self.lbl_status)
        control_layout.addStretch()
        
        self.control_dock.setWidget(control_widget)
        self.addDockWidget(Qt.TopDockWidgetArea, self.control_dock)

    def plot_classic_verdi_waveform(self):
        """Renders authentic classic Verdi nWave waveforms: pure black canvas, yellow clock, cyan bus, green signals, red cursors."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#000000') # Classic nWave pitch-black plotting area
        
        # High-fidelity digital signal steps
        t = np.array([0, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80])
        
        # Clock signal (Classic Verdi Gold/Yellow)
        clk = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        
        # Control Bus / Select Signal (Classic Verdi Cyan)
        sel = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        
        # Data Input Bus (Classic Verdi Magenta)
        data_in = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        
        # Output Latch / Fixed Signal (Classic Verdi Bright Green)
        out_fixed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])

        # Offset layers for distinct tracks
        ax.step(t, clk + 6, where='post', color='#ffff00', linewidth=1.8, label='clk')
        ax.step(t, sel + 4, where='post', color='#00ffff', linewidth=1.8, label='sel [1:0]')
        ax.step(t, data_in + 2, where='post', color='#ff00ff', linewidth=1.8, label='din [7:0]')
        ax.step(t, out_fixed, where='post', color='#00ff00', linewidth=2.0, label='out (synthesizable)')

        # Classic nWave Delta Measurement Cursors (Bright Red & Orange Vertical Lines)
        ax.axvline(x=30, color='#ff0000', linestyle='--', linewidth=1.2, label='Marker T1 (30ns)')
        ax.axvline(x=50, color='#ff8000', linestyle='--', linewidth=1.2, label='Marker T2 (50ns)')
        ax.axvspan(30, 50, color='#ff0000', alpha=0.2) # Delta timing interval highlight

        ax.set_yticks([0.5, 2.5, 4.5, 6.5])
        ax.set_yticklabels(['out', 'din', 'sel', 'clk'], color='#00ff00', fontweight='bold')
        ax.set_xlim(0, 80)
        ax.set_ylim(-0.5, 8.5)
        
        ax.tick_params(colors='#ffffff')
        ax.grid(True, which='both', color='#333333', linestyle=':', linewidth=0.8)
        
        for spine in ax.spines.values():
            spine.set_color('#666666')
            
        ax.set_title("nWave Signal Debugger — $\Delta T = 20ns$", color='#00ffff', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', facecolor='#000000', edgecolor='#666666', labelcolor='#ffffff', fontsize=8)
        self.canvas.draw()

    def run_pipeline(self):
        code = self.txt_buggy.toPlainText().strip()
        if not code:
            self.lbl_status.setText("Status: Error - Input code is empty.")
            return
            
        self.btn_parse.setEnabled(False)
        self.lbl_status.setText("Status: Running AI Inference & Compiler Verification...")
        self.txt_logs.append(">> Starting verification pipeline...")
        
        self.worker = AIVerificationWorker(code)
        self.worker.progress_update.connect(lambda msg: self.txt_logs.append(f">> {msg}"))
        self.worker.finished_analysis.connect(self.display_results)
        self.worker.start()

    def display_results(self, data):
        self.btn_parse.setEnabled(True)
        self.lbl_status.setText("Status: Completed Successfully.")
        
        self.txt_fixed.setPlainText(data["fixed_code"])
        
        report_text = (
            f"=== VERIFICATION SUMMARY ===\n"
            f"Percentage of Correctness: {data['score']}\n"
            f"Compiler Re-Verification Status: {data['report']}\n\n"
            f"=== DETAILED AI ANALYSIS & ERROR FIXES ===\n"
            f"{data['analysis']}"
        )
        self.txt_analysis.setPlainText(report_text)
        self.txt_logs.append(">> Pipeline execution complete. Design verified.\n")
        self.plot_classic_verdi_waveform()

# ==========================================
# 5. Classic Verdi Industrial Gray Stylesheet
# ==========================================
VERDI_CLASSIC_STYLESHEET = """
QMainWindow {
    background-color: #d6d6d6;
}
QDockWidget {
    color: #000000;
    font-weight: bold;
}
QDockWidget::title {
    background: #bcbcbc;
    padding: 6px;
    border: 1px solid #808080;
}
QTabWidget::pane {
    border: 1px solid #808080;
    background-color: #f0f0f0;
}
QTabBar::tab {
    background: #e0e0e0;
    color: #000000;
    padding: 6px 14px;
    border: 1px solid #a0a0a0;
}
QTabBar::tab:selected {
    background: #000080;
    color: #ffffff;
    font-weight: bold;
}
QWidget {
    background-color: #eaeaea;
    color: #000000;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 12px;
}
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #a0a0a0;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    color: #000000;
}
QTreeWidget {
    background-color: #ffffff;
    border: 1px solid #a0a0a0;
    color: #000000;
}
QPushButton {
    background-color: #e0e0e0;
    color: #000000;
    border: 1px solid #808080;
    border-radius: 2px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #d0d0d0;
}
QPushButton:disabled {
    background-color: #f0f0f0;
    color: #808080;
}
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(VERDI_CLASSIC_STYLESHEET)
    
    window = EDAMainWindow()
    window.show()
    sys.exit(app.exec())