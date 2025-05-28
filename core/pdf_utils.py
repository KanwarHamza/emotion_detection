# ---------- core/pdf_utils.py ----------
from fpdf import FPDF
from datetime import datetime
import numpy as np

def generate_pdf_report(results, filename=None):
    pdf = FPDF('P', 'mm', 'A5')
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "NeuroMind Mobile Report", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"MMSE Score: {results['mmse_score']}/30", 0, 1)
    pdf.cell(0, 10, f"Stress Level: {np.mean(results['stress_history']):.2f}/1.0", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Task Performance:", 0, 1)
    for task, perf in results['task_performance'].items():
        pdf.cell(0, 10, f"- {task}: {perf['score']}/{perf['max']}", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Suggestions:", 0, 1)
    pdf.set_font("Arial", '', 10)
    if results['mmse_score'] < 24:
        pdf.multi_cell(0, 8, "Consider cognitive screening with a specialist")
    if np.mean(results['stress_history']) > 0.7:
        pdf.multi_cell(0, 8, "Try breathing exercises for stress")

    if not filename:
        filename = f"NeuroMind_Mobile_Report_{datetime.now().strftime('%Y%m%d')}.pdf"

    pdf.output(filename)
    return filename

