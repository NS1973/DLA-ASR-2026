"""Render the current Markdown report and measured JSONL curves to PDF.

Report tooling only: install reportlab separately from inference dependencies.
"""
import argparse
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether


def markup(text):
    text = html.escape(text).replace('—','-').replace('–','-')
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`(.+?)`', r'<font color="#14636b">\1</font>', text)


def curves(paths, destination):
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    for path in paths:
        records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        name = Path(path).stem if Path(path).parent.name == 'logs' else Path(path).parent.name
        train = [r for r in records if r.get('event')=='train']
        for ax,key,title in [(axes[0,0],'loss','CTC loss'),(axes[0,1],'lr','Learning rate'),(axes[1,0],'gradient_norm','Gradient norm')]:
            if train:
                ax.plot([r['step'] for r in train],[r[key] for r in train],label=name,linewidth=1)
            ax.set(xlabel='Optimizer step',ylabel=title,title=title)
        for split in ['dev-clean','dev-other']:
            valid = [r for r in records if r.get('event')=='validation' and r.get('split')==split]
            if valid:
                axes[1,1].plot([r['step'] for r in valid],[100*r['wer'] for r in valid],label=f'{name}/{split}')
        axes[1,1].set(xlabel='Optimizer step',ylabel='WER (%)',title='Measured dev WER')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination,dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',default='REPORT.md')
    p.add_argument('--output',default='REPORT.pdf')
    p.add_argument('--metrics',nargs='*',default=[])
    args = p.parse_args()
    fonts = [(Path('C:/Windows/Fonts/arial.ttf'),Path('C:/Windows/Fonts/arialbd.ttf')),
             (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))]
    normal,bold = next(pair for pair in fonts if all(path.exists() for path in pair))
    pdfmetrics.registerFont(TTFont('Report',str(normal)))
    pdfmetrics.registerFont(TTFont('ReportBold',str(bold)))
    pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold')
    body = ParagraphStyle('Body',fontName='Report',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#24333b'))
    heading = ParagraphStyle('Heading',parent=body,fontName='ReportBold',fontSize=14,leading=18,spaceBefore=13,spaceAfter=8,keepWithNext=True)
    title = ParagraphStyle('Title',parent=heading,fontSize=22,leading=27,textColor=colors.HexColor('#14636b'))
    small = ParagraphStyle('Small',parent=body,fontSize=8,leading=11,spaceAfter=0)
    story, paragraph, table = [],[],[]
    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(markup(' '.join(paragraph)),body)); paragraph.clear()
    def flush_table():
        if table:
            rows = [[Paragraph(markup(cell),small) for cell in row] for row in table]
            widths = [43*mm,65*mm,58*mm] if len(rows[0])==3 else [166*mm/len(rows[0])]*len(rows[0])
            item = Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2f0f0')),
                                      ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cad5d9')),
                                      ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                                      ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([item,Spacer(1,8)]); table.clear()
    for line in Path(args.source).read_text(encoding='utf-8').splitlines():
        if line.startswith('|'):
            flush_paragraph()
            if not re.match(r'^\|[\s:|-]+\|$',line):
                table.append([cell.strip() for cell in line.strip('|').split('|')])
            continue
        flush_table()
        if line.startswith('#'):
            flush_paragraph()
            story.append(Paragraph(markup(line.lstrip('#').strip()),title if line.startswith('# ') else heading))
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.strip())
    flush_paragraph(); flush_table()
    if args.metrics:
        plot = Path(args.output).parent/'report_curves.png'
        curves(args.metrics,plot)
        story.append(KeepTogether([Paragraph('Кривые из локальных журналов',heading),
                      Paragraph('Источник: '+html.escape(', '.join(str(Path(x)) for x in args.metrics))+'. Только измеренные значения; пилотные запуски не заменяют итоговую оценку.',body),
                      Image(str(plot),width=166*mm,height=116.2*mm)]))
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont('Report',8); canvas.setFillColor(colors.HexColor('#64747b'))
        canvas.drawString(22*mm,13*mm,'DLA 2026 / HW1 ASR')
        canvas.drawRightString(188*mm,13*mm,str(doc.page)); canvas.restoreState()
    SimpleDocTemplate(args.output,pagesize=(210*mm,297*mm),leftMargin=22*mm,rightMargin=22*mm,
                      topMargin=19*mm,bottomMargin=21*mm,title='DLA 2026 HW1 ASR').build(story,onFirstPage=footer,onLaterPages=footer)
    print(Path(args.output).resolve())


if __name__=='__main__':
    main()
