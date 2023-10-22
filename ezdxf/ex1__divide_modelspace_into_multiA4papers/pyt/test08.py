from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus.flowables import PageBreak

# PDFファイルを生成する関数
def generate_pdf(output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter)

    # ドキュメントに挿入するコンテンツを作成
    content = []

    # テキストを作成
    styles = getSampleStyleSheet()
    text = "Hello, ReportLab!"
    p = Paragraph(text, styles['Normal'])
    content.append(p)

    # 改ページを追加（オプション）
    content.append(PageBreak())

    # ドキュメントにコンテンツを追加
    doc.build(content)

# PDFファイルを生成
generate_pdf('hello_reportlab.pdf')
