from reportlab.lib.pagesizes import letter
from reportlab.pdfgen        import canvas


output_file = "output.pdf"
c           = canvas.Canvas(output_file, pagesize=letter)

# 直線の描画
x1, y1, x2, y2 = 100, 100, 400, 100  # 直線の始点と終点座標
c.setLineWidth(1)  # 線の幅
c.setStrokeColorRGB(0, 0, 0)  # 線の色 (RGB)
c.line(x1, y1, x2, y2)  # 直線を描画

c.showPage()
c.save()


