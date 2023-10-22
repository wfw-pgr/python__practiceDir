from fpdf import FPDF
import ezdxf

# 新しいPDFファイルを作成
pdf = FPDF()
pdf.add_page()

# DXFファイルを読み込む
doc = ezdxf.readfile("dxf/original__modelspace_divided_01_01.dxf")

# Paperspaceを取得
msp = doc.modelspace()

# Paperspace内の図形をPDFに描画
for entity in msp.query("*"):
    # Paperspaceの図形のみを取得
    print( entity )
    if ( entity.dxftype() == 'LINE' ):
        start_point = entity.dxf.start
        end_point = entity.dxf.end
        pdf.line(start_point.x, start_point.y, end_point.x, end_point.y)
    if ( entity.dxftype() == 'CIRCLE' ):
        center = entity.dxf.center
        radius = entity.dxf.radius
        pdf.circle(center.x, center.y, 2 * radius, 'D')  # 'D'は直径
    # 他の図形の描画処理も追加

# PDFファイルを保存
pdf.output('output.pdf')
