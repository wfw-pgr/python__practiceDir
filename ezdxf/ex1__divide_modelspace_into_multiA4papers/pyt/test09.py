import ezdxf
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.graphics.shapes import Drawing, Path
from reportlab.graphics import renderPDF

# Splineを描画するための関数
def draw_spline(spline, drawing, color):
    path = Path()
    path.moveTo(spline.control_points[0].x, spline.control_points[0].y)

    for point in spline.control_points[1:]:
        path.curveTo(point.x, point.y, point.x, point.y, point.x, point.y)

    path.strokeColor = color
    path.strokeWidth = 1

    drawing.add(path)

# PDFファイルを生成する関数
def generate_pdf(inpFile, outFile):
    doc = SimpleDocTemplate(outFile, pagesize=letter)

    # DXFファイルを読み込む
    doc_dxf = ezdxf.readfile(inpFile)
    msp = doc_dxf.modelspace()

    # ドキュメントに挿入するコンテンツを作成
    content = []

    for entity in msp.query('*'):
        print( entity )
        if entity.dxftype() == 'LINE':
            # 直線を描画
            x1, y1, x2, y2 = entity.dxf.start.x, entity.dxf.start.y, entity.dxf.end.x, entity.dxf.end.y
            print( x1, y1, x2, y2 )
            # line = Drawing(100, 100)
            # line.add( lines=[(x1, y1, x2, y2)], strokeColor=colors.black)
            # content.append( line )
        # elif entity.dxftype() == 'CIRCLE':
        #     # 円を描画
        #     x, y = entity.dxf.center.x, entity.dxf.center.y
        #     radius = entity.dxf.radius
        #     circle = Drawing(100, 100)
        #     circle.add( [(x - radius, y - radius, x + radius, y + radius)], strokeColor=colors.black)
        #     content.append(circle)
        # elif entity.dxftype() == 'SPLINE':
        #     # スプラインを描画
        #     spline = entity.cast()
        #     spline_color = colors.black
        #     spline_drawing = Drawing(100, 100)
        #     draw_spline(spline, spline_drawing, spline_color)
        #     content.append(spline_drawing)

        # 改ページを追加（オプション）
        content.append( PageBreak() )

    # ドキュメントにコンテンツを追加
    doc.build(content)

# PDFファイルを生成
inpFile = "dxf/original__modelspace_divided_01_01.dxf"
outFile = "pdf/test09.pdf"
generate_pdf( inpFile, outFile )
