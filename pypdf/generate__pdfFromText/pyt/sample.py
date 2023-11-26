from reportlab.platypus import BaseDocTemplate, PageTemplate
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Table, TableStyle, Paragraph, PageBreak, FrameBreak
from reportlab.platypus.flowables import Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4, mm, portrait
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase import cidfonts
from reportlab.lib import colors
from reportlab.platypus.frames import Frame
from reportlab.pdfbase.ttfonts import TTFont



def main():

    outFile  = "test.pdf"

    styles   = getSampleStyleSheet()
    doc      = SimpleDocTemplate( outFile )
    Story    = [ Spacer(1,5*mm) ]
    for i in range(100):
        bogustext = ( "This is Paragraph number %s. " % i ) * 20
        p = Paragraph( bogustext, style )
        Story.append( p )
        Story.append( Spacer( 1, 6*mm ) )
    doc.build( Story )

if __name__ == '__main__':
    main()
