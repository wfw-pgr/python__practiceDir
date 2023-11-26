import os, sys
from reportlab.pdfgen           import canvas
from reportlab.lib.pagesizes    import A4, portrait, landscape
from reportlab.pdfbase          import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.units        import mm
from reportlab.lib.colors       import deeppink, yellow
from reportlab.platypus         import Paragraph, Spacer, PageBreak, FrameBreak, SimpleDocTemplate
from reportlab.lib.styles       import ParagraphStyle, ParagraphStyle
from reportlab.lib.styles       import getSampleStyleSheet
from reportlab.lib.enums        import TA_JUSTIFY, TA_RIGHT, TA_CENTER, TA_LEFT

# ========================================================= #
# ===  generate__pdfFromText                            === #
# ========================================================= #

def generate__pdfFromText( texts=None, outFile=None ):

    # ------------------------------------------------- #
    # --- [1] argumentes                            --- #
    # ------------------------------------------------- #
    if ( texts   is None ): sys.exit( "[generate__pdfFromText.py] texts   == ???" )
    if ( outFile is None ): sys.exit( "[generate__pdfFromText.py] outFile == ???" )

    # ------------------------------------------------- #
    # --- [2] generate pdf file                     --- #
    # ------------------------------------------------- #
    # A4（横）の新規PDFファイルを作成
    p = SimpleDocTemplate( outFile, pagesize=portrait(A4) )

    # スタイルの指定
    stylesheet=getSampleStyleSheet()
    normalStyle = stylesheet['Normal']

    # styles                  = getSampleStyleSheet()
    # my_style                = styles['Normal']
    # my_style.fontSize       = 10.0
    # my_style.leading        = 4.0 # 段落内の行間
    
    # テキストの挿入
    story                   = [ Spacer( 1, 5*mm ) ]
    for ik,atext in enumerate(texts):
        story.append( Paragraph( atext, my_style ) )
        story.append( Spacer(1, 3*mm) )
    
    p.build(story)


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    texts   = [ "全体的なLINACパワーが増加しない限り。核核反応は非効率的ですが、より長い照射時間を使用してラジウムターゲットをプライム位置に配置し、質量RA-226ターゲットを持つことにより、収量を大きく増加させることができます。積み重ねて、多数のラジウム針がこれを行う可能性があります。さらに、貴重なAC-225がターゲットから抽出された後、残留RA-226が継続的なプロセス（RA/ACジェネレーター）で再び照射できることを考慮すると、プロジェクトは時代遅れの放射性科目をゆっくりと減らし、AC-225を生成することもできます。謝辞オーストラリアのVarian CorporationのOle Hagen氏に、2100C CLINACとシドニーのニューサウスウェールズ大学のオレグ・スシコフ教授に関する情報を提供してくれたことに感謝します。参考文献Actinium Pharmaceuticals、2005年。オンライン：http：//www.actiniumpharma ceuticals.com/ ahnesjo、A.、1989。光子用量計算。医薬品。Phys。18、377.allen、B.J.、Raja、C.、Rizvi、S.、Li、Y.、Tsui、W.、Zhang、D.、Song、E.、Fa Qu、C.、Kearsley、J.、Graham、P.、Thompson、J.、2004。がんの標的アルファ療法。Phys。医薬品。Biol。49、3703–3712。Allyn and Bacon、1966。核物理学の基礎、Boston.Berman、B.、1976。ローレンス・リバモア研究所[64HA2]。Boll、R.A.、Malkemus、D.、Mirzadeh、S.、2005。Appl。radiat.isot。62（5）、667–679（Epub 2005 1月28日）。Bueche、F.、1969。科学者とエンジニアのための物理学の紹介。McGraw-Hill Co.、ニューヨーク。光核実験センターデータ（CDFE）、2005年。モスクワ。ref。（T、ヤング、72）。オンライン：http：//cdfe.sinp.msu.ru/index.en.html dimitar、K.、1998。Bremsstrahlungで生成されたいくつかの異性体収率比の研究。Appl。radiat。ISOT。49（8）、989–995。Dowsett、D.J.、Kenny、P.A.、Johnston、R.E.、2001。診断イメージングの物理学。オックスフォード大学出版局、オックスフォード。Emilio、S.、1975。核と粒子、第2版。議会図書館のカタログ。Frauenfelder、H.、Henley、E.、1991。亜原子プロセス、第2版。Prentice-Hall、Englewood Cliffs、ニュージャージー州。Koch、L.、et al。、1999。がん療法におけるAC-225の生産とBI-213娘の適用。Czechoslovak J. Phys。49（Suppl。S1）、817–822。Korea Atomic Energy Research Institute、2005年。オンライン：http：// atom。kaeri.re.kr/cgi-bin/readgam meyerhof、W.、1967。核物理学の要素。ニューヨーク州マクグローヒル。Mohan、R.、et al。、1985。医療線形加速器からの光子のエネルギーと角度分布。医薬品。Phys。12、592。Nordell、B.、et al。、1984。Bremsstrah-Lung標的からの角の分布と収量（放射線療法の場合）。Phys。医薬品。Biol。29、797–810。Spring、K.H.、1960。光子と電子。Methuen＆Co。Ltd.、ロンドン。Vicente、R.、Sordi、G.M.、2004。HiromotoG。Health Phys。86（5）、497–504。Wehr、M.、Richards、J.、1974。原子の物理学。Addison-Wesley Inc.、リーディング、マサチューセッツ州。ウィリアムズ、W.S.C.、1991年。核および粒子物理学。オックスフォードサイエンス出版物、オックスフォード。/応用放射と同位体64（2006）979–988 988" ]
    
    outFile = "pdf/sample.pdf"
    generate__pdfFromText( texts=texts, outFile=outFile )
    
