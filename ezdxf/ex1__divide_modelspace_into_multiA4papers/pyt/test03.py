from typing import cast
import ezdxf
from ezdxf.layouts import Paperspace
from ezdxf.enums import TextEntityAlignment

doc = ezdxf.new( dxfversion="R2010", setup=True )
doc.units = ezdxf.units.M
doc.layers.add("TEXTLAYER", color=0)
msp = doc.modelspace()

msp.add_line( (0, 0), (20, 0), dxfattribs={"color": 0} )
msp.add_text("Test", dxfattribs={"layer": "TEXTLAYER"}).set_placement(
    (0, 0.2), align=TextEntityAlignment.CENTER
)

# get the existing (default) "Layout1":
psp = cast( Paperspace, doc.layout("Layout1") )
a4_width = 210
a4_height = 297

# reset page properties:
psp.page_setup(size=(a4_width, a4_height), margins=(0, 0, 0, 0))

# add a new viewport to the paperspace:
psp.add_viewport(
    # center of viewport in paperspace:
    center=(a4_width * 0.5, a4_height * 0.5),
    # size of the viewport in paperspace:
    size=(a4_width, a4_height),
    # Define the portion of the modelspace to show.
    # center in modelspace:
    view_center_point=(0, 0),
    # the view height to show in modelspace units:
    view_height=100,
    # The view width is proportional to the viewport size!
)
doc.saveas("test.dxf")
