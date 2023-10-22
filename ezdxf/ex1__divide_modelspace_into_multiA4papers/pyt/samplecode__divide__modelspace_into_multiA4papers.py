import ezdxf

# Create a new DXF document
doc = ezdxf.new(dxfversion='R2013')
msp = doc.modelspace()

# Define the radius of the circle (500mm)
radius = 500.0

# Add a circle to the modelspace
msp.add_circle(center=(0, 0), radius=radius)

# Define the A4 paper size (210mm x 297mm)
a4_width = 210.0
a4_height = 297.0

# Calculate the number of rows and columns
rows = 2  # Number of rows of A4 pages
columns = 2  # Number of columns of A4 pages

# Calculate the spacing between A4 pages
spacing_x = a4_width
spacing_y = a4_height

# Create a new layout for paperspace
layout = doc.layout()

# Loop to create multiple A4 pages
for row in range(rows):
    for col in range(columns):
        # Create a new viewport for each A4 page
        viewport = layout.add_viewport(
            center=(col * spacing_x + a4_width / 2, row * spacing_y + a4_height / 2),\
            size=(a4_width,a4_height), view_center_point=None, view_height=None )

# Save the DXF file
# doc.saveas('circles_on_a4_pages.dxf')
outFile = "dxf/out.dxf"
doc.saveas( outFile )
