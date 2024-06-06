# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Image Data Reader'
vortex_15vti = XMLImageDataReader(registrationName='vortex_15.vti', FileName=['/data1/vivekg/actv_img_regr/data/Vortex/vortex_15.vti'])
vortex_15vti.PointArrayStatus = ['ImageScalars']

# Properties modified on vortex_15vti
vortex_15vti.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
vortex_15vtiDisplay = Show(vortex_15vti, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
vortex_15vtiDisplay.Representation = 'Outline'
vortex_15vtiDisplay.ColorArrayName = ['POINTS', '']
vortex_15vtiDisplay.SelectTCoordArray = 'None'
vortex_15vtiDisplay.SelectNormalArray = 'None'
vortex_15vtiDisplay.SelectTangentArray = 'None'
vortex_15vtiDisplay.OSPRayScaleArray = 'ImageScalars'
vortex_15vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
vortex_15vtiDisplay.SelectOrientationVectors = 'None'
vortex_15vtiDisplay.ScaleFactor = 12.700000000000001
vortex_15vtiDisplay.SelectScaleArray = 'ImageScalars'
vortex_15vtiDisplay.GlyphType = 'Arrow'
vortex_15vtiDisplay.GlyphTableIndexArray = 'ImageScalars'
vortex_15vtiDisplay.GaussianRadius = 0.635
vortex_15vtiDisplay.SetScaleArray = ['POINTS', 'ImageScalars']
vortex_15vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
vortex_15vtiDisplay.OpacityArray = ['POINTS', 'ImageScalars']
vortex_15vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
vortex_15vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
vortex_15vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
vortex_15vtiDisplay.ScalarOpacityUnitDistance = 1.7320508075688772
vortex_15vtiDisplay.OpacityArrayName = ['POINTS', 'ImageScalars']
vortex_15vtiDisplay.IsosurfaceValues = [5.792023975634947]
vortex_15vtiDisplay.SliceFunction = 'Plane'
vortex_15vtiDisplay.Slice = 63

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
vortex_15vtiDisplay.ScaleTransferFunction.Points = [0.005305923987179995, 0.0, 0.5, 0.0, 11.578742027282715, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
vortex_15vtiDisplay.OpacityTransferFunction.Points = [0.005305923987179995, 0.0, 0.5, 0.0, 11.578742027282715, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
vortex_15vtiDisplay.SliceFunction.Origin = [63.5, 63.5, 63.5]

# reset view to fit data
renderView1.ResetCamera()

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(vortex_15vtiDisplay, ('POINTS', 'ImageScalars'))

# rescale color and/or opacity maps used to include current data range
vortex_15vtiDisplay.RescaleTransferFunctionToDataRange(True, True)

# change representation type
vortex_15vtiDisplay.SetRepresentationType('Volume')

# get color transfer function/color map for 'ImageScalars'
imageScalarsLUT = GetColorTransferFunction('ImageScalars')

# get opacity transfer function/opacity map for 'ImageScalars'
imageScalarsPWF = GetOpacityTransferFunction('ImageScalars')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
imageScalarsLUT.ApplyPreset('Cool to Warm (Extended)', True)

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 17.041185047943145, 0.2589743733406067, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 17.04118537902832, 0.2589743733406067, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.038768768310547, 0.39743590354919434, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 9.815471649169922, 0.4692307710647583, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.9512820839881897, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 1.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# rescale color and/or opacity maps used to exactly fit the current data range
vortex_15vtiDisplay.RescaleTransferFunctionToDataRange(False, True)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1430, 1262)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-61.0608454576273, 340.6686830542822, -233.56036312283325]
renderView1.CameraFocalPoint = [63.49999999999998, 63.500000000000064, 63.500000000000064]
renderView1.CameraViewUp = [0.6493768794726011, -0.40081301564071514, -0.6462651119311817]
renderView1.CameraParallelScale = 109.9852262806237

outfile = './test01.png'
SaveScreenshot(outfile, 
                renderView1, 
                ImageResolution=[128, 128], 
                CompressionLevel='0')

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).