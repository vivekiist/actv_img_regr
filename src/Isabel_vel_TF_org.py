# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Image Data Reader'
velocityf15binLEraw_corrected_2_subsampledvti = XMLImageDataReader(registrationName='Velocityf15.binLE.raw_corrected_2_subsampled.vti', FileName=['/media/vivek/extra/Work/actv_img_regr/data/Isabel_velocity_raw/Velocityf25.binLE.raw_corrected_2_subsampled.vti'])
velocityf15binLEraw_corrected_2_subsampledvti.PointArrayStatus = ['ImageScalars']

# Properties modified on velocityf15binLEraw_corrected_2_subsampledvti
velocityf15binLEraw_corrected_2_subsampledvti.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
velocityf15binLEraw_corrected_2_subsampledvtiDisplay = Show(velocityf15binLEraw_corrected_2_subsampledvti, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.Representation = 'Outline'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.ColorArrayName = ['POINTS', '']
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SelectTCoordArray = 'None'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SelectNormalArray = 'None'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SelectTangentArray = 'None'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.OSPRayScaleArray = 'ImageScalars'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SelectOrientationVectors = 'None'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.ScaleFactor = 24.900000000000002
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SelectScaleArray = 'ImageScalars'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.GlyphType = 'Arrow'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.GlyphTableIndexArray = 'ImageScalars'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.GaussianRadius = 1.245
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SetScaleArray = ['POINTS', 'ImageScalars']
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.OpacityArray = ['POINTS', 'ImageScalars']
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.ScalarOpacityUnitDistance = 2.4547867246797965
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.OpacityArrayName = ['POINTS', 'ImageScalars']
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.IsosurfaceValues = [32.05048370361328]
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SliceFunction = 'Plane'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.Slice = 24

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SliceFunction.Origin = [124.5, 124.5, 24.5]

# reset view to fit data
renderView1.ResetCamera(False)

# update the view to ensure updated data information
renderView1.Update()

# reset view to fit data
renderView1.ResetCamera(False)

# set scalar coloring
ColorBy(velocityf15binLEraw_corrected_2_subsampledvtiDisplay, ('POINTS', 'ImageScalars'))

# rescale color and/or opacity maps used to include current data range
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.RescaleTransferFunctionToDataRange(True, True)

# change representation type
velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SetRepresentationType('Volume')

# get color transfer function/color map for 'ImageScalars'
imageScalarsLUT = GetColorTransferFunction('ImageScalars')

# get opacity transfer function/opacity map for 'ImageScalars'
imageScalarsPWF = GetOpacityTransferFunction('ImageScalars')

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# invert the transfer function
imageScalarsLUT.InvertTransferFunction()

# invert the transfer function
imageScalarsLUT.InvertTransferFunction()

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
imageScalarsLUT.ApplyPreset('Blue - Green - Orange', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
imageScalarsLUT.ApplyPreset('Linear Green (Gr4L)', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
imageScalarsLUT.ApplyPreset('Viridis (matplotlib)', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
imageScalarsLUT.ApplyPreset('Linear YGB 1211g', True)

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.295251515228301, 0.09487179666757584, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.295251369476318, 0.09487179666757584, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.295251369476318, 0.10000000149011612, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.924701690673828, 0.12051282078027725, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.739427089691162, 0.12564103305339813, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.554152011871338, 0.13076923787593842, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.554152011871338, 0.1358974426984787, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.368877410888672, 0.1358974426984787, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.442503929138184, 0.18205128610134125, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.071954250335693, 0.19230769574642181, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.071954250335693, 0.20256410539150238, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.8866798877716064, 0.20256410539150238, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# invert the transfer function
imageScalarsLUT.InvertTransferFunction()

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 39.08887830050662, 0.6179487109184265, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 39.0888786315918, 0.6179487109184265, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 39.0888786315918, 0.6435897350311279, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 39.0888786315918, 0.6538461446762085, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 39.0888786315918, 0.6589744091033936, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 39.0888786315918, 0.6641026139259338, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.53305435180664, 0.684615433216095, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.162506103515625, 0.7000000476837158, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 37.79195785522461, 0.7102564573287964, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 37.05085754394531, 0.7410256862640381, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 36.86558151245117, 0.7461538910865784, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 36.309757232666016, 0.7615385055541992, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 36.309757232666016, 0.7666667103767395, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 36.12448501586914, 0.7666667103767395, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 36.12448501586914, 0.7717949151992798, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 35.75393295288086, 0.7820513248443604, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 34.82756042480469, 0.8076923489570618, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 34.64228439331055, 0.812820553779602, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 34.08646011352539, 0.8282051682472229, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.6277786758728325, 0.12051282078027725, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.813053131103516, 0.12051282078027725, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.183602809906006, 0.11538461595773697, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.109976291656494, 0.10000000149011612, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.480525970458984, 0.08974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.07948718219995499, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.036350250244141, 0.0743589773774147, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.0743589773774147, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.06923077255487442, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.66580057144165, 0.06923077255487442, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.66580057144165, 0.06410256773233414, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.66580057144165, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.036350250244141, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.036350250244141, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.66580057144165, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.480525970458984, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.295251369476318, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.109976291656494, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.739427089691162, 0.05384615436196327, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.99832820892334, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.62777853012085, 0.03333333507180214, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.257228851318359, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.257228851318359, 0.023076923564076424, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.442503929138184, 0.023076923564076424, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.813053131103516, 0.023076923564076424, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.813053131103516, 0.01794871874153614, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.99832820892334, 0.01794871874153614, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.183602809906006, 0.012820512987673283, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.99832820892334, 0.012820512987673283, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.813053131103516, 0.012820512987673283, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.62777853012085, 0.012820512987673283, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.62777853012085, 0.01794871874153614, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.62777853012085, 0.023076923564076424, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.62777853012085, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.813053131103516, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.99832820892334, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.183602809906006, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.368877410888672, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.554152011871338, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.739427089691162, 0.028205128386616707, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.924701690673828, 0.03333333507180214, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.592174530029297, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.592174530029297, 0.043589744716882706, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.777449131011963, 0.043589744716882706, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.962723731994629, 0.043589744716882706, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 8.147998809814453, 0.04871794953942299, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 8.518548011779785, 0.05384615436196327, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 8.70382308959961, 0.05384615436196327, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 9.074372291564941, 0.06410256773233414, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 9.259647369384766, 0.06923077255487442, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.000746726989746, 0.08974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.000746726989746, 0.09487179666757584, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.927120208740234, 0.11538461595773697, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.112394332885742, 0.12051282078027725, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.297669410705566, 0.12564103305339813, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.297669410705566, 0.13076923787593842, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.297669410705566, 0.14102564752101898, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.48294448852539, 0.14615385234355927, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.853493690490723, 0.1666666716337204, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.853493690490723, 0.17179487645626068, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.038768768310547, 0.17179487645626068, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.224042892456055, 0.17179487645626068, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.224042892456055, 0.17692308127880096, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.779867172241211, 0.18205128610134125, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.335691452026367, 0.19230769574642181, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.7062406539917, 0.19230769574642181, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.1974359005689621, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.19230769574642181, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.18717949092388153, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.076790809631348, 0.18205128610134125, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.262064933776855, 0.18205128610134125, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.262064933776855, 0.17179487645626068, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.17179487645626068, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.16153846681118011, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.15641026198863983, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.15128205716609955, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.14102564752101898, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.13076923787593842, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.11538461595773697, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.11025641113519669, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.1051282063126564, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.10000000149011612, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.262064933776855, 0.09487179666757584, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.7062406539917, 0.06923077255487442, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.335691452026367, 0.04871794953942299, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.965142250061035, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.779867172241211, 0.03333333507180214, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.594593048095703, 0.03333333507180214, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.409317970275879, 0.03333333507180214, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.224042892456055, 0.03333333507180214, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.038768768310547, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.48294448852539, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.297669410705566, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.371295928955078, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.186020851135254, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.000746726989746, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.186020851135254, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.556571006774902, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.74184513092041, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.297669410705566, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.48294448852539, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.668218612670898, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 11.853493690490723, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.038768768310547, 0.03846153989434242, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.038768768310547, 0.043589744716882706, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.224042892456055, 0.043589744716882706, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.594593048095703, 0.05384615436196327, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.779867172241211, 0.058974359184503555, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 12.965142250061035, 0.06410256773233414, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.150416374206543, 0.06410256773233414, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.335691452026367, 0.06410256773233414, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.520966529846191, 0.06923077255487442, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.7062406539917, 0.06923077255487442, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 34.08646011352539, 0.8333333730697632, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 34.27173614501953, 0.8282051682472229, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 35.1981086730957, 0.8230769634246826, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 36.309757232666016, 0.8076923489570618, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 38.53305435180664, 0.7666667103767395, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 38.90360641479492, 0.7564103007316589, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 39.829978942871094, 0.7358974814414978, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 40.015254974365234, 0.7358974814414978, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 40.57107925415039, 0.7153846621513367, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 40.756351470947266, 0.7051282525062561, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 40.941627502441406, 0.6948718428611755, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 41.12690353393555, 0.6897436380386353, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 41.6827278137207, 0.6641026139259338, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 41.6827278137207, 0.6589744091033936, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 41.86800003051758, 0.6538461446762085, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 42.23855209350586, 0.6384615302085876, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 42.609100341796875, 0.6282051205635071, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 42.79437255859375, 0.6230769157409668, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 43.350196838378906, 0.607692301273346, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 43.53547286987305, 0.6025640964508057, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 43.72074890136719, 0.6025640964508057, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 43.72074890136719, 0.5974358916282654, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 43.90602111816406, 0.5974358916282654, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.0912971496582, 0.5923076868057251, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.276573181152344, 0.5820512771606445, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 46.31459426879883, 0.48974359035491943, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 47.055694580078125, 0.45384615659713745, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 47.796791076660156, 0.4076923131942749, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 48.16734313964844, 0.3820512890815735, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 48.35261535644531, 0.36153846979141235, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 48.53789138793945, 0.3512820601463318, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 48.35261535644531, 0.35641026496887207, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 47.9820671081543, 0.39743590354919434, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 47.796791076660156, 0.4076923131942749, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 46.870418548583984, 0.5205128192901611, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 46.49987030029297, 0.5461538434028625, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 45.75876998901367, 0.6333333253860474, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 45.388221740722656, 0.6538461446762085, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.46184539794922, 0.7307692766189575, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.276573181152344, 0.7358974814414978, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.276573181152344, 0.7615385055541992, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.276573181152344, 0.7666667103767395, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.276573181152344, 0.7769231200218201, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.276573181152344, 0.7820513248443604, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 44.46184539794922, 0.7871795296669006, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 45.017669677734375, 0.7923077344894409, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 47.240966796875, 0.812820553779602, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 49.09371566772461, 0.8230769634246826, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 49.834815979003906, 0.8282051682472229, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 51.317012786865234, 0.8384615778923035, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 52.42866134643555, 0.848717987537384, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 53.35503387451172, 0.8538461923599243, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 53.725582122802734, 0.8538461923599243, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 53.725582122802734, 0.8589743971824646, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 53.910858154296875, 0.8589743971824646, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 53.910858154296875, 0.8641026020050049, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 54.83723068237305, 0.8743590116500854, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 55.578330993652344, 0.884615421295166, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 56.1341552734375, 0.884615421295166, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 56.504703521728516, 0.884615421295166, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 56.8752555847168, 0.884615421295166, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.24580383300781, 0.8794872164726257, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.98690414428711, 0.8641026020050049, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 58.172176361083984, 0.8538461923599243, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 58.542728424072266, 0.8384615778923035, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 58.72800064086914, 0.8282051682472229, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 58.72800064086914, 0.8230769634246826, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.09855270385742, 0.7974359393119812, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.09855270385742, 0.7871795296669006, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.09855270385742, 0.7769231200218201, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.09855270385742, 0.7717949151992798, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.2838249206543, 0.7717949151992798, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.46910095214844, 0.7923077344894409, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.65437698364258, 0.8076923489570618, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.210201263427734, 0.8435897827148438, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.210201263427734, 0.8589743971824646, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.39547348022461, 0.8743590116500854, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.58074951171875, 0.8897436261177063, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.76602554321289, 0.9153846502304077, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.951297760009766, 0.9307692646980286, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.951297760009766, 0.9410256743431091, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.951297760009766, 0.9461538791656494, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.951297760009766, 0.9410256743431091, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.76602554321289, 0.9256410598754883, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.39547348022461, 0.9051282405853271, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.210201263427734, 0.9000000357627869, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 60.024925231933594, 0.8743590116500854, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.83964920043945, 0.8692308068275452, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.46910095214844, 0.8025641441345215, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.46910095214844, 0.7974359393119812, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.46910095214844, 0.7820513248443604, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.2838249206543, 0.7820513248443604, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.2838249206543, 0.7717949151992798, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.09855270385742, 0.7512820959091187, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 59.09855270385742, 0.7410256862640381, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 58.72800064086914, 0.7051282525062561, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 58.542728424072266, 0.6897436380386353, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 58.357452392578125, 0.6692308187484741, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.98690414428711, 0.6384615302085876, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.80162811279297, 0.6230769157409668, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.61635208129883, 0.6025640964508057, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.43107986450195, 0.5820512771606445, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.43107986450195, 0.5769230723381042, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.0743589773774147, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.262064933776855, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 15.18843936920166, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 15.558988571166992, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 15.7442626953125, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 16.670637130737305, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 16.855911254882812, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 18.52338409423828, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 18.89393424987793, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 19.635032653808594, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 19.8203067779541, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 20.005582809448242, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 20.746681213378906, 0.06923077255487442, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 20.931955337524414, 0.0743589773774147, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 21.48777961730957, 0.07948718219995499, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 22.414154052734375, 0.08974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 22.599428176879883, 0.08974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.081626892089844, 0.10000000149011612, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.45217514038086, 0.10000000149011612, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 25.563823699951172, 0.11025641113519669, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 26.304922103881836, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 26.860746383666992, 0.12564103305339813, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 27.046022415161133, 0.13076923787593842, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 27.41657066345215, 0.1358974426984787, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 27.60184669494629, 0.14102564752101898, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 27.972394943237305, 0.14102564752101898, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 28.7134952545166, 0.14615385234355927, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 30.010417938232422, 0.15641026198863983, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 30.566242218017578, 0.15641026198863983, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 30.751516342163086, 0.16153846681118011, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 31.307340621948242, 0.16153846681118011, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 31.8631649017334, 0.1666666716337204, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 32.7895393371582, 0.17179487645626068, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 33.530635833740234, 0.18205128610134125, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 34.27173614501953, 0.18717949092388153, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 34.45701217651367, 0.19230769574642181, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 35.01283645629883, 0.20256410539150238, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 35.568660736083984, 0.20769231021404266, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 35.75393295288086, 0.21282051503658295, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 36.86558151245117, 0.2230769246816635, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 37.23613357543945, 0.2230769246816635, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.347782135009766, 0.23333333432674408, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.53305435180664, 0.23333333432674408, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.71833038330078, 0.23846153914928436, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.90360641479492, 0.23846153914928436, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.90360641479492, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 38.53305435180664, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 37.977230072021484, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 37.42140579223633, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 35.568660736083984, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 34.45701217651367, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 33.715911865234375, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 33.530635833740234, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 33.34536361694336, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 32.41899108886719, 0.2282051295042038, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 31.122066497802734, 0.1974359005689621, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 27.787120819091797, 0.1051282063126564, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 27.046022415161133, 0.08461538702249527, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 25.563823699951172, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 25.563823699951172, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 25.378549575805664, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 25.193275451660156, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 25.193275451660156, 0.03846153989434242, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 25.007999420166016, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.822725296020508, 0.01794871874153614, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.822725296020508, 0.012820512987673283, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.822725296020508, 0.01794871874153614, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.822725296020508, 0.023076923564076424, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.637451171875, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 24.45217514038086, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 23.711076736450195, 0.03846153989434242, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 23.15525245666504, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 22.96997833251953, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 22.414154052734375, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 22.228878021240234, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 21.673053741455078, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 21.302505493164062, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 19.8203067779541, 0.03846153989434242, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 19.264482498168945, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 19.079208374023438, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 18.89393424987793, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 16.670637130737305, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 15.7442626953125, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.632615089416504, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.44734001159668, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 14.262064933776855, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.891515731811523, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 13.520966529846191, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.371295928955078, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 10.186020851135254, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 9.259647369384766, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 9.074372291564941, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.962723731994629, 0.028205128386616707, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.777449131011963, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.03333333507180214, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.03846153989434242, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.295251369476318, 0.03846153989434242, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.924701690673828, 0.03846153989434242, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.739427089691162, 0.03846153989434242, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.368877410888672, 0.043589744716882706, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.813053131103516, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.257228851318359, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.8866798877716064, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.5161304473876953, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.33085560798645, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.960306167602539, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.775031328201294, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.589756727218628, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.775031328201294, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.960306167602539, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.33085560798645, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.5161304473876953, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.7014050483703613, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.071954250335693, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.257228851318359, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.62777853012085, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.813053131103516, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.183602809906006, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.368877410888672, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.739427089691162, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.924701690673828, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.480525970458984, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.66580057144165, 0.04871794953942299, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.66580057144165, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.851075172424316, 0.05384615436196327, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.406899452209473, 0.0743589773774147, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.592174530029297, 0.08461538702249527, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.592174530029297, 0.08974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.592174530029297, 0.09487179666757584, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.592174530029297, 0.10000000149011612, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.406899452209473, 0.10000000149011612, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.406899452209473, 0.1051282063126564, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.1051282063126564, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.11025641113519669, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.11538461595773697, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.12564103305339813, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.13076923787593842, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.12564103305339813, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 7.221624851226807, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 6.295251369476318, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 5.739427089691162, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.62777853012085, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 4.442503929138184, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.8866798877716064, 0.12051282078027725, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.7014050483703613, 0.11538461595773697, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.5161304473876953, 0.11538461595773697, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 3.145581007003784, 0.11025641113519669, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.960306167602539, 0.11025641113519669, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.775031328201294, 0.1051282063126564, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 2.219207286834717, 0.10000000149011612, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 0.7370092272758484, 0.08974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 0.1811850517988205, 0.07948718219995499, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.06923077255487442, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.058974359184503555, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.06410256773233414, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.06923077255487442, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.07948718219995499, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.08461538702249527, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.10000000149011612, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.1051282063126564, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.11025641113519669, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.13076923787593842, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.1358974426984787, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.14615385234355927, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.15128205716609955, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.16153846681118011, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.17179487645626068, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.17692308127880096, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.18205128610134125, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.19230769574642181, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.1974359005689621, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.20256410539150238, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.20769231021404266, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.21282051503658295, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.21794871985912323, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.2282051295042038, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.23846153914928436, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

# Properties modified on imageScalarsPWF
imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

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
renderView1.CameraPosition = [162.9084902928369, 124.40258912655527, -661.2603055103629]
renderView1.CameraFocalPoint = [124.50000000000024, 124.50000000000004, 24.49999999999991]
renderView1.CameraViewUp = [-0.20620896973870328, 0.9784381638977673, -0.011688465582294767]
renderView1.CameraParallelScale = 177.7659978736091

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).