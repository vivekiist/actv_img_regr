# trace generated using paraview version 5.8.0
import numpy as np

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def asteroid_volume(phi_values, theta_values):
	# create a new 'XML Image Data Reader'
	asteroid_28649vti = XMLImageDataReader(registrationName='Asteroid_28649.vti', FileName=['../data/asteroid_raw/Asteroid_28649.vti'])
	asteroid_28649vti.PointArrayStatus = ['tev', 'v02', 'v03']


	## Now generate all the images and save their param values also
	###################################################################
	all_params = []
	asteroid_28649vti.TimeArray = 'None'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	# show data in view
	asteroid_28649vtiDisplay = Show(asteroid_28649vti, renderView1, 'UniformGridRepresentation')
	# # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	# asteroid_28649vtiDisplay.ScaleTransferFunction.Points = [0.018699191510677338, 0.0, 0.5, 0.0, 0.4334716200828552, 1.0, 0.5, 0.0]

	# # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	# asteroid_28649vtiDisplay.OpacityTransferFunction.Points = [0.018699191510677338, 0.0, 0.5, 0.0, 0.4334716200828552, 1.0, 0.5, 0.0]

	# # init the 'Plane' selected for 'SliceFunction'
	# asteroid_28649vtiDisplay.SliceFunction.Origin = [149.5, 149.5, 149.5]

	# reset view to fit data
	renderView1.ResetCamera()
	# update the view to ensure updated data information
	renderView1.Update()
	# set scalar coloring
	ColorBy(asteroid_28649vtiDisplay, ('POINTS', 'tev'))
	# rescale color and/or opacity maps used to include current data range
	asteroid_28649vtiDisplay.RescaleTransferFunctionToDataRange(True, True)
	# change representation type
	asteroid_28649vtiDisplay.SetRepresentationType('Volume')
	# get color transfer function/color map for 'tev'
	tevLUT = GetColorTransferFunction('tev')
	# get opacity transfer function/opacity map for 'tev'
	tevPWF = GetOpacityTransferFunction('tev')
	# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
	# tevLUT.ApplyPreset('Cool to Warm (Extended)', True)
	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0
	# Properties modified on tevPWF
	tevPWF.Points = [0.018699191510677338, 0.0, 0.5, 0.0, 0.06664975732564926, 0.012820512987673283, 0.5, 0.0, 0.20211011171340942, 0.11538461595773697, 0.5, 0.0, 0.3111976683139801, 0.45897436141967773, 0.5, 0.0, 0.4334716200828552, 1.0, 0.5, 0.0]
	LoadPalette(paletteName='WhiteBackground')
	# current camera placement for renderView1
	renderView1.CameraPosition = [894.7313043350936, 484.27394761639357, 726.9979911469343]
	renderView1.CameraFocalPoint = [149.5, 149.5, 149.5]
	renderView1.CameraViewUp = [-0.26381906032697583, 0.9423541701219664, -0.20583518033108805]
	renderView1.CameraParallelScale = 258.9415957315472
	# hide color bar/color legend
	asteroid_28649vtiDisplay.SetScalarBarVisibility(renderView1, False)

	camera=GetActiveCamera()
	X, Y = np.meshgrid(phi_values, theta_values)
	XY_pairs = list(zip(X.ravel(), Y.ravel()))
	for i in range(len(XY_pairs)):
		if i%100 == 0:
			print ('generating sample: ' + str(i))

		renderView1.ResetCamera()
		camera.Elevation(XY_pairs[i][0]) 
		camera.Azimuth(XY_pairs[i][1])
		renderView1.Update()

		all_params.append([XY_pairs[i][0],XY_pairs[i][1]])
		outfile = '../data/asteroid_volume_images/hm/' \
					+ str("{:.4f}".format(XY_pairs[i][0])) + '_' + str("{:.4f}".format(XY_pairs[i][1])) + '.png'
		# save image out
		SaveScreenshot(outfile, 
						renderView1, 
						ImageResolution=[128, 128], 
						CompressionLevel='0')
		# undo camera
		camera.Elevation(-XY_pairs[i][0])
		camera.Azimuth(-XY_pairs[i][1])

## write the csv file with phi and theta values
	all_params  = np.asarray(all_params)
	np.savetxt('../data/asteroid_volume_images/hm/asteroid_viewparams_hm.csv', \
				all_params, delimiter=',')

if __name__ == "__main__":
	## regular sampled phi,theta vals
	num_samples = 100
	phi_values = np.linspace(-90, 90, num_samples) #phi -90,90 elevation
	theta_values = np.linspace(0, 360, num_samples) #theta 0 - 360 azimuth
	asteroid_volume(phi_values, theta_values)