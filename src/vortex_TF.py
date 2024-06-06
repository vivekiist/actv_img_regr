# trace generated using paraview version 5.10.1
import numpy as np

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def vortex_volume(phi_values, theta_values):
	# create a new 'XML Image Data Reader'
	vortex_15vti = XMLImageDataReader(registrationName='vortex_15.vti', FileName=['../data/vortex_raw/vortex_15.vti'])
	vortex_15vti.PointArrayStatus = ['ImageScalars']


	## Now generate all the images and save their param values also
	###################################################################
	all_params = []
	# Properties modified on vortex_15vti
	vortex_15vti.TimeArray = 'None'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')

	# show data in view
	vortex_15vtiDisplay = Show(vortex_15vti, renderView1, 'UniformGridRepresentation')
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
	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0
	# Properties modified on imageScalarsPWF
	imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 1.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]
	LoadPalette(paletteName='WhiteBackground')
	# current camera placement for renderView1
	renderView1.CameraPosition = [-61.0608454576273, 340.6686830542822, -233.56036312283325]
	renderView1.CameraFocalPoint = [63.49999999999998, 63.500000000000064, 63.500000000000064]
	renderView1.CameraViewUp = [0.6493768794726011, -0.40081301564071514, -0.6462651119311817]
	renderView1.CameraParallelScale = 109.9852262806237
	# hide color bar/color legend
	vortex_15vtiDisplay.SetScalarBarVisibility(renderView1, False)

	camera=GetActiveCamera()
	for i in range(len(phi_values)):
		if i%100 == 0:
			print ('generating sample: ' + str(i))
		renderView1.ResetCamera()
		camera.Elevation(phi_values[i]) 
		camera.Azimuth(theta_values[i])
		renderView1.Update()

		all_params.append([phi_values[i],theta_values[i]])
		outfile = '../data/vortex_volume_images/train/' \
					+ str("{:.4f}".format(phi_values[i])) + '_' + str("{:.4f}".format(theta_values[i])) + '.png'
		SaveScreenshot(outfile, 
						renderView1, 
						ImageResolution=[128, 128], 
						CompressionLevel='0')
		# undo camera
		camera.Elevation(-phi_values[i])
		camera.Azimuth(-theta_values[i])

## write the csv file with phi and theta values
	all_params  = np.asarray(all_params)
	np.savetxt('../data/vortex_volume_images/train/vortex_viewparams_train.csv', \
				all_params, delimiter=',')

if __name__ == "__main__":
	## regular sampled phi,theta vals
	num_samples = 1024
	## Randomly generate value
	phi_values = np.random.uniform(-90, 90, num_samples) #phi -90,90 elevation
	theta_values = np.random.uniform(0,360, num_samples) #theta 0 - 360 azimuth
	vortex_volume(phi_values, theta_values)