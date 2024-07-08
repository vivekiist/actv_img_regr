# trace generated using paraview version 5.8.0
import numpy as np

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def mixfrac_volume(phi_values, theta_values):
	# create a new 'XML Image Data Reader'
	jet_mixfrac_0041dat_2_subsampledvti = XMLImageDataReader(FileName=['../../../data/mixfrac_raw/jet_mixfrac_0041.dat_2_subsampled.vti'])
	jet_mixfrac_0041dat_2_subsampledvti.PointArrayStatus = ['ImageScalars']


	## Now generate all the images and save their param values also
	###################################################################
	all_params = []
	# Properties modified on jet_mixfrac_0041dat_2_subsampledvti
	jet_mixfrac_0041dat_2_subsampledvti.TimeArray = 'None'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')

	# show data in view
	jet_mixfrac_0041dat_2_subsampledvtiDisplay = Show(jet_mixfrac_0041dat_2_subsampledvti, renderView1, 'UniformGridRepresentation')
	# # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	# jet_mixfrac_0041dat_2_subsampledvtiDisplay.ScaleTransferFunction.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 0.9933253526687622, 1.0, 0.5, 0.0]

	# # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	# jet_mixfrac_0041dat_2_subsampledvtiDisplay.OpacityTransferFunction.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 0.9933253526687622, 1.0, 0.5, 0.0]

	# # init the 'Plane' selected for 'SliceFunction'
	# jet_mixfrac_0041dat_2_subsampledvtiDisplay.SliceFunction.Origin = [119.5, 179.5, 29.5]

	# reset view to fit data
	renderView1.ResetCamera()
	# update the view to ensure updated data information
	renderView1.Update()
	# set scalar coloring
	ColorBy(jet_mixfrac_0041dat_2_subsampledvtiDisplay, ('POINTS', 'ImageScalars'))
	# rescale color and/or opacity maps used to include current data range
	jet_mixfrac_0041dat_2_subsampledvtiDisplay.RescaleTransferFunctionToDataRange(True, True)
	# change representation type
	jet_mixfrac_0041dat_2_subsampledvtiDisplay.SetRepresentationType('Volume')
	# get color transfer function/color map for 'ImageScalars'
	imageScalarsLUT = GetColorTransferFunction('ImageScalars')
	# get opacity transfer function/opacity map for 'ImageScalars'
	imageScalarsPWF = GetOpacityTransferFunction('ImageScalars')
	# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
	imageScalarsLUT.ApplyPreset('Black-Body Radiation', True)
	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0
	# Properties modified on imageScalarsPWF
	imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, 0.07086057960987091, 0.14102564752101898, 0.5, 0.0, 0.9933253526687622, 1.0, 0.5, 0.0]
	LoadPalette(paletteName='WhiteBackground')
	# current camera placement for renderView1
	renderView1.CameraPosition = [87.46580853872983, 287.4587689757011, 862.8540051798168]
	renderView1.CameraFocalPoint = [119.49999999999984, 179.4999999999999, 29.499999999999993]
	renderView1.CameraViewUp = [0.04789805648068396, 0.9908075834107278, -0.12651525141723785]
	renderView1.CameraParallelScale = 217.6482253545845
	# hide color bar/color legend
	jet_mixfrac_0041dat_2_subsampledvtiDisplay.SetScalarBarVisibility(renderView1, False)

	camera=GetActiveCamera()
	for i in range(len(phi_values)):
		if i%100 == 0:
			print ('generating sample: ' + str(i))
		renderView1.ResetCamera()
		camera.Elevation(phi_values[i]) 
		camera.Azimuth(theta_values[i])
		renderView1.Update()

		all_params.append([phi_values[i],theta_values[i]])
		outfile = '../../../data/mixfrac_volume_images/train/' \
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
	np.savetxt('../../../data/mixfrac_volume_images/train/mixfrac_viewparams_train.csv', \
				all_params, delimiter=',')
	
#########################

if __name__ == "__main__":
	## regular sampled phi,theta vals
	num_samples = 10000
	## Randomly generate value
	phi_values = np.random.uniform(-90, 90, num_samples) #phi -90,90 elevation
	theta_values = np.random.uniform(0,360, num_samples) #theta 0 - 360 azimuth
	mixfrac_volume(phi_values, theta_values)