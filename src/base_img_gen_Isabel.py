# trace generated using paraview version 5.8.0
import random
import numpy as np

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def isabel_pressure_volume():

	# create a new 'XML Image Data Reader'
	pf25binLEraw_corrected_2_subsampledvti = XMLImageDataReader(FileName=['../data/Isabel_pressure_raw/Pf25.binLE.raw_corrected_2_subsampled.vti'])
	pf25binLEraw_corrected_2_subsampledvti.PointArrayStatus = ['ImageScalars']

	## regular sampled phi,theta vals
	num_samples = 10000

	## Now generate all the images and save their param values also
	###################################################################
	all_params = []
	for i in range(num_samples):

		if i%100 == 0:
			print ('generating sample: ' + str(i))

		# get active view
		renderView1 = GetActiveViewOrCreate('RenderView')
		# get layout
		layout1 = GetLayout()
		# show data in view
		pf25binLEraw_corrected_2_subsampledvtiDisplay = Show(pf25binLEraw_corrected_2_subsampledvti, renderView1, 'UniformGridRepresentation')
		# reset view to fit data
		renderView1.ResetCamera()
		# update the view to ensure updated data information
		renderView1.Update()
		# set scalar coloring
		ColorBy(pf25binLEraw_corrected_2_subsampledvtiDisplay, ('POINTS', 'ImageScalars'))
		# rescale color and/or opacity maps used to include current data range
		pf25binLEraw_corrected_2_subsampledvtiDisplay.RescaleTransferFunctionToDataRange(True, True)
		# change representation type
		pf25binLEraw_corrected_2_subsampledvtiDisplay.SetRepresentationType('Volume')
		# get color transfer function/color map for 'ImageScalars'
		imageScalarsLUT = GetColorTransferFunction('ImageScalars')
		# get opacity transfer function/opacity map for 'ImageScalars'
		imageScalarsPWF = GetOpacityTransferFunction('ImageScalars')
		# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
		imageScalarsLUT.ApplyPreset('Spectral_lowBlue', True)
		# Hide orientation axes
		renderView1.OrientationAxesVisibility = 0
		# Properties modified on imageScalarsPWF
		imageScalarsPWF.Points = [-4931.54248046875, 0.0, 0.5, 0.0, -4931.54248046875, 1.0, 0.5, 0.0, 0.0, 0.18717949092388153, 0.5, 0.0, 2594.9736328125, 1.0, 0.5, 0.0]
		LoadPalette(paletteName='WhiteBackground')
		# current camera placement for renderView1
		renderView1.CameraPosition = [114.58039117050782, 83.61589529485812, -661.0454102918769]
		renderView1.CameraFocalPoint = [124.5, 124.5, 24.5]
		renderView1.CameraViewUp = [0.02610772339687559, 0.9978636428814608, -0.05988770319834204]
		renderView1.CameraParallelScale = 177.7659978736091

		## Randomly generate value
		phi_val = random.uniform(-90, 90) #phi -90,90 elevation
		theta_val = random.uniform(0,360) #theta 0 - 360 azimuth

		camera=GetActiveCamera()
		renderView1.ResetCamera()
		camera.Elevation(phi_val) 
		camera.Azimuth(theta_val)
		renderView1.Update()

		all_params.append([phi_val,theta_val])
		outfile = '../data/Isabel_pressure_volume_images/train2/' \
					+ str("{:.4f}".format(phi_val)) + '_' + str("{:.4f}".format(theta_val)) + '.png'
		# save image out
		SaveScreenshot(outfile, 
						renderView1, 
						ImageResolution=[128, 128], 
						CompressionLevel='0')
		# undo camera
		camera.Elevation(-phi_val)
		camera.Azimuth(-theta_val)

	## write the csv file out with phi and theta values
	all_params  = np.asarray(all_params)
	np.savetxt('../data/Isabel_pressure_volume_images/train2/isabel_pr_viewparams_train2.csv', \
				all_params, delimiter=',')

	
#########################
isabel_pressure_volume()
