# trace generated using paraview version 5.8.0
import numpy as np
# import pstats
# import cProfile

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def isabel_velocity_volume(phi_values, theta_values):
	# create a new 'XML Image Data Reader'
	velocityf15binLEraw_corrected_2_subsampledvti = XMLImageDataReader(FileName=['../data/Isabel_velocity_raw/Velocityf25.binLE.raw_corrected_2_subsampled.vti'])
	velocityf15binLEraw_corrected_2_subsampledvti.PointArrayStatus = ['ImageScalars']



	## Now generate all the images and save their param values also
	###################################################################
	all_params = []
	# Properties modified on velocityf15binLEraw_corrected_2_subsampledvti
	velocityf15binLEraw_corrected_2_subsampledvti.TimeArray = 'None'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	# show data in view
	velocityf15binLEraw_corrected_2_subsampledvtiDisplay = Show(velocityf15binLEraw_corrected_2_subsampledvti, renderView1, 'UniformGridRepresentation')
	
	# # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	# velocityf15binLEraw_corrected_2_subsampledvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

	# # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	# velocityf15binLEraw_corrected_2_subsampledvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]

	# # init the 'Plane' selected for 'SliceFunction'
	# velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SliceFunction.Origin = [124.5, 124.5, 24.5]


	# reset view to fit data
	renderView1.ResetCamera()
	# update the view to ensure updated data information
	renderView1.Update()
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
	# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
	imageScalarsLUT.ApplyPreset('Linear YGB 1211g', True)
	# invert the transfer function
	imageScalarsLUT.InvertTransferFunction()

	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0
	# Properties modified on imageScalarsPWF
	imageScalarsPWF.Points = [-0.004089686553925276, 0.0, 0.5, 0.0, -0.004089686553925276, 0.24358974397182465, 0.5, 0.0, 57.43107986450195, 0.571794867515564, 0.5, 0.0, 64.10096740722656, 1.0, 0.5, 0.0]
	LoadPalette(paletteName='WhiteBackground')
	# current camera placement for renderView1
	renderView1.CameraPosition = [162.9084902928369, 124.40258912655527, -661.2603055103629]
	renderView1.CameraFocalPoint = [124.50000000000024, 124.50000000000004, 24.49999999999991]
	renderView1.CameraViewUp = [-0.20620896973870328, 0.9784381638977673, -0.011688465582294767]
	renderView1.CameraParallelScale = 177.7659978736091
	# hide color bar/color legend
	velocityf15binLEraw_corrected_2_subsampledvtiDisplay.SetScalarBarVisibility(renderView1, False)

	camera=GetActiveCamera()
	for i in range(len(phi_values)):
		if i%100 == 0:
			print ('generating sample: ' + str(i))

		renderView1.ResetCamera()
		camera.Elevation(phi_values[i]) 
		camera.Azimuth(theta_values[i])
		renderView1.Update()

		all_params.append([phi_values[i],theta_values[i]])
		outfile = '../data/Isabel_velocity_volume_images/train/' \
					+ str("{:.4f}".format(phi_values[i])) + '_' + str("{:.4f}".format(theta_values[i])) + '.png'
		# save image out
		SaveScreenshot(outfile, 
						renderView1, 
						ImageResolution=[128, 128], 
						CompressionLevel='0')
		# undo camera
		camera.Elevation(-phi_values[i])
		camera.Azimuth(-theta_values[i])

## write the csv file with phi and theta values
	all_params  = np.asarray(all_params)
	np.savetxt('../data/Isabel_velocity_volume_images/train/Isabel_velocity_viewparams_train.csv', \
				all_params, delimiter=',')

	
#########################


# isabel_pressure_volume(phi_values, theta_values)

if __name__ == "__main__":
	## regular sampled phi,theta vals
	num_samples = 1024
	## Randomly generate value
	phi_values = np.random.uniform(-90, 90, num_samples) #phi -90,90 elevation
	theta_values = np.random.uniform(0,360, num_samples) #theta 0 - 360 azimuth
	# with cProfile.Profile() as profile:
	isabel_velocity_volume(phi_values, theta_values)
	# profile_result = pstats.Stats(profile)
	# profile_result.sort_stats(pstats.SortKey.TIME)
	# # profile_result.print_stats()
	# profile_result.dump_stats('isabel_pressure_volume.prof')
	# snakeviz isabel_pressure_volume.prof # to visualize the profile