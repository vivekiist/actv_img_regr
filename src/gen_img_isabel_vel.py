# trace generated using paraview version 5.8.0
import numpy as np
import os
import json
import pstats
import cProfile
import argparse


#### import the simple module from the paraview
from paraview.simple import *

# parse arguments
def parse_args():
	parser = argparse.ArgumentParser(description="View Generator for Isabel Velocity Dataset")
	parser.add_argument("--inFile", required=True, type=str, help="Path of the input dataset")
	parser.add_argument("--varName", required=True, type=str, help="Name of the input variable to be visualised")
	parser.add_argument("--view_params", required=True, help="List of [phi_value, theta_value] pairs")
	parser.add_argument("--outPath", required=True, type=str, help="Path of the Output images and csv dataset")
	return parser.parse_args()


def isabel_velocity_volume(args):
	# Extract all phi_values from phi_theta_pairs
	phi_theta_pairs = np.array(json.loads(args.view_params))
	phi_values = phi_theta_pairs[:, 0]
	theta_values = phi_theta_pairs[:,1]
	# print (phi_theta_pairs)
	# create a new 'XML Image Data Reader'
	velocityf15binLEraw_corrected_2_subsampledvti = XMLImageDataReader(FileName=args.inFile)
	velocityf15binLEraw_corrected_2_subsampledvti.PointArrayStatus = [args.varName]



	## Now generate all the images and save their param values also
	###################################################################
	# all_params = []
	# Properties modified on velocityf15binLEraw_corrected_2_subsampledvti
	velocityf15binLEraw_corrected_2_subsampledvti.TimeArray = 'None'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	# get layout
	# layout1 = GetLayout()
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
	for i in range(len(phi_theta_pairs)):
		# if i%10 == 0:
		# 	print ('generating sample: ' + str(i))
		renderView1.ResetCamera()
		camera.Elevation(phi_values[i]) 
		camera.Azimuth(theta_values[i])
		renderView1.Update()

		# all_params.append([phi_values[i],theta_values[i]])
		outfile = os.path.join(args.outPath, f"{phi_values[i]:.4f}_{theta_values[i]:.4f}.png")
		# save image out
		SaveScreenshot(outfile, 
						renderView1, 
						ImageResolution=[128, 128], 
						CompressionLevel='0')
		# undo camera
		camera.Elevation(-phi_values[i])
		camera.Azimuth(-theta_values[i])

# ## write the csv file with phi and theta values
# 	all_params  = np.asarray(all_params)
# 	np.savetxt('../data/Isabel_velocity_volume_images/train2/isabel_pr_viewparams_train2.csv', \
# 				all_params, delimiter=',')

	
#########################


# isabel_velocity_volume(phi_values, theta_values)

if __name__ == "__main__":
	#### disable automatic camera reset on 'Show'
	paraview.simple._DisableFirstRenderCameraReset()
	args = parse_args()

	# with cProfile.Profile() as profile:
	isabel_velocity_volume(args)
	# profile_result = pstats.Stats(profile)
	# profile_result.sort_stats(pstats.SortKey.TIME)
	# profile_result.print_stats()
	# profile_result.dump_stats('./isabel_velocity_volume.prof')
	# snakeviz isabel_velocity_volume.prof # to visualize the profile