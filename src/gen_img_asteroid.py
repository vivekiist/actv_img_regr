# trace generated using paraview version 5.10.1
import numpy as np
import os
import json
# import pstats
import cProfile
import argparse


#### import the simple module from the paraview
from paraview.simple import *

# parse arguments
def parse_args():
	parser = argparse.ArgumentParser(description="View Generator for Asteroid Dataset")
	parser.add_argument("--inFile", required=True, type=str, help="Path of the input dataset")
	parser.add_argument("--varName", required=True, type=str, help="Name of the input variable to be visualised")
	parser.add_argument("--view_params", required=True, help="List of [phi_value, theta_value] pairs")
	parser.add_argument("--outPath", required=True, type=str, help="Path of the Output images and csv dataset")
	return parser.parse_args()


def asteroid_volume(args):
	# Extract all phi_values from phi_theta_pairs
	phi_theta_pairs = np.array(json.loads(args.view_params))
	phi_values = phi_theta_pairs[:, 0]
	theta_values = phi_theta_pairs[:,1]
	# print (phi_theta_pairs)
	# create a new 'XML Image Data Reader'
	asteroid_28649vti = XMLImageDataReader(FileName=args.inFile)
	asteroid_28649vti.PointArrayStatus = [args.varName]



	## Now generate all the images and save their param values also
	###################################################################
	# all_params = []
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
	
#########################

if __name__ == "__main__":
	#### disable automatic camera reset on 'Show'
	paraview.simple._DisableFirstRenderCameraReset()
	args = parse_args()
	mixfrac_volume(args)

	# with cProfile.Profile() as profile:
	# 	vortex_volume(args)
	# profile_result = pstats.Stats(profile)
	# profile_result.sort_stats(pstats.SortKey.TIME)
	# profile_result.print_stats()
	# profile_result.dump_stats('./vortex_volume.prof')
	# snakeviz vortex_volume.prof # to visualize the profile