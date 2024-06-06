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
	parser = argparse.ArgumentParser(description="View Generator for Vortex Dataset")
	parser.add_argument("--inFile", required=True, type=str, help="Path of the input dataset")
	parser.add_argument("--varName", required=True, type=str, help="Name of the input variable to be visualised")
	parser.add_argument("--view_params", required=True, help="List of [phi_value, theta_value] pairs")
	parser.add_argument("--outPath", required=True, type=str, help="Path of the Output images and csv dataset")
	return parser.parse_args()


def vortex_volume(args):
	# Extract all phi_values from phi_theta_pairs
	phi_theta_pairs = np.array(json.loads(args.view_params))
	phi_values = phi_theta_pairs[:, 0]
	theta_values = phi_theta_pairs[:,1]
	# print (phi_theta_pairs)
	# create a new 'XML Image Data Reader'
	vortex_15vti = XMLImageDataReader(FileName=args.inFile)
	vortex_15vti.PointArrayStatus = [args.varName]



	## Now generate all the images and save their param values also
	###################################################################
	# all_params = []
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
	imageScalarsLUT.ApplyPreset('Spectral_lowBlue', True)
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
	vortex_15vtiDisplay.SetScalarBarVisibility(renderView1, False)


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
	vortex_volume(args)

	# with cProfile.Profile() as profile:
	# 	vortex_volume(args)
	# profile_result = pstats.Stats(profile)
	# profile_result.sort_stats(pstats.SortKey.TIME)
	# profile_result.print_stats()
	# profile_result.dump_stats('./vortex_volume.prof')
	# snakeviz vortex_volume.prof # to visualize the profile