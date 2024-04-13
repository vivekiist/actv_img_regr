# trace generated using paraview version 5.8.0
# Use this command to run this code in pvpython 
# /data/paraview_server/ParaView-5.12.0-RC3-egl-MPI-Linux-Python3.10-x86_64/bin/pvpython /data/test/isa_gen/gen_img.py --inFile '/data/active_image_regression/datasets/Isabel_pressure/Pf25.binLE.raw_corrected_2_subsampled.vti' --phi_val -17.6354 --theta_val 311.0394 --outPath '/data/test/isa_gen/imgs/'

import os
import argparse


#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# parse arguments
def parse_args():
	parser = argparse.ArgumentParser(description="View Generator for Isabel Pressure Dataset")

	parser.add_argument("--inFile", required=True, type=  str,
											help="Path of the input dataset")
	parser.add_argument("--varName", required=True, type=  str,
										help="Name of the input variable to be visualised")
	parser.add_argument("--phi_val", type=float, default=0,
											help="Elevation: Phi value")
	parser.add_argument("--theta_val", type=float, default=0,
											help="Azimuth: Theta value")
	parser.add_argument("--outPath", required=True, type=str,
												help="Path of the Output images and csv dataset")

	return parser.parse_args()

# the main function
def generate_view_image(args):
	# create a new 'XML Image Data Reader'
	pf25binLEraw_corrected_2_subsampledvti = XMLImageDataReader(FileName=args.inFile)
	pf25binLEraw_corrected_2_subsampledvti.PointArrayStatus = [args.varName]

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

	camera=GetActiveCamera()
	renderView1.ResetCamera()
	camera.Elevation(args.phi_val) 
	camera.Azimuth(args.theta_val)
	renderView1.Update()

	outfile = os.path.join(args.outPath, f"{args.phi_val:.4f}_{args.theta_val:.4f}.png")
	# save image out
	SaveScreenshot(outfile, 
					renderView1, 
					ImageResolution=[128, 128], 
					CompressionLevel='0')
	# undo camera
	camera.Elevation(-args.phi_val)
	camera.Azimuth(-args.theta_val)
				
if __name__ == "__main__":
	generate_view_image(parse_args())