// /export:mexFunction /dll 
// export http_proxy=http://proxy.rm.hq.corp:80


#ifndef FEATURES_CNN_H
#define FEATURES_CNN_H

#include <stdio.h>
#include <string>
#include <vector>
#include <ctime>

#include "mex.h"
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

// NOTE FROM CAFFE:
// Internally, data is stored with dimensions reversed from Caffe's:
// e.g., if the Caffe blob axes are (num, channels, height, width),
// the matcaffe data is stored as (width, height, channels, num)
// where width is the fastest dimension.

// Data coming in from matlab needs to be in the order
// [width, height, channels, images] where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct format in W x H x C with BGR channels :
// permute channels from RGB to BGR
// im_data = im(:, : , [3, 2, 1]);
// flip width and height to make width the fastest dimension
// im_data = permute(im_data, [2, 1, 3]); convert from uint8 to single
// im_data = single(im_data);
// reshape to a fixed size(e.g., 227x227).
// im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');
// subtract mean_data(already in W x H x C with BGR channels)
// im_data = im_data - mean_data;
// If you have multiple images, cat them with cat(4, ...)

// Wrapper for output blob information
struct blob_info
{
	int width, height, channels, num;
	blob_info() { width = 1; height = 1; channels = 4096; num = 1; }
};

// Save Net state
static std::vector<shared_ptr<Net<float>>> _net;
#define NET_FC _net[0]
#define NET_CONV5 _net[1]
#define NET_CONV3 _net[2]

// Create Network and associate the model with their weights
static void loadNet(string prototxt, string model, string path)
{
	if (_net.empty())
	{
		// Prototxt
		std::string txtProto = path + prototxt + ".prototxt";
		mexPrintf("loading: %s\n", txtProto.c_str());
		std::string txtProtoConv5 = path + prototxt + "-conv5.prototxt";
		mexPrintf("loading: %s\n", txtProtoConv5.c_str());
		std::string txtProtoConv3 = path + prototxt + "-conv3.prototxt";
		mexPrintf("loading: %s\n", txtProtoConv3.c_str());
		// Caffemodel
		std::string binProto = path + model + ".caffemodel";
		mexPrintf("loading: %s\n", binProto.c_str());

		static boost::shared_ptr<Net<float>> netFC(new Net<float>(txtProto, caffe::TEST));
		netFC->CopyTrainedLayersFrom(binProto);
		_net.push_back(netFC);
		static boost::shared_ptr<Net<float>> netConv5(new Net<float>(txtProtoConv5, caffe::TEST));
		netConv5->CopyTrainedLayersFrom(binProto);
		_net.push_back(netConv5);
		static boost::shared_ptr<Net<float>> netConv3(new Net<float>(txtProtoConv3, caffe::TEST));
		netConv3->CopyTrainedLayersFrom(binProto);
		_net.push_back(netConv3);		

		Caffe::SetDevice(0);
		Caffe::set_mode(Caffe::GPU);
	}
	else
		mexPrintf("Net already existing...\n");
}

static void unloadNet()
{
	if (!_net.empty())
		_net.clear();
	else
		mexPrintf("No Net found to be deleted...\n");
}

static blob_info getSize(string layer_id)
{
	shared_ptr<Net<float>> net = NET_FC;
	const boost::shared_ptr<caffe::Blob<float>> feature_blob = net->blob_by_name(layer_id);
	blob_info b;
	b.width = feature_blob->width();
	b.height = feature_blob->height();
	b.channels = feature_blob->channels();
	return b;
}

static blob_info setupFeatures(float *inputImg, int inputW, int inputH, int inputSamples, string layer_id)
{
	if (_net.empty())
		mexErrMsgTxt("Net not initialised!");
	if ((inputW != 227 || inputH != 227) && !layer_id.compare("fc7"))
		mexErrMsgTxt("Fully Connected Layers require an input image of 227x227");

	shared_ptr<Net<float>> net;
	if (!layer_id.compare("fc6") || !layer_id.compare("fc7") || !layer_id.compare("prob"))
		net = NET_FC;
	else if (!layer_id.compare("conv5"))
		net = NET_CONV5;
	else
		net = NET_CONV3;

	// Copy img data to the first layer of the CNN
	boost::shared_ptr<Blob<float>> input_layer = net->blob_by_name("data"); 
	
	// New output dimensions if changed
	// mexPrintf("inputW: %d W: %d inputH: %d H: %d\n", inputW, input_layer->width(), inputH, input_layer->height());
	input_layer->Reshape(inputSamples, 3, inputH, inputW);
	net->Reshape();

	float* input_blob = input_layer->mutable_cpu_data();
	// int dim_features = input_layer->count() / input_layer->num(); // batch = 1 (up to 256?)
	// mexPrintf("before copying: all elems %d, samples %d & dims %d\n", input_layer->count(), input_layer->num(), input_layer->count() / input_layer->num());
	for (int i = 0; i < input_layer->count(); ++i)
		input_blob[i] = inputImg[i];

	// Run network
	net->ForwardPrefilled();

	// Extract last feature layer
	const boost::shared_ptr<caffe::Blob<float>> feature_blob = net->blob_by_name(layer_id);
	blob_info b;
	b.width = feature_blob->width();
	b.height = feature_blob->height();
	b.channels = feature_blob->channels();
	b.num = feature_blob->num();
	return b;
}

// Extract CNN features from specified layer_id
static void getFeatures(float* featCNN, string layer_id)
{
	shared_ptr<Net<float>> net;
	if (!layer_id.compare("fc6") || !layer_id.compare("fc7") || !layer_id.compare("prob"))
		net = NET_FC;
	else if (!layer_id.compare("conv5"))
		net = NET_CONV5;
	else
		net = NET_CONV3;

	const boost::shared_ptr<caffe::Blob<float>> feature_blob = net->blob_by_name(layer_id);

	// Copy the layer to the output data
	// mexPrintf("before getting features: all elems %d, samples %d & dims %d\n", feature_blob->count(), feature_blob->num(), feature_blob->count() / feature_blob->num());
	const float* f = feature_blob->cpu_data();
	// clock_t begin = clock();
	for (int i = 0; i < feature_blob->count(); i++)
		featCNN[i] = f[i];
	// clock_t end = clock();
	// double elapsed = double(end - begin) / CLOCKS_PER_SEC * 1000.0;
	// mexPrintf("Time copy: %f\n", elapsed);
}

bool is_valid(string id)
{
	return id == "conv1" || id == "conv2" || id == "conv3" || id == "conv4" || id == "conv5" || id == "fc6" || id == "fc7" || "prob";
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 1)
		mexErrMsgTxt("Specify an action!");

	// Macros for the input arguments
	#define function_IN prhs[0]

	// Macros for the output arguments
	#define feat_OUT plhs[0]

	string input_call = mxArrayToString(function_IN);
	if (!input_call.compare("loadNet"))
	{	
		#define prototxt_IN prhs[1]
		#define model_IN prhs[2]
		#define path_IN prhs[3]

		string prototxt = "alexnet";
		string model = "alexnet";
		string path = "D:/PhD/Code/Research/TwoStepDetection/features/Caffe/convnet/";
		if (nrhs >= 2)
			prototxt = mxArrayToString(prototxt_IN);
		if (nrhs >= 3)
			model = mxArrayToString(model_IN);
		if (nrhs >= 4)
			path = mxArrayToString(path_IN);
		loadNet(prototxt, model, path);
	}
	else if (!input_call.compare("unloadNet"))
	{
		unloadNet();
	}
	else if (!input_call.compare("getFeatures"))
	{
		#define img_IN prhs[1]
		#define layer_id_IN prhs[2]

		if (nrhs < 2)
			mexErrMsgTxt("Wrong number of inputs");
		if (nlhs != 1)
			mexErrMsgTxt("Wrong number of outputs");

		// For std size: [227 x 227]
		// conv5: 256 x 13 x 13 x #samples
		// fc7: 4096 x 1 x 1 x #samples
		string layer_id;
		if (nrhs == 3)
			layer_id = mxArrayToString(layer_id_IN);
		else if (nrhs < 3) // No output layer specified (default: fc7)
			layer_id = "fc7";

		if (!is_valid(layer_id))
			mexErrMsgTxt("Input layer_id does not exist.");

		float *img = (float*)mxGetData(img_IN);
		mwSize numDims = mxGetNumberOfDimensions(img_IN);
		const mwSize *dims = mxGetDimensions(img_IN);

		int numSamples;
		numDims == 3 ? numSamples = 1 : numSamples = dims[3];
		blob_info feat_size = setupFeatures(img, dims[0], dims[1], numSamples, layer_id);
		
		if (numSamples == 1)
		{
			mwSize out_dims[3] = { feat_size.width, feat_size.height, feat_size.channels };
			feat_OUT = mxCreateNumericArray(3, out_dims, mxSINGLE_CLASS, mxREAL);
		}
		else
		{
			mwSize out_dims[4] = { feat_size.width, feat_size.height, feat_size.channels, feat_size.num };
			feat_OUT = mxCreateNumericArray(4, out_dims, mxSINGLE_CLASS, mxREAL);
		}

		float *featCNN = (float*)mxGetData(feat_OUT);
		getFeatures(featCNN, layer_id);
	}
	else if (!input_call.compare("getSize"))
	{
		if (nrhs != 2)
			mexErrMsgTxt("Wrong number of inputs");
		feat_OUT = mxCreateNumericMatrix(3, 1, mxSINGLE_CLASS, mxREAL);
		float *featDims = (float*)mxGetData(feat_OUT);
		blob_info layer_size = getSize(mxArrayToString(prhs[1]));
		featDims[0] = layer_size.height;
		featDims[1] = layer_size.width;
		featDims[2] = layer_size.channels;
	}
}

#endif