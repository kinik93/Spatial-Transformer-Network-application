//#ifdef _MSC_VER
//#include <windows.h>
//#endif
#ifdef _MSC_VER
#include <windows.h>
#endif

#include <iostream>
#include <stdio.h>
#include <vector>
#include <exception>

#include <DepthSense.hxx>

#include <opencv2/opencv.hpp>

#include <sstream>
#include <string>
#include <fstream>
#include <cstdint>
#include <direct.h>
using namespace DepthSense;
using namespace std;

Context g_context;
DepthNode g_dnode;
ColorNode g_cnode;
AudioNode g_anode;

uint32_t g_aFrames = 0;
uint32_t g_cFrames = 0;
uint32_t g_dFrames = 0;

bool g_bDeviceFound = false;

ProjectionHelper* g_pProjHelper = NULL;
StereoCameraParameters g_scp;

cv::Mat ModDepthForDisplay(const cv::Mat& mat)
{
    const float depth_near = 0;
    const float depth_far  = 32000;

    const float alpha = 255.0 / (depth_far - depth_near);
   

    cv::Mat fmat;
    mat.convertTo(fmat, CV_32F);

    for (int r = 0; r < mat.rows; ++r)
    {
        for (int c = 0; c < mat.cols; ++c)
        {
			
            float v = 255 - fmat.at<float>(r, c) * alpha;
            
            if (v > 255) v = 255;
            if (v < 0)   v = 0;

            fmat.at<float>(r, c) = v;
        }
    }

    cv::Mat bmat;
    fmat.convertTo(bmat, CV_8U);

    return bmat;
}

/*----------------------------------------------------------------------------*/
// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
    printf("A#%u: %d\n",g_aFrames,data.audioData.size());
    g_aFrames++;
}

/*----------------------------------------------------------------------------*/
// New color sample event handler
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{

	//------------UNCOMMENT THIS TO VIEW ALSO THE RGB STREAM----------------//
    /*
	printf("C#%u: %d\n",g_cFrames,data.colorMap.size());
    g_cFrames++;

    int w, h;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat, &w, &h);

    cv::Mat color_mat(h, w, CV_8UC3);
    memcpy(color_mat.data, data.colorMap, w * h * 3);

    cv::imshow("Color", color_mat);
    // wait for 'esc' for terminating the acquisition
	if (cvWaitKey(-1) == 27) g_context.quit();
	*/

}

/*----------------------------------------------------------------------------*/
// New depth sample event handler
void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    int32_t w, h;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat, &w, &h);

	cv::Mat depth_mat(h, w, CV_16U);

    memcpy(depth_mat.data, data.depthMap, w * h * sizeof(int16_t));

	mkdir("img/bin");
	std::stringstream ss;
	ss << "img/bin/frame_" << g_dFrames << ".bin";
	std::string s = ss.str();
	FILE * myfile;
	myfile = fopen(s.c_str(),"wb");
	unsigned short tmp;
	for (int r = 0; r < depth_mat.rows; ++r)
    {
        for (int c = 0; c < depth_mat.cols; ++c)
        {
			tmp = depth_mat.at<unsigned short>(r, c);
			fwrite(&tmp, 2, 1, myfile);
			
        }
    }
	fclose(myfile);
    cv::Mat disp_mat = ModDepthForDisplay(depth_mat);
	cv::line(disp_mat, cv::Point(0,120), cv::Point(320,120), cv::Scalar(255,255,255));
	cv::line(disp_mat, cv::Point(160,0), cv::Point(160,240), cv::Scalar(255,255,255));
    cv::imshow("Depth", disp_mat);

	
    g_dFrames++;


	// wait for 'esc' for terminating the acquisition
	// You can use cvWaitKey(-1) if you want to control the stream frame by frame
	if (cvWaitKey(-1) == 27) g_context.quit();
    
}

/*----------------------------------------------------------------------------*/
void configureAudioNode()
{
    g_anode.newSampleReceivedEvent().connect(&onNewAudioSample);

    AudioNode::Configuration config = g_anode.getConfiguration();
    config.sampleRate = 44100;

    try 
    {
        g_context.requestControl(g_anode,0);

        g_anode.setConfiguration(config);
        
        g_anode.setInputMixerLevel(0.5f);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureDepthNode()
{
	g_context.requestControl(g_dnode);
    g_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);
	g_dnode.setEnableFilter1(true);
	g_dnode.setEnableFilter2(true);
	g_dnode.setEnableFilter3(true);
	g_dnode.setEnableFilter4(true);
	g_dnode.setHighSensitivityMode(2);
    DepthNode::Configuration config = g_dnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_VGA;
    config.framerate = 60;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;

    g_dnode.setEnableDepthMap(true);
    g_dnode.setEnableConfidenceMap(true);
	

    try 
    {
        g_context.requestControl(g_dnode,0);

        g_dnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureColorNode()
{
    // connect new color sample handler
    g_cnode.newSampleReceivedEvent().connect(&onNewColorSample);

    ColorNode::Configuration config = g_cnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_VGA;
    config.compression = COMPRESSION_TYPE_MJPEG;
    config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
    config.framerate = 25;

    g_cnode.setEnableColorMap(true);

    try 
    {
        g_context.requestControl(g_cnode,0);

        g_cnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureNode(Node node)
{
    if ((node.is<DepthNode>())&&(!g_dnode.isSet()))
    {
        g_dnode = node.as<DepthNode>();
        configureDepthNode();
        g_context.registerNode(node);
    }

    if ((node.is<ColorNode>())&&(!g_cnode.isSet()))
    {
        g_cnode = node.as<ColorNode>();
        configureColorNode();
        g_context.registerNode(node);
    }

    if ((node.is<AudioNode>())&&(!g_anode.isSet()))
    {
        g_anode = node.as<AudioNode>();
        configureAudioNode();
        g_context.registerNode(node);
    }
}

/*----------------------------------------------------------------------------*/
void onNodeConnected(Device device, Device::NodeAddedData data)
{
    configureNode(data.node);
}

/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
    if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == g_anode))
        g_anode.unset();
    if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == g_cnode))
        g_cnode.unset();
    if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == g_dnode))
        g_dnode.unset();
    printf("Node disconnected\n");
}

/*----------------------------------------------------------------------------*/
void onDeviceConnected(Context context, Context::DeviceAddedData data)
{
    if (!g_bDeviceFound)
    {
        data.device.nodeAddedEvent().connect(&onNodeConnected);
        data.device.nodeRemovedEvent().connect(&onNodeDisconnected);
        g_bDeviceFound = true;
    }
}

/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
    g_bDeviceFound = false;
    printf("Device disconnected\n");
}

/*----------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{

	mkdir("img");
    g_context = Context::create("localhost");
    g_context.deviceAddedEvent().connect(&onDeviceConnected);
    g_context.deviceRemovedEvent().connect(&onDeviceDisconnected);

    // Get the list of currently connected devices
    vector<Device> da = g_context.getDevices();

    // We are only interested in the first device
    if (da.size() >= 1)
    {
        g_bDeviceFound = true;

        da[0].nodeAddedEvent().connect(&onNodeConnected);
        da[0].nodeRemovedEvent().connect(&onNodeDisconnected);

        vector<Node> na = da[0].getNodes();
        
        printf("Found %u nodes\n",na.size());
        
        for (int n = 0; n < (int)na.size();n++)
            configureNode(na[n]);
    }

    g_context.startNodes();
    g_context.run();
	cvDestroyAllWindows();
    g_context.stopNodes();


	int file_index = 1;
	while(true){
		std::string folderName = "img/bin/";
		std::stringstream sstm;
		std::string imgName = "frame_";
        sstm << folderName << imgName << file_index << ".bin";
		std::string filename = sstm.str();
		FILE* in = fopen(filename.c_str(), "rb"); //mode: read binary file

		if (in !=NULL)
		{
			printf("File frame_%d.bin opened, saving PNG 8 and 16 bit...\n", file_index);
		}
		else{
			std::cout << "End of files\n";
			system("pause");
			break;
		}
		const int H = 240;
		const int W = 320;
		unsigned short temp;

		cv::Mat M_16 = cv::Mat::zeros(H,W, CV_16U);
		cv::Mat M_8 = cv::Mat::zeros(H,W, CV_8UC1); 

		//Thresholding
		int max = -1;
		int rowMax, colMax = 0;
		int min = 70000;
		for(int row=0; row < H; row++){
			for(int col= 0; col< W; col++){
				fread(&temp, 2, 1, in); //read 1 object of 2 bytes of the stream in and save in temp variable
				
				if (temp<min){
					min = temp;
					rowMax = row;
					colMax = col;
				}

			}
		}
		fclose(in);

		FILE* in2 = fopen(filename.c_str(), "rb"); //mode: read binary file
		
		for(int row=0; row < H; row++){
			for(int col= 0; col< W; col++){
				fread(&temp, 2, 1, in); //read 1 object of 2 bytes of the stream in and save in temp variabl

				M_8.at<uint8_t>(row , col) = temp;
				M_16.at<uint16_t>(row, col) = temp;
			}
		}
		fclose(in);

		mkdir("img/16bit");
		mkdir("img/8bit");
		std::stringstream saveStream_16;
		saveStream_16 << "img/" << "16bit/" << imgName << file_index << ".png";
		imwrite(saveStream_16.str() , M_16);

		std::stringstream saveStream_8;
		saveStream_8 << "img/" << "8bit/" << imgName << file_index << ".png";
		imwrite(saveStream_8.str() , M_8);


		

		file_index += 1;
	}



    return 0;
}