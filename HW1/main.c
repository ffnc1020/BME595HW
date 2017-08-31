#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <time.h>

long c_conv(int in_channel, int o_channel, int kernel_size, int stride);
IplImage* img=0;

int main(){
	long num_of_operation;
    int i;
    long num_out_channel;
    double t_0, t_f, dt;
    struct timespec start, end;
    
	// load image
	img = cvLoadImage("720p.jpg", CV_LOAD_IMAGE_UNCHANGED);
	if(!img)
		printf("Could not load image file\n");

	// show image
	//cvNamedWindow("win1", CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win1", 100, 100); // offset from the UL corner of the screen
    //cvShowImage("win1",img);

	// convolution
    for(i=0;i<11;i++){
        num_out_channel = pow(2,i);
        printf("i = %d, o_channel = %ld\n",i,num_out_channel);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        num_of_operation = c_conv(3,num_out_channel,3,1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        dt = (double)(((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_nsec - start.tv_nsec) / 1000.0)/1000000.0);
        printf("dt = %.4lf\n",dt);
    }
    
	//cvWaitKey(0);
	//cvDestroyWindow("win1");
	cvReleaseImage(&img);
	return 0;
}



long c_conv(int in_channel, int o_channel, int kernel_size, int stride){
    long count = 0;
	int i, j, p, q, k;
    int r,g,b,rgb,out;
    long sum;
    // prepare kernel
    int kernel[1][3][3]=
    {
        {
            {-1, -1, -1},
            {0, 0, 0},
            {1, 1, 1},
        },
    };
    
    int single_kernel_size = 1;
	// get input size
    int height = img->height;
	int width = img->width;
	int channel = img->nChannels;
    int depth = img->depth;
    
    // convert input IplImage to CvMat
    CvMat temp, *input_image = cvGetMat(img, &temp,NULL,0);
    
    // output image
    CvSize output_size = cvSize(ceil((width-kernel_size+1)/stride),ceil((height-kernel_size+1)/stride));
    int output_image[output_size.height][output_size.width];

    // convolve
    for(i=0;i<output_size.height;i++){
        for(j=0;j<output_size.width;j++){
            for(p=0;p<kernel_size;p++){
                for(q=0;q<kernel_size;q++){
                    for(k=0;k<o_channel;k++){
                        //kernel is center at (i,j) on output image
                        //current pixel at (i+p,j+q) on input image
                        r=(int)((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 0];
                        g=(int)((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 1];
                        b=(int)((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 2];
                        rgb = r+g+b;
                        out = rgb*kernel[1][p][q];
                        sum += out;
                        count += 5;
                    }
                }
            }
            output_image[i][j] = sum;
            sum = 0;
        }
    }
    
	return count;
}
