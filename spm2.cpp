#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <tchar.h>
#include "windows.h"
#include "shellapi.h"

#define BATCH_BYTES 30730000
#define BATCH_IMGS 10000
#define BATCH_NUM 5
#define IMG_BYTES 3073
#define IMG_CHANNEL_BYTES 1024
#define CLUSTER 500
using namespace cv;
using namespace std;

typedef struct{
	float labels[BATCH_IMGS * BATCH_NUM];
	int labels_i;
}LABELS;

vector<Mat> vecDescriptors;//每个图像的特征是一个Mat
vector<vector<Mat>> vecDescriptors1;//每个图像的特征是4个Mat
vector<vector<Mat>> vecDescriptors2;//每个图像的特征是16个Mat
Mat centers;
float arrFeatures[BATCH_NUM * BATCH_IMGS][CLUSTER * 21];
Mat vecFeatures;
LABELS labels;
Mat responses;
Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> descriptorExtractor;
Ptr<DescriptorMatcher> descriptorMatcher;
CvSVM svm;
int accuracy[10];

void getDescriptors(Mat _img, Mat& _descriptors){//获得L0级描述子
	if(!_img.data){
		cout<<"func::getDescriptors img have no data."<<endl;
		return ;
	}
	vector<KeyPoint> _vecKPs;
	detector->detect(_img, _vecKPs);
	if(_vecKPs.size() == 0){
		return ;
	}
	descriptorExtractor->compute(_img, _vecKPs, _descriptors);
}

void getDescriptors1(Mat _img, vector<Mat>& _descriptors1){
	for(int i = 0; i < 2; i++){
		for(int j = 0; j < 2; j++){
			vector<KeyPoint> _vecKPs;
			Mat _img_t = Mat(_img, Rect(i << 4, j << 4, 16, 16));
			detector->detect(_img_t, _vecKPs);
			Mat _descriptors;
			descriptorExtractor->compute(_img_t, _vecKPs, _descriptors);
			_descriptors1.push_back(_descriptors);
		}
	}
}

void getDescriptors2(Mat _img, vector<Mat>& _descriptors2){
//计算输入图像_img的描述子，结果存放在_descriptors2
	for(int i = 0; i < 2; i++){
		for(int j = 0; j < 2; j++){
			vector<KeyPoint> _vecKPs;
			Mat _img_t = Mat(_img, Rect(i << 3, j << 3, 8, 8));
			detector->detect(_img_t, _vecKPs);
			Mat _descriptors;
			descriptorExtractor->compute(_img_t, _vecKPs, _descriptors);
			_descriptors2.push_back(_descriptors);
		}
	}
}

void calOneBatchDescriptors(string batchPath){
	ifstream f(batchPath, ios::_Nocreate|ios::binary);
	if(!f){
		cout<<batchPath + " open failed."<<endl;
		return;
	}
	char* buffer = (char *)malloc(BATCH_BYTES * sizeof(char));
	if(!buffer){
		cout<<"buffer malloc failed."<<endl;
		return;
	}
	f.read(buffer, BATCH_BYTES);

	Mat img(32, 32, CV_8UC3, Scalar::all(0));
	Vec3b pixel;
	char *worker;//转向buffer的工作指针
	//labels.labels_i = 0;
	//int m = 0, n = 0;//img的行列下标
	for(int i = 0; i < BATCH_IMGS; i++){
		worker = buffer + IMG_BYTES * i;
		labels.labels[labels.labels_i++] = worker[0] + 0.0;
		//构造单幅图像img
		for(int j = 1, m = 0, n = 0; j <= IMG_CHANNEL_BYTES; j++){
			pixel[0] = worker[j];
			pixel[1] = worker[j+1024];
			pixel[2] = worker[j+2048];

			img.at<Vec3b>(m, n) = pixel;
			n++;
			if(n == 32){
				m++;
				n = 0;
			}
		}
		cout<<batchPath + ":"<<i<<endl;

		//获取img的描述子矩阵
		//L0
		Mat _descriptors;
		getDescriptors(img, _descriptors);
		vecDescriptors.push_back(_descriptors);

		//L1
		vector<Mat> _descriptors1;
		getDescriptors1(img, _descriptors1);
		vecDescriptors1.push_back(_descriptors1);

		//L2
		vector<Mat> _descriptors2;
		getDescriptors2(img, _descriptors2);
		vecDescriptors2.push_back(_descriptors2);
	}

	free(buffer);
	buffer = NULL;
	f.close();
}

void calDescriptors(){
	string basePath = "d:\\cv\\data_batch_";
	char t[2];
	for(int i = 1; i <= BATCH_NUM; i++)
	{
		calOneBatchDescriptors(basePath + string(_itoa(i,t,10)) + ".bin");
	}
}

void init(){
	cv::initModule_nonfree();
	detector = FeatureDetector::create("Dense");
	descriptorExtractor = DescriptorExtractor::create("SIFT");
	descriptorMatcher = DescriptorMatcher::create("BruteForce");
}

void calClusterCenters(){
	//将各层特征向量拼接到一个聚类矩阵
	BOWKMeansTrainer trainer(CLUSTER, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1));

	Mat feature_t;
	//L0
	for(vector<Mat>::iterator it = vecDescriptors.begin(); it != vecDescriptors.end(); it++){
		feature_t.push_back(*it);
	}
	//L1
	for(vector<vector<Mat>>::iterator it_r = vecDescriptors1.begin(); it_r != vecDescriptors1.end(); it_r++){
		vector<Mat> t = *it_r;
		for(vector<Mat>::iterator it_c = t.begin(); it_c != t.end(); it_c++){
			feature_t.push_back(*it_c);
		}
	}
	//L2
	for(vector<vector<Mat>>::iterator it_r = vecDescriptors2.begin(); it_r != vecDescriptors2.end(); it_r++){
		vector<Mat> t = *it_r;
		for(vector<Mat>::iterator it_c = t.begin(); it_c != t.end(); it_c++){
			feature_t.push_back(*it_c);
		}
	}
	//聚类中心
	centers = trainer.cluster(feature_t);
}


void calFeatures(){
	int _arr_i = 0, offset = 0;
	//L0
	for(int i = 0; i < vecDescriptors.size(); i++){
		vector<DMatch> matches;
		descriptorMatcher->match(vecDescriptors[i], centers, matches);
		for(vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++){
			arrFeatures[i][it->trainIdx] += 0.125;
		}
	}
	//L1
	_arr_i = 0;
	offset = CLUSTER << 2;
	for(vector<vector<Mat>>::iterator it_r = vecDescriptors1.begin(); it_r != vecDescriptors1.end(); it_r++){
		vector<Mat> _r = *it_r;
		for(vector<Mat>::iterator it_c = _r.begin(); it_c != _r.end(); it_c++){
			vector<DMatch> matches;
			descriptorMatcher->match(*it_c, centers, matches);
			for(vector<DMatch>::iterator it_m = matches.begin(); it_m != matches.end(); it_m++){
				arrFeatures[_arr_i][offset + it_m->trainIdx] += 0.25;
			}	
		}
		_arr_i++;
	}
	//L2
	_arr_i = 0;
	offset = offset + ( CLUSTER << 4 );
	for(vector<vector<Mat>>::iterator it_r = vecDescriptors2.begin(); it_r != vecDescriptors2.end(); it_r++){
		vector<Mat> _r = *it_r;
		for(vector<Mat>::iterator it_c = _r.begin(); it_c != _r.end(); it_c++){
			vector<DMatch> matches;
			descriptorMatcher->match(*it_c, centers, matches);
			for(vector<DMatch>::iterator it_m = matches.begin(); it_m != matches.end(); it_m++){
				arrFeatures[_arr_i][offset + it_m->trainIdx] += 0.5;
			}	
		}
		_arr_i++;
	}

	vecFeatures = Mat(BATCH_NUM * BATCH_IMGS, CLUSTER, CV_32FC1, arrFeatures);
	responses = Mat(BATCH_NUM * BATCH_IMGS, 1, CV_32FC1, labels.labels);
}

void trainSVM(){
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	svm.train(vecFeatures, responses, Mat(), Mat(), params);
}

void predict(){
	ifstream f("d:\\cv\\test_batch.bin", ios::_Nocreate|ios::binary);
	if(!f){
		cout<<"open test file failed."<<endl;
		return ;
	}
	char* buffer = (char *)malloc(BATCH_BYTES * sizeof(char));
	if(!buffer){
		cout<<"buffer malloc failed."<<endl;
		return;
	}
	f.read(buffer, BATCH_BYTES);

	Mat img(32, 32, CV_8UC3, Scalar::all(0));
	Vec3b pixel;
	char *worker;//转向buffer的工作指针
	for(int i = 0; i < BATCH_IMGS; i++){
		worker = buffer + IMG_BYTES * i;
		//构造单幅图像img
		for(int j = 1, m = 0, n = 0; j <= IMG_CHANNEL_BYTES; j++){
			pixel[0] = worker[j];
			pixel[1] = worker[j+1024];
			pixel[2] = worker[j+2048];

			img.at<Vec3b>(m, n) = pixel;
			n++;
			if(n == 32){
				m++;
				n = 0;
			}
		}
		cout<<"test:"<<i<<endl;

		int offset = 0;
		float engin[CLUSTER*21];//用于构造测试Mat
		memset(engin, 0.0, sizeof(engin));
		//获取img的描述子矩阵
		//L0
		Mat _descriptors;
		getDescriptors(img, _descriptors);
		vector<DMatch> _matches;
		descriptorMatcher->match(_descriptors, centers, _matches);
		for(vector<DMatch>::iterator it = _matches.begin(); it != _matches.end(); it++){
			engin[it->trainIdx] += 0.125;
		}
		//L1
		offset = CLUSTER << 2;
		vector<Mat> _descriptors1;
		getDescriptors1(img, _descriptors1);
		for(vector<Mat>::iterator it_d = _descriptors1.begin(); it_d != _descriptors1.end(); it_d++){
			vector<DMatch> _matches1;
			descriptorMatcher->match(*it_d, centers, _matches1);
			for(vector<DMatch>::iterator it = _matches1.begin(); it != _matches1.end(); it++){
				engin[offset + it->trainIdx] += 0.25;
			}	
		}

		//L2
		offset += (CLUSTER << 4);
		vector<Mat> _descriptors2;
		getDescriptors2(img, _descriptors2);
		for(vector<Mat>::iterator it_d = _descriptors2.begin(); it_d != _descriptors2.end(); it_d++){
			vector<DMatch> _matches2;
			descriptorMatcher->match(*it_d, centers, _matches2);
			for(vector<DMatch>::iterator it = _matches2.begin(); it != _matches2.end(); it++){
				engin[offset + it->trainIdx] += 0.5;
			}
		}

		//构造测试矩阵
		Mat test(1, CLUSTER, CV_32FC1, engin);
		int response = (int)svm.predict(test);
		if( response == (worker[0] + 0) ){
			accuracy[response]++;
		}
	}

	free(buffer);
	buffer = NULL;
	f.close();
}

void print(long during){
	ofstream f("log.txt", ios::out|ios::app);
	char buf[2048];

	sprintf_s(buf, "**********CLUSTER:%d**********\n",CLUSTER);
	f<<buf;
	sprintf_s(buf, "during:%ld\n",during);
	f<<buf;
	sprintf_s(buf, "|-class----class0---class1---class2---class3---class4---class5---class6---class7---class8---class9-|\n");
	f<<buf;
	printf("%s", buf);
	sprintf_s(buf, "|Accuracy   %4.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%-|\n",
		accuracy[0] / 10.0, accuracy[1] / 10.0, accuracy[2] / 10.0, accuracy[3] / 10.0, 
		accuracy[4] / 10.0, accuracy[5] / 10.0, accuracy[6] / 10.0, accuracy[7] / 10.0,
		accuracy[8] / 10.0, accuracy[9] / 10.0);
	f<<buf;
	printf("%s",buf);
	f.close();
}

void runnext(){
	SHELLEXECUTEINFO  ShExecInfo  =  {0};  
	ShExecInfo.cbSize  =  sizeof(SHELLEXECUTEINFO);  
	ShExecInfo.fMask  =  SEE_MASK_NOCLOSEPROCESS;  
	ShExecInfo.hwnd  =  NULL;  
	ShExecInfo.lpVerb  =  _T("open");  
	ShExecInfo.lpFile  =  _T("opencv600.exe");                          
	ShExecInfo.lpParameters  =  _T("-f train");              
	ShExecInfo.lpDirectory  =  NULL;  
	ShExecInfo.nShow  =  SW_NORMAL;  
	ShExecInfo.hInstApp  =  NULL;              
	ShellExecuteEx(&ShExecInfo); 
}

int main()
{
	clock_t start, end;
	start = clock();
	init();

	calDescriptors();
	calClusterCenters();
	calFeatures();

	trainSVM();
	predict();

	end = clock();
	print(end - start);
	return 0;
}