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
#define DATASET_SCALE 10000
#define CLUSTER 3
#define FEATURE_SIZE CLUSTER * 21
#define RESPONSES_NUM DATASET_SCALE * 5

using namespace cv;
using namespace std;

int accuracy[10];
float responses[RESPONSES_NUM];
float enginVecs[RESPONSES_NUM * FEATURE_SIZE];
CvSVM svm;
Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> descriptorExtractor;
Ptr<DescriptorMatcher> descriptorMatcher;

void vecNormalize(vector<float>& v1){
	float sum = 0.0;
	bool flag = false;//是否有>1的量
	for(int i = 0; i < v1.size(); i++){
		sum += v1[i];
		if(!flag && v1[i] > 1.0) flag = true;
	}
	if(sum <= 1.0 || !flag) return;//和小于1或者没有大于1的分量则不正规化
	for(int i = 0; i < v1.size(); i++){
		v1[i] /= sum;
	}
}

void calHist(Mat& _img, vector<float>& _hist, float _factor = 0, bool _norm = false){

	//vector<float> hist(CLUSTER);
	//检测关键点，计算描述子向量
	vector<KeyPoint> kps;
	detector->detect(_img, kps);
	Mat descriptors;
	descriptorExtractor->compute(_img, kps, descriptors);

	//K均值聚类,这里设定聚为CLUSTER类
	BOWKMeansTrainer bowkt(CLUSTER, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.1), 3, 2);
	Mat centers = bowkt.cluster(descriptors);//聚类中心
	vector<DMatch> matches;
	descriptorMatcher->match(descriptors, centers, matches);
	//统计聚类频度向量
	_hist.resize(CLUSTER);
	for(vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++){
		_hist[it->trainIdx] += _factor;
	}
}

template<typename T>
void foreach(T& begin, T& end, float factor){
	for(T t = begin; t != end; t++)
		*t *= factor;
}

void calSPMFeature(Mat& img, vector<float>& _vecHistAll){

	//vector<float> _vecHistAll;
	//calc L0
	vector<float> _vecHistL0;
	calHist(img, _vecHistL0, 0.125);
	_vecHistAll.insert(_vecHistAll.end(), _vecHistL0.begin(), _vecHistL0.end());
	//calc L1
	vector<float> _vecHistL1;
	for(int i = 0; i < 2; i++){
		for(int j = 0; j < 2; j++){
			vector<float> t;
			calHist(Mat(img, Rect(i << 4, j << 4, 16, 16)), t, 0.25);
			_vecHistL1.insert(_vecHistL1.end(), t.begin(), t.end());
		}
	}
	_vecHistAll.insert(_vecHistAll.end(), _vecHistL1.begin(), _vecHistL1.end());
	//calc L2
	vector<float> _vecHistL2;
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			vector<float> t;
			calHist(Mat(img, Rect(i << 3, j << 3, 8, 8)), t, 0.5);
			_vecHistL2.insert(_vecHistL2.end(), t.begin(), t.end());
		}
	}
	_vecHistAll.insert(_vecHistAll.end(), _vecHistL2.begin(), _vecHistL2.end());
}


void calSPMFeatures(vector<vector<float>>& _SPMFeatures, vector<float>& _responses, string path){

	ifstream f(path, ios::_Nocreate|ios::binary);
	if(!f){
		cout<<"open file failed"<<endl;
		return ;
	}
	char* f_buffer = (char*)malloc(30730000*sizeof(char));
	if(!f_buffer) {
		cout<<path<<" malloc failed."<<endl;
		return;
	}
	f.read(f_buffer, 30730000);
	
	Mat img(32, 32, CV_8UC3, Scalar::all(0));
	char* buffer;
	Vec3b c;
	for(int l = 0; l < DATASET_SCALE; l++){
		buffer = f_buffer + 3073 * l;
		
		for(int i = 1, k = 0, j = 0; i <= 1024; i++){
			c[0] = buffer[i];
			c[1] = buffer[i+1024];
			c[2] = buffer[i+2048];
		
			img.at<Vec3b>(j, k) = c;
			k++;
			if(k >= 32){
				k = 0;
				j++;
			}
		}
		vector<float> SPMFeature;
		calSPMFeature(img, SPMFeature);
		if(SPMFeature.size() > 0){
			_responses.push_back((int)(buffer[0]+0));
			_SPMFeatures.push_back(SPMFeature);
		}
		cout<<path<<":"<<l<<endl;
	}
	f.close();
	free(f_buffer);
}

void getTrainMat(Mat& train, Mat& label){

	vector<vector<float>> SPMFeatures;
	vector<float> _responses;

	calSPMFeatures(SPMFeatures, _responses, "d:\\cv\\data_batch_1.bin");
	calSPMFeatures(SPMFeatures, _responses, "d:\\cv\\data_batch_2.bin");
	calSPMFeatures(SPMFeatures, _responses, "d:\\cv\\data_batch_3.bin");
	calSPMFeatures(SPMFeatures, _responses, "d:\\cv\\data_batch_4.bin");
	calSPMFeatures(SPMFeatures, _responses, "d:\\cv\\data_batch_5.bin");

	//讲特征SPM特征向量组SPMFeatures转成用于训练的Mat
	for(int i = 0; i < SPMFeatures.size(); i++){
		responses[i] = _responses[i];
		for(int j = 0; j < FEATURE_SIZE; j++){
			enginVecs[i * FEATURE_SIZE + j] = SPMFeatures[i][j];
		}
	}

	train = Mat(SPMFeatures.size(), FEATURE_SIZE, CV_32FC1, enginVecs);
	label = Mat(_responses.size(), 1, CV_32FC1, responses);
}

void test(){
	ifstream _f("d:\\cv\\test_batch.bin", ios::_Nocreate|ios::binary);
	if(!_f){
		cout<<"open file failed"<<endl;
		return ;
	}
	char* _f_buffer = (char*)malloc(30730000*sizeof(char));
	if(!_f_buffer) {
		cout<<" malloc failed."<<endl;
		return;
	}
	_f.read(_f_buffer, 30730000);

	Mat _img(32, 32, CV_8UC3, Scalar::all(0));
	char* _buffer;
	Vec3b _c;
	float *_arrSPMFeatures = (float *)malloc(FEATURE_SIZE * sizeof(float));
	for(int l = 0; l < DATASET_SCALE; l++){
		_buffer = _f_buffer + 3073 * l;
		for(int i = 1, j = 0, k = 0; i <= 1024; i++){
			_c[0] = _buffer[i];
			_c[1] = _buffer[i+1024];
			_c[2] = _buffer[i+2048];
		
			_img.at<Vec3b>(j, k) = _c;
			k++;
			if(k >= 32){
				k = 0;
				j++;
			}
		}
		vector<float> _SPMFeature;
		calSPMFeature(_img, _SPMFeature);

		for(int i = 0; i < _SPMFeature.size(); i++){
			_arrSPMFeatures[i] = _SPMFeature[i];
		}
		Mat _sample(1, FEATURE_SIZE, CV_32FC1, _arrSPMFeatures);
		int _response = (int)svm.predict(_sample);
		if(_response == (_buffer[0] + 0)){
			//分类正确
			accuracy[_response]++;
		}
		cout<<"test:"<<l<<endl;
	}
	free(_arrSPMFeatures);
	free(_f_buffer);
	_f.close();
}


void print(){
	printf("|-class----class0---class1---class2---class3---class4---class5---class6---class7---class8---class9-|\n");
	printf("|Accuracy   %4.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%-|\n",
		accuracy[0] / 10.0, accuracy[1] / 10.0, accuracy[2] / 10.0, accuracy[3] / 10.0, 
		accuracy[4] / 10.0, accuracy[5] / 10.0, accuracy[6] / 10.0, accuracy[7] / 10.0,
		accuracy[8] / 10.0, accuracy[9] / 10.0);
	cout<<"CLUSTER:"<<CLUSTER<<endl;
}

void init(){
	cv::initModule_nonfree();
	detector = FeatureDetector::create("Dense");
	descriptorExtractor = DescriptorExtractor::create("SIFT");
	descriptorMatcher = DescriptorMatcher::create("BruteForce");
}

int main()
{
	clock_t start, end;
	Mat train, lable;
	init();
	start = clock();
	getTrainMat(train, lable);
	//SVM训练
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	svm.train(train, lable, Mat(), Mat(), params);
	//测试
	test();
	end = clock();
	printf("time cost:%dms\n", end - start);
	print();

	system("pause");
	return 0;
}