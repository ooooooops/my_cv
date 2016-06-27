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
#define CLUSTER_CNT 10
typedef int Response_t;

using namespace cv;
using namespace std;


int accuracy[10];
Response_t* responses;
float* enginVecs;
int no_kp_train = 0;
int no_kp_test = 0;

vector<float> getFreqVector(Mat& img){

	vector<float> freq;
	//检测关键点，计算描述子向量
	cv::initModule_nonfree();
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");

	vector<KeyPoint> kps;
	detector->detect(img, kps);
	if(kps.size() == 0) {
		no_kp_train++;
		return freq;
	}else if(kps.size() < CLUSTER_CNT){
		no_kp_train++;//统计无法聚类的图像个数
	}
	int i = 0;
	while(kps.size() < CLUSTER_CNT){
		kps.push_back(kps[i++]);
	}
	Mat descriptors;
	descriptorExtractor->compute(img, kps, descriptors);

	//K均值聚类,这里设定聚为CLUSTER_CNT类
	BOWKMeansTrainer bowkt(CLUSTER_CNT, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.1), 3, 2);
	Mat centers = bowkt.cluster(descriptors);//聚类中心
	vector<DMatch> matches;
	descriptorMatcher->match(descriptors, centers, matches);
	//统计聚类频度向量
	vector<int> freq_cnt(CLUSTER_CNT);//次数统计
	for(vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++){
		freq_cnt[it->trainIdx]++;
	}
	
	freq.resize(CLUSTER_CNT);
	for(int i = 0; i < freq_cnt.size(); i++){
		freq[i] = freq_cnt[i] / (float)matches.size();
	}

	return freq;
}


void getSubtrainMat(vector<vector<float>>& enginVectors, vector<Response_t>& lables, string path){

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
		vector<float> freqVec = getFreqVector(img);
		if(freqVec.size() > 0){
			lables.push_back((int)(buffer[0]+0));
			enginVectors.push_back(freqVec);
		}
		cout<<l<<":"<<"class"<<buffer[0] + 0<<" his_size:"<<freqVec.size()<<endl;
	}
	f.close();
	free(f_buffer);
}

void getTrainMat(Mat& train, Mat& lable){

	vector<vector<float>> enginVectors;
	vector<Response_t> lables;

	getSubtrainMat(enginVectors, lables, "d:\\cv\\data_batch_1.bin");
	getSubtrainMat(enginVectors, lables, "d:\\cv\\data_batch_2.bin");
	getSubtrainMat(enginVectors, lables, "d:\\cv\\data_batch_3.bin");
	getSubtrainMat(enginVectors, lables, "d:\\cv\\data_batch_4.bin");
	getSubtrainMat(enginVectors, lables, "d:\\cv\\data_batch_5.bin");

	Response_t* _lables = (Response_t *)malloc(lables.size() * sizeof(Response_t));
	float* _engins = (float *)malloc(enginVectors.size() * CLUSTER_CNT * sizeof(float));
	if(!responses) responses = (Response_t *)malloc(lables.size() * sizeof(Response_t));
	if(!enginVecs) enginVecs = (float *)malloc(enginVectors.size() * CLUSTER_CNT * sizeof(float));
	for(int i = 0; i < enginVectors.size(); i++){
		responses[i] = lables[i];
		for(int j = 0; j < CLUSTER_CNT; j++){
			enginVecs[i * CLUSTER_CNT + j] = enginVectors[i][j];
		}
	}

	train = Mat(enginVectors.size(), CLUSTER_CNT, CV_32FC1, enginVecs);
	lable = Mat(lables.size(), 1, CV_32SC1, responses);
}

void test(CvSVM& svm){
	ifstream f("d:\\cv\\test_batch.bin", ios::_Nocreate|ios::binary);
	if(!f){
		cout<<"open file failed"<<endl;
		return ;
	}
	char* f_buffer = (char*)malloc(30730000*sizeof(char));
	if(!f_buffer) {
		cout<<" malloc failed."<<endl;
		return;
	}
	f.read(f_buffer, 30730000);

	Mat img(32, 32, CV_8UC3, Scalar::all(0));
	char* buffer;
	Vec3b c;
	float *frequencies = (float *)malloc(CLUSTER_CNT * sizeof(float));
	for(int l = 0; l < DATASET_SCALE; l++){
		buffer = f_buffer + 3073 * l;
		for(int i = 1, j = 0, k = 0; i <= 1024; i++){
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
		vector<float> freq = getFreqVector(img);
		if(freq.size() < CLUSTER_CNT){
			//cout<<"feature inadequate.."<<endl;
			no_kp_test++;
			continue;
		}
		for(int i = 0; i < freq.size(); i++){
			frequencies[i] = freq[i];
		}
		Mat sample(1, CLUSTER_CNT, CV_32FC1, frequencies);
		float response = svm.predict(sample);
		if(response == (buffer[0] + 0.0)){
			//分类正确
			accuracy[(int)response]++;
		}
	}
	free(frequencies);
	f.close();
}


void print(){
	printf("|-class----class0---class1---class2---class3---class4---class5---class6---class7---class8---class9-|\n");
	printf("|Accuracy   %4.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%-|\n",
		accuracy[0] / 10.0, accuracy[1] / 10.0, accuracy[2] / 10.0, accuracy[3] / 10.0, 
		accuracy[4] / 10.0, accuracy[5] / 10.0, accuracy[6] / 10.0, accuracy[7] / 10.0,
		accuracy[8] / 10.0, accuracy[9] / 10.0);
	printf("train have no adequate keypoints:%d\n", no_kp_train);
	printf("train have no adequate keypoints:%d\n", no_kp_test);
}

int main()
{
	clock_t start, end;
	Mat train, lable;
	start = clock();
	getTrainMat(train, lable);
	//SVM训练
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM svm;

	for(int i=0; i<lable.rows; i++)
		for(int j=0; j<lable.cols; j++)
			printf("label(%d, %d) = %d \n", i, j, lable.at<Response_t>(i,j));

	cout<<"train size:"<<train.size()<<endl;
	svm.train(train, lable, Mat(), Mat(), params);
	//测试
	test(svm);
	end = clock();
	printf("time cost:%dms\n", end - start);
	print();

	if(responses) {free(responses);responses = NULL;}
	if(enginVecs) {free(enginVecs);enginVecs = NULL;}
	system("pause");
	return 0;
}