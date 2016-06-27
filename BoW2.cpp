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

#define BATCH_BYTES 30730000  //每个数据文件的字节数
#define BATCH_IMGS 10000      //每个数据文件的图像数 
#define BATCH_NUM 5           //数据文件中测试数据文件的个数
#define IMG_BYTES 3073        //每幅图像的字节数
#define IMG_CHANNEL_BYTES 1024//单幅图像的单通道字节数
#define CLUSTER 300           //聚类数，即词典大小
using namespace cv;
using namespace std;

typedef struct{
	float labels[BATCH_IMGS * BATCH_NUM];
	int labels_i;
}LABELS;

vector<Mat> vecDescriptors;//每个图像的特征是一个Mat
Mat centers;
float arrFeatures[BATCH_NUM * BATCH_IMGS][CLUSTER];//保存图像特征向量
Mat vecFeatures;//图像特征向量，数据内容来自arrFeatures
LABELS labels;//保存图像的标签
Mat responses;//图像标签，实际的标签内容来自labels
Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> descriptorExtractor;
Ptr<DescriptorMatcher> descriptorMatcher;
CvSVM svm;
int accuracy[10];        //每一类的正确个数统计

/*
	@description 计算单幅图像的描述子
	@param _img 待计算图像
	       _descriptors 计算结果
	@return
**/
void calDescriptors(Mat _img, Mat& _descriptors){//获得L0级描述子
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

/*
	@description 计算batchPath给出的数据集文件的描述子
	@param batchPath 数据集文件在本地的绝对路径
	@return
**/
void calOneBatchDescriptors(string batchPath){
	ifstream f(batchPath, ios::_Nocreate|ios::binary);
	if(!f){
		cout<<batchPath + " open failed."<<endl;
		return;
	}
	char* buffer = (char *)malloc(BATCH_BYTES * sizeof(char));//分配文件缓存
	if(!buffer){
		cout<<"buffer malloc failed."<<endl;
		return;
	}
	f.read(buffer, BATCH_BYTES);

	Mat img(32, 32, CV_8UC3, Scalar::all(0));
	Vec3b pixel;
	char *worker;//指向buffer的工作指针
	
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
		calDescriptors(img, _descriptors);
		vecDescriptors.push_back(_descriptors);
	}

	free(buffer);
	buffer = NULL;
	f.close();
}

/*
	@description 计算所有训练集文件的描述子
	@params
	@return
**/
void calDescriptors(){
	string basePath = "d:\\cv\\data_batch_";
	char t[2];
	for(int i = 1; i <= BATCH_NUM; i++)
	{
		calOneBatchDescriptors(basePath + string(_itoa(i,t,10)) + ".bin");
	}
}

/*
	@description 初始化模块
	@params
	@return
**/
void init(){
	cv::initModule_nonfree();
	detector = FeatureDetector::create("Dense");
	descriptorExtractor = DescriptorExtractor::create("SIFT");
	descriptorMatcher = DescriptorMatcher::create("BruteForce");
}

/*
	@description 计算聚类中心，结果保存在全局变量centers中
	@params
	@return
**/
void calClusterCenters(){
	Mat feature_t;
	//将所有图像的特征由向量组vecDescriptors转为Mat型数据feature_t
	for(vector<Mat>::iterator it = vecDescriptors.begin(); it != vecDescriptors.end(); it++){
		feature_t.push_back(*it);
	}
	//聚类中心计算
	BOWKMeansTrainer trainer(CLUSTER, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1));
	centers = trainer.cluster(feature_t);
}

/*
	@description 计算每幅图像的描述子在centers上的投影arrFeatures[i]，并根据svm的接口转换数据格式
	@params
	@return
**/
void calFeatures(){
	int _arr_i = 0, offset = 0;
	//一次计算所有图像在centers上的投影，每幅图像的结果保存在arrFeatures[i]
	for(int i = 0; i < vecDescriptors.size(); i++){
		vector<DMatch> matches;
		descriptorMatcher->match(vecDescriptors[i], centers, matches);
		for(vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++){
			arrFeatures[i][it->trainIdx] ++;
		}
	}
	//构造用于svm训练的训练矩阵
	vecFeatures = Mat(BATCH_NUM * BATCH_IMGS, CLUSTER, CV_32FC1, arrFeatures);//特征矩阵
	responses = Mat(BATCH_NUM * BATCH_IMGS, 1, CV_32FC1, labels.labels);//与特征矩阵相对应的响应矩阵
}

/*
	@description 构造svm并训练
	@params
	@return
**/
void trainSVM(){
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	svm.train(vecFeatures, responses, Mat(), Mat(), params);
}

/*
	@description 利用训练好的svm在测试集上测试
	@params
	@return
**/
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
		float engin[CLUSTER];//用于构造测试Mat
		memset(engin, 0.0, sizeof(engin));
		//获取img的描述子矩阵
		Mat _descriptors;
		calDescriptors(img, _descriptors);
		vector<DMatch> _matches;
		descriptorMatcher->match(_descriptors, centers, _matches);
		for(vector<DMatch>::iterator it = _matches.begin(); it != _matches.end(); it++){
			engin[it->trainIdx] ++;
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

/*
	@description 输出结果到日志和屏幕
	@params during 程序耗时
	@return
**/
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

int main()
{
	clock_t start, end;
	start = clock();
	init();

	calDescriptors();//计算训练集描述子
	calClusterCenters();//根据描述子计算聚类中心，即词典
	calFeatures();//计算图像在词典上的投影，以投影结果作为图像的特征向量

	trainSVM();//训练svm
	predict();//测试

	end = clock();
	print(end - start);//输出结果
	system("pause");
	return 0;
}