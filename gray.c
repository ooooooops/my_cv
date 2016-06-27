#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BATCHSCALE 10000
typedef unsigned char uchar;

FILE *train1, *train2, *train3, *train4, *train5, *test;
uchar *test_c;//测试集在内存中的缓存
uchar *train_c;//训练集在内存中的缓存
int matched[3][10];//每一类knn匹配成功计数

typedef struct{//近邻数组元素定义
	uchar type;//邻居类型，取值0-9
	unsigned int dis;//邻居与测试集元素的距离
}Neighbor;

typedef struct{
	uchar test_type;//测试集元素的类型
	Neighbor nbrs[5];//5近邻
}Neighbors;

Neighbors dis_5kn[BATCHSCALE];//近邻数组

typedef struct{
	int his[256];
	int len;//直方图横向尺寸，即统计维度的数目
}singleChannelHis;

typedef struct{
	uchar gray_type;//灰度类型，取值范围0-9
	singleChannelHis gray_his;//灰度直方图
}Gray_his;


void init(){
	int i, j;
	for(i = 0; i < BATCHSCALE; i++){
		dis_5kn[i].test_type = 'a';
		for(j = 0; j < 5; j++){
			dis_5kn[i].nbrs[j].type = 'a';
			dis_5kn[i].nbrs[j].dis = ~0;
		}
	}
}
	

void rgb2gray(uchar* gray, uchar *rgb, int clen){
	//pic-图像首指针，clen-单通道长度
	int i;
	for(i = 0; i < clen; i++){
		//gray[i] = ( 299 * rgb[i] + 587 * rgb[i + clen] + 114 * rgb[i + 2 * clen] + 500 ) / 1000;
		gray[i] = ( rgb[i] + rgb[i + clen] + rgb[i + 2 * clen] ) / 3;
	}
}	
	
int calManhattanDis(uchar *v1, uchar *v2, int len){//计算v1，v2之间的Manhattan距离，len指定数组长度
	int i, res;
	if(len <= 0) return 0;
	res = 0;
	for(i = 0; i < len; i++){
		res += abs( *(v1+i) - *(v2+i) );
	}
	return res;
}

int calManhattanDis(int *v1, int *v2, int len){//计算v1，v2之间的Manhattan距离，len指定数组长度
	int i, res;
	if(len <= 0) return 0;
	res = 0;
	for(i = 0; i < len; i++){
		res += abs( *(v1+i) - *(v2+i) );
	}
	return res;
}


bool openFile(){
	if( ( !fopen_s(&train1, "D:\\cv\\data_batch_1.bin", "rb") ) &&
		( !fopen_s(&train2, "D:\\cv\\data_batch_2.bin", "rb") ) &&
		( !fopen_s(&train3, "D:\\cv\\data_batch_3.bin", "rb") ) &&
		( !fopen_s(&train4, "D:\\cv\\data_batch_4.bin", "rb") ) &&
		( !fopen_s(&train5, "D:\\cv\\data_batch_5.bin", "rb") ) &&
		( !fopen_s(&test, "D:\\cv\\test_batch.bin", "rb") ) )
		return true;
	else return false;
}

void closeFile(){
	if(train1) fclose(train1);
	if(train2) fclose(train2);
	if(train3) fclose(train3);
	if(train4) fclose(train4);
	if(train5) fclose(train5);
	if(test) fclose(test);
}

void updateNeighbor(int n, Neighbor neighbor){

	int i, max_i;
	max_i = 0;
	for(i = 1; i < 5; i++){
		//寻找5近邻中的最大距离
		if(dis_5kn[n].nbrs[i].dis > dis_5kn[n].nbrs[max_i].dis)
			max_i = i;
	}
	//替换最远邻居
	dis_5kn[n].nbrs[max_i] = neighbor;
}



void getHis(singleChannelHis *sch, uchar* start, int cnt){
	//进行统计，数据以start开始，数据个数cnt
	int i;
	uchar *p;
	
	sch->len = 256;
	//直方图初始化
	for(i = 0; i < sch->len; i++){
		sch->his[i] = 0;
	}
	
	p = start;
	//计算直方图
	for(i = 0; i < cnt; i++){
		sch->his[*(p + i)]++;
	}
}


void knn_pre_proc(){
	int i, j, k;
	Neighbor tmp_nb;
	bool swap_flag = false;

	for(i = 0; i < BATCHSCALE; i++){
		swap_flag = false;
		for(j = 0; j < 5; j++){
			for(k = 0; k < (5-j)-1; k++){
				if(dis_5kn[i].nbrs[k].dis > dis_5kn[i].nbrs[k+1].dis){
					tmp_nb = dis_5kn[i].nbrs[k];
					dis_5kn[i].nbrs[k] = dis_5kn[i].nbrs[k+1];
					dis_5kn[i].nbrs[k+1] = tmp_nb;
					swap_flag = true;
				}
			}
			if(swap_flag == false) break;
		}
	}
}

double knnAccuracy(int k){
	int i, j;
	int match = 0;
	int cnt_of_type[10];//每种类型的近邻数量统计
	int i_of_majority;

	int _g_k = k / 2;
	for(i = 0; i < 3; i++)
		matched[_g_k][i] = 0;

	for(i = 0; i < BATCHSCALE; i++){
		//对每一个测试集元素初始化统计数组
		for(j = 0; j < 10; j++){
			cnt_of_type[j] = 0;
		}
		//统计，结果存入cnt_of_type
		for(j = 0; j < k; j++){
			cnt_of_type[dis_5kn[i].nbrs[j].type]++;
		}
		//寻找近邻中的众数
		i_of_majority = 0;//最多的近邻类型在cnt_of_type中的下标
		for(j = 1; j < 10; j++){
			if(cnt_of_type[i_of_majority] < cnt_of_type[j])
				i_of_majority = j;
		}

		if(i_of_majority == (dis_5kn[i].test_type) ) {
			match++;
			matched[_g_k][i_of_majority]++;
		}
	}
	return ((double)match) / ((double)BATCHSCALE);
}

void print(double _a1, double _a2, double _a3, double duration){
	int i;
	char buffer[1024] = {0};
	printf("--Accuracy--------1-NN--------3-NN--------5-NN--\n");
	printf("|-Raw data  %9.2lf%%  %9.2lf%%  %9.2lf%%-|\n", _a1, _a2, _a3);
	printf("------------------------------------------------\n\n");
	printf("--Accuracy--Raw,1-NN--------Raw,3-NN--------Raw,5-NN--\n");
	for(i = 0; i < 10; i++){
		sprintf_s(buffer, sizeof(buffer), 
		   "|-class %d %9.2lf%% %14.2lf%% %14.2lf%%-|\n", 
			i, ((float)matched[0][i]) / ((float)100.0), ((float)matched[1][i]) / ((float)100.0), ((float)matched[2][i]) / ((float)100.0));
		printf("%s", buffer);
	}
	printf("------------------------------------------------------\n\n");
	printf("cost time:%lfs\n", duration);
}


void getBatchGrayHis(Gray_his* gray_his, uchar* dataset){
	//计算dataset的灰度直方图，结果保存在gray_his
	int i;
	uchar *p = dataset;
	uchar gray[32*32];//临时数组，用于存放灰度图
	for(i = 0; i < BATCHSCALE; i++){

		rgb2gray(gray, p + 1, 32*32);//彩图转灰度图
		getHis(&gray_his[i].gray_his, gray, 32*32);//根据rgb2gray得到的结果计算灰度直方图
		gray_his[i].gray_his.len = 256;
		gray_his[i].gray_type = *p;
		
		p += 3073;
	}
}

void calGrayHisDis(Gray_his* test, Gray_his* train){
	//计算距离并根据距离保存结果
	int i, j;
	int gray_dis;
	Neighbor tmp_neibr;
	for(i = 0; i < BATCHSCALE; i++){
		
		for(j = 0; j < BATCHSCALE; j++){
			//计算距离
			gray_dis = calManhattanDis(test[i].gray_his.his, train[j].gray_his.his, 256);

			dis_5kn[i].test_type = test[i].gray_type;

			tmp_neibr.type = train[j].gray_type;
			tmp_neibr.dis = gray_dis;
			//更新近邻数组以保存结果
			updateNeighbor(i, tmp_neibr);
		}
	}
}

int main()
{  
    double a1, a2, a3;
	clock_t start, end;
	Gray_his* test_gray_his;
	Gray_his* train_gray_his;
	//初始化近邻数组
	init();
	//打开文件
	if(!openFile()) return -1;

	//分配内存
	test_c = (uchar *)malloc(30730000 * sizeof(uchar));
	train_c = (uchar *)malloc(30730000 * sizeof(uchar));
	test_gray_his = (Gray_his*)malloc(BATCHSCALE * sizeof(Gray_his));
	train_gray_his = (Gray_his*)malloc(BATCHSCALE * sizeof(Gray_his));
	if(!test_c || !train_c  || !test_gray_his || !train_gray_his){
		printf("malloc error");
		return -1;
	}
	
	//读入测试集
	fread(test_c, sizeof(uchar), 30730000, test);
	//获得测试集的灰度直方图
	getBatchGrayHis(test_gray_his, test_c);
	
	//time record
	start = clock();
	//处理第一个训练集
	printf("正在处理第一个训练集\n");
	fread(train_c, sizeof(uchar), 30730000, train1);
	printf("\t正在计算第一个训练集的灰度直方图\n");
	getBatchGrayHis(train_gray_his, train_c);
	printf("\t正在计算第一个训练集的灰度直方图与测试集灰度直方图之间的距离\n");
	calGrayHisDis(test_gray_his, train_gray_his);
	
	//处理第二个训练集
	printf("正在处理第二个训练集\n");
	fread(train_c, sizeof(uchar), 30730000, train2);
	printf("\t正在计算第二个训练集的灰度直方图\n");
	getBatchGrayHis(train_gray_his, train_c);
	printf("\t正在计算第二个训练集的灰度直方图与测试集灰度直方图之间的距离\n");
	calGrayHisDis(test_gray_his, train_gray_his);

	//处理第三个训练集
	printf("正在处理第三个训练集\n");
	fread(train_c, sizeof(uchar), 30730000, train3);
	printf("\t正在计算第三个训练集的灰度直方图\n");
	getBatchGrayHis(train_gray_his, train_c);
	printf("\t正在计算第三个训练集的灰度直方图与测试集灰度直方图之间的距离\n");
	calGrayHisDis(test_gray_his, train_gray_his);

	//处理第四个训练集
	printf("正在处理第四个训练集\n");
	fread(train_c, sizeof(uchar), 30730000, train4);
	printf("\t正在计算第四个训练集的灰度直方图\n");
	getBatchGrayHis(train_gray_his, train_c);
	printf("\t正在计算第四个训练集的灰度直方图与测试集灰度直方图之间的距离\n");
	calGrayHisDis(test_gray_his, train_gray_his);

	//处理第五个训练集
	printf("正在处理第五个训练集\n");
	fread(train_c, sizeof(uchar), 30730000, train5);
	printf("\t正在计算第五个训练集的灰度直方图\n");
	getBatchGrayHis(train_gray_his, train_c);
	printf("\t正在计算第五个训练集的灰度直方图与测试集灰度直方图之间的距离\n");
	calGrayHisDis(test_gray_his, train_gray_his);

	knn_pre_proc();
	a1 = knnAccuracy(1);//1nn
	a2 = knnAccuracy(3);//3nn
	a3 = knnAccuracy(5);//5nn
	end = clock();

	print(a1*100, a2*100, a3*100, (double)(end - start) / CLOCKS_PER_SEC);
	closeFile();
	if(test_c) {free(test_c); test_c = NULL;}
	if(train_c) {free(train_c); train_c = NULL;}
	if(test_gray_his) {free(test_gray_his); test_gray_his = NULL;}
	if(train_gray_his) {free(train_gray_his); train_gray_his = NULL;}
	system("pause");
    return 0;
}