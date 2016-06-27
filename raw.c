#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BATCHSCALE 10000
FILE *train1, *train2, *train3, *train4, *train5, *test;
char *test_c;
char *train_c;
int matched[3][10];//每一类knn匹配成功计数

typedef struct{
	char type;
	unsigned int dis;
}Neighbor;

typedef struct{
	char test_type;
	Neighbor nbrs[5];
}Neighbors;

Neighbors dis_5kn[BATCHSCALE];

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
	
int calManhattanDis(char *v1, char *v2, int len){//计算v1，v2之间的Manhattan距离，len指定数组长度
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
	else false;
}

void updateNeighbor(char test_type, int n, Neighbor neighbor){

	int i, max_i;
	max_i = 0;
	for(i = 1; i < 5; i++){
		if(dis_5kn[n].nbrs[i].dis > dis_5kn[n].nbrs[max_i].dis)
			max_i = i;
	}
	dis_5kn[n].test_type = test_type;
	dis_5kn[n].nbrs[max_i] = neighbor;
}

void closeFile(){
	if(train1) fclose(train1);
	if(train2) fclose(train2);
	if(train3) fclose(train3);
	if(train4) fclose(train4);
	if(train5) fclose(train5);
	if(test) fclose(test);
}

void processData(char* train_c, int serial){
	//train_c-单个训练集
	//serial-训练集编号，只是用于printf
	char* offset_test = test_c;//test_c是全局变量，offset_test在test_c进行偏移以遍历测试集
	char* offset_train = train_c;//同offset_test
	int i, j, tmp_dis;
	Neighbor tmp_neibr;

	printf("%正在处理第%d个训练集\n", serial);
	for(i = 0; i < BATCHSCALE; i++){
		
		for(j = 0; j < BATCHSCALE; j++){
			tmp_dis = calManhattanDis(offset_test + 1, offset_train + 1, 3072);//计算距离
			
			tmp_neibr.type = *offset_train;
			tmp_neibr.dis = tmp_dis;
			updateNeighbor(*offset_test, j, tmp_neibr);//根据距离更新当前测试集的5nn数组

			offset_test += 3073;
		}
		offset_train += 3073;
		offset_test = test_c;
	}
}

void knn_pre_proc(){
	//对近邻数组预处理，保证每个测试集元素的5近邻按照距离由小到大有序
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
	//knn准确率计算
	//k-近邻数
	int i, j;
	int match = 0;
	int cnt_of_type[10];//每种类型的近邻数量统计
	int i_of_majority;

	//matched是一个3*10全局数组，记录了测试集中1nn，3nn，5nn的10个类的准确个数
	//matched的行号是0,1,2,对应1nn，3nn，5nn，当k=1时，需要更新matched的第1/2=0行，依此类推
	int _g_k = k / 2;
	for(i = 0; i < 10; i++)
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

int main()
{  
    double a1, a2, a3;
	clock_t start, end;
	
	init();
	//goto ls;
	if(!openFile()) return -1;

	//读入测试集
	test_c = (char *)malloc(30730000 * sizeof(char));
	train_c = (char *)malloc(30730000 * sizeof(char));
	if(test_c == NULL || train_c == NULL){
		printf("malloc error");
		return -1;
	}
	
	fread(test_c, sizeof(char), 30730000, test);

	start = clock();
	//处理第一个训练集
	fread(train_c, sizeof(char), 30730000, train1);
	processData(train_c, 1);

	fread(train_c, sizeof(char), 30730000, train2);
	processData(train_c, 2);

	fread(train_c, sizeof(char), 30730000, train3);
	processData(train_c,3);

	fread(train_c, sizeof(char), 30730000, train4);
	processData(train_c, 4);

	fread(train_c, sizeof(char), 30730000, train5);
	processData(train_c, 5);

	knn_pre_proc();
	a1 = knnAccuracy(1);//1nn
	a2 = knnAccuracy(3);//3nn
	a3 = knnAccuracy(5);//5nn
	end = clock();

	print(a1*100, a2*100, a3*100, (double)(end - start) / CLOCKS_PER_SEC);
	closeFile();
	free(test_c);
	free(train_c);
	system("pause");
    return 0;
}