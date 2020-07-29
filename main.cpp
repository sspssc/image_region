#include<iostream>
#include<opencv2\opencv.hpp>
#include<vector>
#include<stack>

using namespace std;
using namespace cv;

struct myRegion {
	int rowNum, cBegin, cEnd;
	int regionNum;
};


RNG rng(12345);

//ÿһ�б����ͨ��

vector<myRegion> mark_region_first(const Mat & inputImg)	//�����ֵͼ��
{
	vector<myRegion> each_row_group;	//���б����ͨ��
	int regionNum_current = 0;
	for (int r = 0; r < inputImg.rows; r++)
	{
		const uchar *ptr = inputImg.ptr<uchar>(r);
		int change = 0;		//������û����ͨ����change=0����
		for (int c = 0; c < inputImg.cols; c++)
		{
			if (ptr[c] == 255)
			{
				myRegion mRr;
				mRr.rowNum = r;
				mRr.cBegin = c;
				mRr.regionNum = ++regionNum_current;
				while (ptr[c] == ptr[c + 1])
				{
					c++;
				}
				if (c >= inputImg.cols) c = inputImg.cols - 1;
				mRr.cEnd = c;
				each_row_group.push_back(mRr);
				change = 1;
			}

		}
		if (change == 0)
		{		//������û����ͨ�����������������ͨ�򣬱������кϲ�ʱ�ж�
			myRegion non_mRr;
			non_mRr.cBegin = -1;
			non_mRr.cEnd = -1;
			non_mRr.regionNum = -1;
			non_mRr.rowNum = r;
			each_row_group.push_back(non_mRr);
		}
	}
	return each_row_group;
}

//���кŴ�����ͨ��ṹ

vector<vector<myRegion>>add_group_rownum(const Mat&inputImg, vector<myRegion>group)	//�����ֵͼ��
{
	int rows = inputImg.rows;
	vector<vector<myRegion>>group_with_rownum(rows);		//���кŴ洢����ͨ��ṹ
	vector<myRegion>non_mRr_vector;		//������û����ͨ�������ýṹ
	myRegion non_mRr;
	non_mRr.cBegin = -1;		//����ʼλ������Ϊ-1�������ж�
	non_mRr_vector.push_back(non_mRr);
	int temp = 0;
	for (int r = 0; r < rows; r++)
	{
		for (int i = 0; i < group.size(); i++)
		{
			if (group[i].cBegin == -1)
			{
				group.erase(group.begin() + i);		//������û����ͨ����ɾ��
				break;
			}
			else if (group[i].rowNum == r)
			{
				group_with_rownum[r].push_back(group[i]);
			}
		}
		if (group_with_rownum.empty())
		{
			group_with_rownum[r].push_back(non_mRr_vector[0]);	//��ֹ��������û����ͨ����ͬһ�д�����non_mRr�ṹ
		}
	}
	return group_with_rownum;
}

//���кϲ���ͨ��(4��ͨ��
vector<vector<myRegion>>merge_row_group(vector<vector<myRegion>>&group_with_row, vector<myRegion>&group)
{
	vector<vector<myRegion>>group_final;
	group_final = group_with_row;
	for (int i = 0; i < group_final.size(); i++)
	{
		if (i == group_final.size() - 1)
		{
			return group_final;
		}
		for (int j = 0; j < group_final[i].size(); j++)
		{
			for (int m = 0; m < group_final[i + 1].size(); m++)
			{
				if (group_final[i][j].cEnd >= group_final[i + 1][m].cBegin&&group_final[i][j].cBegin <= group_final[i + 1][m].cEnd)
				{
					group_final[i][j].regionNum = min(group_final[i][j].regionNum, group_final[i + 1][m].regionNum);
					group_final[i + 1][m].regionNum = min(group_final[i][j].regionNum, group_final[i + 1][m].regionNum);
				}
				else continue;
			}
		}
	}
}

//�����ͳһΪ����,�����±�Ϊ��ͨ���Ŵ洢��ͨ��
vector<vector<myRegion>> label_process(vector<vector<myRegion>>&group_final, vector<vector<myRegion>>&group_with_rownum)
{

	int size = 0;
	for (int i = 0; i < group_with_rownum.size(); i++)
	{
		for (int j = 0; j < group_with_rownum[i].size(); j++)
		{
			if (group_with_rownum[i][j].regionNum)
			{
				++size;
			}
		}
	}

	vector<vector<myRegion>>img_region(size);		//����ͨ���Ŵ���������
	for (int m = 0; m < group_final.size(); m++)
	{
		for (int n = 0; n < group_final[m].size(); n++)
		{
			img_region[group_final[m][n].regionNum - 1].push_back(group_final[m][n]);
		}
	}
	for (int i = 0; i < img_region.size(); i++) //ɾ����Ԫ��
	{
		if (img_region[i].empty())
		{
			img_region.erase(img_region.begin() + i);
			--i;
		}
	}
	return img_region;
}

void mark_area_above_100(vector<vector<myRegion>>image_region, Mat&inputImg)  //����vec3bͼ��
{
	vector<Vec3b> colors(image_region.size());

	for (int i = 0; i < image_region.size(); i++)   //�����ɫֵ
	{
		colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	}

	for (int i = 0; i < image_region.size(); i++) //ÿ����ͨ�����Ͻǵ�����Ϊ(image_region[i][0].rowNum,image_region[i][0].cbegin)
	{
		int area_size = 0;
		for (int j = 0; j < image_region[i].size(); j++) {
			area_size += image_region[i][j].cEnd - image_region[i][j].cBegin;
		}
		if (area_size > 100)
		{
			stringstream ss;
			ss << area_size;
			string write_area_size;
			write_area_size = ss.str();
			putText(inputImg, write_area_size, Point(image_region[i][0].cBegin, image_region[i][0].rowNum), 0, 0.25, Scalar(0, 255, 0));
			for (int m = 0; m < image_region[i].size(); m++)
			{
				for (int n = image_region[i][m].cBegin; n < image_region[i][m].cEnd; n++)
				{
					inputImg.at<Vec3b>(image_region[i][m].rowNum, n) = colors[i];
				}
			}
		}
	}
	imshow("image with area size", inputImg);
	imwrite("area with size.bmp", inputImg);
	return;
}

void find_103B_A101W(vector<vector<myRegion>>image_region, Mat&inputImg_INV, Mat&inputImg)
{		//�����ֵͼ��
		//103B
	mark_area_above_100(image_region, inputImg_INV);

	return;
}

vector<vector<myRegion>>final_img_region(Mat&inputImg_threshold)
{
	vector<myRegion>group;
	group = mark_region_first(inputImg_threshold);		//��ȡδ���б�ǵ���ͨ��
	vector<vector<myRegion>>group_with_rownum;
	group_with_rownum = add_group_rownum(inputImg_threshold, group);		//��ȡ������ͨ���б��
	vector<vector<myRegion>>group_final;		//���н���ͨ��ϲ��������δͳһ
	group_final = merge_row_group(group_with_rownum, group);
	vector<vector<myRegion>>img_region;
	img_region = label_process(group_final, group_with_rownum);		//����ͬһ����ͨ���Сregion����ͨ���ŷ���img_region
	return img_region;
}

int main()
{
	Mat src = imread("1.png");

	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	//��ֵ��ֵ��ͼ��
	Mat src_threshold;
	Mat src_threshold_INV;
	threshold(src_gray, src_threshold, 135, 255, CV_THRESH_BINARY);
	threshold(src_gray, src_threshold_INV, 50, 255, CV_THRESH_BINARY_INV);

	medianBlur(src_threshold, src_threshold, 3);
	medianBlur(src_threshold_INV, src_threshold_INV, 3);


	imshow("src_threshold", src_threshold);

	vector<vector<myRegion>>img_region;
	img_region = final_img_region(src_threshold);
	mark_area_above_100(img_region, src);		//���������100����ͨ���ǳ���������vec3bͼ��

	imshow("src_threshold_INV", src_threshold_INV);
	imwrite("src_threshold_INV.bmp", src_threshold_INV);

	waitKey();
	return 0;
}