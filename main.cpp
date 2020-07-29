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

//每一行标记连通域

vector<myRegion> mark_region_first(const Mat & inputImg)	//输入二值图像
{
	vector<myRegion> each_row_group;	//逐行标记连通域
	int regionNum_current = 0;
	for (int r = 0; r < inputImg.rows; r++)
	{
		const uchar *ptr = inputImg.ptr<uchar>(r);
		int change = 0;		//若该行没有连通域则change=0不变
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
		{		//若该行没有连通域则存入如下类型连通域，便于逐行合并时判断
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

//按行号存入连通域结构

vector<vector<myRegion>>add_group_rownum(const Mat&inputImg, vector<myRegion>group)	//输入二值图像
{
	int rows = inputImg.rows;
	vector<vector<myRegion>>group_with_rownum(rows);		//按行号存储的连通域结构
	vector<myRegion>non_mRr_vector;		//若该行没有连通域则存入该结构
	myRegion non_mRr;
	non_mRr.cBegin = -1;		//把起始位置设置为-1，便于判断
	non_mRr_vector.push_back(non_mRr);
	int temp = 0;
	for (int r = 0; r < rows; r++)
	{
		for (int i = 0; i < group.size(); i++)
		{
			if (group[i].cBegin == -1)
			{
				group.erase(group.begin() + i);		//若该行没有连通域则删除
				break;
			}
			else if (group[i].rowNum == r)
			{
				group_with_rownum[r].push_back(group[i]);
			}
		}
		if (group_with_rownum.empty())
		{
			group_with_rownum[r].push_back(non_mRr_vector[0]);	//防止连续多行没有连通域在同一行存入多个non_mRr结构
		}
	}
	return group_with_rownum;
}

//按行合并连通域(4连通）
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

//将标号统一为连续,并以下标为连通域编号存储连通域
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

	vector<vector<myRegion>>img_region(size);		//按连通域标号存入数组中
	for (int m = 0; m < group_final.size(); m++)
	{
		for (int n = 0; n < group_final[m].size(); n++)
		{
			img_region[group_final[m][n].regionNum - 1].push_back(group_final[m][n]);
		}
	}
	for (int i = 0; i < img_region.size(); i++) //删除空元素
	{
		if (img_region[i].empty())
		{
			img_region.erase(img_region.begin() + i);
			--i;
		}
	}
	return img_region;
}

void mark_area_above_100(vector<vector<myRegion>>image_region, Mat&inputImg)  //输入vec3b图像
{
	vector<Vec3b> colors(image_region.size());

	for (int i = 0; i < image_region.size(); i++)   //随机颜色值
	{
		colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	}

	for (int i = 0; i < image_region.size(); i++) //每个连通域左上角的坐标为(image_region[i][0].rowNum,image_region[i][0].cbegin)
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
{		//输入二值图像
		//103B
	mark_area_above_100(image_region, inputImg_INV);

	return;
}

vector<vector<myRegion>>final_img_region(Mat&inputImg_threshold)
{
	vector<myRegion>group;
	group = mark_region_first(inputImg_threshold);		//获取未按行标记的连通域
	vector<vector<myRegion>>group_with_rownum;
	group_with_rownum = add_group_rownum(inputImg_threshold, group);		//将取出的连通域按行标记
	vector<vector<myRegion>>group_final;		//逐行将连通域合并起来标号未统一
	group_final = merge_row_group(group_with_rownum, group);
	vector<vector<myRegion>>img_region;
	img_region = label_process(group_final, group_with_rownum);		//将在同一个连通域的小region按连通域标号放入img_region
	return img_region;
}

int main()
{
	Mat src = imread("1.png");

	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	//阈值二值化图像
	Mat src_threshold;
	Mat src_threshold_INV;
	threshold(src_gray, src_threshold, 135, 255, CV_THRESH_BINARY);
	threshold(src_gray, src_threshold_INV, 50, 255, CV_THRESH_BINARY_INV);

	medianBlur(src_threshold, src_threshold, 3);
	medianBlur(src_threshold_INV, src_threshold_INV, 3);


	imshow("src_threshold", src_threshold);

	vector<vector<myRegion>>img_region;
	img_region = final_img_region(src_threshold);
	mark_area_above_100(img_region, src);		//将面积大于100的连通域标记出来（输入vec3b图像）

	imshow("src_threshold_INV", src_threshold_INV);
	imwrite("src_threshold_INV.bmp", src_threshold_INV);

	waitKey();
	return 0;
}