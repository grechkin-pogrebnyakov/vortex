#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"


#define BLOCK_SIZE      (64)                                // ������ ����� ��� ���� ���������� �� GPU, ����� �������� ��
#define QUANT           (2576)                              // ���������� ����� �������� ��
#define R               (0.5)                               // ������ ����������� �����
#define DELT            (1E-12)                             // 
#define dt              (0.001)                            // ��� �� �������
#define INCR_STEP       (8192)                              // ��� ���������� ������� ������� ��
#define VINF            {1.0, 0.0}                          // �������� ����������� ������
#define EPS             (0.0005)                            // ������ ��
#define EPS2            (EPS * EPS)                         // ������� ������� ��
#define R_COL_1         (8.0 * EPS2 / 9.0)                  // ������ �������� ��� �� ������ �����
#define R_COL_2			(4.0 * EPS2 / 9.0)					// ������ �������� ��� �� ������ ������
#define MAX_VE_G		(1e-3)								// ������������ ������������� �� ����� ��������

#define STEPS           (100)                            // ���������� ����� �� �������
#define SAVING_STEP     (1)                                // ��� ����������
#define VISCOSITY       (0.001)                             // ����������� ��������
#define N_OF_POINTS     (20.0)                              // ����� ��������� ������ ��� ���������� ���������
#define COUNT_AREA      (10.0)                              // ������� ���������
#define NCOL            (2)                                 // ���������� �������� � ��������

#define Ndx             (10 * (COUNT_AREA + 2))				// ���������� ����� �� ��� x (��� ��������)
#define Ndy             (200)								// ���������� ����� �� ��� y (��� ��������)
#define HX              ((COUNT_AREA + 2.0)/Ndx)            // ������ ������ �� ��� x
#define HY              (20.0/Ndy)                          // ������ ������ �� ��� y
#define RHO             (1.0)                               // ���������

#define RC              {0.0,0.0}							// �����, ������������ ������� ������� ������
#define PR_FILE			"Profile_file_plas_2576.txt"		// ���� � ��������


typedef double TVars;									    // ��� ������, ����������� ��� ���� ����� � ��������� ������
typedef double TVctr[2];								    // ������

                                                            // ��� ������ ��
struct Vortex{
    double r[2];  //���������
    double g;     //�������������
    Vortex(): g(0.0){ r[0] = 0.0; r[1] = 0.0;}
};//POS
                                                            // ��� ������ ������ ��������� ��
struct PVortex{
    double v[2]; //��������
    PVortex() {v[0] = 0.0; v[1] = 0.0;}
};//VEL
						                                    // ��� ������ � ���������
struct Eps_Str{
    double eps; //
};


struct tPanel {
	// number
	unsigned int n;
	// left side
	TVctr left;
	// right side
	TVctr right;
	// control point
	TVctr contr;
	// birth point
	TVctr birth;
	// normal
	TVctr norm;
	// tangent
	TVctr tang;
	// length
	TVars length;
	// number of left panel
	unsigned int n_of_lpanel;
	// number of right panel
	unsigned int n_of_rpanel;
	tPanel():n(0), length(0), n_of_lpanel(0), n_of_rpanel(0) {
		left[0] = 0.0;
		left[1] = 0.0;
		right[0] = 0.0;
		right[1] = 0.0;
		contr[0] = 0.0;
		contr[1] = 0.0;
		birth[0] = 0.0;
		birth[1] = 0.0;
		norm[0] = 0.0;
		norm[1] = 0.0;
		tang[0] = 0.0;
		tang[1] = 0.0;
	}
};// panel

#endif