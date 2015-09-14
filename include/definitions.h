/*
 ============================================================================
 Name        : definitions.h
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 20, 2015
 Copyright   : All rights reserved
 Description : definitions file of vortex project for Profile_file_plas
 ============================================================================
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>

#define _strncmp(_str_, _etalon_) strncmp(_str_, _etalon_, sizeof(_etalon_) - 1 )

#define BLOCK_SIZE      (32)                                // ������ ����� ��� ���� ���������� �� GPU, ����� �������� ��
#define QUANT           (2576)                              // ���������� ����� �������� ��
#define R               (0.5)                               // ������ ����������� �����
#define DELT            (1E-12)                             // 
#define dt              (0.001)                            // ��� �� �������
#define INCR_STEP       (8192)                              // ��� ���������� ������� ������� ��
#define VINF            {1.0, 0.0}                          // �������� ����������� ������
#define EPS             (0.0005)                            // ������ ��
#define EPS2            (EPS * EPS)                         // ������� ������� ��
#define R_COL_1         (16.0 * EPS2 / 9.0)                  // ������ �������� ��� �� ������ ������
#define R_COL_2			(4.0 * EPS2 / 9.0)					// ������ �������� ��� �� ������ �����
#define MAX_VE_G		(0.00026)								// ������������ ������������� �� ����� ��������

#define STEPS           (3500)                            // ���������� ����� �� �������
#define SAVING_STEP     (10)                                // ��� ����������
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
//#define PR_FILE			"Profile_file_krug_1000.txt"		// ���� � ��������

typedef double TVars;									    // ��� ������, ����������� ��� ���� ����� � ��������� ������
typedef TVars TVctr[2];								    // ������

                                                            // ��� ������ ��
typedef struct Vortex{
    TVars r[2];         //���������
    TVars g;        //�������������
} Vortex;//POS
                                                            // ��� ������ ������ ��������� ��
typedef struct PVortex{
    TVars v[2]; //��������
} PVortex;//VEL
						                                    // ��� ������ � ���������
typedef struct Eps_Str{
    TVars eps; //
} Eps_Str;


typedef struct tPanel {
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
} tPanel;// panel

struct conf_t {
    size_t steps, saving_step;
    TVars ddt;
    size_t inc_step;
    TVars viscosity, rho;
    TVctr v_inf;
    size_t n_of_points;
    TVars x_max, x_min, y_max, y_min;
    TVars ve_size;
    size_t n_col;
    size_t n_dx, n_dy;
    TVctr rc;
    char pr_file[256];
    size_t birth_quant;
    TVars r_col_diff_sign, r_col_same_sign;
    unsigned matrix_load;
};

#endif // DEFINITIONS_H_
