#define _USE_MATH_DEFINES
#pragma once
#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>


#define BLOCK_SIZE      (64)                                // ������ ����� ��� ���� ���������� �� GPU, ����� �������� ��
#define QUANT           (1000)                               // ���������� ����� �������� ��
#define R               (0.5)                               // ������ ����������� �����
#define DELT            (1E-12)                             // 
#define dt              (0.001)                             // ��� �� �������
#define INCR_STEP       (8192)                              // ��� ���������� ������� ������� ��
#define VINF            {1.0, 0.0}                          // �������� ����������� ������
#define EPS             (0.0015)                            // ������ ��
#define EPS2            (EPS * EPS)                         // ������� ������� ��
#define STEPS           (60000)                               // ���������� ����� �� �������
#define SAVING_STEP     (100)                                 // ��� ����������
#define VISCOSITY       (0.001)                             // ����������� ��������
#define N_OF_POINTS     (20.0)                              // ����� ��������� ������ ��� ���������� ���������
#define COUNT_AREA      (10.0)                              // ������� ���������
#define NCOL            (2)                                 // ���������� �������� � ��������
#define Ndx             (10 * (COUNT_AREA + 2))				// ���������� ����� �� ��� x (��� ��������)
#define Ndy             (200)								// ���������� ����� �� ��� y (��� ��������)
#define HX              ((COUNT_AREA + 2.0)/Ndx)            // ������ ������ �� ��� x
#define HY              (20.0/Ndy)                          // ������ ������ �� ��� y
#define RHO             (1.0)                               // ���������
#define RC              {0.0,0.0}                         // �����, ������������ ������� ������� ������

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
