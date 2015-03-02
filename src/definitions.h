/*
 ============================================================================
 Name        : definitions.h
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 22, 2014
 Copyright   : All rights reserved
 Description : definitions of vortex project
 ============================================================================
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>


#define BLOCK_SIZE      (64)                                // размер блока для всех вычислений на GPU, кроме рождения ВЭ
#define QUANT           (2576)                              // количество точек рождения ВЭ
#define R               (0.5)                               // радиус обтекаемого круга
#define DELT            (1E-12)                             //
#define dt              (0.0001)                            // шаг по времени
#define INCR_STEP       (8192)                              // шаг увеличения размера массива ВЭ
#define VINF            {1.0, 0.0}                          // скорость набегающего потока
#define EPS             (0.0005)                            // радиус ВЭ
#define EPS2            (EPS * EPS)                         // квадрат радиуса ВЭ
#define R_COL           (4.0 * EPS2 / 9.0)                  // радиус коллапса

#define STEPS           (10000)                            // количество шагов по времени
#define SAVING_STEP     (100)                                // шаг сохранения
#define VISCOSITY       (0.001)                             // коэффициент вязкости
#define N_OF_POINTS     (20.0)                              // число разбиений панели при вычислении интеграла
#define COUNT_AREA      (10.0)                              // граница отрисовки
#define NCOL            (2)                                 // количество проходов в коллапсе

#define Ndx             (10 * (COUNT_AREA + 2))				// количество ячеек по оси x (для коллапса)
#define Ndy             (200)								// количество ячеек по оси y (для коллапса)
#define HX              ((COUNT_AREA + 2.0)/Ndx)            // размер ячейки по оси x
#define HY              (20.0/Ndy)                          // размер ячейки по оси y
#define RHO             (1.0)                               // плотность

#define RC              {0.0,0.0}							// точка, относительно которой считаем момент
#define PR_FILE			"Profile_file_plas_2576.txt"		// файл с профилем


typedef double TVars;									    // тип данных, применяемый для ВСЕХ чисел с плавающей точкой
typedef double TVctr[2];								    // вектор

                                                            // тип данных ВЭ
struct Vortex{
    double r[2];  //положение
    double g;     //интенсивность
    Vortex(): g(0.0){ r[0] = 0.0; r[1] = 0.0;}
};//POS
                                                            // тип данных данных скоростей ВЭ
struct PVortex{
    double v[2]; //скорость
    PVortex() {v[0] = 0.0; v[1] = 0.0;}
};//VEL
						                                    // тип данных с точностью
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

#endif /* DEFINITIONS_H_ */
