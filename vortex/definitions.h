#define _USE_MATH_DEFINES
#pragma once
#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>


#define BLOCK_SIZE      (64)                                // размер блока дл€ всех вычислений на GPU, кроме рождени€ ¬Ё
#define QUANT           (1000)                               // количество точек рождени€ ¬Ё
#define R               (0.5)                               // радиус обтекаемого круга
#define DELT            (1E-12)                             // 
#define dt              (0.001)                             // шаг по времени
#define INCR_STEP       (8192)                              // шаг увеличени€ размера массива ¬Ё
#define VINF            {1.0, 0.0}                          // скорость набегающего потока
#define EPS             (0.0015)                            // радиус ¬Ё
#define EPS2            (EPS * EPS)                         // квадрат радиуса ¬Ё
#define STEPS           (60000)                               // количество шагов по времени
#define SAVING_STEP     (100)                                 // шаг сохранени€
#define VISCOSITY       (0.001)                             // коэффициент в€зкости
#define N_OF_POINTS     (20.0)                              // число разбиений панели при вычислении интеграла
#define COUNT_AREA      (10.0)                              // граница отрисовки
#define NCOL            (2)                                 // количество проходов в коллапсе
#define Ndx             (10 * (COUNT_AREA + 2))				// количество €чеек по оси x (дл€ коллапса)
#define Ndy             (200)								// количество €чеек по оси y (дл€ коллапса)
#define HX              ((COUNT_AREA + 2.0)/Ndx)            // размер €чейки по оси x
#define HY              (20.0/Ndy)                          // размер €чейки по оси y
#define RHO             (1.0)                               // плотность
#define RC              {0.0,0.0}                         // точка, относительно которой считаем момент

typedef double TVars;									    // тип данных, примен€емый дл€ ¬—≈’ чисел с плавающей точкой
typedef double TVctr[2];								    // вектор

                                                            // тип данных ¬Ё
struct Vortex{
    double r[2];  //положение
    double g;     //интенсивность
    Vortex(): g(0.0){ r[0] = 0.0; r[1] = 0.0;}
};//POS
                                                            // тип данных данных скоростей ¬Ё
struct PVortex{
    double v[2]; //скорость
    PVortex() {v[0] = 0.0; v[1] = 0.0;}
};//VEL
						                                    // тип данных с точностью
struct Eps_Str{
    double eps; //
};
