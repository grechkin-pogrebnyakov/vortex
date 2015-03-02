/*
 ============================================================================
 Name        : main.h
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 22, 2014
 Copyright   : All rights reserved
 Description : main file header of vortex project
 ============================================================================
 */

#ifndef MAIN_H_
#define MAIN_H_

//#include "kernel.cuh"
#include "unit_1.cuh"
//#include "definitions.h"
#include "unita.h"

#include <string>


const TVars TVarsZero = 0.0;                    // для обнуления переменных в памяти GPU
TVars       *M = NULL;                          // "матрица формы" (host)
size_t      n = 0;                              // количество ВЭ
size_t      size = 0;					        // размер массива ВЭ
Eps_Str     Psp;                                // структура с радиусом ВЭ (0.008)
TVars       *d_host = NULL;                     // характерное расстояние до ближайших ВЭ (host)
PVortex     *VEL_host = NULL;                   // скорости ВЭ (host)
PVortex     *VEL_device = NULL;                 // скорости ВЭ (device)
Vortex      *POS_host = NULL;                   // ВЭ (координаты точки + интенсивность вихря) (host)
Vortex      *POS_device = NULL;                 // ВЭ (device)
TVctr       *V_inf_device = NULL;               // скорость потока (device)
TVars       *d_device = NULL;                   // характерное расстояние до ближайших ВЭ (device)
TVars       *M_device = NULL;                   // "матрица формы" (device)
TVars       *d_g_device = NULL;                 // суммарная интенсивность "убитых" ВЭ (device)
PVortex     *F_p_device = NULL;                 // главный вектор сил давления (device)
PVortex     F_p_host;                           // главный вектор сил давления (host)
TVars       *Momentum_device = NULL;            // момент сил (device)
TVars       Momentum_host = 0.0;                // момент сил (host)
TVars       *R_p_device = NULL;                 // правые части системы "рождения" ВЭ (device)
int         err = 0;                            // обработка ошибок
int         st = STEPS;                         // количество шагов по времени
TVctr       V_inf_host = VINF;                  // вектор скорости потока (host)
int         sv = SAVING_STEP;                   // шаг сохранения
TVars       nu = VISCOSITY;                     // коэффициент вязкости
PVortex     *V_contr_device = NULL;             // скорости в контрольных точках (device)
PVortex     *V_contr_host = NULL;               // скорости в контрольных точках (host)
PVortex     *Contr_points_device = NULL;        // контрольные точки для скорости (device)
PVortex     *Contr_points_host = NULL;          // контрольные точки для скорости (host)


tPanel		*panels_host = NULL;				//
tPanel		*panels_device = NULL;				//


// освобождение памяти
void mem_clear();
// вывод результата в файл
void save_to_file(Vortex *POS, size_t size, Eps_Str Psp, int _step);

// вывод сил в файл
void save_forces(PVortex F_p, TVars M, int step);

// загрузка профиля из файла
void load_profile(tPanel *&panels, size_t &p);

#endif
