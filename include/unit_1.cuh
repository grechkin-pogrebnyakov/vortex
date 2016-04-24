/*
 ============================================================================
 Name        : unit_1.cuh
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Mar. 02, 2015
 Copyright   : All rights reserved
 Description : unit_1 header of vortex project
 ============================================================================
 */

#ifndef UNIT_1_CUH_
#define UNIT_1_CUH_

#ifdef __cplusplus
extern "C" {
#endif
#include "incl.h"

//создание "матрицы формы" и её обращение
TVars *matr_creation(tPanel *panels, size_t s, size_t birth);

// Загрузка матрицы
TVars   *load_matrix(size_t *p);

// выделение памяти под массивы
int allocate_arrays(Vortex **p_host, Vortex **p_dev, PVortex **v_host, PVortex **v_dev, TVars **d_dev, size_t size);

int randomize_tail(Vortex **p_dev, size_t new_size, size_t increased);
// расширение массивов
int incr_vort_quant(Vortex **p_host, Vortex **p_dev, PVortex **v_host, PVortex **v_dev, TVars **d_dev, size_t *size);

// рождение вихрей на профиле
int vort_creation(Vortex *pos, TVctr *V_infDev, size_t n_of_birth, size_t n_of_birth_BLOCK_S,
                     size_t n, TVars *M_Dev, TVars *d_g, tPanel *panels);

// запуск таймера
void start_timer(cudaEvent_t *start, cudaEvent_t *stop);
// остановка таймера
float stop_timer(cudaEvent_t start, cudaEvent_t stop);

// определение скоростей в каждой точке через интенсивности вихрей
int Speed(Vortex *pos, TVctr *v_inf, size_t s, PVortex *v, TVars *d, TVars nu, tPanel *panels);

// движение на одном временном шаге + сортировка ВЭ + коллапс
int Step(Vortex *pos, PVortex *V, size_t *n, size_t s, TVars *d_g, PVortex *F_p, TVars *M, tPanel *panels);

int init_device_conf_values();

//
int velocity_control(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, int *n_v);

#ifdef __cplusplus
}
#endif

#endif // UNIT_1_CUH_
