#ifndef UNIT_1_CUH_
#define UNIT_1_CUH_

#include "kernel.cuh"
//#include "unita.h"

//создание "матрицы формы" и её обращение
TVars *matr_creation(tPanel *panels, size_t s);

// Загрузка матрицы
TVars   *load_matrix(size_t &p);

// сохранение матрицы
int save_matr(TVars* M, size_t size, char *name);
// сохранение матрицы
int save_matr(TVars** M, size_t size, char *name);
// обращение матрицы
TVars **inverse_matrix(TVars **M, size_t size);
// перестановка строк
int move_line(TVars **M, size_t s, size_t st, size_t fin);
// перестановка всех строк в порядке mov
int move_all_back(TVars **M, size_t size, size_t *mov);
// освобождение памяти вектора векторов
void clear_memory (TVars **M, size_t s);
// расширение массивов
int incr_vort_quont(Vortex *&p_host, Vortex *&p_dev, PVortex *&v_host, PVortex *&v_dev, TVars *&d_dev, size_t &size);
// рождение вихрей на профиле
int vort_creation(Vortex *pos, TVctr *V_infDev, size_t n_of_birth, size_t n_of_birth_BLOCK_S,
                     size_t n, TVars * M_Dev, TVars *d_g, tPanel *panels);

// запуск таймера
void start_timer(cudaEvent_t &start, cudaEvent_t &stop);
// остановка таймера
float stop_timer(cudaEvent_t start, cudaEvent_t stop);

// определение скоростей в каждой точке через интенсивности вихрей
int Speed(Vortex *pos, TVctr *v_inf, size_t s, PVortex *v, TVars *d, TVars nu, tPanel *panels);

void save_vel_to_file(Vortex *POS, PVortex *VEL, size_t size, int _step, int stage);

void save_d(double *d, size_t size, int _step);

// движение на одном временном шаге + сортировка ВЭ + коллапс
int Step(Vortex *pos, PVortex *V, size_t &n, size_t s, TVars *d_g, PVortex *F_p, TVars *M, tPanel *panels);

//
int velocity_control(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, int *n_v);

#endif