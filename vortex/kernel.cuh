#include "definitions.h"

// CUDA ЯДРО обнуление ВЭ, начиная с элемента s, при этом у них случайные координаты
__global__ void zero_Kernel(float *randoms, Vortex *pos, int s);

// CUDA ЯДРО Рождение новых ВЭ на границе профиля
__global__ void birth_Kernel(Vortex *pos, size_t n_vort, size_t n_birth, size_t n_birth_BLOCK_S, TVars * M, TVars *d_g, TVars *R_p, tPanel *panel);

// CUDA ЯДРО Расчёт скоростей ВЭ (с использованием разделяемой памяти) + процедура поиска хорактерного расстояния до 3-х соседних ВЭ
__global__ void shared_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *V, TVars *d);

// CUDA ЯДРО Расчёт и прибавление диффузионной скорости ВЭ
__global__ void diffusion_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu);

// CUDA ЯДРО Расчёт и прибавление диффузионной скорости от профиля ВЭ
__global__ void diffusion_2_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu, tPanel *panels);

// CUDA ЯДРО Движение на одном временном шаге + контроль "протыкания" профиля + удаление дельнего следа
__global__ void step_Kernel(Vortex *pos, PVortex *V, TVars *d_g_Dev, PVortex *F_p, TVars *M, size_t n, tPanel *panels);

// CUDA ЯДРО Суммирует d_g_Dev и записывает результат в d_g
__global__ void summ_Kernel(TVars *d_g_Dev, TVars *d_g, PVortex *F_p_dev, PVortex *F_p, TVars *M_dev, TVars *M, size_t n);

// CUDA ЯДРО сортировка ВЭ
__global__ void sort_Kernel(Vortex *pos, size_t *s);

// CUDA ЯДРО поиск элементов для коллапса
__global__ void setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL);

// CUDA ЯДРО коллапс
__global__ void collapse_Kernel(Vortex *pos, int *COL, size_t n);

// CUDA ЯДРО расчёт правых частей для рождения
__global__ void Right_part_Kernel(Vortex *pos, TVctr *V_inf, size_t n_vort, size_t n_birth_BLOCK_S, TVars *R_p, tPanel *panels);

/*
// x - координата точки рождения ВЭ
__device__ __host__ TVars R_birth_x(size_t n, size_t j);

// y - координата точки рождения ВЭ
__device__ __host__ TVars R_birth_y(size_t n, size_t j);

// x - координата точки контроля ВЭ
__device__ __host__ TVars R_contr_x(size_t n, size_t i);

// y - координата точки контроля ВЭ
__device__ __host__ TVars R_contr_y(size_t n, size_t i);

// x - координата нормали к точке контроля ВЭ
__device__ __host__ TVars N_contr_x(size_t n, size_t i);

// y - координата нормали к точке контроля ВЭ
__device__ __host__ TVars N_contr_y(size_t n, size_t i);
*/

// x - координата точки рождения ВЭ
__device__ __host__ TVars R_birth_x(tPanel *panel, size_t j);

// y - координата точки рождения ВЭ
__device__ __host__ TVars R_birth_y(tPanel *panel, size_t j);

// x - координата точки контроля ВЭ
__device__ __host__ TVars R_contr_x(tPanel *panel, size_t j);

// y - координата точки контроля ВЭ
__device__ __host__ TVars R_contr_y(tPanel *panel, size_t j);

// x - координата нормали к точке контроля ВЭ
__device__ __host__ TVars N_contr_x(tPanel *panel, size_t j);

// y - координата нормали к точке контроля ВЭ
__device__ __host__ TVars N_contr_y(tPanel *panel, size_t j);

//расстояние между точками на плоскости
__device__ __host__ TVars Ro2(TVctr a, TVctr b);

// Вчисление I_0, I_3
__device__ void I_0_I_3(TVctr &Ra, TVctr &Rb, TVctr &Norm, TVctr &Rj, TVars &dL, TVars &d, size_t N, TVars &RES_0, TVctr &RES_3);

__device__ bool hitting(tPanel *Panel, double* a, double* b, int* hitpan);
