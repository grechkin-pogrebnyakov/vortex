/*
 ============================================================================
 Name        : kernel.cuh
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Mar. 04, 2015
 Copyright   : All rights reserved
 Description : kernel header of vortex project
 ============================================================================
 */

#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "incl.h"

// CUDA ЯДРА поиск границ области, занимаемой ВЭ
template <unsigned int block_size>
__global__ void first_find_range_Kernel( Vortex *pos, unsigned int s, node_t *tree );
template <unsigned int block_size>
__global__ void second_find_range_Kernel( node_t *input, unsigned int s, node_t *tree );

// CUDA ЯДРА поиск размеров ячеек и определение принадлежности ВЭ ячейкам нижнего уровня
template <size_t block_size, size_t level>
__global__ void calculate_tree_index_Kernel( Vortex *pos, unsigned int s, node_t *tree );
template <size_t block_size, size_t level>
__global__ void first_tree_reduce_Kernel( Vortex *pos, unsigned int s, node_t *tree, node_t *output, unsigned start_undex );
template <size_t block_size, size_t level>
__global__ void second_tree_reduce_Kernel( node_t *input, unsigned int s, node_t *output );

// CUDA ЯДРА определение параметров листьев нижнего уровня
template <size_t block_size, size_t level>
__global__ void first_find_leaves_params_Kernel( Vortex *pos, unsigned int s, node_t *output );
template <size_t block_size, size_t level>
__global__ void second_find_leaves_params_Kernel( node_t *input, unsigned int s, node_t *output );

// CUDA ЯДРО определение параметров всего дерева
template <size_t block_size, size_t level>
__global__ void find_tree_params_Kernel( node_t *tree );

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

// CUDA ЯДРО поиск элементов разных знаков для коллапса
__global__ void first_setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL);

// CUDA ЯДРО поиск элементов одного знака для коллапса
__global__ void second_setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL);

// CUDA ЯДРО коллапс ВЭ разных знаков
__global__ void first_collapse_Kernel(Vortex *pos, int *COL, size_t n);

// CUDA ЯДРО коллапс ВЭ одного знака
__global__ void second_collapse_Kernel(Vortex *pos, int *COL, size_t n);

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

__device__ __host__ inline TVars R_birth_x(tPanel *panel, size_t j) {
    return panel[j].birth[0];
}
__device__ __host__ inline TVars R_birth_y(tPanel *panel, size_t j) {
    return panel[j].birth[1];
}
__device__ __host__ inline TVars R_left_x(tPanel *panel, size_t j) {
    return panel[j].left[0];
}
__device__ __host__ inline TVars R_left_y(tPanel *panel, size_t j) {
    return panel[j].left[1];
}
__device__ __host__ inline TVars R_right_x(tPanel *panel, size_t j) {
    return panel[j].right[0];
}
__device__ __host__ inline TVars R_right_y(tPanel *panel, size_t j) {
    return panel[j].right[1];
}
__device__ __host__ inline TVars R_contr_x(tPanel *panel, size_t j) {
    return panel[j].contr[0];
}
__device__ __host__ inline TVars R_contr_y(tPanel *panel, size_t j) {
    return panel[j].contr[1];
}
__device__ __host__ inline TVars N_contr_x(tPanel *panel, size_t j) {
	return panel[j].norm[0];
}
__device__ __host__ inline TVars N_contr_y(tPanel *panel, size_t j) {
    return panel[j].norm[1];
}
__device__ __host__ inline float R_birth_xf(tPanel *panel, size_t j) {
    return (float)panel[j].birth[0];
}
__device__ __host__ inline float R_birth_yf(tPanel *panel, size_t j) {
    return (float)panel[j].birth[1];
}
__device__ __host__ inline float R_left_xf(tPanel *panel, size_t j) {
    return (float)panel[j].left[0];
}
__device__ __host__ inline float R_left_yf(tPanel *panel, size_t j) {
    return (float)panel[j].left[1];
}
__device__ __host__ inline float R_right_xf(tPanel *panel, size_t j) {
    return (float)panel[j].right[0];
}
__device__ __host__ inline float R_right_yf(tPanel *panel, size_t j) {
    return (float)panel[j].right[1];
}
__device__ __host__ inline float R_contr_xf(tPanel *panel, size_t j) {
    return (float)panel[j].contr[0];
}
__device__ __host__ inline float R_contr_yf(tPanel *panel, size_t j) {
    return (float)panel[j].contr[1];
}
__device__ __host__ inline float N_contr_xf(tPanel *panel, size_t j) {
	return (float)panel[j].norm[0];
}
__device__ __host__ inline float N_contr_yf(tPanel *panel, size_t j) {
    return (float)panel[j].norm[1];
}
__device__ __host__ inline TVars Tau_x(tPanel *panel, size_t j) {
	return panel[j].tang[0];
}
__device__ __host__ inline TVars Tau_y(tPanel *panel, size_t j) {
    return panel[j].tang[1];
}
__device__ __host__ inline TVars Panel_length(tPanel *panel, size_t j) {
    return panel[j].length;
}

__device__ __host__ inline TVars sp_vec(TVctr a, TVctr b) {
   return a[0]*b[0]+a[1]*b[1];
}

__device__ __host__ inline TVars sp(TVars a0, TVars a1, TVars b0, TVars b1) {
   return a0*b0+a1*b1;
}

__device__ __host__ inline float spf(float a0, float a1, float b0, float b1) {
   return a0*b0+a1*b1;
}

__device__ inline TVars Ro2(TVars a0, TVars a1, TVars b0, TVars b1) {
    return (a0 - b0) * (a0 - b0) + (a1 - b1) * (a1 - b1);
}

__device__ inline float Ro2f(float a0, float a1, float b0, float b1) {
    return (a0 - b0) * (a0 - b0) + (a1 - b1) * (a1 - b1);
}

// Вчисление I_0, I_3
__device__ void I_0_I_3(TVars Ra_0, TVars Ra_1, TVars Rb_0, TVars Rb_1, TVars Norm_0, TVars Norm_1, TVars Rj_0, TVars Rj_1,
                                TVars dL, TVars d, size_t N,TVars *RES_0, TVars *RES_3_0, TVars *RES_3_1);

__device__ void I_0_I_3f(float Ra_0, float Ra_1, float Rb_0, float Rb_1, float Norm_0, float Norm_1, float Rj_0, float Rj_1,
                                float dL, float d, size_t N, float *RES_0, float *RES_3_0, float *RES_3_1);

__device__ bool hitting(tPanel *Panel, TVars a0, TVars a1, TVars* b, int* hitpan);

// вычисление скоростей в контрольных точках
__global__ void velocity_control_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, int *n_v);

#endif // KERNEL_CUH_
