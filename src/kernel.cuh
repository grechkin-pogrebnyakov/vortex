/*
 ============================================================================
 Name        : kernel.cuh
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 22, 2014
 Copyright   : All rights reserved
 Description : kernel header of vortex project
 ============================================================================
 */

#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "definitions.h"

// CUDA ���� ��������� ��, ������� � �������� s, ��� ���� � ��� ��������� ����������
__global__ void zero_Kernel(float *randoms, Vortex *pos, int s);

// CUDA ���� �������� ����� �� �� ������� �������
__global__ void birth_Kernel(Vortex *pos, size_t n_vort, size_t n_birth, size_t n_birth_BLOCK_S, TVars * M, TVars *d_g, TVars *R_p, tPanel *panel);

// CUDA ���� ������ ��������� �� (� �������������� ����������� ������) + ��������� ������ ������������ ���������� �� 3-� �������� ��
__global__ void shared_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *V, TVars *d);

// CUDA ���� ������ � ����������� ������������ �������� ��
__global__ void diffusion_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu);

// CUDA ���� ������ � ����������� ������������ �������� �� ������� ��
__global__ void diffusion_2_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu, tPanel *panels);

// CUDA ���� �������� �� ����� ��������� ���� + �������� "����������" ������� + �������� �������� �����
__global__ void step_Kernel(Vortex *pos, PVortex *V, TVars *d_g_Dev, PVortex *F_p, TVars *M, size_t n, tPanel *panels);

// CUDA ���� ��������� d_g_Dev � ���������� ��������� � d_g
__global__ void summ_Kernel(TVars *d_g_Dev, TVars *d_g, PVortex *F_p_dev, PVortex *F_p, TVars *M_dev, TVars *M, size_t n);

// CUDA ���� ���������� ��
__global__ void sort_Kernel(Vortex *pos, size_t *s);

// CUDA ���� ����� ��������� ������ ������ ��� ��������
__global__ void first_setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL);

// CUDA ���� ����� ��������� ������ ����� ��� ��������
__global__ void second_setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL);

// CUDA ���� ������� �� ������ ������
__global__ void first_collapse_Kernel(Vortex *pos, int *COL, size_t n);

// CUDA ���� ������� �� ������ �����
__global__ void second_collapse_Kernel(Vortex *pos, int *COL, size_t n);

// CUDA ���� ������ ������ ������ ��� ��������
__global__ void Right_part_Kernel(Vortex *pos, TVctr *V_inf, size_t n_vort, size_t n_birth_BLOCK_S, TVars *R_p, tPanel *panels);

/*
// x - ���������� ����� �������� ��
__device__ __host__ TVars R_birth_x(size_t n, size_t j);

// y - ���������� ����� �������� ��
__device__ __host__ TVars R_birth_y(size_t n, size_t j);

// x - ���������� ����� �������� ��
__device__ __host__ TVars R_contr_x(size_t n, size_t i);

// y - ���������� ����� �������� ��
__device__ __host__ TVars R_contr_y(size_t n, size_t i);

// x - ���������� ������� � ����� �������� ��
__device__ __host__ TVars N_contr_x(size_t n, size_t i);

// y - ���������� ������� � ����� �������� ��
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

__device__ __host__ inline TVars Ro2(TVctr a, TVctr b) {
	TVars x;
	x = (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]);
    return x;
}

__device__ __host__ inline TVars sp(TVctr a, TVctr b) {
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

// ��������� I_0, I_3
__device__ void I_0_I_3(TVars &Ra_0, TVars &Ra_1, TVars &Rb_0, TVars &Rb_1, TVars &Norm_0, TVars &Norm_1, TVars &Rj_0, TVars &Rj_1,
                                TVars &dL, TVars &d, size_t N,TVars &RES_0, TVars &RES_3_0, TVars &RES_3_1);

__device__ void I_0_I_3f(float &Ra_0, float &Ra_1, float &Rb_0, float &Rb_1, float &Norm_0, float &Norm_1, float &Rj_0, float &Rj_1,
                                float &dL, float &d, size_t N, float &RES_0, float &RES_3_0, float &RES_3_1);

__device__ bool hitting(tPanel *Panel, TVars a0, TVars a1, TVars* b, int* hitpan);

// ���������� ��������� � ����������� ������
__global__ void velocity_control_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, int *n_v);

#endif
