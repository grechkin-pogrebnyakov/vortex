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

// x - ���������� ����� �������� ��
__device__ __host__ TVars R_birth_x(tPanel *panel, size_t j);

// y - ���������� ����� �������� ��
__device__ __host__ TVars R_birth_y(tPanel *panel, size_t j);

// x - ���������� ����� �������� ��
__device__ __host__ TVars R_left_x(tPanel *panel, size_t j);

// y - ���������� ����� �������� ��
__device__ __host__ TVars R_left_y(tPanel *panel, size_t j);

// x - ���������� ����� �������� ��
__device__ __host__ TVars R_right_x(tPanel *panel, size_t j);

// y - ���������� ����� �������� ��
__device__ __host__ TVars R_right_y(tPanel *panel, size_t j);

// x - ���������� ����� �������� ��
__device__ __host__ TVars R_contr_x(tPanel *panel, size_t j);

// y - ���������� ����� �������� ��
__device__ __host__ TVars R_contr_y(tPanel *panel, size_t j);

// x - ���������� ������� � ����� �������� ��
__device__ __host__ TVars N_contr_x(tPanel *panel, size_t j);

// y - ���������� ������� � ����� �������� ��
__device__ __host__ TVars N_contr_y(tPanel *panel, size_t j);

//���������� ����� ������� �� ���������
__device__ __host__ TVars Ro2(TVctr a, TVctr b);

// ��������� I_0, I_3
__device__ void I_0_I_3(TVctr &Ra, TVctr &Rb, TVctr &Norm, TVctr &Rj, TVars &dL, TVars &d, size_t N, TVars &RES_0, TVctr &RES_3);

__device__ bool hitting(tPanel *Panel, double* a, double* b, int* hitpan);

// ���������� ��������� � ����������� ������
__global__ void velocity_control_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V);

#endif