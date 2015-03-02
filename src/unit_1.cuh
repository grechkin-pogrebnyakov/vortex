#ifndef UNIT_1_CUH_
#define UNIT_1_CUH_

#include "kernel.cuh"
//#include "unita.h"

//�������� "������� �����" � � ���������
TVars *matr_creation(tPanel *panels, size_t s);

// �������� �������
TVars   *load_matrix(size_t &p);

// ���������� �������
int save_matr(TVars* M, size_t size, char *name);
// ���������� �������
int save_matr(TVars** M, size_t size, char *name);
// ��������� �������
TVars **inverse_matrix(TVars **M, size_t size);
// ������������ �����
int move_line(TVars **M, size_t s, size_t st, size_t fin);
// ������������ ���� ����� � ������� mov
int move_all_back(TVars **M, size_t size, size_t *mov);
// ������������ ������ ������� ��������
void clear_memory (TVars **M, size_t s);
// ���������� ��������
int incr_vort_quont(Vortex *&p_host, Vortex *&p_dev, PVortex *&v_host, PVortex *&v_dev, TVars *&d_dev, size_t &size);
// �������� ������ �� �������
int vort_creation(Vortex *pos, TVctr *V_infDev, size_t n_of_birth, size_t n_of_birth_BLOCK_S,
                     size_t n, TVars * M_Dev, TVars *d_g, tPanel *panels);

// ������ �������
void start_timer(cudaEvent_t &start, cudaEvent_t &stop);
// ��������� �������
float stop_timer(cudaEvent_t start, cudaEvent_t stop);

// ����������� ��������� � ������ ����� ����� ������������� ������
int Speed(Vortex *pos, TVctr *v_inf, size_t s, PVortex *v, TVars *d, TVars nu, tPanel *panels);

void save_vel_to_file(Vortex *POS, PVortex *VEL, size_t size, int _step, int stage);

void save_d(double *d, size_t size, int _step);

// �������� �� ����� ��������� ���� + ���������� �� + �������
int Step(Vortex *pos, PVortex *V, size_t &n, size_t s, TVars *d_g, PVortex *F_p, TVars *M, tPanel *panels);

//
int velocity_control(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, int *n_v);

#endif