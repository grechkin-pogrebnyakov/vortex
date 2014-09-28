#include "kernel.cuh"
//#include "unita.h"

//�������� "������� �����" � � ���������
TVars *matr_creation(size_t s);
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
                     size_t n, TVars * M_Dev, TVars *d_g);
// ������ �������
void start_timer(cudaEvent_t &start, cudaEvent_t &stop);
// ��������� �������
float stop_timer(cudaEvent_t start, cudaEvent_t stop);
// ����������� ��������� � ������ ����� ����� ������������� ������
int Speed(Vortex *pos, TVctr *v_inf, size_t s, PVortex *v, TVars *d, TVars nu);
// �������� �� ����� ��������� ���� + ���������� �� + �������
int Step(Vortex *pos, PVortex *V, size_t &n, size_t s, TVars *d_g, PVortex *F_p, TVars *M);
