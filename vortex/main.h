//#include "kernel.cuh"
#include "unit_1.cuh"
//#include "definitions.h"
#include "unita.h"

#include <string>


const TVars TVarsZero = 0.0;                    // ��� ��������� ���������� � ������ GPU
TVars       *M = NULL;                          // "������� �����" (host)
size_t      n = 0;                              // ���������� ��
size_t      size = 0;					        // ������ ������� ��
Eps_Str     Psp;                                // ��������� � �������� �� (0.008)
TVars       *d_host = NULL;                     // ����������� ���������� �� ��������� �� (host)
PVortex     *VEL_host = NULL;                   // �������� �� (host)
PVortex     *VEL_device = NULL;                 // �������� �� (device)
Vortex      *POS_host = NULL;                   // �� (���������� ����� + ������������� �����) (host)
Vortex      *POS_device = NULL;                 // �� (device)
TVctr       *V_inf_device = NULL;               // �������� ������ (device)
TVars       *d_device = NULL;                   // ����������� ���������� �� ��������� �� (device)
TVars       *M_device = NULL;                   // "������� �����" (device)
TVars       *d_g_device = NULL;                 // ��������� ������������� "������" �� (device)
PVortex     *F_p_device = NULL;                 // ������� ������ ��� �������� (device)
PVortex     F_p_host;                           // ������� ������ ��� �������� (host)
TVars       *Momentum_device = NULL;            // ������ ��� (device)
TVars       Momentum_host = 0.0;                // ������ ��� (host)
TVars       *R_p_device = NULL;                 // ������ ����� ������� "��������" �� (device)
int         err = 0;                            // ��������� ������
int         st = STEPS;                         // ���������� ����� �� �������
TVctr       V_inf_host = VINF;                  // ������ �������� ������ (host)
int         sv = SAVING_STEP;                   // ��� ����������
TVars       nu = VISCOSITY;                     // ����������� ��������
PVortex     *V_contr_device = NULL;             // �������� � ����������� ������ (device)
PVortex     *V_contr_host = NULL;               // �������� � ����������� ������ (host)
PVortex     *Contr_points_device = NULL;        // ����������� ����� ��� �������� (device)
PVortex     *Contr_points_host = NULL;          // ����������� ����� ��� �������� (host)


tPanel		*panels_host = NULL;				// 
tPanel		*panels_device = NULL;				// 


// ������������ ������
void mem_clear();
// ����� ���������� � ����
void save_to_file(Vortex *POS, size_t size, Eps_Str Psp, int _step);

// ����� ��� � ����
void save_forces(PVortex F_p, TVars M, int step);

// �������� ������� �� �����
void load_profile(tPanel *&panels, size_t &p);

