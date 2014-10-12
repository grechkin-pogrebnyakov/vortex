//#include "kernel.cuh"
#include "unit_1.cuh"
//#include "definitions.h"
#include "unita.h"

#include <string>


const TVars TVarsZero = 0.0;                    // дл€ обнулени€ переменных в пам€ти GPU
TVars       *M = NULL;                          // "матрица формы" (host)
size_t      n = 0;                              // количество ¬Ё
size_t      size = 0;					        // размер массива ¬Ё
Eps_Str     Psp;                                // структура с радиусом ¬Ё (0.008)
TVars       *d_host = NULL;                     // характерное рассто€ние до ближайших ¬Ё (host)
PVortex     *VEL_host = NULL;                   // скорости ¬Ё (host)
PVortex     *VEL_device = NULL;                 // скорости ¬Ё (device)
Vortex      *POS_host = NULL;                   // ¬Ё (координаты точки + интенсивность вихр€) (host)
Vortex      *POS_device = NULL;                 // ¬Ё (device)
TVctr       *V_inf_device = NULL;               // скорость потока (device)
TVars       *d_device = NULL;                   // характерное рассто€ние до ближайших ¬Ё (device)
TVars       *M_device = NULL;                   // "матрица формы" (device)
TVars       *d_g_device = NULL;                 // суммарна€ интенсивность "убитых" ¬Ё (device)
PVortex     *F_p_device = NULL;                 // главный вектор сил давлени€ (device)
PVortex     F_p_host;                           // главный вектор сил давлени€ (host)
TVars       *Momentum_device = NULL;            // момент сил (device)
TVars       Momentum_host = 0.0;                // момент сил (host)
TVars       *R_p_device = NULL;                 // правые части системы "рождени€" ¬Ё (device)
int         err = 0;                            // обработка ошибок
int         st = STEPS;                         // количество шагов по времени
TVctr       V_inf_host = VINF;                  // вектор скорости потока (host)
int         sv = SAVING_STEP;                   // шаг сохранени€
TVars       nu = VISCOSITY;                     // коэффициент в€зкости
PVortex     *V_contr_device = NULL;             // скорости в контрольных точках (device)
PVortex     *V_contr_host = NULL;               // скорости в контрольных точках (host)
PVortex     *Contr_points_device = NULL;        // контрольные точки дл€ скорости (device)
PVortex     *Contr_points_host = NULL;          // контрольные точки дл€ скорости (host)


tPanel		*panels_host = NULL;				// 
tPanel		*panels_device = NULL;				// 


// освобождение пам€ти
void mem_clear();
// вывод результата в файл
void save_to_file(Vortex *POS, size_t size, Eps_Str Psp, int _step);

// вывод сил в файл
void save_forces(PVortex F_p, TVars M, int step);

// загрузка профил€ из файла
void load_profile(tPanel *&panels, size_t &p);

