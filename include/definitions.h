/*
 ============================================================================
 Name        : definitions.h
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 20, 2015
 Copyright   : All rights reserved
 Description : definitions file of vortex project for Profile_file_plas
 ============================================================================
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#define _strncmp(_str_, _etalon_) strncmp(_str_, _etalon_, sizeof(_etalon_) - 1 )

#define LEV_AHTUNG  0
#define LEV_ERROR   1
#define LEV_WARN    2
#define LEV_INFO    3
#define LEV_DEBUG   4

#define LOG_BUF_SIZ (10240)
#define TIMINGS_BUF_SIZ (1024)

#define log_lev_file( lev, file, line, fmt, ... ) log_lev( lev, fmt " (%s:%d)", ##__VA_ARGS__, file, line )

#define log_a( fmt, ... ) log_lev( LEV_AHTUNG, fmt " (%s:%d)", ##__VA_ARGS__, __FILE__, __LINE__ )
#define log_e( fmt, ... ) log_lev( LEV_ERROR, fmt " (%s:%d)", ##__VA_ARGS__, __FILE__, __LINE__ )
#define log_w( fmt, ... ) log_lev( LEV_WARN, fmt " (%s:%d)", ##__VA_ARGS__, __FILE__, __LINE__ )
#define log_i( fmt, ... ) log_lev( LEV_INFO, fmt " (%s:%d)", ##__VA_ARGS__, __FILE__, __LINE__ )
#define log_d( fmt, ... ) log_lev( LEV_DEBUG, fmt " (%s:%d)", ##__VA_ARGS__, __FILE__, __LINE__ )

#define cuda_safe( cuerr ) cusafe( cuerr, __FILE__, __LINE__ )

#define BLOCK_SIZE      (32)                                // размер блока для всех вычислений на GPU, кроме рождения ВЭ
#define DELT            (1E-12)                             // zero threshold

typedef double TVars;									    // тип данных, применяемый для ВСЕХ чисел с плавающей точкой
typedef TVars TVctr[2];								    // вектор

#ifndef NO_TREE
// тип данных ячеек дерева
typedef struct node_t {
    float x_min, x_max, y_min, y_max;
    float med;
    uint8_t axe;
    float g_above;
    float xg_above;
    float yg_above;
    float g_below;
    float xg_below;
    float yg_below;
    float2 rc_above;
    float2 rc_below;
    float2 rc;
    float2 dim;
} tree_t;
#endif // NO_TREE

// тип данных ВЭ
typedef struct Vortex{
    TVars r[2];         //положение
    TVars g;        //интенсивность

#ifndef NO_TREE
    unsigned int tree_id; // id of tree block
#endif // NO_TREE

} Vortex;//POS

// тип данных данных скоростей ВЭ
typedef struct PVortex{
    TVars v[2]; //скорость
} PVortex;//VEL

// тип данных с точностью
typedef struct Eps_Str{
    TVars eps; //
} Eps_Str;


typedef struct tPanel {
	// number
	unsigned int n;
	// left side
	TVctr left;
	// right side
	TVctr right;
	// control point
	TVctr contr;
	// birth point
	TVctr birth;
	// normal
	TVctr norm;
	// tangent
	TVctr tang;
	// length
	TVars length;
	// number of left panel
	unsigned int n_of_lpanel;
	// number of right panel
	unsigned int n_of_rpanel;
} tPanel;// panel

struct conf_t {
    size_t steps, saving_step;
    TVars dt;
    size_t inc_step;
    TVars viscosity, rho;
    TVars v_inf_x;
    TVars v_inf_y;
    size_t n_of_points;
    TVars x_max, x_min, y_max, y_min;
    TVars profile_x_min;
    TVars profile_x_max;
    TVars profile_y_min;
    TVars profile_y_max;
    TVars ve_size;
    size_t n_col;
    TVars h_col_x, h_col_y;
    TVars rc_x;
    TVars rc_y;
    char pr_file[256];
    size_t birth_quant;
    TVars r_col_diff_sign, r_col_same_sign;
    uint8_t matrix_load;
    size_t v_inf_incr_steps;
    TVars max_ve_g;
    char log_file[256];
    char config_file[256];
    uint8_t log_level;
    char timings_file[256];
    char kadr_file[256];
    char second_points_file[256];
    TVars rel_t;
    uint8_t steady_flow;
#ifndef NO_TREE
    size_t tree_depth;
    TVars theta;
#endif // NO_TREE
    uint8_t no_log_buf;
};

#endif // DEFINITIONS_H_
