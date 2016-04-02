/*
 ============================================================================
 Name        : main.cpp
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 22, 2015
 Copyright   : All rights reserved
 Description : main file of vortex project
 ============================================================================
 */

#include "main.h"
#include "unita.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

struct conf_t conf;
int current_step = 0;
size_t      n = 0;                              // количество ВЭ
PVortex     *VEL_host = NULL;                   // скорости ВЭ (host)
PVortex     *VEL_device = NULL;                 // скорости ВЭ (device)
Vortex      *POS_host = NULL;                   // ВЭ (координаты точки + интенсивность вихря) (host)
Vortex      *POS_device = NULL;                 // ВЭ (device)
TVctr       *V_inf_device = NULL;               // скорость потока (device)
TVars       *d_device = NULL;                   // характерное расстояние до ближайших ВЭ (device)
TVars       *M_device = NULL;                   // "матрица формы" (device)
TVars       *d_g_device = NULL;                 // суммарная интенсивность "убитых" ВЭ (device)
PVortex     *V_contr_device = NULL;             // скорости в контрольных точках (device)
PVortex     *V_contr_host = NULL;               // скорости в контрольных точках (host)
PVortex     *Contr_points_device = NULL;        // контрольные точки для скорости (device)
PVortex     *Contr_points_host = NULL;          // контрольные точки для скорости (host)
PVortex     *F_p_device = NULL;                 // главный вектор сил давления (device)
TVars       *Momentum_device = NULL;            // момент сил (device)
tPanel      *panels_host = NULL;                // массив панелей (host)
tPanel      *panels_device = NULL;              // массив панелей (device)
FILE *log_file = NULL;
FILE *timings_file = NULL;
cudaError_t cuda_error = cudaSuccess;

static int read_config( const char *fname ) {
    FILE *conf_f = fopen( fname, "r" );
    if ( !conf_f ) {
        log_e( "unable to open config file %s", fname );
        return 1;
    }
    char buf[236];
    while ( fgets( buf, sizeof(buf), conf_f ) ) {
        if( !_strncmp( buf, "steps " ) ) {
            sscanf( buf, "%*s %zu", &conf.steps);
        }
        else if( !_strncmp( buf, "saving_step ") ) {
            sscanf( buf, "%*s %zu", &conf.saving_step );
        }
        else if( !_strncmp( buf, "tree_depth ") ) {
            sscanf( buf, "%*s %zu", &conf.tree_depth );
        }
        else if( !_strncmp( buf, "dt ") ) {
            sscanf( buf, "%*s %lf", &conf.dt );
        }
        else if( !_strncmp( buf, "v_inf_incr_steps ") ) {
            sscanf( buf, "%*s %zu", &conf.v_inf_incr_steps );
        }
        else if( !_strncmp( buf, "increment_step ") ) {
            sscanf( buf, "%*s %zu", &conf.inc_step );
        }
        else if( !_strncmp( buf, "viscosity ") ) {
            sscanf( buf, "%*s %lf", &conf.viscosity );
        }
        else if( !_strncmp( buf, "rho ") ) {
            sscanf( buf, "%*s %lf", &conf.rho );
        }
        else if( !_strncmp( buf, "v_inf_x ") ) {
            sscanf( buf, "%*s %lf", &(conf.v_inf[0]) );
        }
        else if( !_strncmp( buf, "v_inf_y ") ) {
            sscanf( buf, "%*s %lf", &(conf.v_inf[1]) );
        }
        else if( !_strncmp( buf, "n_of_points ") ) {
            sscanf( buf, "%*s %zu", &conf.n_of_points );
        }
        else if( !_strncmp( buf, "x_max ") ) {
            sscanf( buf, "%*s %lf", &conf.x_max );
        }
        else if( !_strncmp( buf, "x_min ") ) {
            sscanf( buf, "%*s %lf", &conf.x_min );
        }
        else if( !_strncmp( buf, "y_max ") ) {
            sscanf( buf, "%*s %lf", &conf.y_max );
        }
        else if( !_strncmp( buf, "y_min ") ) {
            sscanf( buf, "%*s %lf", &conf.y_min );
        }
        else if( !_strncmp( buf, "ve_size ") ) {
            sscanf( buf, "%*s %f", &conf.ve_size );
        }
        else if( !_strncmp( buf, "n_col ") ) {
            sscanf( buf, "%*s %zu", &conf.n_col );
        }
        else if( !_strncmp( buf, "h_col_x ") ) {
            sscanf( buf, "%*s %zu", &conf.h_col_x );
        }
        else if( !_strncmp( buf, "h_col_y ") ) {
            sscanf( buf, "%*s %zu", &conf.h_col_y );
        }
        else if( !_strncmp( buf, "rc_x ") ) {
            sscanf( buf, "%*s %lf", &(conf.rc_x) );
        }
        else if( !_strncmp( buf, "rc_y ") ) {
            sscanf( buf, "%*s %lf", &(conf.rc_y) );
        }
        else if( !_strncmp( buf, "pr_file ") ) {
            sscanf( buf, "%*s %s", conf.pr_file );
        }
        else if( !_strncmp( buf, "timings_file ") ) {
            sscanf( buf, "%*s %s", conf.timings_file );
        }
        else if( !_strncmp( buf, "r_col_different_signs_ve ") ) {
            sscanf( buf, "%*s %lf", &conf.r_col_diff_sign );
        }
        else if( !_strncmp( buf, "r_col_same_sign_ve ") ) {
            sscanf( buf, "%*s %lf", &conf.r_col_same_sign );
        }
        else if( !_strncmp( buf, "matrix_load ") ) {
            sscanf( buf, "%*s %u", &conf.matrix_load );
        }
        else if( !_strncmp( buf, "max_ve_g ") ) {
            sscanf( buf, "%*s %lf", &conf.max_ve_g );
        }
    }
    fclose( conf_f );
    return 0;
}

static void mem_clear() {
    cudaFree( V_inf_device );
    cudaFree( d_g_device );
    cudaFree( M_device );
    cudaFree( POS_device );
    cudaFree( d_device );
    cudaFree( VEL_device );
    cudaFree( V_contr_device );
    cudaFree( Contr_points_device );
    cudaFree( F_p_device );
    cudaFree( Momentum_device );
    cudaFree( panels_device );
    cudaDeviceReset();
    free( POS_host );
    free( VEL_host );
    free( Contr_points_host );
    free( V_contr_host );
    free( panels_host );
}

static void save_contr_vels( PVortex *contr_points, PVortex *v_contr, int _step ) {
    if( !contr_points || !v_contr )
        return;
    char fname1[] = "output/vels/contr_Vel";
    char fname2[] = ".txt";
    char fzero[] = "0";
    char fstep[6];
    char fname[ sizeof(fname1) + 10];
    fname[0] = '\0';
    itoaxx(_step,fstep,10);
    strcat(fname,fname1);
    if (_step<10) strcat(fname,fzero);
    if (_step<100) strcat(fname,fzero);
    if (_step<1000) strcat(fname,fzero);
    if (_step<10000) strcat(fname,fzero);
    if (conf.steps >= 10000 && _step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname,fname2);
    FILE *outfile = fopen(fname, "w");
    if( !outfile ) {
        log_e("error file opening %s : %s", fname, strerror(errno) );
    } else {
        for (size_t i = 0; i < (500 * conf.saving_step); ++i) {
            fprintf( outfile, "%zu %lf %lf %lf %lf\n", i, contr_points[i%500].v[0], contr_points[i%500].v[1], v_contr[i].v[0], v_contr[i].v[1] );
        }//for i
        fclose(outfile);
    }
}

static void save_to_file(Vortex *POS, size_t size, Eps_Str Psp, int _step) {
    char fname1[] = "output/kadrs/Kadr";
    char fname2[] = ".txt";
    char fzero[] = "0";
    char fstep[6];
    char fname[ sizeof(fname1) + 10 ];
    fname[0] = '\0';
    itoaxx(_step,fstep,10);
    strcat(fname,fname1);
    if (_step<10) strcat(fname,fzero);
    if (_step<100) strcat(fname,fzero);
    if (_step<1000) strcat(fname,fzero);
    if (_step<10000) strcat(fname,fzero);
    if (conf.steps >= 10000 && _step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname,fname2);
    FILE *outfile = fopen(fname, "w");
    if( !outfile ) {
        log_e("error file opening %s : %s", fname, strerror(errno) );
        return;
    }
    log_i( "Output %d", _step );
    fprintf( outfile, "%zu\n", size );
    for (size_t i = 0; i < size; ++i) {
        fprintf( outfile, "%zu %lf %lf %lf %lf %lf %lf %lf\n", i, Psp.eps, POS[i].r[0], POS[i].r[1], 0.0, 0.0, 0.0, POS[i].g );
    }//for i
    fclose(outfile);
} //save_to_file

void save_forces(PVortex F_p, TVars M, int step) {
    static FILE *outfile = NULL;
    static char *mod = "w";
    if( !outfile ) {
        outfile = fopen("output/F_p.txt", mod);
        if( !outfile ) {
            log_e( "unable to output forces %s", strerror(errno) );
            mod = "a";
            return;
        }
    }
    fprintf( outfile, "%d %lf %lf %lf\n", step, F_p.v[0], F_p.v[1], M );
    fflush(outfile);
}

static int load_profile(const char *fname, tPanel **panels, size_t *p, size_t *birth) {
    FILE *infile = fopen(fname, "r");
    if ( !infile ) {
        log_e( "can't open profile file %s error = %s", fname, strerror(errno) );
        return 1;
    }
    char buf[255];
    fgets( buf, sizeof(buf), infile );
    fgets( buf, sizeof(buf), infile );
    fscanf( infile, "%zu", p );
    float rash = (float)(*p) / BLOCK_SIZE;
    *birth = (size_t)(BLOCK_SIZE * ceil(rash));
    *panels = (tPanel*)malloc( sizeof(tPanel) * (*birth) );
    memset(*panels, 0, sizeof(tPanel) * (*birth) );
    for (size_t i = 0; i < *p; ++i) {
        fscanf( infile, "\n%u %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %u %u", &(*panels)[i].n, &(*panels)[i].left[0], &(*panels)[i].left[1],
                &(*panels)[i].right[0], &(*panels)[i].right[1], &(*panels)[i].contr[0], &(*panels)[i].contr[1], &(*panels)[i].birth[0], &(*panels)[i].birth[1],
                &(*panels)[i].norm[0], &(*panels)[i].norm[1], &(*panels)[i].tang[0], &(*panels)[i].tang[1], &(*panels)[i].length, &(*panels)[i].n_of_lpanel, &(*panels)[i].n_of_rpanel );
        if( LEV_DEBUG < conf.log_level )
            log_d( "%u", (*panels)[i].n_of_rpanel );
    }
    fclose( infile );
}

void flush_log() {
    if( log_file )
        fflush( log_file );
    if( timings_file )
        fflush( timings_file );
}

__attribute__((destructor))
static void close_logs() {
    fflush(NULL);
    if( timings_file )
        fclose( timings_file );
    if( log_file )
        fclose( log_file );
}

static void set_log_file() {
    static char log_buf[LOG_BUF_SIZ];
    static char timings_buf[TIMINGS_BUF_SIZ];
    if( *conf.log_file ) {
        log_file = fopen( conf.log_file, "w" );
        if( !log_file ) {
            log_e( "Log file %s open failed: '%s'.", conf.log_file, strerror(errno) );
        } else {
            setvbuf( log_file, log_buf, _IOFBF, LOG_BUF_SIZ );
            log_a("\n\n\n\n\nnew run\n\n\n\n");
        }
    }
    if( *conf.timings_file ) {
        timings_file = fopen( conf.timings_file, "w" );
        if( !timings_file ) {
            log_e( "timings file %s open failed: '%s'.", conf.timings_file, strerror(errno) );
        } else {
            setvbuf( timings_file, timings_buf, _IOFBF, TIMINGS_BUF_SIZ );
        }
    }
}

static inline void vwrite_to_log( FILE *file, char *fmt, va_list ap ) {
    vfprintf(file, fmt, ap);
}

static void write_to_log( FILE *file, char *fmt, ... ) {
    va_list ap;
    va_start( ap, fmt );
    vwrite_to_log( file, fmt, ap );
    va_end( ap );
}

static inline void log_t( char *fmt, ... ) {
    va_list ap;
    va_start( ap, fmt );
    vwrite_to_log( timings_file ? : stdout, fmt, ap );
    va_end( ap );
}

void log_lev( uint8_t lev, char *fmt, ... ) {
    if( conf.log_level < lev )
        return;
    char p[16 * 1024];
    va_list ap;
    va_start( ap, fmt );
    vsnprintf( p, sizeof(p), fmt, ap );
    va_end( ap );
    char timestr[64];
    struct timespec t;
    clock_gettime( CLOCK_REALTIME, &t );
    struct tm bdtime;
    localtime_r(&t.tv_sec, &bdtime);
    size_t timestr_len = strftime(timestr, sizeof(timestr), "%d.%m.%Y %H:%M:%S", &bdtime);
    snprintf(&timestr[ timestr_len ], sizeof(timestr) - timestr_len, ".%06ld", t.tv_nsec / 1000 );
    char new_fmt[4096];
    char symlev[] = "AEWIDB";
    snprintf(new_fmt, sizeof(new_fmt), "[%s] %c %s\n", timestr, symlev[lev % sizeof(symlev)], "%s");
    write_to_log( log_file ? : stdout, new_fmt, p );
}

static int parse_params( int argc, char **argv ) {
    for( int i = 0; i < argc; ++i ) {
        char *param = argv[i];
        if( *param != '-' )
            continue;
        param++;
        switch( *param ) {
        case 'c':
            strncpy( conf.config_file, argv[++i], sizeof(conf.config_file) );
            log_d( "config file %s", conf.config_file );
            break;
        case 'l':
            conf.log_level = atoi( argv[++i] );
            log_d( "log_level %s", argv[i] );
            break;
        case 'o':
            strncpy( conf.log_file, argv[++i], sizeof(conf.log_file) );
            log_d( "log_file %s", conf.log_file );
            break;
        }
    }
    if( !(*conf.config_file) ) {
        printf( "Usage:\nvortex -c file.conf\n" );
        return 1;
    }
    return 0;
}

static int create_dir(const char *dir_name) {
    struct stat st = {0};
    if (stat(dir_name, &st) != 0) {
        if (errno == ENOENT) {
            if( mkdir(dir_name, 0777) ) {
                log_e("cannot create %d directory: %s", dir_name, strerror(errno));
                return -1;
            }
            log_i("directory %s created", dir_name);
        } else {
            log_e("cannot stat %s directory: %s", dir_name, strerror(errno));
            return -1;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        log_e("%s is not directory", dir_name);
        return -1;
    }
    return 0;
}

static int create_output_dirs() {
    if (create_dir("output"))
        return -1;
    if (create_dir("output/kadrs"))
        return -1;
    if (create_dir("output/vels"))
        return -1;
    return 0;
}

int main( int argc, char **argv ) {
    const TVars TVarsZero = 0.0;                    // для обнуления переменных в памяти GPU
    TVars       *M = NULL;                          // "матрица формы" (host)
    size_t      size = 0;                           // размер массива ВЭ
    Eps_Str     Psp;                                // структура с радиусом ВЭ (0.008)
    // TVars       *d_host = NULL;                     // характерное расстояние до ближайших ВЭ (host)
    TVars       *R_p_device = NULL;                 // правые части системы "рождения" ВЭ (device)
    int         err = 0;                            // обработка ошибок
    PVortex     F_p_host = {0.0, 0.0};                           // главный вектор сил давления (host)
    TVars       Momentum_host = 0.0;                // момент сил (host)


// расширенное количество рождаемых ВЭ
// (кратно BLOCK_SIZE)
    size_t birth = 0;

    memset( &conf, 0, sizeof(conf) );
    conf.log_level = 3;
    conf.v_inf_incr_steps = 1;

    if ( parse_params( argc, argv ) )
        return 1;
    log_d("ok read params");

    if ( read_config( conf.config_file ) )
        return 1;
    log_d("ok read config");

    if ( create_output_dirs() )
        return 1;

    set_log_file();

    cudaDeviceReset();

    load_profile(conf.pr_file, &panels_host, &conf.birth_quant, &birth);
    log_d( "birth = %zu", birth );

    if( init_device_conf_values() )
        return 1;

    if ( !conf.matrix_load ) {
        log_i( "generate matrix" );
        int cnt = 0;
        do {
            // генерация матрицы
            M = matr_creation( panels_host, conf.birth_quant, birth );
            ++cnt;
        } while (M == NULL && cnt < 10);
        if (M == NULL) {
            log_e( "Matrix creation error!" );
            return 1;
        } else {
            log_i( "Matrix created!" );
        }
    } else {
        log_i( "load matrix" );
        size_t p = 0;
        // загрузка матрицы из файла
        M = load_matrix(&p);
        if (M == NULL) {
            log_e( "Matrix loading error!" );
            return 1;
        } else {
            log_i( "Matrix loaded!" );
        }
        if (conf.birth_quant != p ) {
            log_e( "martix and profile mismatch");
            return 1;
        }
    }

// количество ВЭ
    n = 0;

// размер массива ВЭ
    size = 0;
    Psp.eps = 0.008;

//-----------------------------------------------------------------------------------------------------------------------------
    // Вычисление скоростей при x = 0.35
    Contr_points_host = (PVortex*)malloc( sizeof(PVortex) * 500 );
    for (int i = 0; i < 500; ++i) {
        Contr_points_host[i].v[1] = 0.01 + 0.002 * i;
        Contr_points_host[i].v[0] = -0.15;
    }

    V_contr_host = (PVortex*)malloc( sizeof(PVortex) * 500 * conf.saving_step );
    memset(V_contr_host, 0, sizeof(PVortex) * 500 *conf.saving_step );
    cuda_safe( cudaMalloc( (void**)&V_contr_device, 500 * conf.saving_step * sizeof(PVortex) ) );
    cuda_safe( cudaMalloc( (void**)&Contr_points_device, 500 * sizeof(PVortex) ) );
    cuda_safe( cudaMemcpy( Contr_points_device, Contr_points_host, 500 * sizeof(PVortex), cudaMemcpyHostToDevice ) );
    int v_n_host = 0;
    int *v_n_device = NULL;
    cuda_safe( cudaMalloc( (void**)&v_n_device, sizeof(int) ) );
    cuda_safe( cudaMemcpy( v_n_device, &v_n_host, sizeof(int), cudaMemcpyHostToDevice ) );

    F_p_host.v[0] = 0.0;
    F_p_host.v[1] = 0.0;

    // выделение памяти и копирование на device

    cuda_safe( cudaMalloc( (void**)&V_inf_device, sizeof(TVctr) ) );
    cuda_safe( cudaMalloc( (void**)&d_g_device, sizeof(TVars) ) );
    cuda_safe( cudaMalloc( (void**)&Momentum_device, sizeof(TVars) ) );
    cuda_safe( cudaMalloc( (void**)&F_p_device, sizeof(PVortex) ) );
    cuda_safe( cudaMalloc( (void**)&M_device, (birth+1) * (birth+1) * sizeof(TVars) ) );
    cuda_safe( cudaMalloc( (void**)&panels_device, birth * sizeof(tPanel) ) );
    cuda_safe( cudaMemcpy( V_inf_device, &conf.v_inf, sizeof(TVctr), cudaMemcpyHostToDevice ) );
    cuda_safe( cudaMemcpy( d_g_device, &TVarsZero, sizeof(TVars), cudaMemcpyHostToDevice ) );
    cuda_safe( cudaMemcpy( Momentum_device, &Momentum_host, sizeof(TVars), cudaMemcpyHostToDevice ) );
    cuda_safe( cudaMemcpy( F_p_device, &F_p_host , sizeof(PVortex), cudaMemcpyHostToDevice ) );
    cuda_safe( cudaMemcpy( M_device, M, (birth+1) * (birth+1) * sizeof(TVars), cudaMemcpyHostToDevice ) );
    cuda_safe( cudaMemcpy( panels_device, panels_host, birth * sizeof(tPanel), cudaMemcpyHostToDevice ) );
    // все массивы имеют переменную длину и память для них выделяется в incr_vort_quant()

    free( M );
    log_i( "dt = %lf", conf.dt );

    // увеличение массива ВЭ на INCR_STEP элементов
    err = incr_vort_quant( &POS_host, &POS_device, &VEL_host, &VEL_device, &d_device, &size );
    if (err != 0)
    {
        log_e( "Increase ERROR!" );
        mem_clear();
        return 1;
    }
    float creation_time = 0.0;
    float speed_time = 0.0;
    float step_time = 0.0;
    cudaEvent_t start = 0, stop = 0;
//------------------------------------------------------------------------------------------
    TVars d_V_inf = conf.v_inf[0] / (TVars)conf.v_inf_incr_steps;
    log_d( "delta V = %lf", d_V_inf );
    conf.v_inf[0] = 0.0;
    cuda_safe( cudaMemcpy( V_inf_device, &conf.v_inf, sizeof(TVctr), cudaMemcpyHostToDevice ) );

    // цикл шагов выполнения расчётов
    for (current_step = 0; current_step < conf.steps; current_step++) {
        log_d( "step %d", current_step );
        if (current_step < conf.v_inf_incr_steps) {
            conf.v_inf[0] += d_V_inf;
            cuda_safe( cudaMemcpy( V_inf_device, &conf.v_inf, sizeof(TVctr), cudaMemcpyHostToDevice ) );
            log_d( "increase v_inf = %lf", conf.v_inf[0] );
        }
        // количество ВЭ на текущем шаге, увеличенное до кратности BLOCK_SIZE
        size_t s = 0;
        float rashirenie = 0;
        rashirenie = (TVars)(n + conf.birth_quant) / BLOCK_SIZE;
        s = (int)( BLOCK_SIZE * ceil(rashirenie) );
        if (s > size) {
            // увеличение массива ВЭ на INCR_STEP элементов, если это необходимо
            err = incr_vort_quant( &POS_host, &POS_device, &VEL_host, &VEL_device, &d_device, &size );
            if (err != 0) {
                log_e( "Increase ERROR!" );
                mem_clear();
                return 1;
            }// if err
        }// if size

        // "рождение" ВЭ
        start = 0; stop = 0;
        start_timer( &start, &stop );
        err = vort_creation( POS_device, V_inf_device, conf.birth_quant, birth, n, M_device, d_g_device, panels_device );
        creation_time += stop_timer( start, stop );
        if (err ) {
            log_e( "Creation ERROR!" );
            mem_clear();
            return 1;
        }// if err
        n += conf.birth_quant;

        log_t( "%d\tN = %zu\tCreation time = %f\tSpeed time = %f\tStep time = %f\n", current_step, n, creation_time, speed_time, step_time );

        // вывод данных в файл
        if (current_step % conf.saving_step == 0) {
            cudaDeviceSynchronize();
            if( cuda_safe( cudaMemcpy( POS_host, POS_device, n * sizeof(Vortex), cudaMemcpyDeviceToHost ) ) ) {
                log_e("Saving ERROR at POS copy" );
                log_e( "n = %zu, sizeof(POS_host) = %zu, size = %zu", n, sizeof(POS_host), size );
                mem_clear();
                return 1;
            }// if cuda_safe
//////////////////////////////////////////////////////////////////////////


            cuda_safe( cudaMemcpy( V_contr_host, V_contr_device, 500 * conf.saving_step * sizeof(PVortex), cudaMemcpyDeviceToHost ) );
            cuda_safe( cudaMemcpy( v_n_device, &v_n_host, sizeof(int), cudaMemcpyHostToDevice ) );
            save_contr_vels( Contr_points_host, V_contr_host, current_step );


//////////////////////////////////////////////////////////////////////////

            save_to_file(POS_host, n, Psp, current_step);
        }// if saving_step
        velocity_control(POS_device, V_inf_device, n, Contr_points_device, V_contr_device, v_n_device);
        if( cuda_safe( cudaMemcpy( &F_p_host, F_p_device, sizeof(PVortex), cudaMemcpyDeviceToHost ) ) ) {
            log_e( "Saving ERROR at F_p copy, step =  %d F_p_host = %p F_p_device = %p", current_step, &F_p_host, F_p_device );
            mem_clear();
            return 1;
        }// if cuda_safe
        if( cuda_safe( cudaMemcpy( &Momentum_host, Momentum_device, sizeof(TVars), cudaMemcpyDeviceToHost ) ) ) {
            log_e( "Saving ERROR Momentum copy, step = %d M_host = %p M_device = %p", current_step, &Momentum_host, Momentum_device );
            mem_clear();
            return 1;
        }// if cuda_safe
        save_forces(F_p_host, Momentum_host, current_step);

        // расчёт скоростей
        start = 0; stop = 0;
        start_timer( &start, &stop );
        err = Speed( POS_device, V_inf_device, s, VEL_device, d_device, conf.viscosity, panels_device );
        if (err != 0) {
            log_e( "Speed evaluation ERROR!" );
            mem_clear();
            return 1;
        }
        speed_time += stop_timer(start, stop);
        F_p_host.v[0] = 0.0;
        F_p_host.v[1] = 0.0;
        Momentum_host = 0.0;
        cuda_safe( cudaMemcpy( F_p_device, &F_p_host , sizeof(PVortex), cudaMemcpyHostToDevice ) );
        cuda_safe( cudaMemcpy( d_g_device, &TVarsZero , sizeof(TVars), cudaMemcpyHostToDevice ) );
        cuda_safe( cudaMemcpy( Momentum_device, &Momentum_host , sizeof(TVars), cudaMemcpyHostToDevice ) );
        // перемещение ВЭ
        start = 0; stop = 0;
        start_timer( &start, &stop );
        err = Step( POS_device, VEL_device, &n, s, d_g_device, F_p_device , Momentum_device, panels_device );
        if (err != 0) {
            log_e( "Moving ERROR!" );
            mem_clear();
            return 1;
        }
        step_time += stop_timer( start, stop );
    }// for current_step
    log_t( "%d\tN = %zu\tCreation time = %f\tSpeed time = %f\tStep time = %f\n", conf.steps, n, creation_time, speed_time, step_time );
    cuda_safe( cudaMemcpy( POS_host , POS_device , n  * sizeof(Vortex) , cudaMemcpyDeviceToHost ) );
    if( cuda_safe( cudaMemcpy( POS_host, POS_device, n * sizeof(Vortex), cudaMemcpyDeviceToHost ) ) ) {
        log_e("Saving ERROR at POS copy" );
        log_e( "n = %zu, sizeof(POS_host) = %zu, size = %zu", n, sizeof(POS_host), size );
        mem_clear();
        return 1;
    }// if cuda_safe
    save_to_file( POS_host, n,  Psp, conf.steps );

    if( cuda_safe( cudaMemcpy( &F_p_host, F_p_device, sizeof(PVortex), cudaMemcpyDeviceToHost ) ) ) {
        log_e( "Saving ERROR at F_p copy, step =  %d F_p_host = %p F_p_device = %p", conf.steps, &F_p_host, F_p_device );
        mem_clear();
        return 1;
    }// if cuda_safe
    if( cuda_safe( cudaMemcpy( &Momentum_host, Momentum_device, sizeof(TVars), cudaMemcpyDeviceToHost ) ) ) {
        log_e( "Saving ERROR Momentum copy, step = %d M_host = %p M_device = %p", conf.steps, &Momentum_host, Momentum_device );
        mem_clear();
        return 1;
    }// if cuda_safe
    save_forces(F_p_host, Momentum_host, conf.steps);
    // вывод в файл последнего шага
    log_i( "ready!" );
    mem_clear();
    return 0;
}
