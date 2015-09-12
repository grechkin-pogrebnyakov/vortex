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

struct conf_t conf;
FILE *outfile222 = NULL;
int current_step = 0;
const TVars TVarsZero = 0.0;                    // для обнуления переменных в памяти GPU
TVars       *M = NULL;                          // "матрица формы" (host)
size_t      n = 0;                              // количество ВЭ
size_t      size = 0;                           // размер массива ВЭ
Eps_Str     Psp;                                // структура с радиусом ВЭ (0.008)
TVars       *d_host = NULL;                     // характерное расстояние до ближайших ВЭ (host)
PVortex     *VEL_host = NULL;                   // скорости ВЭ (host)
PVortex     *VEL_device = NULL;                 // скорости ВЭ (device)
Vortex      *POS_host = NULL;                   // ВЭ (координаты точки + интенсивность вихря) (host)
Vortex      *POS_device = NULL;                 // ВЭ (device)
TVctr       *V_inf_device = NULL;               // скорость потока (device)
TVars       *d_device = NULL;                   // характерное расстояние до ближайших ВЭ (device)
TVars       *M_device = NULL;                   // "матрица формы" (device)
TVars       *d_g_device = NULL;                 // суммарная интенсивность "убитых" ВЭ (device)
PVortex     *F_p_device = NULL;                 // главный вектор сил давления (device)
PVortex     F_p_host;                           // главный вектор сил давления (host)
TVars       *Momentum_device = NULL;            // момент сил (device)
TVars       Momentum_host = 0.0;                // момент сил (host)
TVars       *R_p_device = NULL;                 // правые части системы "рождения" ВЭ (device)
int         err = 0;                            // обработка ошибок
int         st = STEPS;                         // количество шагов по времени
TVctr       V_inf_host = VINF;                  // вектор скорости потока (host)
int         sv = SAVING_STEP;                   // шаг сохранения
TVars       nu = VISCOSITY;                     // коэффициент вязкости
PVortex     *V_contr_device = NULL;             // скорости в контрольных точках (device)
PVortex     *V_contr_host = NULL;               // скорости в контрольных точках (host)
PVortex     *Contr_points_device = NULL;        // контрольные точки для скорости (device)
PVortex     *Contr_points_host = NULL;          // контрольные точки для скорости (host)


tPanel      *panels_host = NULL;                // 
tPanel      *panels_device = NULL;              // 


static int read_config( const char *fname ) {
    FILE *conf_f = fopen( fname, "r" );
    if ( !conf_f ) {
        printf( "unable to open config file %s", fname );
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
        else if( !_strncmp( buf, "dt ") ) {
            sscanf( buf, "%*s %lf", &conf.ddt );
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
            sscanf( buf, "%*s %lf", &conf.ve_size );
        }
        else if( !_strncmp( buf, "n_col ") ) {
            sscanf( buf, "%*s %zu", &conf.n_col );
        }
        else if( !_strncmp( buf, "n_dx ") ) {
            sscanf( buf, "%*s %zu", &conf.n_dx );
        }
        else if( !_strncmp( buf, "n_dy ") ) {
            sscanf( buf, "%*s %zu", &conf.n_dy );
        }
        else if( !_strncmp( buf, "rc_x ") ) {
            sscanf( buf, "%*s %lf", &(conf.rc[0]) );
        }
        else if( !_strncmp( buf, "rc_y ") ) {
            sscanf( buf, "%*s %lf", &(conf.rc[1]) );
        }
        else if( !_strncmp( buf, "pr_file ") ) {
            sscanf( buf, "%*s %s", conf.pr_file );
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
    }
    fclose( conf_f );
    return 0;
}

static void mem_clear() {
    cudaFree(V_inf_device);
    cudaFree(d_g_device);
    cudaFree(M_device);
    cudaFree(POS_device);
    cudaFree(d_device);
    cudaFree(VEL_device);
    cudaDeviceReset();
    free( POS_host );
    free( VEL_host );
}

static void save_to_file(Vortex *POS, size_t size, Eps_Str Psp, int _step) {
    char *fname1;
    fname1 = "Kadr";
    char *fname2;
    fname2 = ".txt";
    char *fzero;
    fzero = "0";
    char fstep[6];
    char fname[15];
    fname[0] = '\0';
    itoaxx(_step,fstep,10);
    strcat(fname,fname1);
    if (_step<10) strcat(fname,fzero);
    if (_step<100) strcat(fname,fzero);
    if (_step<1000) strcat(fname,fzero);
    if (_step<10000) strcat(fname,fzero);
    if (conf.steps >=10000 && _step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname,fname2);
    FILE *outfile = fopen(fname, "w");
    if( !outfile ) {
        printf( "unable to save to file\n" );
        return;
    }
    fprintf( outfile, "%zu\n", size );
    for (size_t i = 0; i < size; ++i) {
        fprintf( outfile, "%zu %lf %lf %lf %lf %lf %lf %lf\n", i, Psp.eps, POS[i].r[0], POS[i].r[1], 0.0, 0.0, 0.0, POS[i].g );
    }//for i
    fclose(outfile);
} //save_to_file

void save_forces(PVortex F_p, TVars M, int step) {
    FILE *outfile = NULL;
    if (step == 0) {
        outfile = fopen("F_p.txt", "w");
    } else {
        outfile = fopen("F_p.txt", "a");
    }
    if( !outfile ) {
        printf( "unable to output forces\n" );
        return;
    }
    fprintf( outfile, "%d %lf %lf %lf\n", step, F_p.v[0], F_p.v[1], M );
    fclose(outfile);
}

static int load_profile(tPanel **panels, size_t *p) {
    FILE *infile = fopen(conf.pr_file, "r");
    if ( !infile ) {
        printf( "can't open profile file %s error = %s\n", conf.pr_file, strerror(errno) );
        return 1;
    }
    char buf[255];
    fgets( buf, sizeof(buf), infile );
    fgets( buf, sizeof(buf), infile );
    fscanf( infile, "%zu", p );
    float rash = (float)(*p) / BLOCK_SIZE;
    size_t birth = (size_t)(BLOCK_SIZE * ceil(rash));
    *panels = (tPanel*)malloc( sizeof(tPanel) * birth );
    for (size_t i = 0; i < *p; ++i) {
        fscanf( infile, "\n%u %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %u %u", &(*panels)[i].n, &(*panels)[i].left[0], &(*panels)[i].left[1],
                &(*panels)[i].right[0], &(*panels)[i].right[1], &(*panels)[i].contr[0], &(*panels)[i].contr[1], &(*panels)[i].birth[0], &(*panels)[i].birth[1],
                &(*panels)[i].norm[0], &(*panels)[i].norm[1], &(*panels)[i].tang[0], &(*panels)[i].tang[1], &(*panels)[i].length, &(*panels)[i].n_of_lpanel, &(*panels)[i].n_of_rpanel );
    }
    fclose( infile );
}

int main( int argc, char **argv ) {
// расширенное количество рождаемых ВЭ
// (кратно BLOCK_SIZE)
    size_t birth;
    float rash;

// количество точек рождения ВЭ
    size_t p = 0;

// ошибки CUDA
    cudaError_t cuerr = cudaSuccess;


    if( argc < 3 || strcmp( argv[1], "-c" ) ) {
        printf( "Usage:\nvortex -c file.conf\n" );
        return 1;
    }

    memset( &conf, 0, sizeof(conf) );

    if ( read_config( argv[2] ) )
        return 1;

    printf("%lf %zu\n", conf.ve_size, conf.n_col );

    cudaDeviceReset();

    load_profile(&panels_host, &p);
    if ( !conf.matrix_load ) {
        printf( "generate matrix\n" );
        int cnt = 0;
        do {
            M = matr_creation(panels_host, p);                                       // генерация матрицы
            ++cnt;
        } while (M == NULL && cnt < 10);
        if (M == NULL) {
            printf( "Matrix creation error!\n" );
            return 1;
        } else {
            printf( "Matrix created!\n" );
        }
    } else {
        printf( "load matrix\n" );
        M = load_matrix(&p);
        if (M == NULL) {
            printf( "Matrix loading error!\n" );
            return 1;
        } else {
            printf( "Matrix loaded!\n" );
        }
    }

// количество ВЭ
    n = 0;

// размер массива ВЭ
    size = 0;
    Psp.eps = 0.008;
    rash = (TVars)(p) / BLOCK_SIZE;
    birth = (size_t)(BLOCK_SIZE * ceil(rash));

//-----------------------------------------------------------------------------------------------------------------------------
    // Вычисление скоростей при x = 0.35
    Contr_points_host = (PVortex*)malloc( sizeof(PVortex) * 500 );
    for (int i = 0; i < 500; ++i) {
        Contr_points_host[i].v[1] = 0.01 + 0.002 * i;
        Contr_points_host[i].v[0] = -0.15;
    }

    V_contr_host = (PVortex*)malloc( sizeof(PVortex) * 500 * SAVING_STEP );
    cuerr = cudaMalloc( (void**)&V_contr_device, 500 * SAVING_STEP * sizeof(PVortex) );
    cuerr = cudaMalloc( (void**)&Contr_points_device, 500 * sizeof(PVortex) );
    cuerr = cudaMemcpy( Contr_points_device, Contr_points_host, 500 * sizeof(PVortex), cudaMemcpyHostToDevice );
    int v_n_host = 0;
    int *v_n_device = NULL;
    cuerr = cudaMalloc( (void**)&v_n_device, sizeof(int) );
    cuerr=cudaMemcpy( v_n_device, &v_n_host, sizeof(int), cudaMemcpyHostToDevice );

    F_p_host.v[0] = 0.0;
    F_p_host.v[1] = 0.0;

    // РІС‹РґРµР»РµРЅРёРµ РїР°РјСЏС‚Рё Рё РєРѕРїРёСЂРѕРІР°РЅРёРµ РЅР° device

    cuerr = cudaMalloc( (void**)&V_inf_device, sizeof(TVctr) );
    cuerr += cudaMalloc( (void**)&d_g_device, sizeof(TVars) );
    cuerr += cudaMalloc( (void**)&Momentum_device, sizeof(TVars) );
    cuerr += cudaMalloc( (void**)&F_p_device, sizeof(PVortex) );
    cuerr += cudaMalloc( (void**)&M_device, (birth+1) * (birth+1) * sizeof(TVars) );
    cuerr += cudaMalloc( (void**)&panels_device, birth * sizeof(tPanel) );
    cuerr += cudaMemcpy( V_inf_device, &V_inf_host, sizeof(TVctr), cudaMemcpyHostToDevice );
    cuerr += cudaMemcpy( d_g_device, &TVarsZero, sizeof(TVars), cudaMemcpyHostToDevice );
    cuerr += cudaMemcpy( Momentum_device, &Momentum_host, sizeof(TVars), cudaMemcpyHostToDevice );
    cuerr += cudaMemcpy( F_p_device, &F_p_host , sizeof(PVortex), cudaMemcpyHostToDevice );
    cuerr += cudaMemcpy( M_device, M, (birth+1) * (birth+1) * sizeof(TVars), cudaMemcpyHostToDevice );
    cuerr += cudaMemcpy( panels_device, panels_host, birth * sizeof(tPanel), cudaMemcpyHostToDevice );
    // РІСЃРµ РјР°СЃСЃРёРІС‹ РёРјРµСЋС‚ РїРµСЂРµРјРµРЅРЅСѓСЋ РґР»РёРЅСѓ Рё РїР°РјСЏС‚СЊ РґР»СЏ РЅРёС… РІС‹РґРµР»СЏРµС‚СЃСЏ РІ incr_vort_quont()

    free( M );
    printf( "dt = %lf\n", dt );

    // СѓРІРµР»РёС‡РµРЅРёРµ РјР°СЃСЃРёРІР° Р’Р­ РЅР° INCR_STEP СЌР»РµРјРµРЅС‚РѕРІ
    err = incr_vort_quont( &POS_host, &POS_device, &VEL_host, &VEL_device, &d_device, &size );
    if (err != 0)
    {
        printf( "Increase ERROR!\n" );
        mem_clear();
        return 1;
    }
    float creation_time = 0.0;
    float speed_time = 0.0;
    float step_time = 0.0;
    cudaEvent_t start = 0, stop = 0;
//------------------------------------------------------------------------------------------
    V_inf_host[0] = 0.0;
    cuerr = cudaMemcpy( V_inf_device, &V_inf_host, sizeof(TVctr), cudaMemcpyHostToDevice );
    TVars d_V_inf = 1.0/100;

    // цикл шагов выполнения расчётов
    for (current_step = 0; current_step < st; current_step++) {
        if (current_step < 100) {
            V_inf_host[0] += d_V_inf;
            cuerr = cudaMemcpy( V_inf_device, &V_inf_host, sizeof(TVctr), cudaMemcpyHostToDevice );
        }
        // количество ВЭ на текущем шаге, увеличенное до кратности BLOCK_SIZE
        size_t s = 0;
        float rashirenie = 0;
        rashirenie = (TVars)(n + p) / BLOCK_SIZE;
        s = (int)( BLOCK_SIZE * ceil(rashirenie) );
        if (s > size) {
            //СѓРІРµР»РёС‡РµРЅРёРµ РјР°СЃСЃРёРІР° Р’Р­ РЅР° INCR_STEP СЌР»РµРјРµРЅС‚РѕРІ, РµСЃР»Рё СЌС‚Рѕ РЅРµРѕР±С…РѕРґРёРјРѕ
            err = incr_vort_quont( &POS_host, &POS_device, &VEL_host, &VEL_device, &d_device, &size );
            if (err != 0) {
                printf( "Increase ERROR!\n" );
                mem_clear();
                return 1;
            }// if err
        }// if size

        // "рождение" ВЭ
        start = 0; stop = 0;
        start_timer( &start, &stop );
        err = vort_creation( POS_device, V_inf_device, p, birth, n, M_device, d_g_device, panels_device );
        creation_time += stop_timer( start, stop );
        if (err ) {
            printf( "Creation ERROR!\n" );
            mem_clear();
            return 1;
        }// if err
        n += p;

        if ( current_step % 1 == 0 ) {
            if (current_step == 0) {
                outfile222 = fopen("Log.log", "w");
            } else {
                fclose(outfile222);
                outfile222 = fopen("Log.log", "a");
            }

            fprintf( outfile222,"%d\tN = %zu\tCreation time = %f\tSpeed time = %f\tStep time = %f\n", current_step, n, creation_time, speed_time, step_time );
        }

        // вывод данных в файл
        if (current_step % sv == 0) {
//            cuerr=cudaMemcpy ( d , d_Dev , size  * sizeof(TVars) , cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cuerr = cudaMemcpy( POS_host, POS_device, n * sizeof(Vortex), cudaMemcpyDeviceToHost );
            if (cuerr != cudaSuccess) {
                printf("%s\nSaving ERROR at POS copy\n", cudaGetErrorString(cuerr) );
                printf( "n = %zu, sizeof(POS_host) = %zu, size = %zu\n", n, sizeof(POS_host), size );
                mem_clear();
                return 1;
            }// if cuerr
//            cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
            printf( "\nOutput %d\n", current_step );
/*
            double gamma = 0.0;
            for( int i = 0; i < n; ++i ) {
            gamma += POS_host[i].g;
            }
            std::cout << gamma<<'\n';
*/
//////////////////////////////////////////////////////////////////////////


            cuerr = cudaMemcpy( V_contr_host, V_contr_device, 500 * SAVING_STEP * sizeof(PVortex), cudaMemcpyDeviceToHost );
            cuerr = cudaMemcpy( v_n_device, &v_n_host, sizeof(int), cudaMemcpyHostToDevice );
            char *fname1;
            fname1 = "Vel";
            char *fname2;
            fname2 = ".txt";
            char *fzero;
            fzero = "0";
            char fstep[6];
            char fname[15];
            fname[0] = '\0';
            itoaxx(current_step,fstep,10);
            strcat(fname,fname1);
            if (current_step<10) strcat(fname,fzero);
            if (current_step<100) strcat(fname,fzero);
            if (current_step<1000) strcat(fname,fzero);
            if (current_step<10000) strcat(fname,fzero);
            //	if (_step<100000) strcat(fname,fzero);
            strcat(fname,fstep);
            strcat(fname,fname2);
            FILE *outfile = fopen(fname, "w");
            for (size_t i = 0; i < (500 * SAVING_STEP); ++i) {
                fprintf( outfile, "%zu %lf %lf %lf %lf\n", i, Contr_points_host[i%500].v[0], Contr_points_host[i%500].v[1], V_contr_host[i].v[0], V_contr_host[i].v[1] );
            }//for i
            fclose(outfile);


//////////////////////////////////////////////////////////////////////////

            save_to_file(POS_host, n, Psp, current_step);
        }// if sv
        velocity_control(POS_device, V_inf_device, n, Contr_points_device, V_contr_device, v_n_device);
        cuerr = cudaMemcpy( &F_p_host, F_p_device, sizeof(PVortex), cudaMemcpyDeviceToHost );
        if (cuerr != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(cuerr) );
            printf( "Saving ERROR at F_p copy, step =  %d F_p_host = %p F_p_device = %p\n", current_step, &F_p_host, F_p_device );
            mem_clear();
            return 1;
        }// if cuerr
        cuerr = cudaMemcpy( &Momentum_host, Momentum_device, sizeof(TVars), cudaMemcpyDeviceToHost );
        if (cuerr != cudaSuccess) {
            printf( "%s\n", cudaGetErrorString(cuerr) );
            printf( "Saving ERROR Momentum copy, step = %d M_host = %p M_device = %p\n", current_step, &Momentum_host, Momentum_device );
            mem_clear();
            return 1;
        }// if cuerr
        save_forces(F_p_host, Momentum_host, current_step);
/*
			if(j==1)
			{
				TVars *gg=new TVars[p];
				for( int i=0;i<p;i++)
				{
					gg[i]=0.0;

					for (size_t k=0;k<p;k++)
					{
						gg[i]+=M[(birth+1)*i+k]*POS[k].g;
					}
				}
				for (int i=0;i<p;i++) {
                    POS[i].g=gg[i];
                }
				save_to_file(0);
			}
			TVars gamma=0.0;

//			for (int k=(*n)-1.0;k>((*n)-QUANT)-1.0;k--)
			for (int k=0;k<(*n);k++)
				gamma+=POS[k].g;
			cout<<" j= "<<j <<";  gamma= "<<gamma<<endl;
*/

//			cout << j;

//		if ((j%100 == 0) && (j%1000 != 0)) cout<<"j= "<<j<<endl;

        // расчёт скоростей
        start = 0; stop = 0;
        start_timer( &start, &stop );
        err = Speed( POS_device, V_inf_device, s, VEL_device, d_device, nu, panels_device );
        if (err != 0) {
            printf( "Speed evaluation ERROR!\n" );
            mem_clear();
            return 1;
        }
        speed_time += stop_timer(start, stop);
/*
		if (j==0)																		//РІС‹РІРѕРґ СЃРєРѕСЂРѕСЃС‚РµР№ РІ С„Р°Р№Р»
		{
			cuerr=cudaMemcpy ( VEL , VDev , (*n)  * sizeof(PVortex) , cudaMemcpyDeviceToHost);
			stf();

		}
*/
        F_p_host.v[0] = 0.0;
        F_p_host.v[1] = 0.0;
        Momentum_host = 0.0;
        cuerr = cudaMemcpy( F_p_device, &F_p_host , sizeof(PVortex), cudaMemcpyHostToDevice );
        cuerr = cudaMemcpy( d_g_device, &TVarsZero , sizeof(TVars), cudaMemcpyHostToDevice );
        cuerr = cudaMemcpy( Momentum_device, &Momentum_host , sizeof(TVars), cudaMemcpyHostToDevice );
        // перемещение ВЭ
        start = 0; stop = 0;
        start_timer( &start, &stop );
        err = Step( POS_device, VEL_device, &n, s, d_g_device, F_p_device , Momentum_device, panels_device );
        if (err != 0) {
            printf( "Moving ERROR!\n" );
            mem_clear();
            return 1;
        }
        step_time += stop_timer( start, stop );
//        cout << n << '\n';
/*
		if (j==0)																		//РІС‹РІРѕРґ РІ С„Р°Р№Р» Р’Р­ РїРѕСЃР»Рµ РїРµСЂРµРјРµС‰РµРЅРёСЏ
		{

			cuerr=cudaMemcpy ( POS , posDev , (*n)  * sizeof(Vortex) , cudaMemcpyDeviceToHost);

			save_to_file(30);
		}
*/
	}//j
//------------------------------------------------------------------------------------------
//	float time = stop_timer(start, stop);
//	cout << "Computing time = "<< time << " sec\n";
    printf( "Creation time = %lf speed time = %lf step time = %lf\n", creation_time, speed_time, step_time );
    cuerr = cudaMemcpy( POS_host , POS_device , n  * sizeof(Vortex) , cudaMemcpyDeviceToHost );
    cuerr = cudaMemcpy( &F_p_host, F_p_device, sizeof(PVortex), cudaMemcpyDeviceToHost );
    cuerr = cudaMemcpy( &Momentum_host, Momentum_device, sizeof(PVortex), cudaMemcpyDeviceToHost );
    // РІС‹РІРѕРґ РІ С„Р°Р№Р» РїРѕСЃР»РµРґРЅРµРіРѕ С€Р°РіР°
    save_to_file( POS_host, n,  Psp, st );
    save_forces( F_p_host, Momentum_host, st );
    printf( "ready!\n" );
    mem_clear();
    fclose(outfile222);
    return 0;
}
