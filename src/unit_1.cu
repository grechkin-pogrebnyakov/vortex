/*
 ============================================================================
 Name        : unit_1.cu
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Mar. 02, 2015
 Copyright   : All rights reserved
 Description : unit_1 file of vortex project
 ============================================================================
 */

#include "unit_1.cuh"
#include "kernel.cuh"


extern struct conf_t conf;
extern cudaError_t cuda_error;
extern int current_step;

__constant__ TVars dt;
__constant__ size_t quant;
__constant__ TVars ve_size;
__constant__ TVars ve_size2;
__constant__ TVars r_col_diff_sign2, r_col_same_sign2;
__constant__ TVars max_ve_g;
__constant__ size_t n_of_points;
__constant__ TVars x_max;
__constant__ TVars x_min;
__constant__ TVars y_max;
__constant__ TVars y_min;
__constant__ TVars h_col_x;
__constant__ TVars h_col_y;
__constant__ TVars rho;
__constant__ TVars rc_x;
__constant__ TVars rc_y;

#ifndef NO_TREE
__constant__ float theta;
#endif // NO_TREE

//#include "kernel.cuh"
static int save_matr( TVars** M, size_t size, char *name ) {
    if ( !name ) name = "D.txt";
    if (M == NULL) return 1;
    FILE *outfile = fopen( name, "w" );
    if( !outfile ) return 2;
    fprintf( outfile, "%zu\n", size );
    for (size_t i = 0; i < size; ++i) {
        if (M[i] == NULL) {
            fclose( outfile );
            return 1;
        }
        for (size_t j = 0; j < size; j++) {
            fprintf( outfile, "%lf\t", M[i][j] );
        }// for j
        fprintf( outfile, "\n");
    }// for i
    fclose( outfile );
    return 0;
}

void    clear_memory(TVars **M, size_t s) {
    if (M != NULL) {
        for (size_t i = 0; i < s; ++i) {
            if (M[i] != NULL) {
                free( M[i] );
            }
        }
        free( M );
    }
}

static int move_line(TVars **M, size_t s, size_t st, size_t fin) {
    TVars *Ln = (TVars*)malloc( sizeof(TVars) * s );
    if (!Ln) return 1;
    for (size_t i = 0; i < s ; i++) {
        Ln[i] = M[st][i];
    }
    for (size_t i = 0; i < s ; i++) {
        M[st][i] = M[fin][i];
    }
    for (size_t i = 0; i < s ; i++) {
        M[fin][i] = Ln[i];
    }
    free( Ln );
    return 0;
}

static int move_all_back(TVars **M, size_t size, size_t *mov) {
    if (M == NULL || mov == NULL) return 1;
    int err = 0;
    int cnt = 0;
    for (size_t i = 0; i < size; ++i) {
        if (mov[i] != i) {
            err = move_line( M, size, i, mov[i] );
            if (err || cnt < 10) {
                --i;
                ++cnt;
                continue;
            }
            cnt = 0;
            if (err) return 1;
            mov[mov[i]] = mov[i];
            mov[i] = i;
        }
    }
    return 0;
}

static TVars **inverse_matrix(TVars **M, size_t size) {
    int err = 0;
    // порядок строк в матрице
    size_t *POR = (size_t*)malloc( sizeof(size_t) * size );
    if (!POR) return NULL;
    size_t PR;
    for (size_t i = 0; i < size; i++) {
        POR[i]=i;
    }
    TVars b;
    TVars **M_inv = (TVars**)malloc( sizeof(TVars*) * size );
    {
        size_t i;
        for(i = 0; i < size; ++i) {
            M_inv[i] = (TVars*)malloc( sizeof(TVars) * size );
            if (!M_inv[i]) break;
            memset(M_inv[i], 0, sizeof(TVars) * size );
        }
        if (i != size) {
            while (i != 0) {
                free( M_inv[i--] );
            }
            free( M_inv );
            free( POR );
            return NULL;
        }
    }
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            if (i != j) {
                M_inv[i][j]=0;
            }
            else {
                M_inv[i][j]=1;
            }
        }
    }
    if (fabs(M[0][0]) < DELT) {
        TVars mx = fabs( M[0][0] );
        size_t num = 0;
        for (size_t i = 1; i < size; i++) {
            if (fabs( M[i][0] ) > mx) {
                mx = fabs( M[i][0] );
                num = i;
            }
        }//i
        if (num != 0) {
            err = move_line( M, size, 0, num );
            if (err) {
                move_all_back(M, size, POR);
                free( POR );
                clear_memory(M_inv, size);
                return NULL;
            }
            err = move_line( M_inv, size, 0, num );
            PR = POR[0];
            POR[0] = POR[num];
            POR[num] = PR;
            if (err) {
                move_all_back(M, size, POR);
                free( POR );
                clear_memory( M_inv, size );
                return NULL;
            }
        }
    }//if
    for (size_t k = 0; k < size-1; k++) {
        if (fabs(M[k][k]) < DELT) {
            move_all_back( M, size, POR );
            free( POR );
            clear_memory( M_inv, size );
            return NULL;
        }//if
        TVars mx = fabs( M[k+1][k+1] );
        size_t line = k +1;
        for (size_t i = k + 1; i < size; i++) {               // Выбор главного элемента
            if (fabs( M[i][k + 1] ) > mx) {
                mx = fabs( M[i][k + 1] );
                line = i;
            }//if
        }//i
        if (mx < DELT) {
            move_all_back( M, size, POR );
            free( POR );
            clear_memory( M_inv, size );
            return NULL;
        }
        err = move_line( M, size, k + 1, line );
        if (err) {
            move_all_back( M, size, POR );
            free( POR );
            clear_memory( M_inv, size );
            return NULL;
        }
        err = move_line( M_inv, size, k + 1, line );                      // перестановка строк
        PR = POR[k + 1];
        POR[k + 1] = POR[line];
        POR[line] = PR;
        if (err) {
            move_all_back( M, size, POR );
            free( POR );
            clear_memory( M_inv, size );
            return NULL;
        }
        for (size_t i = 0; i < size; i++) {
            if (i != k) {
                TVars c = M[i][k] / M[k][k];
                for (size_t j = 0; j < size; j++) {
                    b = M[i][j] - c * (M[k][j]);                  // преобразование матрицы
                    M[i][j] = b;
                    b = M_inv[i][j] - c * (M_inv[k][j]);          // преобразование матрицы
                    M_inv[i][j] = b;
                }//j
            }//if
        }//i
    }//k
    if (fabs( M[size - 1][size - 1] ) < DELT) {
        move_all_back( M, size, POR );
        free( POR );
        clear_memory( M_inv, size );
        return NULL;
    }
    for (size_t i = 0; i < size-1; ++i) {
        TVars c = M[i][size - 1] / M[size - 1][size - 1];
    //		   b=M[i][size-1]-c*(M[size-1][size-1]);        // преобразование матрицы
    //		   M[i][size-1]=b;
        for (size_t j = 0; j < size; j++) {
            b = M_inv[i][j] - c * (M_inv[size - 1][j]);                 // преобразование матрицы
            M_inv[i][j] = b;
        }// j
    }// i
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            M_inv[i][j] = M_inv[i][j] / M[i][i];
        }// j
    }// i
    move_all_back( M, size, POR );
    free( POR );
    return M_inv;
}

TVars   *matr_creation(tPanel *panels, size_t s, size_t birth) {
    TVars **M = NULL;
    TVars *MM = NULL;
    TVars **L = NULL;
    L = (TVars**)malloc(sizeof(TVars*) * (s + 1));
    if (L == NULL) {
        return NULL;
    }
    {
        size_t i;
        for(i = 0; i < s + 1; i++) {
            L[i] = (TVars*)malloc(sizeof(TVars) * (s + 1));
            if (L[i] == NULL) break;
            memset(L[i], 0, sizeof(TVars) * (s + 1) );
        }
        if (i != s+1) {
            while (i != 0) {
                free( L[i--] );
            }
            free( L );
        }
    }
    TVctr a_0 = {0.0, 0.0};
    TVctr a_1 = {0.0, 0.0};
    TVctr b_0 = {0.0, 0.0};
    TVctr b_1 = {0.0, 0.0};
    TVctr d_i = {0.0, 0.0};
    TVctr d_j = {0.0, 0.0};
    TVctr p_1 = {0.0, 0.0}, p_2 = {0.0, 0.0}, s_1 = {0.0, 0.0}, s_2 = {0.0, 0.0};
    TVars q1_1 = 0.0, q1_2 = 0.0, q1_3 = 0.0, q2_1 = 0.0, q2_2 = 0.0, q2_3 = 0.0, z_1 = 0.0, z_2 = 0.0, z_3 = 0.0;
    TVctr c_1 = {0.0, 0.0}, c_2 = {0.0, 0.0}, c_3 = {0.0, 0.0};
    TVctr tau = {0.0, 0.0}, v = {0.0, 0.0};
    for (size_t i = 0; i < s; ++i) {
        tau[0] = Tau_x( panels, i );
        tau[1] = Tau_y( panels, i );
        for (size_t j = 0;j < s; ++j) {
            if ( j != i ) {
                a_0[0] = R_left_x( panels, i );
                a_0[1] = R_left_y( panels, i );
                a_1[0] = R_right_x( panels, i );
                a_1[1] = R_right_y( panels, i );

                b_0[0] = R_left_x( panels, j );
                b_0[1] = R_left_y( panels, j );
                b_1[0] = R_right_x( panels, j );
                b_1[1] = R_right_y( panels, j );

                if ( ( j == i +1 ) || ( ( j == 0 ) && ( i == s - 1 ) ) ) {
                    a_1[0] = R_left_x( panels, i );
                    a_1[1] = R_left_y( panels, i );
                    a_0[0] = R_right_x( panels, i );
                    a_0[1] = R_right_y( panels, i );

                    b_1[0] = R_left_x( panels, j );
                    b_1[1] = R_left_y( panels, j );
                    b_0[0] = R_right_x( panels, j );
                    b_0[1] = R_right_y( panels, j );
                }

                d_j[0] = b_1[0] - b_0[0];
                d_j[1] = b_1[1] - b_0[1];
                d_i[0] = a_1[0] - a_0[0];
                d_i[1] = a_1[1] - a_0[1];

                p_1[0] = a_0[0] - b_1[0];
                p_1[1] = a_0[1] - b_1[1];
                p_2[0] = a_1[0] - b_1[0];
                p_2[1] = a_1[1] - b_1[1];

                s_1[0] = a_0[0] - b_0[0];
                s_1[1] = a_0[1] - b_0[1];
                s_2[0] = a_1[0] - b_0[0];
                s_2[1] = a_1[1] - b_0[1];

                z_1 = p_1[0] * p_2[1] - p_1[1] * p_2[0];
                z_2 = s_1[0] * s_2[1] - s_1[1] * s_2[0];
                z_3 = s_2[0] * p_2[1] - s_2[1] * p_2[0];

                if ( ( j == i - 1 ) || ( j == i + 1 )\
                  || ( (j == 0 ) && ( i == s - 1 ) )\
                  || ( (i == 0 ) && ( j == s - 1 ) ) ) {
                    q1_1 = 0.0; q2_1 = 0.0;
                } else {
                    q1_1 = atan( sp_vec( d_i, p_1 ) / z_1 ) - atan( sp_vec( d_i, p_2 ) / z_1 );
                    q2_1 = 0.5 * log( sp_vec( p_2, p_2 ) / sp_vec( p_1, p_1 ) );
                }

                q1_2 = atan( sp_vec( d_i, s_2 ) / z_2 ) - atan( sp_vec( d_i, s_1 ) / z_2 );
                q1_3 = atan( sp_vec( d_j, p_2 ) / z_3 ) - atan( sp_vec( d_j, s_2 ) / z_3 );

                q2_2 = 0.5 * log( sp_vec( s_1, s_1 ) / sp_vec( s_2, s_2 ) );
                q2_3 = 0.5 * log( sp_vec( p_2, p_2 ) / sp_vec( s_2, s_2 ) );

                c_1[0] = sp_vec( d_j, p_1 ) * d_i[0] + sp_vec( d_i, s_1 ) * d_j[0] \
                - sp_vec( d_i, d_j ) * s_1[0];
                c_1[1] = sp_vec( d_j, p_1 ) * d_i[1] + sp_vec( d_i, s_1 ) * d_j[1] \
                - sp_vec( d_i, d_j ) * s_1[1];
                c_2[0] = c_1[0] + sp_vec( d_j, d_j ) * d_i[0];
                c_2[1] = c_1[1] + sp_vec( d_j, d_j ) * d_i[1];
                c_3[0] = sp_vec( d_i, d_i ) * d_j[0];
                c_3[1] = sp_vec( d_i, d_i ) * d_j[1];

                v[0] = 1.0 / ( 2 * M_PI * sqrt( sp_vec( d_j, d_j ) ) * sp_vec( d_i, d_i ) )\
                     * (q1_1 * c_1[0] + q1_2 * c_2[0] + q1_3 * c_3[0]\
                      + ( q2_1 * c_1[1] + q2_2 * c_2[1] + q2_3 * c_3[1] ) );
                v[1] = 1.0 / ( 2 * M_PI * sqrt( sp_vec( d_j, d_j ) ) * sp_vec( d_i, d_i ) )\
                     * (q1_1 * c_1[1] + q1_2 * c_2[1] + q1_3 * c_3[1]\
                      - ( q2_1 * c_1[0] + q2_2 * c_2[0] + q2_3 * c_3[0] ) );

                L[i][j] = sp_vec( v, tau );
            } else L[i][j] = -0.5;
        }
    }
    for (size_t i=0; i<s; i++) {
        L[s][i] = Panel_length( panels, i );
        L[i][s]=1;
    }
    L[s][s]=0;
    save_matr(L, s+1, "L.txt");
    M=inverse_matrix(L,s+1);
    clear_memory(L, s+1);
    if (M == NULL) {
        return NULL;
    }
    save_matr(M, s+1, "M.txt");
    MM = (TVars*)malloc( sizeof(TVars) * (birth + 1) * (birth + 1) );
    memset(MM, 0, sizeof(TVars) * (birth + 1) * (birth + 1) );
    if (MM == NULL) {
        clear_memory(M, s+1);
        return NULL;
    }
    for (size_t i=0; i < s+1; i++) {
        for (size_t j=0; j < s+1 ; j++) {
            MM[(birth+1)*i+j]=M[i][j];
        }
        for (size_t j=(s+1); j<(birth+1);j++) {
            MM[(birth+1)*i+j]=0;
        }
    }
    for (size_t i=s+1; i < birth+1; i++) {
        for (size_t j=0; j<(birth+1);j++) {
            MM[(birth+1)*i+j]=0;
        }
    }
    clear_memory(M, s+1);
    return MM;
}

TVars   *load_matrix(size_t *p) {
    FILE *infile = fopen( "M.txt", "r" );
    if( !infile ) {
        log_e("unable to open matrix file %s", "M.txt");
        return NULL;
    }
    fscanf( infile, "%zu", p );
    (*p)--;
    TVars **M = (TVars**)malloc( sizeof(TVars*) * (*p + 1) );
    if (M == NULL) {
        return NULL;
    }
    {
        size_t i;
        for(i = 0; i < *p + 1; i++) {
            M[i] = (TVars*)malloc( sizeof(TVars) * (*p + 1) );
            if (M[i] == NULL) break;
            memset(M[i], 0, sizeof(TVars) * (*p + 1) );
        }
        if (i != *p + 1) {
            while (i != 0) {
                free( M[i--] );
            }
            free( M );
        }
    }
    for (size_t i = 0; i < *p + 1; ++i) {
        for (size_t j = 0; j < *p + 1; ++j) {
            fscanf( infile, "%lf ", &(M[i][j]) );
        }
    }
    fclose( infile );
    TVars rash = 0.0;
    size_t birth = 0;
    rash = (TVars)(*p) / BLOCK_SIZE;
    birth = (size_t)(BLOCK_SIZE * ceil(rash));

    TVars *MM = NULL;
    MM = (TVars*)malloc( sizeof(TVars) * (birth + 1) * (birth + 1) );
    if (MM == NULL) {
        clear_memory(M, *p + 1);
        return NULL;
    }
    memset( MM, 0, sizeof(TVars) * (birth + 1) * (birth + 1) );

    for (size_t i=0; i < *p + 1; i++) {
        for (size_t j=0; j < *p + 1 ; j++) {
            MM[(birth + 1) * i + j] = M[i][j];
        }
    }
    clear_memory(M, *p + 1);
    return MM;
}

int allocate_arrays(Vortex **p_host, Vortex **p_dev, PVortex **v_host, PVortex **v_dev, TVars **d_dev, size_t size) {
    if ( !p_host || !p_dev || !v_host || !v_dev || !d_dev || !size ) {
        log_e( "wrong parameters" );
        return 2;
    }
    *p_host = (Vortex*)malloc( sizeof(Vortex) * size );
    memset( *p_host, 0, sizeof(Vortex) * size );
    *v_host = (PVortex*)malloc( sizeof(PVortex) * size );
    memset( *v_host, 0, sizeof(PVortex) * size );
    if( cuda_safe( cudaMalloc( (void**)p_dev, size * sizeof(Vortex) ) ) ) {
        return 1;
    }
    if( cuda_safe( cudaMemset( (void*)(*p_dev), 0, size * sizeof(Vortex) ) ) ) {
        return 1;
    }
    if( cuda_safe( cudaMalloc( (void**)d_dev, size * sizeof(TVars) ) ) ) {
        return 1;
    }
    if( cuda_safe( cudaMemset( (void*)(*d_dev), 0, size * sizeof(TVars) ) ) ) {
        return 1;
    }
    if( cuda_safe( cudaMalloc( (void**)v_dev, size  * sizeof(PVortex) ) ) ) {
        return 1;
    }
    if( cuda_safe( cudaMemset( (void*)(*v_dev), 0, size  * sizeof(PVortex) ) ) ) {
        return 1;
    }
    return 0;
}

int randomize_tail(Vortex **p_dev, size_t new_size, size_t increased) {
    float *rnd_dev = NULL, *rnd_host = NULL;
    rnd_host = (float*)malloc( sizeof(float) * increased );
    for (int i = 0; i < increased; ++i) {
        rnd_host[i] = (float)rand();
    }
    if( cuda_safe( cudaMalloc( (void**)&rnd_dev, increased * sizeof(float) ) ) ) {
        return 1;
    }
    if( cuda_safe( cudaMemcpy( rnd_dev, rnd_host, increased * sizeof(float), cudaMemcpyHostToDevice ) ) ) {
        return 1;
    }
    dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(increased/BLOCK_SIZE);
    // generate random numbers
    zero_Kernel <<< blocks, threads >>> (rnd_dev, *p_dev, new_size - increased );
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }
}

int incr_vort_quant(Vortex **p_host, Vortex **p_dev, PVortex **v_host, PVortex **v_dev, TVars **d_dev, size_t *size) {
    if ( !p_host || !p_dev || !v_host || !v_dev || !d_dev || !size ) {
        log_e( "wrong parameters" );
        return 2;
    }
    log_d("increase vortex quantity");
    if ( *p_host && *p_dev && *v_host && *v_dev && *d_dev ) {
        Vortex *p_dev_new = NULL;
        size_t size_n = *size + conf.inc_step;
        if( cuda_safe( cudaMalloc( (void**)&p_dev_new , size_n * sizeof(Vortex) ) ) ) {
            return 1;
        }
        if( cuda_safe( cudaMemcpy( p_dev_new, *p_dev, *size  * sizeof(Vortex), cudaMemcpyDeviceToDevice ) ) ) {
            return 1;
        }
        *size += conf.inc_step;
        free( *p_host );
        *p_host = (Vortex*)malloc( sizeof(Vortex) * (*size) );
        memset( *p_host, 0, sizeof(Vortex) * (*size) );
        free( *v_host );
        *v_host = (PVortex*)malloc( sizeof(PVortex) * (*size) );
        memset( *v_host, 0, sizeof(PVortex) * (*size) );
        cudaFree(*p_dev);
        cudaFree(*d_dev);
        cudaFree(*v_dev);
        if( cuda_safe( cudaMalloc( (void**)d_dev, *size * sizeof(TVars) ) ) ) {
            return 1;
        }
        if( cuda_safe( cudaMalloc( (void**)v_dev, *size  * sizeof(PVortex) ) ) ) {
            return 1;
        }
        *p_dev = p_dev_new;
        cudaDeviceSynchronize();
    }
    else if ( !(*p_host) && !(*p_dev) && !(*v_host) && !(*v_dev) && !(*d_dev) ) {
        *size = conf.inc_step;
        if( allocate_arrays( p_host, p_dev, v_host, v_dev, d_dev, *size ) )
            return 1;
    }
    else {
        log_e( "wrong parameters" );
        return 2;
    }
    if( randomize_tail( p_dev, *size, conf.inc_step ) )
        return 1;

    return 0;
}

int vort_creation(Vortex *pos, TVctr *V_infDev, size_t n_of_birth, size_t n_of_birth_BLOCK_S,
                     size_t n, TVars *M_Dev, TVars *d_g, tPanel *panels) {
    log_d("vortex creation");
    cudaEvent_t start, stop;
    start_timer( &start, &stop );
    TVars *R_p = NULL;
    if( cuda_safe( cudaMalloc( (void**)&R_p, (n_of_birth_BLOCK_S) * sizeof(TVars) ) ) ) {
        return 1;
    }

    dim3 threads1 = dim3(BLOCK_SIZE);
    dim3 blocks1  = dim3(n_of_birth_BLOCK_S/BLOCK_SIZE);
    log_d( "n = %zu", n );
    Right_part_Kernel <<< blocks1, threads1 >>> (pos, V_infDev, n, n_of_birth_BLOCK_S, R_p, panels);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }
    if( LEV_DEBUG < conf.log_level ) {
        TVars *R_p_host = (TVars*)malloc( (n_of_birth_BLOCK_S) * sizeof(TVars) );
        if( cuda_safe( cudaMemcpy(R_p_host, R_p, (n_of_birth_BLOCK_S) * sizeof(TVars), cudaMemcpyDeviceToHost) ) ) {
            return 1;
        }
        for( size_t i = 0; i < (n_of_birth_BLOCK_S); ++i )
            if( R_p_host[i] > DELT ) log_d( "R_p[%zu] = %lf", i, R_p_host );
        free( R_p_host );
    }

    birth_Kernel<<< blocks1, threads1 >>>(pos, n, n_of_birth, n_of_birth_BLOCK_S, M_Dev, d_g, R_p, panels);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }
    cudaFree(R_p);
    return 0;
}

void start_timer(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventCreate(start);
    cudaEventCreate(stop);
    cudaEventRecord(*start,0);
    cudaEventSynchronize(*start);
}

float stop_timer(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time=time/1000;
    return time;
}

 __attribute__((unused))
static void save_vel_to_file(Vortex *POS, PVortex *VEL, size_t size, int _step, int stage) {
    if( !POS || !VEL ) return;
    char fname1[] = "output/vels/Vel";
    char fname2[] = ".txt";
    char fzero[] = "0";
    char fstep[8];
    char fname[ sizeof(fname1) + 10 ];
    fname[0] = '\0';
    char stage_str[6];
    sprintf(stage_str, "_%d", stage);
    sprintf(fstep,"%d", _step);
    strcat(fname,fname1);
    if (_step<10) strcat(fname,fzero);
    if (_step<100) strcat(fname,fzero);
    if (_step<1000) strcat(fname,fzero);
    if (_step<10000) strcat(fname,fzero);
    if (conf.steps >= 10000 && _step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname, stage_str);
    strcat(fname,fname2);
    FILE *outfile = fopen( fname, "w" );
    if( !outfile ) {
        log_e("error file opening %s : %s", fname, strerror(errno) );
        return;
    }
    // сохранение числа вихрей в пелене
    fprintf( outfile, "%zu\n", size );
    for (size_t i = 0; i < size; ++i) {
        fprintf( outfile, "%zu %lf %lf %lf %lf\n", i, POS[i].r[0], POS[i].r[1], VEL[i].v[0], VEL[i].v[1] );
    }//for i
    fclose( outfile );
} //save_to_file

 __attribute__((unused))
static void save_d(TVars *d, size_t size, int _step) {
    char *fname1;
    fname1 = "output/ddd/d";
    char *fname2;
    fname2 = ".txt";
    char *fzero;
    fzero = "0";
    char fstep[8];
    char fname[20];
    fname[0] = '\0';
    sprintf(fstep,"%d", _step);
    strcat(fname,fname1);
    if (_step<10) strcat(fname,fzero);
    if (_step<100) strcat(fname,fzero);
    if (_step<1000) strcat(fname,fzero);
    if (_step<10000) strcat(fname,fzero);
    if (conf.steps >= 10000 && _step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname,fname2);
    FILE *outfile = fopen( fname, "w" );
    if( !outfile ) {
        log_e( "error open file %s : %s", fname, strerror(errno) );
        return;
    }
    // сохранение числа вихрей в пелене
    fprintf( outfile, "%zu\n", size );
    for (size_t i = 0; i < size; ++i) {
        fprintf( outfile, "%zu %lf\n", i, d[i] );
    }//for i
    fclose( outfile );
} //save_to_file


#ifndef NO_TREE
static void output_tree( tree_t *t, size_t depth, int current_step ) {
    char filename[64];
    snprintf(filename, sizeof(filename), "tree_%d.txt", current_step);
    FILE *f = fopen(filename, "w");
    for( size_t j = 0; j < depth; ++j ) {
        unsigned count_on_level = 1 << j;
        fprintf( f, "\nlevel = %zu\n", j );
        for( size_t jj = 0; jj < count_on_level; ++jj ) {
            fprintf( f, "i = %zu\nx_min = %f\nx_max = %f\ny_min = %f\ny_max = %f\naxe = %u\nmed = %f\ng_above = %f\nxg_above = %f\nyg_above = %f\ng_below = %f\n"
                    "xg_below = %f\nyg_below = %f\nrc_above.x = %f\nrc_above.y = %f\nrc_below.x = %f\nrc_below.y = %f\nrc.x = %f\nrc.y = %f\ndimx = %f\ndimy = %f\n\n",
            jj, t[jj].x_min, t[jj].x_max, t[jj].y_min, t[jj].y_max, t[jj].axe, t[jj].med, t[jj].g_above, t[jj].xg_above, t[jj].yg_above, t[jj].g_below,
            t[jj].xg_below, t[jj].yg_below, t[jj].rc_above.x, t[jj].rc_above.y, t[jj].rc_below.x, t[jj].rc_below.y, t[jj].rc.x, t[jj].rc.y, t[jj].dim.x, t[jj].dim.y );
        }
        t += count_on_level;
    }
    fclose(f);
}

#define BUILD_LEVEL (2)
#define BUILD_COUNT (1 << (BUILD_LEVEL-1))

#define BUILD_TREE_STEP_IMPL( _level_, _start_index_) ({ \
    first_tree_reduce_Kernel<BLOCK_SIZE, _level_> <<< dim3(second_reduce_size), dim3(BLOCK_SIZE) >>> ( pos, s, tree_pointer, tmp_tree_2, _start_index_ ); \
    log_d("first_tree_reduce_Kernel %p lev = %u start_index = %u", tree_pointer, _level_, _start_index_); \
    cudaDeviceSynchronize(); \
    if( cuda_safe( cudaGetLastError() ) ) { \
        log_e("first_tree_reduce_Kernel %u",_level_); \
        return 1; \
    } \
    second_tree_reduce_Kernel<BLOCK_SIZE, _level_> <<< dim3(1), dim3(BLOCK_SIZE) >>> ( tmp_tree_2, second_reduce_size, _next_level_tree_ ); \
    cudaDeviceSynchronize(); \
    if( cuda_safe( cudaGetLastError() ) ) { \
        log_e("second_tree_reduce_Kernel %u",_level_); \
        return 1; \
    } \
})

#define BUILD_TREE_STEP( _lev_ ) ({ \
    log_d("tree build: level = %u", _lev_); \
    size_t __prev_level_count =  1 << (_lev_ - 1); \
    node_t *_next_level_tree_ = tree_pointer + __prev_level_count; \
    calculate_tree_index_Kernel<BLOCK_SIZE, _lev_> <<< dim3(second_reduce_size), dim3(BLOCK_SIZE) >>> ( pos,  s, tree_pointer ); \
    log_d("calculate_tree_index_Kernel %p lev = %u", tree_pointer, _lev_); \
    cudaDeviceSynchronize(); \
    if( cuda_safe( cudaGetLastError() ) ) { \
        log_e("calculate_tree_index_Kernel %u",_lev_); \
        return 1; \
    } \
    if( _lev_ <= BUILD_LEVEL ) { \
        BUILD_TREE_STEP_IMPL( _lev_, 0 ); \
        tree_pointer = _next_level_tree_; \
    } else { \
        node_t *orig_ptr = _next_level_tree_; \
        for( size_t iii = 0; iii < __prev_level_count; iii += BUILD_COUNT ) { \
            BUILD_TREE_STEP_IMPL( BUILD_LEVEL, _next_level_tree_ - orig_ptr ); \
            log_d("new tree build: level = %u, step %u ok", _lev_, iii); \
            tree_pointer += BUILD_COUNT; \
            _next_level_tree_ += BUILD_COUNT * 2; \
        } \
    } \
})

#define BUILD_TREE_STEP_CASE( _i_ ) \
    case _i_: \
        BUILD_TREE_STEP( _i_ ); \
        break;

#define BUILD_TREE_STEP_SIMPLE( __level__ ) \
    switch(( __level__ )) { \
        BUILD_TREE_STEP_CASE( 1 )\
        BUILD_TREE_STEP_CASE( 2 )\
        BUILD_TREE_STEP_CASE( 3 )\
        BUILD_TREE_STEP_CASE( 4 )\
        BUILD_TREE_STEP_CASE( 5 )\
        BUILD_TREE_STEP_CASE( 6 )\
        BUILD_TREE_STEP_CASE( 7 )\
        BUILD_TREE_STEP_CASE( 8 )\
        BUILD_TREE_STEP_CASE( 9 )\
        BUILD_TREE_STEP_CASE( 10 )\
        BUILD_TREE_STEP_CASE( 11 )\
        BUILD_TREE_STEP_CASE( 12 )\
        BUILD_TREE_STEP_CASE( 13 )\
        BUILD_TREE_STEP_CASE( 14 )\
        BUILD_TREE_STEP_CASE( 15 )\
        BUILD_TREE_STEP_CASE( 16 )\
        BUILD_TREE_STEP_CASE( 17 )\
        BUILD_TREE_STEP_CASE( 18 )\
        BUILD_TREE_STEP_CASE( 19 )\
        BUILD_TREE_STEP_CASE( 20 )\
        BUILD_TREE_STEP_CASE( 21 )\
        BUILD_TREE_STEP_CASE( 22 )\
        BUILD_TREE_STEP_CASE( 23 )\
        BUILD_TREE_STEP_CASE( 24 )\
        BUILD_TREE_STEP_CASE( 25 )\
        BUILD_TREE_STEP_CASE( 26 )\
        default: \
            log_e("tree_depth %u unsuported\n", __level__ + 1); \
            return 1; \
    } \

#define FIND_NODES_PARAMS_SIMPLE( _last_level_, _start_index_ ) ({\
    first_find_leaves_params_Kernel<BLOCK_SIZE, _last_level_> <<< dim3(second_reduce_size), dim3(BLOCK_SIZE) >>> ( pos, s, tmp_tree_2, _start_index_ ); \
    cudaDeviceSynchronize(); \
    if( cuda_safe( cudaGetLastError() ) ) { \
        return 1; \
    } \
    second_find_leaves_params_Kernel<BLOCK_SIZE, _last_level_> <<< dim3(1), dim3(BLOCK_SIZE) >>> ( tmp_tree_2, second_reduce_size, tree_pointer ); \
    cudaDeviceSynchronize(); \
    if( cuda_safe( cudaGetLastError() ) ) { \
        return 1; \
    } \
})

#define FIND_NODES_PARAMS_IMPL( _last_lev_ ) ({ \
    log_d("find nodes params: last level = %u", _last_lev_); \
    size_t __last_level_count =  1 << (_last_lev_); \
    node_t *orig_tree = tree_pointer; \
    if( _last_lev_ <= BUILD_LEVEL ) { \
        FIND_NODES_PARAMS_SIMPLE( _last_lev_, 0 ); \
    } else { \
        for( size_t iii = 0; iii < __last_level_count; iii += BUILD_COUNT ) { \
            FIND_NODES_PARAMS_SIMPLE( BUILD_LEVEL, tree_pointer - orig_tree ); \
            log_d("new find nodes params: tree_pointer = %p level = %u, step %u ok", tree_pointer, _last_lev_, iii); \
            tree_pointer += BUILD_COUNT; \
        } \
    } \
    find_tree_params_Kernel<BLOCK_SIZE> <<< dim3(1), dim3(BLOCK_SIZE) >>> ( orig_tree, _last_lev_ ); \
    cudaDeviceSynchronize(); \
    if( cuda_safe( cudaGetLastError() ) ) { \
        return 1; \
    } \
    tree_pointer = orig_tree; \
})

#define FIND_NODES_PARAMS_CASE( _i_ ) \
    case ((_i_) + 1): \
        FIND_NODES_PARAMS_IMPL( _i_ ); \
        break;

#define FIND_NODES_PARAMS( _depth_ ) \
    switch(( _depth_ )) { \
        FIND_NODES_PARAMS_CASE( 1 ) \
        FIND_NODES_PARAMS_CASE( 2 ) \
        FIND_NODES_PARAMS_CASE( 3 ) \
        FIND_NODES_PARAMS_CASE( 4 ) \
        FIND_NODES_PARAMS_CASE( 5 ) \
        FIND_NODES_PARAMS_CASE( 6 ) \
        FIND_NODES_PARAMS_CASE( 7 ) \
        FIND_NODES_PARAMS_CASE( 8 ) \
        FIND_NODES_PARAMS_CASE( 9 ) \
        FIND_NODES_PARAMS_CASE( 10 ) \
        FIND_NODES_PARAMS_CASE( 11 ) \
        FIND_NODES_PARAMS_CASE( 12 ) \
        FIND_NODES_PARAMS_CASE( 13 ) \
        FIND_NODES_PARAMS_CASE( 14 ) \
        FIND_NODES_PARAMS_CASE( 15 ) \
        FIND_NODES_PARAMS_CASE( 16 ) \
        FIND_NODES_PARAMS_CASE( 17 ) \
        FIND_NODES_PARAMS_CASE( 18 ) \
        FIND_NODES_PARAMS_CASE( 19 ) \
        FIND_NODES_PARAMS_CASE( 20 ) \
        FIND_NODES_PARAMS_CASE( 21 ) \
        FIND_NODES_PARAMS_CASE( 22 ) \
        FIND_NODES_PARAMS_CASE( 23 ) \
        FIND_NODES_PARAMS_CASE( 24 ) \
        FIND_NODES_PARAMS_CASE( 25 ) \
        FIND_NODES_PARAMS_CASE( 26 ) \
        default: \
            log_e("tree_depth %u unsuported\n", _depth_); \
            return 1; \
    }

#define FIND_NEAR_AND_FAR_LEAVES_IMPL( _last_lev_ ) \
    find_near_and_far_leaves<BLOCK_SIZE, _last_lev_> <<< dim3(ceil(float(last_level_size) / float(BLOCK_SIZE))), dim3(BLOCK_SIZE) >>> ( tree + 3, tree_pointer, leaves_params, is_fast_lists, rc); \
    log_d("find_near_and_far_leaves tree + 3 = %p, tree_pointer = %p last_level_size = %zu", tree+3, tree_pointer, last_level_size); \
    cudaDeviceSynchronize(); \
    if( cuda_safe( cudaGetLastError() ) ) { \
        log_e("find_near_and_far_leaves"); \
        return 1; \
    }//if

#define FIND_NEAR_AND_FAR_LEAVES_CASE( _i_ ) \
    case ((_i_) + 1): \
        FIND_NEAR_AND_FAR_LEAVES_IMPL( _i_ ); \
        break;

#define FIND_NEAR_AND_FAR_LEAVES( _depth_ ) \
    switch(( _depth_ )) { \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 1 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 2 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 3 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 4 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 5 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 6 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 7 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 8 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 9 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 10 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 11 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 12 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 13 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 14 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 15 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 16 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 17 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 18 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 19 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 20 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 21 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 22 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 23 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 24 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 25 ) \
        FIND_NEAR_AND_FAR_LEAVES_CASE( 26 ) \
        default: \
            log_e("tree_depth %u unsuported\n", _depth_); \
            return 1; \
    }

static int build_tree( Vortex *pos, size_t s, float4 *leaves_params, uint8_t *is_fast_lists, float2 *rc ) {
    static node_t *tree = NULL;
    static size_t tree_size = 0;
    if ( !tree_size )
    {
        for( size_t i = 0; i < conf.tree_depth; ++i ) {
            tree_size += 1 << i;
        }
        log_i( "tree_size = %zu", tree_size );
        cuda_safe( cudaMalloc( (void**)&tree, tree_size * sizeof( node_t ) ) );
    }

    unsigned int second_reduce_size = 2 * BLOCK_SIZE;
    static node_t *tmp_tree = NULL;
    if( !tmp_tree )
        cuda_safe( cudaMalloc( (void**)&tmp_tree, second_reduce_size * sizeof( node_t ) ) );
    log_d("start tree_building");
    first_find_range_Kernel<BLOCK_SIZE> <<< dim3(second_reduce_size), dim3(BLOCK_SIZE) >>> ( pos, (unsigned int)s, tmp_tree );
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if
    second_find_range_Kernel<BLOCK_SIZE> <<< dim3(1), dim3(BLOCK_SIZE) >>> ( tmp_tree, second_reduce_size, tree );
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if
    node_t *tree_pointer = tree;
    static node_t *tmp_tree_2 = NULL;
    if( !tmp_tree_2 )
        cuda_safe( cudaMalloc( (void**)&tmp_tree_2, second_reduce_size * sizeof( node_t ) * (2 << conf.tree_depth) ) );

    for( size_t i = 1; i < conf.tree_depth; ++i ) {
        BUILD_TREE_STEP_SIMPLE( i );
    }

    FIND_NODES_PARAMS( conf.tree_depth );

    if( LEV_DEBUG < conf.log_level ) {
        node_t *host_tree = (node_t*)malloc(sizeof(node_t) * tree_size);
        cuda_safe( cudaMemcpy( (void*)host_tree, (void*)tree, tree_size * sizeof( node_t ), cudaMemcpyDeviceToHost ) );
        output_tree(host_tree, conf.tree_depth, current_step);
        free(host_tree);
    }

    size_t last_level_size = 1 << (conf.tree_depth - 1);

    cuda_safe( cudaMemset( (void*)leaves_params, 0, last_level_size * sizeof(float4) ) );
    cuda_safe( cudaMemset( (void*)is_fast_lists, 0, last_level_size * last_level_size ) );

    FIND_NEAR_AND_FAR_LEAVES( conf.tree_depth );

    if( LEV_DEBUG < conf.log_level ) {
        static float4 *leaves_params_host = NULL;
        static uint8_t *leaves_lists_host = NULL;
        if( !leaves_params_host ) {
            leaves_params_host = (float4*)malloc(last_level_size * sizeof(float4) );
            leaves_lists_host = (uint8_t*)malloc(last_level_size * last_level_size );
        }
        cuda_safe( cudaMemcpy( (void*)leaves_params_host, (void*)leaves_params,  last_level_size * sizeof(float4), cudaMemcpyDeviceToHost ) );
        cuda_safe( cudaMemcpy( (void*)leaves_lists_host, (void*)is_fast_lists,  last_level_size * last_level_size, cudaMemcpyDeviceToHost ) );
        for( size_t k = 0; k < last_level_size; ++k ) {
            printf("leave %zu\nA = %f\tB = %f\tC = %f\tD = %f\n", k, leaves_params_host[k].x, leaves_params_host[k].y, leaves_params_host[k].z, leaves_params_host[k].w);
            for( size_t m = 0; m < last_level_size; ++m ) {
                printf("%u", leaves_lists_host[k * last_level_size + m]);
            }
            printf("\n");
        }
        printf("\n");
    }

    log_d("finish tree_building");
    return 0;
}
#endif // NO_TREE

int Speed(Vortex *pos, TVctr *v_inf, size_t s, PVortex *v, TVars *d, TVars nu, tPanel *panels) {
    log_d("speed");
    extern int current_step;
    extern size_t n;
    cudaDeviceSynchronize();
    dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(s / BLOCK_SIZE);
    PVortex * VEL = NULL;
    PVortex * VELLL = NULL;
    Vortex *POS = NULL;

    log_d( "s = %zu", s );

#ifndef NO_TREE
    static float4 *leaves_params = NULL;
    static uint8_t *is_fast_lists = NULL;
    static float2 *rc = NULL;

    if( conf.tree_depth > 1 ) {
        if( !leaves_params ) {
            size_t last_level_size = 1 << (conf.tree_depth - 1);
            cuda_safe( cudaMalloc( (void**)&leaves_params, last_level_size * sizeof(float4) ) );
            cuda_safe( cudaMalloc( (void**)&is_fast_lists, last_level_size * last_level_size ) );
            cuda_safe( cudaMalloc( (void**)&rc, last_level_size * sizeof(float2) ) );
        }

        cudaEvent_t start_tree = 0, stop_tree = 0;
        start_timer( &start_tree, &stop_tree );
        if( build_tree( pos, n, leaves_params, is_fast_lists, rc ) ) {
            log_e( "error tree building" );
            return 1;
        }
        log_i("tree_time = %f", stop_timer( start_tree, stop_tree ));
    }
#endif // NO_TREE

    cudaEvent_t start_convective = 0, stop_convective = 0;
    start_timer( &start_convective, &stop_convective );
    shared_Kernel <<< blocks, threads >>> (pos, v_inf, s, v, d
#ifndef NO_TREE
            , leaves_params, is_fast_lists, conf.tree_depth - 1, rc
#endif // NO_TREE
            );
//	simple_Kernel <<< blocks, threads >>> (pos, v_inf, *n, v);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if
    log_i("convective_time = %f", stop_timer( start_convective, stop_convective ));

    if( LEV_DEBUG < conf.log_level ) {
        VEL = (PVortex*)malloc( sizeof(PVortex) * s );
        VELLL = (PVortex*)malloc( sizeof(PVortex) * s );
        POS = (Vortex*)malloc( sizeof(Vortex) * s );
        cuda_safe( cudaMemcpy( POS  , pos , s  * sizeof(Vortex) , cudaMemcpyDeviceToHost ) );
        cuda_safe( cudaMemcpy( VEL  , v , s  * sizeof(PVortex) , cudaMemcpyDeviceToHost ) );
        save_vel_to_file( POS, VEL, n, current_step, 0 );
        TVars *dd = (TVars*)malloc( sizeof(TVars) * s );
        cudaMemcpy(dd,d,s * sizeof(TVars),cudaMemcpyDeviceToHost);
        save_d(dd, s, current_step);
        free(dd);
    }

    diffusion_Kernel <<< blocks, threads >>> (pos, s, v, d, nu);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if

    if( LEV_DEBUG < conf.log_level ) {
        cuda_safe( cudaMemcpy( VELLL  , v , s  * sizeof(PVortex) , cudaMemcpyDeviceToHost ) );
        for (size_t sss = 0; sss < s; ++sss) {
            VEL[sss].v[0] = VELLL[sss].v[0] - VEL[sss].v[0];
            VEL[sss].v[1] = VELLL[sss].v[1] - VEL[sss].v[1];
        }
        save_vel_to_file(POS, VEL, n, current_step, 1);
    }

    diffusion_2_Kernel <<< blocks, threads >>> (pos, s, v, d, nu, panels);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if

    if( LEV_DEBUG < conf.log_level ) {
        cuda_safe( cudaMemcpy( VEL  , v , s  * sizeof(PVortex) , cudaMemcpyDeviceToHost ) );
        for (size_t sss = 0; sss < s; ++sss) {
            VELLL[sss].v[0] = VEL[sss].v[0] - VELLL[sss].v[0];
            VELLL[sss].v[1] = VEL[sss].v[1] - VELLL[sss].v[1];
        }
        save_vel_to_file(POS, VELLL, n, current_step, 2);
        save_vel_to_file(POS, VEL, n, current_step, 3);
        free( POS );
        free( VEL );
        free( VELLL );
    }
/*	
	TVars *dd=new TVars[size];
    cudaMemcpy(dd,d,size * sizeof(TVars),cudaMemcpyDeviceToHost);
    cout<<"d= "<<dd[0]<<endl;
    delete[]dd;
	TVars *ddt=new TVars;
	cuerr=cudaMemcpy (ddt  , den ,  sizeof(TVars) , cudaMemcpyDeviceToHost);
	cout<<"nu*y=  "<<(*ddt)<<endl;
	TVctr V_inf;
	cuerr=cudaMemcpy (VEL  , VDev , size  * sizeof(PVortex) , cudaMemcpyDeviceToHost);
	cuerr=cudaMemcpy (V_inf  , V_infDev , sizeof(TVctr) , cudaMemcpyDeviceToHost);
	cout<<"V=  "<<VEL[0].v[0]<<endl;
*/
//	cudaDeviceSynchronize();
//    cuerr=cudaMemcpy2D ( M3 , nb , cDev, pitch , nb , &n , cudaMemcpyDeviceToHost);
    return 0;
}

int Step(Vortex *pos, PVortex *V, size_t *n, size_t s, TVars *d_g, PVortex *F_p, TVars *M, tPanel *panels) {
    log_d("step");
    TVars *d_g_Dev = NULL;
    if( cuda_safe( cudaMalloc( (void**)&d_g_Dev, *n * sizeof(TVars) ) ) ) {
        return 1;
    }//if
    PVortex *F_p_dev = NULL;
    TVars *M_dev = NULL;
    if( cuda_safe( cudaMalloc( (void**)&F_p_dev, *n * sizeof(PVortex) ) ) ) {
        return 1;
    }//if
    if( cuda_safe( cudaMalloc((void**)&M_dev, *n * sizeof(TVars)) ) ) {
        return 1;
    }//if
//	TVars d_g_h;
//	cuerr=cudaMemcpy ( &d_g_h, d_g , sizeof(TVars) , cudaMemcpyDeviceToHost);
//  std::cout << "D_g_before = " << d_g_h << '\n';
    dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(s/BLOCK_SIZE);
    step_Kernel <<< blocks, threads >>> (pos, V, d_g_Dev, F_p_dev, M_dev, *n, panels);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if

//	cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
//	save_to_file_size((*n)+1);

    summ_Kernel <<< dim3(1),dim3(1) >>> (d_g_Dev, d_g, F_p_dev, F_p, M_dev, M, *n);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if
    cudaFree(d_g_Dev);
    cudaFree(F_p_dev);
    cudaFree(M_dev);
    if( LEV_DEBUG <= conf.log_level ) {
        TVars d_g_h = 0.0;
        cuda_safe( cudaMemcpy ( &d_g_h, d_g , sizeof(TVars) , cudaMemcpyDeviceToHost) );
        log_d( "d_g = %lf", d_g_h );
    }

    size_t *n_dev = NULL;
    if( cuda_safe( cudaMalloc( (void**)&n_dev ,  sizeof(size_t) ) ) ) {
        return 1;
    }//if
    if( cuda_safe( cudaMemcpy( n_dev, n, sizeof(size_t), cudaMemcpyHostToDevice ) ) ) {
        return 1;
    }//if
    log_d( "n_old =  %zu", *n );
    sort_Kernel <<< dim3(1), dim3(1) >>> (pos,n_dev);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }//if
    if( cuda_safe( cudaMemcpy( n,n_dev,sizeof(size_t), cudaMemcpyDeviceToHost ) ) ) {
        return 1;
    }//if
    log_d( "n_new =  %zu", *n );
    cudaFree(n_dev);
    log_d( "first collapse" );
    for (int cc = 0; cc < conf.n_col; ++cc) {
        int *Setx = NULL;
        int *Sety = NULL;
        int *COL = NULL;
        cuda_safe( cudaMalloc ( (void**)&Setx, *n * sizeof( int ) ) );
        cuda_safe( cudaMalloc ( (void**)&Sety, *n * sizeof( int ) ) );
        cuda_safe( cudaMalloc ( (void**)&COL, *n * sizeof( int ) ) );

        first_setka_Kernel <<< blocks, threads >>> (pos, *n, Setx, Sety, COL);
        cudaFree(Setx);
        cudaFree(Sety);
//		int *COLD;
//		COLD= new int [n];
//		cudaMemcpy(COLD, COL, n * sizeof(int), cudaMemcpyDeviceToHost);
/*
        int sss = 0;
		for(int gg = 0; gg < n; gg++) {
			if (COLD[gg] >= 0) sss += 1;
		}
		std::cout << cc << ' ' << sss << '\n';
		if (sss==0) cc=10;
		delete[] COLD;
*/
        cudaDeviceSynchronize();
        first_collapse_Kernel <<< dim3(1), dim3(1) >>> (pos, COL, *n);
        cudaFree(COL);
        cuda_safe( cudaMalloc( (void**)&n_dev ,  sizeof(size_t) ) );
        cuda_safe( cudaMemcpy( n_dev, n, sizeof(size_t), cudaMemcpyHostToDevice ) );
        cudaDeviceSynchronize();
        sort_Kernel <<< dim3(1), dim3(1) >>> (pos, n_dev);
        cudaDeviceSynchronize();
        cuda_safe( cudaMemcpy( n, n_dev, sizeof(size_t), cudaMemcpyDeviceToHost ) );
        log_d( "n_after collapse %d =  %zu", cc, *n );
        cudaFree(n_dev);
    }
    log_d( "second collapse" );
    for (int cc = 0; cc < conf.n_col; ++cc) {
        int *Setx = NULL;
        int *Sety = NULL;
        int *COL = NULL;
        cuda_safe( cudaMalloc( (void**)&Setx, *n * sizeof( int ) ) );
        cuda_safe( cudaMalloc( (void**)&Sety, *n * sizeof( int ) ) );
        cuda_safe( cudaMalloc( (void**)&COL, *n * sizeof( int ) ) );

        second_setka_Kernel <<< blocks, threads >>> (pos, *n, Setx, Sety, COL);
        cudaFree(Setx);
        cudaFree(Sety);
/*
        int *COLD;
        COLD= new int [n];
        cudaMemcpy(COLD, COL, n * sizeof(int), cudaMemcpyDeviceToHost);
        int sss = 0;
        for(int gg = 0; gg < n; gg++) {
            if (COLD[gg] >= 0) sss += 1;
        }
        std::cout << cc << ' ' << sss << '\n';
        if (sss==0) cc=10;
        delete[] COLD;
*/
        cudaDeviceSynchronize();
        second_collapse_Kernel <<< dim3(1), dim3(1) >>> (pos, COL, *n);
        cudaFree(COL);
        cuda_safe( cudaMalloc( (void**)&n_dev ,  sizeof(size_t) ) );
        cuda_safe( cudaMemcpy( n_dev, n, sizeof(size_t), cudaMemcpyHostToDevice ) );
        cudaDeviceSynchronize();
        sort_Kernel <<< dim3(1), dim3(1) >>> (pos, n_dev);
        cudaDeviceSynchronize();
        cuda_safe( cudaMemcpy( n, n_dev, sizeof(size_t), cudaMemcpyDeviceToHost ) );
        log_d( "n_after collapse %d =  %zu", cc, *n );
        cudaFree(n_dev);
    }
    return 0;
}

int init_device_conf_values() {
    srand((unsigned int)time(NULL));
    if( cuda_safe( cudaMemcpyToSymbol( dt, &conf.dt, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( quant, &conf.birth_quant, sizeof(size_t) ) ) ) return 1;
    TVars ve_s2 = conf.ve_size * conf.ve_size;
    if( cuda_safe( cudaMemcpyToSymbol( ve_size, &conf.ve_size, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( ve_size2, &ve_s2, sizeof(TVars) ) ) ) return 1;
    TVars r_col_diff2 = conf.r_col_diff_sign * conf.r_col_diff_sign * ve_s2;
    TVars r_col_same2 = conf.r_col_same_sign * conf.r_col_same_sign * ve_s2;
    if( cuda_safe( cudaMemcpyToSymbol( r_col_diff_sign2, &r_col_diff2, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( r_col_same_sign2, &r_col_same2, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( max_ve_g, &conf.max_ve_g, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( n_of_points, &conf.n_of_points, sizeof(size_t) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( x_max, &conf.x_max, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( x_min, &conf.x_min, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( y_max, &conf.y_max, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( y_min, &conf.y_min, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( h_col_x, &conf.h_col_x, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( h_col_y, &conf.h_col_y, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( rho, &conf.rho, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( rc_x, &conf.rc_x, sizeof(TVars) ) ) ) return 1;
    if( cuda_safe( cudaMemcpyToSymbol( rc_y, &conf.rc_y, sizeof(TVars) ) ) ) return 1;
#ifndef NO_TREE
    float h_theta = (float)conf.theta;
    if( cuda_safe( cudaMemcpyToSymbol( theta, &h_theta, sizeof(float) ) ) ) return 1;
#endif // NO_TREE
    return 0;
}

int velocity_control(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, unsigned n_contr) {
    log_d("velocity control");
    TVars rash = 0.0;
    size_t birth = 0;
    rash = (TVars)(n_contr) / BLOCK_SIZE;
    birth = (size_t)(BLOCK_SIZE * ceil(rash));
    dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(birth / BLOCK_SIZE);
    velocity_control_Kernel <<< blocks, threads >>> (pos, V_inf, n, Contr_points, V, n_contr);
    cudaDeviceSynchronize();
    if( cuda_safe( cudaGetLastError() ) ) {
        return 1;
    }
    return 0;
}
