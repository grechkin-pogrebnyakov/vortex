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

TVars   *matr_creation(tPanel *panels, size_t s) {
    TVars rash = 0.0;
    size_t birth = 0;
    rash = (TVars)(s) / BLOCK_SIZE;
    birth = (size_t)(BLOCK_SIZE * ceil(rash));
    TVars **M = NULL;
    TVars *MM = NULL;
    TVars **L = NULL;
    L = new TVars*[s + 1];
    if (L == NULL) {
        return NULL;
    }
    {
        size_t i;
        for(i = 0; i < s+1; i++) {
            L[i] = NULL;
            L[i]=new TVars[s+1];
            if (L[i] == NULL) break;
        }
        if (i != s+1) {
            while (i != 0) {
                delete[] L[i--];
            }
            delete[] L;
        }
    }
    TVctr a_0 = {0.0, 0.0};
    TVctr a_1 = {0.0, 0.0};
    TVctr b_0 = {0.0, 0.0};
    TVctr b_1 = {0.0, 0.0};
    TVctr d_i = {0.0, 0.0};
    TVctr d_j = {0.0, 0.0};
    TVctr p_1, p_2, s_1, s_2;
    TVars q1_1, q1_2, q1_3, q2_1, q2_2, q2_3, z_1, z_2, z_3;
    TVctr c_1, c_2, c_3;
    TVctr tau, v;
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
                q1_1 = atan( sp( d_i, p_1 ) / z_1 ) - atan( sp( d_i, p_2 ) / z_1 );
                q2_1 = 0.5 * log( sp( p_2, p_2 ) / sp( p_1, p_1 ) );
            }

            q1_2 = atan( sp( d_i, s_2 ) / z_2 ) - atan( sp( d_i, s_1 ) / z_2 );
            q1_3 = atan( sp( d_j, p_2 ) / z_3 ) - atan( sp( d_j, s_2 ) / z_3 );

            q2_2 = 0.5 * log( sp( s_1, s_1 ) / sp( s_2, s_2 ) );
            q2_3 = 0.5 * log( sp( p_2, p_2 ) / sp( s_2, s_2 ) );

            c_1[0] = sp( d_j, p_1 ) * d_i[0] + sp( d_i, s_1 ) * d_j[0] \
	 	   - sp( d_i, d_j ) * s_1[0];
            c_1[1] = sp( d_j, p_1 ) * d_i[1] + sp( d_i, s_1 ) * d_j[1] \
	 	   - sp( d_i, d_j ) * s_1[1];
            c_2[0] = c_1[0] + sp( d_j, d_j ) * d_i[0];
            c_2[1] = c_1[1] + sp( d_j, d_j ) * d_i[1];
            c_3[0] = sp( d_i, d_i ) * d_j[0];
            c_3[1] = sp( d_i, d_i ) * d_j[1];

            v[0] = 1.0 / ( 2 * M_PI * sqrt( sp( d_j, d_j ) ) * sp( d_i, d_i ) )\
                 * (q1_1 * c_1[0] + q1_2 * c_2[0] + q1_3 * c_3[0]\
                  + ( q2_1 * c_1[1] + q2_2 * c_2[1] + q2_3 * c_3[1] ) );
            v[1] = 1.0 / ( 2 * M_PI * sqrt( sp( d_j, d_j ) ) * sp( d_i, d_i ) )\
                 * (q1_1 * c_1[1] + q1_2 * c_2[1] + q1_3 * c_3[1]\
                  - ( q2_1 * c_1[0] + q2_2 * c_2[0] + q2_3 * c_3[0] ) );

            L[i][j] = sp( v, tau );
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
	MM = new TVars[(birth + 1) * (birth + 1)];
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

TVars   *load_matrix(size_t &p) {
	using namespace std;
	ifstream infile;
	infile.open("M.txt");
	infile >> p;
	p--;
	TVars **M = NULL;
	M = new TVars*[p + 1];
    if (M == NULL) {
        return NULL;
    }
	{
        size_t i;
        for(i = 0; i < p+1; i++) {
            M[i] = NULL;
            M[i]=new TVars[p+1];
            if (M[i] == NULL) break;
        }
        if (i != p+1) {
            while (i != 0) {
                delete[] M[i--];
            }
            delete[] M;
        }
    }
	for (size_t i = 0; i < p + 1; ++i) {
		for (size_t j = 0; j < p + 1; ++j) {
			infile >> M[i][j];
		}
	}
    TVars rash = 0.0;
    size_t birth = 0;
    rash = (TVars)(p) / BLOCK_SIZE;
    birth = (size_t)(BLOCK_SIZE * ceil(rash));
    
    TVars *MM = NULL;
    MM = new TVars[(birth + 1) * (birth + 1)];
    if (MM == NULL) return NULL;

    for (size_t i=0; i < p+1; i++) {
        for (size_t j=0; j < p+1 ; j++) {
            MM[(birth+1)*i+j]=M[i][j];
        }
        for (size_t j=(p+1); j<(birth+1);j++) {
            MM[(birth+1)*i+j]=0;
        }
    }
    for (size_t i=p+1; i < birth+1; i++) {
        for (size_t j=0; j<(birth+1);j++) {
            MM[(birth+1)*i+j]=0;
        }
    }
    clear_memory(M, p+1);
    return MM;
}

int     save_matr(TVars* M, size_t size, char *name = "D.txt") {
    using namespace std;
    if (M == NULL) return 1;
    ofstream outfile;
    outfile.open(name);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; j++) {
            outfile<<(TVars)(M[i*size+j])<<"    ";
        }// for j
        outfile<<endl;
    }// for i
    outfile.close();
    return 0;
}
int     save_matr(TVars** M, size_t size, char *name = "D.txt") {
    using namespace std;
    if (M == NULL) return 1;
    ofstream outfile;
    outfile.open(name);
	outfile << size << '\n';
    for (size_t i = 0; i < size; ++i) {
        if (M[i] == NULL) {
            outfile.close();
            return 1;
        }
        for (size_t j = 0; j < size; j++) {
            outfile<<(TVars)(M[i][j])<<"    ";
        }// for j
        outfile<<endl;
    }// for i
    outfile.close();
    return 0;
}
TVars   **inverse_matrix(TVars **M, size_t size) {
    int err = 0;
    size_t *POR = NULL;                                        // массив для учёта перестановки строк
    POR = new size_t[size];
    if (!POR) return NULL;
    size_t PR;                                                 // переменная для учёта перестановок строк
    for (size_t i = 0; i < size; i++) {
        POR[i]=i;
    }
    TVars b;
    TVars **M_inv = NULL;                                   // обратная матрица
    M_inv = new TVars*[size];
    {
        size_t i;
        for(i = 0; i < size; ++i) {
            M_inv[i] = NULL;
            M_inv[i]=new TVars[size];
            if (!M_inv[i]) break;
        }
        if (i != size) {
            while (i != 0) {
                delete[] M_inv[i--];
            }
            delete [] M_inv;
            delete[] POR;
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
    if (fabs(M[0][0])<DELT) {
        TVars mx=fabs(M[0][0]);
        size_t num=0;
        for (size_t i = 1; i < size; i++) {
            if (fabs(M[i][0])>mx) {
                mx=fabs(M[i][0]);
                num=i;
            }
        }//i
        if (num!=0) {
            err = move_line(M,size,0,num);
            if (err) {
                move_all_back(M, size, POR);
                delete[] POR;
                clear_memory(M_inv, size);
                return NULL;
            }
            err = move_line(M_inv,size,0,num);
            PR=POR[0];
            POR[0]=POR[num];
            POR[num]=PR;
            if (err) {
                move_all_back(M, size, POR);
                delete[] POR;
                clear_memory(M_inv, size);
                return NULL;
            }
        }
    }//if
    for (size_t k = 0; k < size-1; k++) {
        if (fabs(M[k][k])<DELT) {
            move_all_back(M, size, POR);
            delete[] POR;
            clear_memory(M_inv, size);
            return NULL;
        }//if
        TVars mx=fabs(M[k+1][k+1]);
        size_t line=k+1;
        for (size_t i = k+1; i < size; i++) {               // Выбор главного элемента
            if (fabs(M[i][k+1])>mx) {
                mx=fabs(M[i][k+1]);
                line=i;
            }//if
        }//i
        if (mx<DELT) {
            move_all_back(M, size, POR);
            delete[] POR;
            clear_memory(M_inv, size);
            return NULL;
        }
        err = move_line(M,size,k+1,line);
        if (err) {
            move_all_back(M, size, POR);
            delete[] POR;
            clear_memory(M_inv, size);
            return NULL;
        }
        err = move_line(M_inv,size,k+1,line);                      // перестановка строк
        PR=POR[k+1];
        POR[k+1]=POR[line];
        POR[line]=PR;
        if (err) {
            move_all_back(M, size, POR);
            delete[] POR;
            clear_memory(M_inv, size);
            return NULL;
        }
        for (size_t i = 0; i < size; i++) {
            if (i!=k) {
                TVars c=M[i][k]/M[k][k];
                for (size_t j = 0; j < size; j++) {
                    b=M[i][j]-c*(M[k][j]);                  // преобразование матрицы
                    M[i][j]=b;
                    b=M_inv[i][j]-c*(M_inv[k][j]);          // преобразование матрицы
                    M_inv[i][j]=b;
                }//j
            }//if
        }//i
    }//k
    if (fabs(M[size-1][size-1])<DELT) {
        move_all_back(M, size, POR);
        delete[] POR;
        clear_memory(M_inv, size);
        return NULL;
    }
    for (size_t i = 0; i < size-1; ++i) {
        TVars c=M[i][size-1]/M[size-1][size-1];
    //		   b=M[i][size-1]-c*(M[size-1][size-1]);        // преобразование матрицы
    //		   M[i][size-1]=b;
        for (size_t j = 0; j < size; j++) {
            b=M_inv[i][j]-c*(M_inv[size-1][j]);                 // преобразование матрицы
            M_inv[i][j]=b;
        }// j
    }// i
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            M_inv[i][j]=M_inv[i][j]/M[i][i];
        }// j
    }// i
    move_all_back(M, size, POR);
    delete[] POR;
    return M_inv;
}
int     move_line(TVars **M, size_t s, size_t st, size_t fin) {
    TVars *Ln = NULL;
    Ln=new TVars[s];
    if (! Ln) return 1;
    for (size_t i = 0; i < s ; i++) {
        Ln[i]=M[st][i];
    }
    for (size_t i = 0; i < s ; i++) {
        M[st][i]=M[fin][i];
    }
    for (size_t i = 0; i < s ; i++) {
        M[fin][i]=Ln[i];
    }
    delete[] Ln;
    return 0;
}
int     move_all_back(TVars **M, size_t size, size_t *mov) {
    if (M == NULL || mov == NULL) return 1;
    int err = 0;
    int cnt = 0;
    for (size_t i = 0; i < size; ++i) {
        if (mov[i] != i) {
            err = move_line(M, size, i, mov[i]);
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
void    clear_memory(TVars **M, size_t s) {
    if (M != NULL) {
        for (size_t i = 0; i < s; ++i) {
            if (M[i] != NULL) {
                delete[] M[i];
            }
        }
        delete[] M;
    }
}
int     incr_vort_quont(Vortex *&p_host, Vortex *&p_dev, PVortex *&v_host, PVortex *&v_dev, TVars *&d_dev, size_t &size)
{
    using namespace std;
    cudaError_t cuerr;
    if (p_host != NULL && p_dev != NULL && v_host != NULL && v_dev != NULL && d_dev != NULL)
    {
        Vortex *p_dev_new = NULL;
        size_t size_n = size + INCR_STEP;
        cuerr=cudaMalloc( (void**)&p_dev_new , size_n * sizeof(Vortex));
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            return 1;
        }
        cuerr = cudaMemcpy (p_dev_new, p_dev, size  * sizeof(Vortex), cudaMemcpyDeviceToDevice);
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            return 1;
        }
        size += INCR_STEP;
        delete[] p_host;
        p_host = new Vortex[size];
        delete[] v_host;
        v_host = new PVortex[size];
        cudaFree(p_dev);
        cudaFree(d_dev);
        cudaFree(v_dev);
        cuerr=cudaMalloc( (void**)&d_dev, size * sizeof(TVars));
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            return 1;
        }
        cuerr=cudaMalloc( (void**)&v_dev, size  * sizeof(PVortex));
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            return 1;
        }
        p_dev = p_dev_new;
        cudaDeviceSynchronize();
    }
    else {
        size = INCR_STEP;
        p_host=new Vortex[size];
        v_host=new PVortex[size];
        cuerr=cudaMalloc((void**)&p_dev , size * sizeof(Vortex));
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            return 1;
        }
        cuerr=cudaMalloc((void**)&d_dev , size * sizeof(TVars));
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            return 1;
        }
        cuerr=cudaMalloc((void**)&v_dev , size  * sizeof(PVortex));
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            return 1;
        }
    }
    srand((unsigned int)time(NULL));
    float *rnd_dev = NULL, *rnd_host = NULL;
    rnd_host = new float[INCR_STEP];
    for (int i = 0; i < INCR_STEP; ++i) {
        rnd_host[i] = (float)rand();
    }
    cuerr = cudaMalloc((void**)&rnd_dev, INCR_STEP * sizeof(float));
    if (cuerr != cudaSuccess) {
        cout << cudaGetErrorString(cuerr) << '\n';
        return 1;
    }
    cuerr = cudaMemcpy(rnd_dev, rnd_host, INCR_STEP * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        cout << cudaGetErrorString(cuerr) << '\n';
        return 1;
    }
    dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(INCR_STEP/BLOCK_SIZE);
    // generate random numbers
    zero_Kernel <<< blocks, threads >>> (rnd_dev, p_dev, (size-INCR_STEP) );
    cudaDeviceSynchronize();
    //	cuerr=cudaMemcpy ( p_host , p_dev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
    //	save_to_file_size(1);
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        cout << cudaGetErrorString(cuerr) << '\n';
        return 1;
    }
    return 0;
}
int     vort_creation(Vortex *pos, TVctr *V_infDev, size_t n_of_birth, size_t n_of_birth_BLOCK_S,
                     size_t n, TVars * M_Dev, TVars *d_g, tPanel *panels) {
    using namespace std;
    cudaError_t cuerr = cudaSuccess;
    cudaEvent_t start, stop;
start_timer(start, stop);
    TVars *R_p = NULL;
    cuerr=cudaMalloc((void**)&R_p, (n_of_birth_BLOCK_S) * sizeof(TVars));
    if (cuerr != cudaSuccess) {
        cout << cudaGetErrorString(cuerr) << '\n';
        return 1;
    }
    extern int current_step;

	dim3 threads1 = dim3(BLOCK_SIZE);
    dim3 blocks1  = dim3(n_of_birth_BLOCK_S/BLOCK_SIZE);
    Right_part_Kernel <<< blocks1, threads1 >>> (pos, V_infDev, n, n_of_birth_BLOCK_S, R_p, panels);
	cudaDeviceSynchronize();
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        cout << cudaGetErrorString(cuerr) << '\n';
        return 1;
    }


	birth_Kernel<<< blocks1, threads1 >>>(pos, n, n_of_birth, n_of_birth_BLOCK_S, M_Dev, d_g, R_p, panels);
	cudaDeviceSynchronize();
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        cout << cudaGetErrorString(cuerr) << '\n';
        return 1;
    }
    cudaFree(R_p);
    return 0;
}
void start_timer(cudaEvent_t &start, cudaEvent_t &stop) {
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	cudaEventSynchronize(start);
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
int Speed(Vortex *pos, TVctr *v_inf, size_t s, PVortex *v, TVars *d, TVars nu, tPanel *panels) {
    //extern int current_step;
    //extern size_t n;
    cudaError_t cuerr = cudaSuccess;
	cudaDeviceSynchronize();
	dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(s/BLOCK_SIZE);
    //PVortex * VEL = new PVortex[s];
    //PVortex * VELLL = new PVortex[s];
	shared_Kernel <<< blocks, threads >>> (pos, v_inf, s, v, d);
//	simple_Kernel <<< blocks, threads >>> (pos, v_inf, *n, v);
    cudaDeviceSynchronize();
    //Vortex *POS = new Vortex[s];
    //cuerr=cudaMemcpy (POS  , pos , s  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
    //cuerr=cudaMemcpy (VEL  , v , s  * sizeof(PVortex) , cudaMemcpyDeviceToHost);
    //save_vel_to_file(POS, VEL, n, current_step, 0);
    cuerr=cudaGetLastError(); 
	if (cuerr != cudaSuccess) {               
		std::cout <<cudaGetErrorString(cuerr);
		return 1;
	}//if

//	TVars* dd=new TVars[s];
//    cudaMemcpy(dd,d,s * sizeof(TVars),cudaMemcpyDeviceToHost);
//    save_d(dd, s, current_step);
//    delete[]dd;

	diffusion_Kernel <<< blocks, threads >>> (pos, s, v, d, nu);
//	cuerr=cudaMemcpy (POS  , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
//	save_to_file(j);
	cudaDeviceSynchronize();
//	cuerr=cudaMemcpy (VELLL  , v , s  * sizeof(PVortex) , cudaMemcpyDeviceToHost);
/*
    for (size_t sss = 0; sss < s; ++sss) {
        VEL[sss].v[0] = VELLL[sss].v[0] - VEL[sss].v[0];
        VEL[sss].v[1] = VELLL[sss].v[1] - VEL[sss].v[1];
    }
	save_vel_to_file(POS, VEL, n, current_step, 1);
*/
    cuerr=cudaGetLastError(); 
	if (cuerr != cudaSuccess) {               
		std::cout <<cudaGetErrorString(cuerr);
		return 1;
	}//if
	diffusion_2_Kernel <<< blocks, threads >>> (pos, s, v, d, nu, panels);
//	cuerr=cudaMemcpy (VEL  , VDev , size  * sizeof(PVortex) , cudaMemcpyDeviceToHost);
//	stf(j,1);
	cudaDeviceSynchronize();
/*
    cuerr=cudaMemcpy (VEL  , v , s  * sizeof(PVortex) , cudaMemcpyDeviceToHost);
    for (size_t sss = 0; sss < s; ++sss) {
        VELLL[sss].v[0] = VEL[sss].v[0] - VELLL[sss].v[0];
        VELLL[sss].v[1] = VEL[sss].v[1] - VELLL[sss].v[1];
    }
    save_vel_to_file(POS, VELLL, n, current_step, 2);
    save_vel_to_file(POS, VEL, n, current_step, 3);
*/
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
   	cuerr=cudaGetLastError();
	if (cuerr != cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr);
		return 1;
	}//if
	return 0;
}

void save_vel_to_file(Vortex *POS, PVortex *VEL, size_t size, int _step, int stage) {
    using namespace std;
    char *fname1;
    fname1 = "velocities/Vel";
    char *fname2;
    fname2 = ".txt";
    char *fzero;
    fzero = "0";
    char fstep[8];
    char fname[20];
    fname[0] = '\0';
    char stage_str[5];
    sprintf(stage_str, "%d", stage);
    sprintf(fstep,"%d", _step);
    strcat(fname,fname1);
    strcat(fname, stage_str);
    if (_step<10) strcat(fname,fzero);
    if (_step<100) strcat(fname,fzero);
    if (_step<1000) strcat(fname,fzero);
    if (_step<10000) strcat(fname,fzero);
    //	if (_step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname,fname2);
    ofstream outfile;
    outfile.open(fname);
    // ����������� ����� ������ � ������
    outfile << (size) << endl;
    for (size_t i = 0; i < (size); ++i) {
        outfile<<(int)(i)<<" "<<(TVars)(POS[i].r[0])<<" "<<(TVars)(POS[i].r[1])<<" "<<(TVars)(VEL[i].v[0])<<" "<<(TVars)(VEL[i].v[1])<<endl;
        //      outfile<<(TVars)(d[i])<<" "<<(TVars)(POS[i].r[0])<<" "<<(TVars)(POS[i].r[1])<<" "<<(TVars)(POS[i].g)<<endl;     
    }//for i
    outfile.close();
} //save_to_file

void save_d(TVars *d, size_t size, int _step) {
    using namespace std;
    char *fname1;
    fname1 = "ddd/d";
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
    //	if (_step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname,fname2);
    ofstream outfile;
    outfile.open(fname);
    // ����������� ����� ������ � ������
    outfile << (size) << endl;
    for (size_t i = 0; i < (size); ++i) {
        outfile<<(int)(i)<<" "<<d[i]<<endl;
        //      outfile<<(TVars)(d[i])<<" "<<(TVars)(POS[i].r[0])<<" "<<(TVars)(POS[i].r[1])<<" "<<(TVars)(POS[i].g)<<endl;     
    }//for i
    outfile.close();
} //save_to_file

int Step(Vortex *pos, PVortex *V, size_t &n, size_t s, TVars *d_g, PVortex *F_p, TVars *M, tPanel *panels) {
	cudaError_t cuerr = cudaSuccess;
	TVars *d_g_Dev = NULL;
	cuerr=cudaMalloc((void**)&d_g_Dev, n * sizeof(TVars));
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
    PVortex *F_p_dev = NULL;
    TVars *M_dev = NULL;
    cuerr=cudaMalloc((void**)&F_p_dev, n * sizeof(PVortex));
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
    cuerr=cudaMalloc((void**)&M_dev, n * sizeof(TVars));
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
//	TVars d_g_h;
//	cuerr=cudaMemcpy ( &d_g_h, d_g , sizeof(TVars) , cudaMemcpyDeviceToHost);
//  std::cout << "D_g_before = " << d_g_h << '\n';
    dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(s/BLOCK_SIZE);
	step_Kernel <<< blocks, threads >>> (pos, V, d_g_Dev, F_p_dev, M_dev, n, panels);
    cudaDeviceSynchronize();
    cuerr=cudaGetLastError();
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if

//	cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
//	save_to_file_size((*n)+1);

	summ_Kernel <<< dim3(1),dim3(1) >>> (d_g_Dev, d_g, F_p_dev, F_p, M_dev, M, n);
	cudaDeviceSynchronize();
    cuerr=cudaGetLastError();
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
	cudaFree(d_g_Dev);
    cudaFree(F_p_dev);
    cudaFree(M_dev);
	TVars d_g_h = 0.0;
	cuerr=cudaMemcpy ( &d_g_h, d_g , sizeof(TVars) , cudaMemcpyDeviceToHost);
//	std::cout << "d_g = " << d_g_h << '\n';

	size_t *n_dev = NULL;
	cuerr = cudaMalloc( (void**)&n_dev ,  sizeof(size_t)); 
	if (cuerr!= cudaSuccess) {               
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
	cuerr = cudaMemcpy(n_dev, &n, sizeof(size_t), cudaMemcpyHostToDevice);
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
	sort_Kernel <<< dim3(1), dim3(1) >>> (pos,n_dev);
    cudaDeviceSynchronize();
    cuerr=cudaGetLastError();
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
//	std::cout << "n_old =  " << n;
	cuerr = cudaMemcpy(&n,n_dev,sizeof(size_t), cudaMemcpyDeviceToHost);
	if (cuerr!= cudaSuccess) {
		std::cout <<cudaGetErrorString(cuerr) << '\n';
		return 1;
	}//if
//        std::cout << "   n_new =  " << n <<'\n';
	cudaFree(n_dev);
//    std::cout << "first collapse\n";
	for (int cc = 0; cc < NCOL; ++cc) {
		int *Setx = NULL;
		int *Sety = NULL;
		int *COL = NULL;
		cuerr=cudaMalloc (&Setx, n * sizeof( int ));
		cuerr=cudaMalloc (&Sety, n * sizeof( int ));
		cuerr=cudaMalloc (&COL, n * sizeof( int ));
		
		first_setka_Kernel <<< blocks, threads >>> (pos, n, Setx, Sety, COL);
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
		first_collapse_Kernel <<< dim3(1), dim3(1) >>> (pos, COL, n);
		cudaFree(COL);
		cudaMalloc( (void**)&n_dev ,  sizeof(size_t));
		cudaMemcpy(n_dev, &n, sizeof(size_t), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		sort_Kernel <<< dim3(1), dim3(1) >>> (pos, n_dev);
		cudaDeviceSynchronize();
		cudaMemcpy(&n, n_dev, sizeof(size_t), cudaMemcpyDeviceToHost);
		cudaFree(n_dev);
	}
//    std::cout << "second collapse\n";
    for (int cc = 0; cc < NCOL; ++cc) {
        int *Setx = NULL;
        int *Sety = NULL;
        int *COL = NULL;
        cuerr=cudaMalloc (&Setx, n * sizeof( int ));
        cuerr=cudaMalloc (&Sety, n * sizeof( int ));
        cuerr=cudaMalloc (&COL, n * sizeof( int ));

        second_setka_Kernel <<< blocks, threads >>> (pos, n, Setx, Sety, COL);
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
        second_collapse_Kernel <<< dim3(1), dim3(1) >>> (pos, COL, n);
        cudaFree(COL);
        cudaMalloc( (void**)&n_dev ,  sizeof(size_t));
        cudaMemcpy(n_dev, &n, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        sort_Kernel <<< dim3(1), dim3(1) >>> (pos, n_dev);
        cudaDeviceSynchronize();
        cudaMemcpy(&n, n_dev, sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaFree(n_dev);
    }
	return 0;
}



int velocity_control(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, int *n_v) {
    cudaError_t cuerr;
    size_t nummm = 500;
    TVars rash = 0.0;
    size_t birth = 0;
    rash = (TVars)(nummm) / BLOCK_SIZE;
    birth = (size_t)(BLOCK_SIZE * ceil(rash));
    dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3(birth / BLOCK_SIZE);
    velocity_control_Kernel <<< blocks, threads >>> (pos, V_inf, n, Contr_points, V, n_v);
    cudaDeviceSynchronize();    
    cuerr = cudaGetLastError();    
    if (cuerr != cudaSuccess) {        
        std::cout << cudaGetErrorString(cuerr) << '\n';
        return 1;
    }
    return 0;
}
