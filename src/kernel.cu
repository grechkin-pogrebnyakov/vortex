/*
 ============================================================================
 Name        : kernel.cu
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Mar. 4, 2015
 Copyright   : All rights reserved
 Description : kernel file of vortex project
 ============================================================================
 */

#include "kernel.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"

__global__ void zero_Kernel(float *randoms, Vortex *pos, int s ) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    TVars a = 0.0;
	pos[s+ind].r[0]=(2.0e+5)*randoms[ind]+2.0e+5;
	pos[s+ind].r[1]=(2.0e+5)*randoms[ind]+2.0e+5;
	pos[s+ind].g = a;
}

__global__ void Right_part_Kernel(Vortex *pos, TVctr *V_inf, size_t n_vort, size_t n_birth_BLOCK_S, TVars *R_p, tPanel *panels) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    R_p[i] = 0.0;
    float y0 = 0.0f, y1 = 0.0f;
//	TVars dist2;
    float mnog = 0.0f;
    // координаты и нормали расчётной точки
    float a_left0 = 0.0f, a_left1 = 0.0f;
    float a_right0 = 0.0f, a_right1 = 0.0f;
    float d0 = 0.0f, d1 = 0.0f;
    float kd0 = 0.0f, kd1 = 0.0f;
    float tau0 = 0.0f, tau1 = 0.0f;
    float s_00 = 0.0f, s_01 = 0.0f;
    float s_10 = 0.0f, s_11 = 0.0f;
    float z = 0.0f;
    float alpha = 0.0f, beta = 0.0f;

    // координаты воздействующей точки
    __shared__ float b_sh_0 [BLOCK_SIZE];
    __shared__ float b_sh_1 [BLOCK_SIZE];
    // интенсивность воздействующей точки
    __shared__ float g [BLOCK_SIZE];

    a_left0 = (float)panels[i].left[0];
    a_left1 = (float)panels[i].left[1];
    a_right0 = (float)panels[i].right[0];
    a_right1 = (float)panels[i].right[1];
    tau0 = (float)panels[i].tang[0];
    tau1 = (float)panels[i].tang[1];

    d0 = a_right0 - a_left0;
    d1 = a_right1 - a_left1;

    kd0 = -d1;
    kd1 = d0;

    for (int f = 0 ; f < n_vort ; f += BLOCK_SIZE) {
        b_sh_0[threadIdx.x] = (float)pos[threadIdx.x+f].r[0];
        b_sh_1[threadIdx.x] = (float)pos[threadIdx.x+f].r[1];
        g[threadIdx.x] = (float)pos[threadIdx.x+f].g;
        __syncthreads();
        for (int j = 0 ; j < BLOCK_SIZE ; ++j) {
            s_00 = a_left0 - b_sh_0[j];
            s_01 = a_left1 - b_sh_1[j];

            s_10 = a_right0 - b_sh_0[j];
            s_11 = a_right1 - b_sh_1[j];

            z = d0 * s_01 - d1 * s_00;

            alpha = atanf( spf( s_00, s_01, d0, d1 ) / z )\
                   -atanf( spf( s_10, s_11, d0, d1 ) / z );

            beta = 0.5 * logf( spf( s_10, s_11, s_10, s_11 )\
                             /spf( s_00, s_01, s_00, s_01 ) );

            y0 += g[j] * ( alpha * d0 + beta * kd0 );
            y1 += g[j] * ( alpha * d1 + beta * kd1 );
        }//j
        __syncthreads();
    }//f

    mnog = 1 / (2 * M_PI * spf( d0, d1, d0, d1 ) );

    R_p[i] = -(TVars)( ( mnog * y0 + (float)(*V_inf)[0] ) * tau0 + ( mnog * y1 + (float)(*V_inf)[1] ) * tau1 );
//    R_p[i] = -( (  (*V_inf)[0] ) * tau0 + ( (*V_inf)[1] ) * tau1 );
//	V[i].v[k] =  (*V_inf)[k];
    __syncthreads(); 
}

__global__ void birth_Kernel(Vortex *pos, size_t n_vort, size_t n_birth, size_t n_birth_BLOCK_S, TVars * M, TVars *d_g, TVars *R_p, tPanel *panel) {
	int i= blockIdx.x * blockDim.x + threadIdx.x;
	int i_next = panel[i].n_of_rpanel;
	register TVars g;
	register TVars g_next;
	for (size_t j = 0; j < n_birth; ++j) {
//		pos_N.g += M[(pp+1)*i+j]*R_p[j]; 
            g += M[(n_birth_BLOCK_S + 1) * i + j] * R_p[j]; 
            g_next += M[(n_birth_BLOCK_S + 1) * i_next + j] * R_p[j]; 
	}
    	g += M[(n_birth_BLOCK_S + 1) * i + n_birth] * (*d_g);
    	g_next += M[(n_birth_BLOCK_S + 1) * i_next + n_birth] * (*d_g);
	if (i < n_birth)
	{
//		pos[i+n_vort].r[0] = pos_N.r[0];
//		pos[i+n_vort].r[1] = pos_N.r[1];

		//pos[i+n_vort].r[0] = R_birth_x(panel, i);
        pos[i + n_vort].r[0] = panel[i].right[0] + panel[i].norm[0] * 1e-7;
		//pos[i+n_vort].r[1] = R_birth_y(panel, i);
        pos[i + n_vort].r[1] = panel[i].right[1] + panel[i].norm[1] * 1e-7;
            pos[i + n_vort].g = 0.5 * ( g * Panel_length( panel, i )\
                                 + g_next * Panel_length( panel, i_next ) );
	}
}
__global__ void shared_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *V, TVars *d) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    float y0 = 0.0f, y1 = 0.0f;
//	TVars dist2;
    float mnog = 0.0f;
    float d_1 = 0.0f;      //  четыре наименьших
    float d_2 = 0.0f;      //  расстояния от этой
    float d_3 = 0.0f;      //  точки до остальных.
    float d_0 = 0.0f;
    float dist2 = 0.0f;
    float dst = 0.0f;
    // координаты расчётной точки
    float a0 = 0.0f, a1 = 0.0f;
    // координаты воздействующей точки
    __shared__ float b_sh_0 [BLOCK_SIZE];
    __shared__ float b_sh_1 [BLOCK_SIZE];
    // интенсивность воздействующей точки
    __shared__ float g [BLOCK_SIZE];
    a0 = (float)pos[i].r[0];
    a1 = (float)pos[i].r[1];
    d_1 = 1e+5f;
    d_2 = 1e+5f;
    d_3 = 1e+5f;
    d_0 = 1e+5f;
    for (int f = 0 ; f < n ; f += BLOCK_SIZE) {
        b_sh_0[threadIdx.x] = (float)pos[threadIdx.x+f].r[0];
        b_sh_1[threadIdx.x] = (float)pos[threadIdx.x+f].r[1];
        g[threadIdx.x] = (float)pos[threadIdx.x+f].g;
        __syncthreads();
        for (int j = 0 ; j < BLOCK_SIZE ; ++j) {		
            //dist2= Ro2(a0, a1, b_sh_0[j], b_sh_1[j]);
            dist2 = (a0 - b_sh_0[j]) * (a0 - b_sh_0[j]) + (a1 - b_sh_1[j]) * (a1 - b_sh_1[j]);
            if (d_3 > dist2) {
                    d_3 = dist2;
                    dst = fminf(d_3, d_2);
                    d_3 = fmaxf(d_3, d_2);
                    d_2 = dst;
                    dst = fminf(d_1, d_2);
                    d_2 = fmaxf(d_1, d_2);
                    d_1 = dst;
                    dst = fminf(d_1, d_0);
                    d_1 = fmaxf(d_1, d_0);
                    d_0 = dst;
            }
	//		if (dist2 < EPS2) dist2=EPS2;
            dist2 = fmaxf(dist2, EPS2);
            mnog = g[j] / dist2;
            y1 +=  mnog * (a0 - b_sh_0[j]);	
            y0 += -mnog * (a1 - b_sh_1[j]);
        }//j
        __syncthreads();
    }//f
//    d[i] = sqrt(d_1 + d_2 + d_3) / 3;
    d[i] = (sqrtf(d_1) + sqrtf(d_2) + sqrtf(d_3)) / 3;
//    d[i] = max(d[i], 4.0 * EPS / 3.0);
    V[i].v[0] = (TVars)( y0 / (2 * M_PI) + (*V_inf)[0] );
    V[i].v[1] = (TVars)( y1 / (2 * M_PI) + (*V_inf)[1] );
//	V[i].v[k] =  (*V_inf)[k];
    __syncthreads(); 
}

__global__ void diffusion_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu_d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float nu = (float) nu_d;
    float II_2_0 = 0.0f, II_2_1 = 0.0f;
    float e = 0.0f;
    // знаменатель
    float II_1 = 0.0f;
    float dist = 0.0;
    float mnog = 0.0;
    // координаты расчётной точки
    float a0 = 0.0, a1 = 0.0;
    // координаты воздействующих точек
    __shared__ float b_sh_0[BLOCK_SIZE];
    __shared__ float b_sh_1[BLOCK_SIZE];
    // интенсивности воздействующих точек
    __shared__ TVars g[BLOCK_SIZE];
    float dd = 0.0;
    a0 = (float)pos[i].r[0];
    a1 = (float)pos[i].r[1];
    dd = (float)d[i];
    __syncthreads();
    for (int f = 0; f < n; f += BLOCK_SIZE) {
        b_sh_0[threadIdx.x] = (float)pos[threadIdx.x + f].r[0];
        b_sh_1[threadIdx.x] = (float)pos[threadIdx.x + f].r[1];
        g[threadIdx.x] = (float)pos[threadIdx.x + f].g;
        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            dist= sqrtf(Ro2f(a0, a1, b_sh_0[j], b_sh_1[j]));
            if(dist > 0.001 * EPS) {
                mnog=g[j] / dist;
                e = expf(-(dist) / (dd));
                II_2_0 += -mnog * (a0-b_sh_0[j]) * e;
                II_2_1 += -mnog * (a1-b_sh_1[j]) * e;
                II_1 += g[j] * e;
//                II_1 += mnog;
            }
        }//j
        __syncthreads();
    }//f
    if (fabs(II_1) > 1e-5) {
        V[i].v[0] += (TVars)( -nu * II_2_0 / (II_1 * dd) );
        V[i].v[1] += (TVars)( -nu * II_2_1 / (II_1 * dd) );
//            V[i].v[k] = II_2[k];
    }
}

__global__ void diffusion_2_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu_d, tPanel *panels) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    float nu = (float) nu_d;
    if (i < n) {
    float Ra_0 = 0.0f, Ra_1 = 0.0f;
    float Rb_0 = 0.0f, Rb_1 = 0.0f;
    float Rk_0 = 0.0f, Rk_1 = 0.0f;
    float Norm_0 = 0.0f, Norm_1 = 0.0f;
    float dL = 0.0f;
    float RES_0 = 0.0f;
    float RES_3_0 = 0.0f, RES_3_1 = 0.0f;
    // координаты расчётной точки
    float a0 = 0.0f, a1 = 0.0f;
    float dd = 0.0f;
//    F_vis[i].v[0] = 0.0;
//    F_vis[i].v[1] = 0.0;
    a0 = (float)pos[i].r[0];
    a1 = (float)pos[i].r[1];
    dd = (float)d[i];
    float II_0 = 2 * M_PI * dd * dd;
    float II_3_0 = 0.0f, II_3_1 = 0.0f;
    //	TVars denomenator = 2 * M_PI * dd; // знаменатель
    for (int f = 0; f < QUANT; ++f) {

        Ra_0 = R_left_xf(panels, f);
        Ra_1 = R_left_yf(panels, f);
        Rb_0 = R_right_xf(panels, f);
        Rb_1 = R_right_yf(panels, f);
        Rk_0 = R_birth_xf(panels, f);
        Rk_1 = R_birth_yf(panels, f);
        //dL = sqrt((Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1]));
        dL = (float)panels[f].length;
        float r = sqrtf(Ro2f(a0, a1, Rk_0, Rk_1));
      //  II_0 = 1; II_3[0]=r; II_3[1]=1;
        if ((r < 5 * dL) && (r > 0.1 * dL)) {
            Norm_0 = -N_contr_xf(panels, f);
            Norm_1 = -N_contr_yf(panels, f);
            I_0_I_3f(Ra_0, Ra_1, Rb_0, Rb_1, Norm_0, Norm_1, a0, a1, dL, dd, N_OF_POINTS, RES_0, RES_3_0, RES_3_1);
            II_0 += (-dd) * RES_0;
            II_3_0 -= RES_3_0;
            II_3_1 -= RES_3_1;
        //  II_0 = 1;  II_3[0] = r ; II_3[1]=2;
        } else if (r <= 0.1 * dL) {
         //   printf("...");
            Norm_0 = N_contr_xf(panels, f);
            Norm_1 = N_contr_yf(panels, f);
            II_0 = M_PI * dd * dd;
            II_3_0 = 2 * Norm_0 * dd * (1 - expf(-dL / (2 * dd)));
            II_3_1 = 2 * Norm_1 * dd * (1 - expf(-dL / (2 * dd)));
          //  II_0=1;II_3[0]=r;II_3[1]=0;
           // f = QUANT + 5;
            break;
        }

    }//f
    V[i].v[0] += (float)( nu * II_3_0 / II_0 );
    V[i].v[1] += (float)( nu * II_3_1 / II_0 );
    }
}

__global__ void step_Kernel(Vortex *pos, PVortex *V, TVars *d_g_Dev, PVortex *F_p, TVars *M, size_t n, tPanel *panels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
        TVars rc[2] = RC;
		d_g_Dev[i] = 0.0;
        TVars d_g = 0.0;
        F_p[i].v[0] = 0.0;
        F_p[i].v[1] = 0.0;
        M[i] = 0.0;
        if (i >= n - QUANT) {
            F_p[i].v[0] = pos[i].g * (-pos[i].r[1]);
            F_p[i].v[1] = pos[i].g * ( pos[i].r[0]);
            M[i] = pos[i].g * Ro2(pos[i].r, rc);
        }
	    //__syncthreads;

		TVars r_new_0 = pos[i].r[0] + V[i].v[0] * dt;
        TVars r_new_1 = pos[i].r[1] + V[i].v[1] * dt;
//	    TVctr Zero  = {0, 0};
		int hitpan = 0;

	    if ( (pos[i].g != 0) && (hitting(panels, r_new_0, r_new_1, pos[i].r, &hitpan))) {
            F_p[i].v[0] -= pos[i].g * (-panels[hitpan].contr[1]);
            F_p[i].v[1] -= pos[i].g * ( panels[hitpan].contr[0]);
            M[i] -= pos[i].g * Ro2(panels[hitpan].contr, rc);
		    r_new_0 =  2e+5;
		    r_new_1 =  2e+5;
		    d_g = pos[i].g;
//printf( "d_g[%d] =  %lf \n", i, d_g];
//		    d_g_Dev[i] = pos[i].g;
		    pos[i].g = 0;
		}

		pos[i].r[0] = r_new_0;
		pos[i].r[1] = r_new_1;

	    if ((pos[i].g != 0) && ((pos[i].r[0] > COUNT_AREA) || (fabs(pos[i].r[1]) > 10))) {
		    pos[i].r[0]= -2.0e+5; 
		    pos[i].r[1]= -2.0e+5; 
		    pos[i].g=0;
	    }
        //__syncthreads;
        d_g_Dev[i] = d_g;
    }
	//__syncthreads;
}
__global__ void summ_Kernel(TVars *d_g_Dev, TVars *d_g, PVortex *F_p_dev, PVortex *F_p, TVars *M_dev, TVars *M, size_t n) {
       *d_g = 0.0;
		for (int k = 0; k < n; ++k) {
		(*d_g) += d_g_Dev[k];
        (*F_p).v[0] += F_p_dev[k].v[0];
        (*F_p).v[1] += F_p_dev[k].v[1];
        (*M) += M_dev[k];
        }
//printf("d_g =  %lf\n", *d_g);
//        printf("%.6f\n",*d_g);
        (*F_p).v[0] *= RHO / dt;
        (*F_p).v[1] *= RHO / dt;
        (*M) *= RHO / (2 * dt);
}
__global__ void sort_Kernel(Vortex *pos, size_t *s) {
    TVars r0 = 0.0, r1 = 0.0;
	size_t n = 0;
    n = (*s);
	for (size_t i = 0 ; i < n ; ++i) {
		if (fabs(pos[i].g) < DELT) {
			r0=pos[i].r[0];
			r1=pos[i].r[1];
			pos[i].g=pos[n-1].g;
			pos[i].r[0]=pos[n-1].r[0];
			pos[i].r[1]=pos[n-1].r[1];
			pos[n-1].g=0;
			pos[n-1].r[0]=r0;
			pos[n-1].r[1]=r1;
			n--;
			i--;
		}
    }
	(*s)=n;
}
__global__ void second_setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<n) {
        TVars r0 = pos[i].r[0];
        TVars r1 = pos[i].r[1];
        TVars g = pos[i].g;
        int Setx_i = floor((r0+2)/HX);
        int Sety_i = floor((r1+10)/HY);
		Setx[i] = Setx_i;
		Sety[i] = Sety_i;
		COL[i] = -1;

		__syncthreads();

		//	for (int j = (i+1); j < n; j++ ) {
		for (int j = 0; j < n; ++j) {
			if ((abs(Setx_i - Setx[j]) < 2) && (abs(Sety_i - Sety[j]) < 2) && (g * pos[j].g > 0) &&
				(Ro2(r0, r1, pos[j].r[0], pos[j].r[1]) < R_COL_2) && (j != i) && (fabs(g + pos[j].g) < MAX_VE_G)) {
				COL[i] = j;
				//j = n + 5;
				break;
			}
		}
	}
}

__global__ void first_setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) {
        TVars r0 = pos[i].r[0];
        TVars r1 = pos[i].r[1];
        TVars g = pos[i].g;
        int Setx_i = floor((r0+2)/HX);
        int Sety_i = floor((r1+10)/HY);
        Setx[i] = Setx_i;
        Sety[i] = Sety_i;
        COL[i] = -1;

        __syncthreads();

        //  for (int j = (i+1); j < n; j++ ) {
        for (int j = 0; j < n; ++j) {
            if ((abs(Setx_i - Setx[j]) < 2) && (abs(Sety_i - Sety[j]) < 2) &&
                (g * pos[j].g < 0) && (Ro2(r0, r1, pos[j].r[0], pos[j].r[1]) < R_COL_1)) {
                COL[i] = j;
                //j = n + 5;
                break;
            }
        }
    }
}

__global__ void second_collapse_Kernel(Vortex *pos, int *COL, size_t n) {
	for (int i = 0; i < n; i++) {
		if ((COL[i] > (-1))) {
			int j = COL[i];
			TVars new_g = pos[i].g + pos[j].g;
        //    if (fabs(new_g) < 2 * MAX_VE_G) {
                pos[i].r[0] = (pos[i].r[0] * pos[i].g + pos[j].r[0] * pos[j].g) / new_g;
                pos[i].r[1] = (pos[i].r[1] * pos[i].g + pos[j].r[1] * pos[j].g) / new_g;
                pos[i].g = new_g;
                pos[j].g = 0;
                pos[j].r[0] = (TVars)(1e+10);
                pos[j].r[1] = (TVars)(1e+10);
                COL[j] = -1;
        //    } else {
        //        printf("%.6f  %.6f  %.6f\n", pos[i].r[0], pos[i].r[1], new_g);
        //    }
		}
	}
}

__global__ void first_collapse_Kernel(Vortex *pos, int *COL, size_t n) {
	for (int i = 0; i < n; i++) {
		if ((COL[i] > (-1))) {
			int j = COL[i];
			TVars mnog = fabs(pos[i].g) + fabs(pos[j].g);
			pos[i].r[0] = (pos[i].r[0] * fabs(pos[i].g) + pos[j].r[0] * fabs(pos[j].g)) / fabs(mnog);
			pos[i].r[1] = (pos[i].r[1] * fabs(pos[i].g) + pos[j].r[1] * fabs(pos[j].g)) / fabs(mnog);
			pos[i].g = pos[i].g + pos[j].g;
			pos[j].g = 0;
			pos[j].r[0] = (TVars)(1e+10);
			pos[j].r[1] = (TVars)(1e+10);
			COL[j] = -1;
		}
	}
}

/*
__device__ __host__ TVars R_birth_x(size_t n, size_t j) {
    TVars arg=(TVars)(j*2*M_PI/n);
    return R*cos(arg);
}
__device__ __host__ TVars R_birth_y(size_t n, size_t j) {
    TVars arg=(TVars)(j*2*M_PI/n);
    return R*sin(arg);
}
__device__ __host__ TVars R_contr_x(size_t n, size_t i) {
    TVars arg=(TVars)((i+0.5)*2*M_PI/n);
    return R*cos(arg);
}
__device__ __host__ TVars R_contr_y(size_t n, size_t i) {
    TVars arg=(TVars)((i+0.5)*2*M_PI/n);
    return R*sin(arg);
}
__device__ __host__ TVars N_contr_x(size_t n, size_t i) {
    TVars arg=(TVars)((i+0.5)*2*M_PI/n);
    return cos(arg);
}
__device__ __host__ TVars N_contr_y(size_t n, size_t i) {
    TVars arg=(TVars)((i+0.5)*2*M_PI/n);
    return sin(arg);
}
*/

inline __device__ void I_0_I_3(TVars &Ra_0, TVars &Ra_1, TVars &Rb_0, TVars &Rb_1, TVars &Norm_0, TVars &Norm_1, TVars &Rj_0, TVars &Rj_1,
                                TVars &dL, TVars &d, size_t N,TVars &RES_0, TVars &RES_3_0, TVars &RES_3_1) {
    TVars Rk_0 = 0.0, Rk_1 = 0.0;
    TVars Eta_0 = 0.0, Eta_1 = 0.0;
    TVars dR_0 = 0.0, dR_1 = 0.0;
    RES_0 = 0.0;
    RES_3_0 = 0.0;
    RES_3_1 = 0.0;
    dR_0 = (Rb_0 - Ra_0) / N;
    dR_1 = (Rb_1 - Ra_1) / N;
    TVars delt = dL / N;
    for (size_t k = 0; k < N; ++k) {
        Rk_0 = (Ra_0 + k * dR_0 + Ra_0 + (k + 1) * dR_0) / 2;
        Rk_1 = (Ra_1 + k * dR_1 + Ra_1 + (k + 1) * dR_1) / 2;
        Eta_0 = (Rj_0 - Rk_0) / d;
        Eta_1 = (Rj_1 - Rk_1) / d;
        TVars mod_eta = sqrt(Eta_0 * Eta_0 + Eta_1 * Eta_1);
        RES_0 += (Eta_0 * Norm_0 + Eta_1 * Norm_1) / (Eta_0 * Eta_0 + Eta_1 * Eta_1) * 
            (mod_eta + 1) * exp(-mod_eta) * delt;
        RES_3_0 += Norm_0 * exp(-mod_eta) * delt;
        RES_3_1 += Norm_1 * exp(-mod_eta) * delt;
    }
}

inline __device__ void I_0_I_3f(float &Ra_0, float &Ra_1, float &Rb_0, float &Rb_1, float &Norm_0, float &Norm_1, float &Rj_0, float &Rj_1,
                                float &dL, float &d, size_t N, float &RES_0, float &RES_3_0, float &RES_3_1) {
    float Rk_0 = 0.0f, Rk_1 = 0.0f;
    float Eta_0 = 0.0f, Eta_1 = 0.0f;
    float dR_0 = 0.0f, dR_1 = 0.0f;
    RES_0 = 0.0f;
    RES_3_0 = 0.0f;
    RES_3_1 = 0.0f;
    dR_0 = (Rb_0 - Ra_0) / N;
    dR_1 = (Rb_1 - Ra_1) / N;
    float delt = dL / N;
    for (size_t k = 0; k < N; ++k) {
        Rk_0 = (Ra_0 + k * dR_0 + Ra_0 + (k + 1) * dR_0) / 2;
        Rk_1 = (Ra_1 + k * dR_1 + Ra_1 + (k + 1) * dR_1) / 2;
        Eta_0 = (Rj_0 - Rk_0) / d;
        Eta_1 = (Rj_1 - Rk_1) / d;
        TVars mod_eta = sqrtf(Eta_0 * Eta_0 + Eta_1 * Eta_1);
        RES_0 += (Eta_0 * Norm_0 + Eta_1 * Norm_1) / (Eta_0 * Eta_0 + Eta_1 * Eta_1) * 
            (mod_eta + 1) * expf(-mod_eta) * delt;
        RES_3_0 += Norm_0 * expf(-mod_eta) * delt;
        RES_3_1 += Norm_1 * expf(-mod_eta) * delt;
    }
}


//------------------------------------------------------
//-----------------Контроль протыкания------------------
// Вход:  Panel    - контролируемый профиль
//        a[]    - конечное положение
//        b[]    - начальное положение
// Выход: return - признак протыкания
//		  hitpan - номер панели, которая пересекается
//------------------------------------------------------
__device__ inline bool hitting(tPanel *Panel, TVars a0, TVars a1, TVars* b, int* hitpan) {
	const TVars porog_r=1e-12;
	
	TVars x1=a0;//конечное положение
	TVars y1=a1;
	TVars x2=b[0];//начальное положение
	TVars y2=b[1];
	TVars minDist=25.0; //расстояние до пробиваемой панели
	int minN=-1;          //номер пробиваемой панели
  
	bool hit=true; //по умолчанию устанавливаем пробивание
    
	//if ( ((x1<Profile[prf].low_left[0]) && (x2<Profile[prf].low_left[0])) ||   <-- Было
	//     ((x1>Profile[prf].up_right[0]) && (x2>Profile[prf].up_right[0])) ||
	//     ((y1<Profile[prf].low_left[1]) && (y2<Profile[prf].low_left[1])) ||
	//     ((y1>Profile[prf].up_right[1]) && (y2>Profile[prf].up_right[1])) ) hit=false;

	//если вихрь вне габ. прямоугольника - возвращаем false
	hit = !( ((x1<-0.5) && (x2<-0.5)) ||   
			 ((x1>0.5) && (x2>0.5)) ||
			 ((y1<-0.01) && (y2<-0.01)) ||
			 ((y1>0.01) && (y2>0.01))   );
  
	//если внутри габ. прямоугольника - проводим контроль
	if (hit)
	{
		hit=false;
        //Определение прямой: Ax+By+D=0 - перемещение вихря
        TVars A=y2-y1;
        TVars B=x1-x2;
        TVars D=y1*x2-x1*y2;
        TVars A1, B1, D1;
        //Проверка на пересечение
        TVars r0=0, r1=0, r2=0, r3=0;
        bool hitt=false;
        for(int i=0; i<QUANT; ++i)
		{ 
			
			r0=A*Panel[i].left[0] + B*Panel[i].left[1] + D;
            r1=A*Panel[i].right[0] + B*Panel[i].right[1] + D;
			if (fabs(r0)<porog_r) r0=0.0;
			if (fabs(r1)<porog_r) r1=0.0;
            hitt=false;
            if (r0*r1<=0) 
				hitt=true;
            if (hitt)
            {
				A1=Panel[i].right[1]-Panel[i].left[1]; //Определение прямой:A1x+B1y+D1=0 -панель
                B1=Panel[i].left[0]-Panel[i].right[0];
                D1=Panel[i].left[1]*Panel[i].right[0]-Panel[i].left[0]*Panel[i].right[1];
				r2=A1*x1+B1*y1+D1;
                r3=A1*x2+B1*y2+D1;
                if (fabs(r2)<porog_r) r2=0.0;
			    if (fabs(r3)<porog_r) r3=0.0;
				
				if (r2*r3<=0)
				{
					hit=true;// пробила!
                    TVars d2=(x2-(B*D1-D*B1)/(A*B1-B*A1))*(x2-(B*D1-D*B1)/(A*B1-B*A1))+(y2-(A1*D-D1*A)/(A*B1-B*A1))*(y2-(A1*D-D1*A)/(A*B1-B*A1)); 
					if (d2<minDist) 
					{
						minDist=d2;
						minN=i;
					}//if d2
				}//if r2*r3
			}//if hitt                              
		}//for i=0;i<Profile[prf].n
	}; //if hit

	hitpan[0]=minN;
	return hit;
	
}//hitting


__global__ void velocity_control_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V, int *n_v) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
if ( i < 500 ) {
    TVars y0 = 0.0, y1 = 0.0;
//	TVars dist2;
    TVars mnog = 0.0;
    TVars dist2 = 0.0;
    // координаты расчётной точки
    TVars a0 = 0.0, a1 = 0.0;
    // координаты воздействующей точки
    __shared__ TVars b_sh_0 [BLOCK_SIZE];
    __shared__ TVars b_sh_1 [BLOCK_SIZE];
    // интенсивность воздействующей точки
    __shared__ TVars g [BLOCK_SIZE];
    a0=Contr_points[i].v[0];
    a1=Contr_points[i].v[1];
    for (int f = 0 ; f < n ; f += BLOCK_SIZE) {
        b_sh_0[threadIdx.x]=pos[threadIdx.x+f].r[0];
        b_sh_1[threadIdx.x]=pos[threadIdx.x+f].r[1];
        g[threadIdx.x]=pos[threadIdx.x+f].g;
        __syncthreads();
        for (int j = 0 ; j < BLOCK_SIZE ; ++j) {		
            dist2 = Ro2(a0, a1, b_sh_0[j], b_sh_1[j]);
	//		if (dist2 < EPS2) dist2=EPS2;
            dist2 = max(dist2, EPS2);
            mnog=g[j] / dist2;
            y1 +=  mnog * (a0 - b_sh_0[j]);	
            y0 += -mnog * (a1 - b_sh_1[j]);
        }//j
        __syncthreads();
    }//f
    V[i + (*n_v)].v[0] = y0/(2*M_PI) + (*V_inf)[0];
    V[i + (*n_v)].v[1] = y1/(2*M_PI) + (*V_inf)[1];
//	V[i].v[k] =  (*V_inf)[k];
    __syncthreads(); 
    if (i == 1) {
        (*n_v)+=500;
    }
}
}
