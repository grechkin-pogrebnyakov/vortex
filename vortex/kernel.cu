#include "kernel.cuh"
#include "device_launch_parameters.h"
#include "math_functions.h"

__global__ void zero_Kernel(float *randoms, Vortex *pos, int s ) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    double a = 0.0;
	pos[s+ind].r[0]=(2.0e+5)*randoms[ind]+2.0e+5; 
	pos[s+ind].r[1]=(2.0e+5)*randoms[ind]+2.0e+5; 
	pos[s+ind].g = a; 
}

__global__ void Right_part_Kernel(Vortex *pos, TVctr *V_inf, size_t n_vort, size_t n_birth_BLOCK_S, TVars *R_p, tPanel *panels) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    R_p[i] = 0.0;
    TVctr y = {0.0, 0.0};
//	TVars dist2;
    TVars mnog = 0.0;
    TVars dist2 = 0.0;
    // координаты и нормали расчётной точки
    TVctr a = {0.0, 0.0};
    TVctr n = {0.0, 0.0};
    // координаты воздействующей точки
    __shared__ TVctr b_sh [BLOCK_SIZE];
    // интенсивность воздействующей точки
    __shared__ TVars g [BLOCK_SIZE];
    a[0]=R_contr_x(panels,i);
    a[1]=R_contr_y(panels,i);
    n[0]=N_contr_x(panels,i);
    n[1]=N_contr_y(panels,i);
    for (int f = 0 ; f < n_vort ; f += BLOCK_SIZE) {
        b_sh[threadIdx.x][0]=pos[threadIdx.x+f].r[0];
        b_sh[threadIdx.x][1]=pos[threadIdx.x+f].r[1];
        g[threadIdx.x]=pos[threadIdx.x+f].g;
        __syncthreads();
        for (int j = 0 ; j < BLOCK_SIZE ; ++j) {		
            dist2= Ro2(a, b_sh[j]);
	//		if (dist2 < EPS2) dist2=EPS2;
            dist2 = max(dist2, EPS2);
            mnog=g[j] / dist2;
            y[1] +=  mnog * (a[0] - b_sh[j][0]);	
            y[0] += -mnog * (a[1] - b_sh[j][1]);
        }//j
        __syncthreads();
    }//f                                                                         
    R_p[i] = -((y[0]/(2*M_PI) + (*V_inf)[0]) * n[0] + (y[1]/(2*M_PI) + (*V_inf)[1]) * n[1]);
//	V[i].v[k] =  (*V_inf)[k];
    __syncthreads(); 
}

__global__ void birth_Kernel(Vortex *pos, size_t n_vort, size_t n_birth, size_t n_birth_BLOCK_S, TVars * M, TVars *d_g, TVars *R_p, tPanel *panel) {
	int i= blockIdx.x * blockDim.x + threadIdx.x;
	register TVars g = 0.0;
	for (size_t j = 0; j < n_birth; ++j) {
//		pos_N.g += M[(pp+1)*i+j]*R_p[j]; 
        g += M[(n_birth_BLOCK_S + 1) * i + j] * R_p[j]; 
	}
    g += M[(n_birth_BLOCK_S + 1) * i + n_birth] * (*d_g);
	__syncthreads;
	if (i < n_birth)
	{
//		pos[i+n_vort].r[0] = pos_N.r[0];
//		pos[i+n_vort].r[1] = pos_N.r[1];

		pos[i+n_vort].r[0] = R_birth_x(panel, i);
		pos[i+n_vort].r[1] = R_birth_y(panel, i);
		pos[i+n_vort].g = g;
	}
}
__global__ void shared_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *V, TVars *d) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    TVctr y = {0.0, 0.0};
//	TVars dist2;
    TVars mnog = 0.0;
    TVars d_1 = 0.0;      //  четыре наименьших
    TVars d_2 = 0.0;      //  расстояния от этой
    TVars d_3 = 0.0;      //  точки до остальных.
    TVars d_0 = 0.0;
    TVars dist2 = 0.0;
    TVars dst = 0.0;
    // координаты расчётной точки
    TVctr a = {0.0, 0.0};
    // координаты воздействующей точки
    __shared__ TVctr b_sh [BLOCK_SIZE];
    // интенсивность воздействующей точки
    __shared__ TVars g [BLOCK_SIZE];
    a[0]=pos[i].r[0];
    a[1]=pos[i].r[1];
    d_1=1e+15;
    d_2=1e+15;
    d_3=1e+15;
    d_0=1e+15;
    for (int f = 0 ; f < n ; f += BLOCK_SIZE) {
        b_sh[threadIdx.x][0]=pos[threadIdx.x+f].r[0];
        b_sh[threadIdx.x][1]=pos[threadIdx.x+f].r[1];
        g[threadIdx.x]=pos[threadIdx.x+f].g;
        __syncthreads();
        for (int j = 0 ; j < BLOCK_SIZE ; ++j) {		
            dist2= Ro2(a, b_sh[j]);
            if (d_3 > dist2) {
                    d_3 = dist2;
                    dst = min(d_3, d_2);
                    d_3 = max(d_3, d_2);
                    d_2 = dst;
                    dst = min(d_1, d_2);
                    d_2 = max(d_1, d_2);
                    d_1 = dst;
                    dst = min(d_1, d_0);
                    d_1 = max(d_1, d_0);
                    d_0 = dst;
            }
	//		if (dist2 < EPS2) dist2=EPS2;
            dist2 = max(dist2, EPS2);
            mnog=g[j] / dist2;
            y[1] +=  mnog * (a[0] - b_sh[j][0]);	
            y[0] += -mnog * (a[1] - b_sh[j][1]);
        }//j
        __syncthreads();
    }//f
    d[i] = sqrt(d_1 + d_2 + d_3) / 3;
    d[i] = max(d[i], 4.0 * EPS / 3.0);                                                                          // это ещё почему???!!!
    V[i].v[0] = y[0]/(2*M_PI) + (*V_inf)[0];
    V[i].v[1] = y[1]/(2*M_PI) + (*V_inf)[1];
//	V[i].v[k] =  (*V_inf)[k];
    __syncthreads(); 
}
__global__ void diffusion_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    TVctr II_2 = {0.0,0.0};
    TVars e = 0.0;
    // знаменатель
    TVars II_1 = 0.0;
	TVars dist = 0.0;
    TVars mnog = 0.0;
    // координаты расчётной точки
    TVctr a = {0.0, 0.0};
    // координаты воздействующих точек
    __shared__ TVctr b_sh[BLOCK_SIZE];
    // интенсивности воздействующих точек
    __shared__ TVars g[BLOCK_SIZE];
    TVars dd = 0.0;
    a[0] = pos[i].r[0];
    a[1] = pos[i].r[1];
    dd = d[i];
    __syncthreads();
    for (int f = 0; f < n; f += BLOCK_SIZE) {
        b_sh[threadIdx.x][0] = pos[threadIdx.x + f].r[0];
        b_sh[threadIdx.x][1] = pos[threadIdx.x + f].r[1];
        g[threadIdx.x] = pos[threadIdx.x + f].g;
        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            dist= sqrt(Ro2(a, b_sh[j]));
            if(dist > 0.001 * EPS) {
                mnog=g[j] / dist;
                e = exp(-(dist) / (dd));
                II_2[0] += -mnog * (a[0]-b_sh[j][0]) * e;
                II_2[1] += -mnog * (a[1]-b_sh[j][1]) * e;
                II_1 += g[j] * e;
//                II_1 += mnog;
            }
        }//j
        __syncthreads();
    }//f
    if (fabs(II_1) > 1e-5) {
        V[i].v[0] += -nu * II_2[0] / (II_1 * dd);
        V[i].v[1] += -nu * II_2[1] / (II_1 * dd);
//            V[i].v[k] = II_2[k];
    }
}

__global__ void diffusion_2_Kernel(Vortex *pos, int n, PVortex *V, TVars *d, TVars nu, tPanel *panels) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    TVctr Ra = {0.0, 0.0};
    TVctr Rb = {0.0, 0.0};
    TVctr Rk = {0.0, 0.0};
    TVctr Norm = {0.0, 0.0};
    TVars dL = 0.0;
    TVars RES_0 = 0.0;
    TVctr RES_3 = {0.0, 0.0};
    // координаты расчётной точки
    TVctr a = {0.0, 0.0};
    TVars dd = 0.0;
//    F_vis[i].v[0] = 0.0;
//    F_vis[i].v[1] = 0.0;
    a[0] = pos[i].r[0];
    a[1] = pos[i].r[1];
    dd = d[i];
    TVars II_0 = 2 * M_PI * dd;                                                     // НУЖЕН ЛИ ТУТ КВАДРАТ?!!!
    TVctr II_3 = {0, 0};
    //	TVars denomenator = 2 * M_PI * dd; // знаменатель
    for (int f = 0; f < QUANT; ++f) {

        Ra[0] = R_left_x(panels, f);
        Ra[1] = R_left_y(panels, f);
        Rb[0] = R_right_x(panels, f);
        Rb[1] = R_right_y(panels, f);
        Rk[0] = R_contr_x(panels, f);
        Rk[1] = R_contr_y(panels, f);
        //dL = sqrt((Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1]));
        dL = panels[f].length;
        if ((Ro2(a, Rk) < 25 * dL * dL) && (Ro2(a, Rk) > 0.01 * dL * dL)) {
            Norm[0] = -N_contr_x(panels, f);
            Norm[1] = -N_contr_y(panels, f);
            I_0_I_3(Ra, Rb, Norm, a, dL, dd, N_OF_POINTS, RES_0, RES_3);
            II_0 += (-dd) * RES_0;
            II_3[0] -= RES_3[0];
            II_3[1] -= RES_3[1];
        } else if (Ro2(a, Rk) <= 0.01 * dL * dL) {

            Norm[0] = -N_contr_x(panels, f);
            Norm[1] = -N_contr_y(panels, f);
            II_0 = M_PI * dd * dd;
            II_3[0] = 2 * Norm[0] * dd * (1 - exp(-dL / (2 * dd)));
            II_3[0] = 2 * Norm[1] * dd * (1 - exp(-dL / (2 * dd)));
            f = QUANT + 5;
        }

    }//f
    V[i].v[0] += nu * II_3[0] / II_0;
    V[i].v[1] += nu * II_3[1] / II_0;
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
	    __syncthreads;

		TVctr r_new = {pos[i].r[0] + V[i].v[0] * dt,pos[i].r[1] + V[i].v[1] * dt};
//	    TVctr Zero  = {0, 0};
		int hitpan = 0;
	    if ((pos[i].g != 0) && (hitting(panels, r_new, pos[i].r,&hitpan))) {
            F_p[i].v[0] -= pos[i].g * (-panels[hitpan].contr[1]);
            F_p[i].v[1] -= pos[i].g * ( panels[hitpan].contr[0]);
            M[i] -= pos[i].g * Ro2(panels[hitpan].contr, rc);
		    pos[i].r[0] =  2e+5;
		    pos[i].r[1] =  2e+5;
		    d_g = pos[i].g;
//		    d_g_Dev[i] = pos[i].g;
		    pos[i].g = 0;
		}

		pos[i].r[0] += V[i].v[0] * dt;
		pos[i].r[1] += V[i].v[1] * dt;

	    if ((pos[i].g != 0) && ((pos[i].r[0] > COUNT_AREA) || (fabs(pos[i].r[1]) > 10))) {
		    pos[i].r[0]= -2.0e+5; 
		    pos[i].r[1]= -2.0e+5; 
		    pos[i].g=0;
	    }
        __syncthreads;
        d_g_Dev[i] = d_g;
    }
	__syncthreads;
}
__global__ void summ_Kernel(TVars *d_g_Dev, TVars *d_g, PVortex *F_p_dev, PVortex *F_p, TVars *M_dev, TVars *M, size_t n) {
		for (int k = 0; k < n; ++k) {
		(*d_g) += d_g_Dev[k];
        (*F_p).v[0] += F_p_dev[k].v[0];
        (*F_p).v[1] += F_p_dev[k].v[1];
        (*M) += M_dev[k];
        }
        (*F_p).v[0] *= RHO / dt;
        (*F_p).v[1] *= RHO / dt;
        (*M) *= RHO / (2 * dt);
}
__global__ void sort_Kernel(Vortex *pos, size_t *s) {
    TVctr r = {0.0, 0.0};
	size_t n = 0;
    n = (*s);
	for (size_t i = 0 ; i < n ; ++i) {
		if (fabs(pos[i].g) < DELT) {
			r[0]=pos[i].r[0];
			r[1]=pos[i].r[1];
			pos[i].g=pos[n-1].g;
			pos[i].r[0]=pos[n-1].r[0];
			pos[i].r[1]=pos[n-1].r[1];
			pos[n-1].g=0;
			pos[n-1].r[0]=r[0];
			pos[n-1].r[1]=r[1];
			n--;
			i--;
		}
    }
	(*s)=n;
}
__global__ void setka_Kernel(Vortex *pos, size_t n, int *Setx, int *Sety, int *COL) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<n) {
		Setx[i] = floor((pos[i].r[0]+2)/HX);
		Sety[i] = floor((pos[i].r[1]+10)/HY);
		COL[i] = -1;
	}
	__syncthreads();
	for (int j = (i+1); j < n; j++ ) {
		if ((abs(Setx[i] - Setx[j]) < 2) && (abs(Sety[i] - Sety[j]) < 2)) {
			if (Ro2(pos[i].r,pos[j].r) < R_COL) {
				COL[i] = j;
				j = n + 5;
			}
        }
	}
}
__global__ void collapse_Kernel(Vortex *pos, int *COL, size_t n) {
	for (int i = 0; i < n; i++)
	{
		if ((pos[i].g != 0) && (COL[i] > (-1))) {
			for(int k = 0; k < 2; ++k)
				pos[i].r[k] = (pos[i].r[k] * fabs(pos[i].g) + pos[COL[i]].r[k] * fabs(pos[COL[i]].g)) /
                              (fabs(pos[i].g) + fabs(pos[COL[i]].g));
				pos[i].g=pos[i].g+pos[COL[i]].g;
				pos[COL[i]].g = 0;
				pos[COL[i]].r[0] = (double)(1e+10);
				pos[COL[i]].r[1] = (double)(1e+10);
		}
	}
}

/*
__device__ __host__ TVars R_birth_x(size_t n, size_t j) {
    double arg=(double)(j*2*M_PI/n);
    return R*cos(arg);
}
__device__ __host__ TVars R_birth_y(size_t n, size_t j) {
    double arg=(double)(j*2*M_PI/n);
    return R*sin(arg);
}
__device__ __host__ TVars R_contr_x(size_t n, size_t i) {
    double arg=(double)((i+0.5)*2*M_PI/n);
    return R*cos(arg);
}
__device__ __host__ TVars R_contr_y(size_t n, size_t i) {
    double arg=(double)((i+0.5)*2*M_PI/n);
    return R*sin(arg);
}
__device__ __host__ TVars N_contr_x(size_t n, size_t i) {
    double arg=(double)((i+0.5)*2*M_PI/n);
    return cos(arg);
}
__device__ __host__ TVars N_contr_y(size_t n, size_t i) {
    double arg=(double)((i+0.5)*2*M_PI/n);
    return sin(arg);
}
*/
__device__ __host__ TVars R_birth_x(tPanel *panel, size_t j) {
    return panel[j].birth[0];
}
__device__ __host__ TVars R_birth_y(tPanel *panel, size_t j) {
    return panel[j].birth[1];
}
__device__ __host__ TVars R_left_x(tPanel *panel, size_t j) {
    return panel[j].left[0];
}
__device__ __host__ TVars R_left_y(tPanel *panel, size_t j) {
    return panel[j].left[1];
}
__device__ __host__ TVars R_right_x(tPanel *panel, size_t j) {
    return panel[j].right[0];
}
__device__ __host__ TVars R_right_y(tPanel *panel, size_t j) {
    return panel[j].right[1];
}
__device__ __host__ TVars R_contr_x(tPanel *panel, size_t j) {
    return panel[j].contr[0];
}
__device__ __host__ TVars R_contr_y(tPanel *panel, size_t j) {
    return panel[j].contr[1];
}
__device__ __host__ TVars N_contr_x(tPanel *panel, size_t j) {
	return -panel[j].norm[0];
}
__device__ __host__ TVars N_contr_y(tPanel *panel, size_t j) {
    return -panel[j].norm[1];
}

__device__ __host__ TVars Ro2(TVctr a, TVctr b) { 
	TVars x;
	x = (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]);
    return x;
}
__device__ void I_0_I_3(TVctr &Ra, TVctr &Rb, TVctr &Norm, TVctr &Rj, TVars &dL, TVars &d, size_t N, TVars &RES_0, TVctr &RES_3) {
    TVctr Rk = {0.0, 0.0};
    TVctr Eta = {0.0, 0.0};
    TVctr dR = {0.0, 0.0};
    RES_0 = 0.0;
    RES_3[0] = 0.0;
    RES_3[1] = 0.0;
    dR[0] = (Rb[0] - Ra[0]) / N;
    dR[1] = (Rb[1] - Ra[1]) / N;
    TVars delt = dL / N;
    for (size_t k = 0; k < N; ++k) {
        Rk[0] = (Ra[0] + k * dR[0] + Ra[0] + (k + 1) * dR[0]) / 2;
        Rk[1] = (Ra[1] + k * dR[1] + Ra[1] + (k + 1) * dR[1]) / 2;
        Eta[0] = (Rj[0] - Rk[0]) / d;
        Eta[1] = (Rj[1] - Rk[1]) / d;
        RES_0 += (Eta[0] * Norm[0] + Eta[1] * Norm[1]) / (Eta[0] * Eta[0] + Eta[1] * Eta[1]) * 
            (sqrt(Eta[0] * Eta[0] + Eta[1] * Eta[1]) + 1) * exp(-sqrt(Eta[0] * Eta[0] + Eta[1] * Eta[1])) * delt;
        RES_3[0] += Norm[0] * exp(-sqrt(Eta[0] * Eta[0] + Eta[1] * Eta[1])) * delt;
        RES_3[1] += Norm[1] * exp(-sqrt(Eta[0] * Eta[0] + Eta[1] * Eta[1])) * delt;
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
__device__ bool hitting(tPanel *Panel, double* a, double* b, int* hitpan) {
	const double porog_r=1e-12;
	
	double x1=a[0];//конечное положение
	double y1=a[1];
	double x2=b[0];//начальное положение
	double y2=b[1];
	double minDist=25.0; //расстояние до пробиваемой панели
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
        double A=y2-y1;
        double B=x1-x2;
        double D=y1*x2-x1*y2;
        double A1, B1, D1;
        //Проверка на пересечение
        double r0=0, r1=0, r2=0, r3=0;
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
                    double d2=(x2-(B*D1-D*B1)/(A*B1-B*A1))*(x2-(B*D1-D*B1)/(A*B1-B*A1))+(y2-(A1*D-D1*A)/(A*B1-B*A1))*(y2-(A1*D-D1*A)/(A*B1-B*A1)); 
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


__global__ void velocity_control_Kernel(Vortex *pos, TVctr *V_inf, int n, PVortex *Contr_points, PVortex *V) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    TVctr y = {0.0, 0.0};
//	TVars dist2;
    TVars mnog = 0.0;
    TVars dist2 = 0.0;
    // координаты расчётной точки
    TVctr a = {0.0, 0.0};
    // координаты воздействующей точки
    __shared__ TVctr b_sh [BLOCK_SIZE];
    // интенсивность воздействующей точки
    __shared__ TVars g [BLOCK_SIZE];
    a[0]=Contr_points[i].v[0];
    a[1]=Contr_points[i].v[1];
    for (int f = 0 ; f < n ; f += BLOCK_SIZE) {
        b_sh[threadIdx.x][0]=pos[threadIdx.x+f].r[0];
        b_sh[threadIdx.x][1]=pos[threadIdx.x+f].r[1];
        g[threadIdx.x]=pos[threadIdx.x+f].g;
        __syncthreads();
        for (int j = 0 ; j < BLOCK_SIZE ; ++j) {		
            dist2= Ro2(a, b_sh[j]);
	//		if (dist2 < EPS2) dist2=EPS2;
            dist2 = max(dist2, EPS2);
            mnog=g[j] / dist2;
            y[1] +=  mnog * (a[0] - b_sh[j][0]);	
            y[0] += -mnog * (a[1] - b_sh[j][1]);
        }//j
        __syncthreads();
    }//f
    V[i].v[0] = y[0]/(2*M_PI) + (*V_inf)[0];
    V[i].v[1] = y[1]/(2*M_PI) + (*V_inf)[1];
//	V[i].v[k] =  (*V_inf)[k];
    __syncthreads(); 
}