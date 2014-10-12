/*
 ============================================================================
 Name        : main.cpp
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 22, 2014
 Copyright   : All rights reserved
 Description : main file of vortex project
 ============================================================================
 */
#include "main.h"

int main() {
    using namespace std;
    int cnt = 0;
    size_t birth;                                                   // расширенное количество рождаемых ВЭ
    double rash;                                                    // (кратно BLOCK_SIZE)
	size_t p = 0;												// количество точек рождения ВЭ
    cudaError_t cuerr;												// ошибки CUDA
    cudaDeviceReset();
    load_profile(panels_host, p);
	int menu = 0;
	cout << "1 - generate matrix\n2 - load matrix\n";
	cin >> menu;
    cin.get();
	if (menu == 0) {
		return 0;
	} else if (menu == 1) {
		cout << "generate matrix\n";
		do {
	        M = matr_creation(panels_host, p);                                       // генерация "матрицы формы"
			++cnt;
		} while (M == NULL && cnt < 10);
		cnt = 0;
		if (M == NULL) {
	        cout << "Matrix creation error!\n";
            cin.get();
			return 1;
		} else {
            cout << "Matrix created!\n";
        }
	} else if (menu == 2) {
		cout << "load matrix\n";
		M = load_matrix(p);
		if (M == NULL) {
			cout << "Matrix loading error!\n";
            cin.get();
			return 1;
		} else {
            cout << "Matrix loaded!\n";
        }
	}
//	n=new size_t;
	n = 0;															// количество ВЭ
	size = 0;                                                       // размер массива ВЭ
	Psp.eps = 0.008;                                                // 
	rash = (double)(p) / BLOCK_SIZE;
	birth = (size_t)(BLOCK_SIZE * ceil(rash));

//-----------------------------------------------------------------------------------------------------------------------------
    // Вычисление скоростей при x = 0.35
    Contr_points_host = new PVortex[500];
    for (int i = 0; i < 500; ++i) {
        Contr_points_host[i].v[1] = 0.021 + 0.002 * i;
        Contr_points_host[i].v[0] = 1.0;
    }

    V_contr_host = new PVortex[500];
    cuerr=cudaMalloc((void**)&V_contr_device, 500 * sizeof(PVortex));
    cuerr=cudaMalloc((void**)&Contr_points_device, 500 * sizeof(PVortex));
    cuerr=cudaMemcpy(Contr_points_device, Contr_points_host, 500 * sizeof(PVortex), cudaMemcpyHostToDevice);

//-----------------------------------------------------------------------------------------------------------------------------
/*
	TVars rrr=0;
	TVctr rrr1={0,0};
	TVctr aa={0,0};
	TVctr bb={1,1};
	TVctr rr={0.51,0.5};
	TVctr Nm={1.4121356,-1.4121356};
	I_0(aa, bb,Nm, rr, 0.001, 20, rrr,rrr1);

//	d=new TVars[size];												// характерное расстояние до 3-х ближайших ВЭ (host)
*/
//	save_matr(M, birth + 1, "M.txt");
    F_p_host.v[0] = 0.0;
    F_p_host.v[1] = 0.0;
    // выделение памяти и копирование на device
	cuerr=cudaMalloc((void**)&V_inf_device, sizeof(TVctr));
	cuerr=cudaMalloc((void**)&d_g_device, sizeof(TVars));
    cuerr=cudaMalloc((void**)&Momentum_device, sizeof(TVars));
    cuerr=cudaMalloc((void**)&F_p_device, sizeof(PVortex));
    cuerr=cudaMalloc((void**)&M_device, (birth+1) * (birth+1) * sizeof(TVars));
	cuerr = cudaMalloc((void**)&panels_device, birth * sizeof(tPanel));
    cuerr=cudaMemcpy(V_inf_device, &V_inf_host, sizeof(TVctr), cudaMemcpyHostToDevice);
    cuerr=cudaMemcpy(d_g_device, &TVarsZero, sizeof(TVars), cudaMemcpyHostToDevice);
    cuerr=cudaMemcpy(Momentum_device, &Momentum_host, sizeof(TVars), cudaMemcpyHostToDevice);
    cuerr=cudaMemcpy ( F_p_device, &F_p_host , sizeof(PVortex), cudaMemcpyHostToDevice);
	cuerr=cudaMemcpy(M_device, M, (birth+1) * (birth+1) * sizeof(TVars), cudaMemcpyHostToDevice);
	cuerr=cudaMemcpy(panels_device, panels_host, birth * sizeof(tPanel), cudaMemcpyHostToDevice);
    // все массивы имеют переменную длину и память для них выделяется в incr_vort_quont()

	delete[] M;
	cout << "dt = " << dt << '\n';

//	cout << "input saving step" << endl;
//	cin >> sv;

	// увеличение массива ВЭ на INCR_STEP элементов
    err = incr_vort_quont(POS_host, POS_device, VEL_host, VEL_device, d_device, size);
	if (err != 0) 
	{
		cout << "Increase ERROR!\n";
        mem_clear();
        system("pause");
		return 1;
	}

	cudaEvent_t start = 0, stop = 0;	    
	start_timer(start, stop);
//------------------------------------------------------------------------------------------
    // цикл шагов выполнения расчётов
	for (int j = 0; j < st; j++) {
        // количество ВЭ на текущем шаге, увеличенное до кратности BLOCK_SIZE
        size_t s = 0;
        double rashirenie = 0;
        rashirenie = (double)(n+p)/BLOCK_SIZE;
        s = (int)(BLOCK_SIZE*ceil(rashirenie));
        if (s > size) {
            //увеличение массива ВЭ на INCR_STEP элементов, если это необходимо
		    err=incr_vort_quont(POS_host, POS_device, VEL_host, VEL_device, d_device, size);
            if (err != 0) {
                cout << "Increase ERROR!" << endl;
                cin.get();
                mem_clear();
                return 1;
            }// if err
        }// if size

//	    cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
//	    save_to_file_size(2*j);

        // "рождение" ВЭ

		err = vort_creation(POS_device, V_inf_device, p, birth, n, M_device, d_g_device, panels_device);

        if (err != 0) {
            cout << "Creation ERROR!" << endl;
            mem_clear();
            cin.get();
            return 1;
        }// if err
		n += p;

//	cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
//	save_to_file_size(2*j+1);

        // вывод данных в файл
		if (j%sv == 0) {
//			cuerr=cudaMemcpy ( d , d_Dev , size  * sizeof(TVars) , cudaMemcpyDeviceToHost);
			cuerr=cudaMemcpy(POS_host, POS_device, n  * sizeof(Vortex), cudaMemcpyDeviceToHost);
            if (cuerr != cudaSuccess) {
                cout << cudaGetErrorString(cuerr) << '\n';
                cout << "Saving ERROR\n";
                mem_clear();
                cin.get();
                return 1;
            }// if cuerr
//			cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
            cout << "Output " << j << '\n';


//////////////////////////////////////////////////////////////////////////

            velocity_control(POS_device, V_inf_device, n, Contr_points_device, V_contr_device);

            cuerr=cudaMemcpy(V_contr_host, V_contr_device, 500 * sizeof(PVortex), cudaMemcpyDeviceToHost);

            char *fname1;
            fname1 = "Vel";
            char *fname2;
            fname2 = ".txt";
            char *fzero;
            fzero = "0";
            char fstep[6];
            char fname[15];
            fname[0] = '\0';
            itoaxx(j,fstep,10);
            strcat(fname,fname1);
            if (j<10) strcat(fname,fzero);
            if (j<100) strcat(fname,fzero);
            if (j<1000) strcat(fname,fzero);
            if (j<10000) strcat(fname,fzero);
            //	if (_step<100000) strcat(fname,fzero);
            strcat(fname,fstep);
            strcat(fname,fname2);
            ofstream outfile;
            outfile.open(fname);
            // Сохранен­ие числа вихрей в пелене
            outfile << (100) << endl;
            for (size_t i = 0; i < (500); ++i) {
                outfile<<(int)(i)<<" "<<(double)(Contr_points_host[i].v[0])<<" "<<(double)(Contr_points_host[i].v[1])<<" "<<(double)(V_contr_host[i].v[0])<<" "<<(double)(V_contr_host[i].v[1])<<endl;
                //      outfile<<(double)(d[i])<<" "<<(double)(POS[i].r[0])<<" "<<(double)(POS[i].r[1])<<" "<<(double)(POS[i].g)<<endl;     
                // нули пишутся для совместимо­сти с трехмерной­ программой­ и обработчик­ами ConMDV, ConMDV-p и Construct
            }//for i
            outfile.close();


//////////////////////////////////////////////////////////////////////////

			save_to_file(POS_host, n, Psp, j);
        }// if sv

        cuerr=cudaMemcpy(&F_p_host, F_p_device, sizeof(PVortex), cudaMemcpyDeviceToHost);
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            cout << "Saving ERROR\n";
            mem_clear();
            cin.get();
            return 1;
        }// if cuerr
        cuerr=cudaMemcpy(&Momentum_host, Momentum_device, sizeof(TVars), cudaMemcpyDeviceToHost);
        if (cuerr != cudaSuccess) {
            cout << cudaGetErrorString(cuerr) << '\n';
            cout << "Saving ERROR\n";
            mem_clear();
            cin.get();
            return 1;
        }// if cuerr
        save_forces(F_p_host, Momentum_host, j);
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
			double gamma=0.0;

//			for (int k=(*n)-1.0;k>((*n)-QUANT)-1.0;k--) 
			for (int k=0;k<(*n);k++) 
				gamma+=POS[k].g;
			cout<<" j= "<<j <<";  gamma= "<<gamma<<endl;
*/

//			cout << j;

//		if ((j%100 == 0) && (j%1000 != 0)) cout<<"j= "<<j<<endl;

        // расчёт скоростей

		err = Speed(POS_device, V_inf_device, s, VEL_device, d_device, nu, panels_device);									
		if (err != 0) {
            cout << "Speed evaluation ERROR!" << endl;
            mem_clear();
			cin.get();
			return 1;
		}
/*
		if (j==0)																		//вывод скоростей в файл
		{
			cuerr=cudaMemcpy ( VEL , VDev , (*n)  * sizeof(PVortex) , cudaMemcpyDeviceToHost);
			stf();

		}
*/
        F_p_host.v[0] = 0.0;
        F_p_host.v[1] = 0.0;
        Momentum_host = 0.0;
        cuerr=cudaMemcpy (F_p_device, &F_p_host , sizeof(PVortex), cudaMemcpyHostToDevice);
        cuerr=cudaMemcpy (d_g_device, &TVarsZero , sizeof(TVars), cudaMemcpyHostToDevice);
        cuerr=cudaMemcpy (Momentum_device, &Momentum_host , sizeof(TVars), cudaMemcpyHostToDevice);
        // перемещение ВЭ

		err = Step(POS_device, VEL_device, n, s, d_g_device, F_p_device , Momentum_device, panels_device);
		if (err != 0) {
            cout << "Moving ERROR!" << endl;
            mem_clear();
			cin.get();
			return 1;
		}
/*
		if (j==0)																		//вывод в файл ВЭ после перемещения
		{

			cuerr=cudaMemcpy ( POS , posDev , (*n)  * sizeof(Vortex) , cudaMemcpyDeviceToHost);

			save_to_file(30);
		}
*/
	}//j
//------------------------------------------------------------------------------------------
	float time = stop_timer(start, stop);
	cout << "Computing time = "<< time << " sec\n";
	cuerr=cudaMemcpy ( POS_host , POS_device , n  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
    cuerr=cudaMemcpy(&F_p_host, F_p_device, sizeof(PVortex), cudaMemcpyDeviceToHost);
    cuerr=cudaMemcpy(&Momentum_host, Momentum_device, sizeof(PVortex), cudaMemcpyDeviceToHost);
    // вывод в файл последнего шага
	save_to_file(POS_host, n,  Psp, st);
    save_forces(F_p_host, Momentum_host, st);
	cout<<"ready!";
//  cin.get();
    mem_clear();
	cin.get();
	return 0;
}
void mem_clear() {
    cudaFree(V_inf_device);
    cudaFree(d_g_device);
    cudaFree(M_device);
    cudaFree(POS_device);
    cudaFree(d_device);
    cudaFree(VEL_device);
    cudaDeviceReset();
    delete[] POS_host;
    delete[] VEL_host;
}
void save_to_file(Vortex *POS, size_t size, Eps_Str Psp, int _step) {
    using namespace std;
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
    //	if (_step<100000) strcat(fname,fzero);
    strcat(fname,fstep);
    strcat(fname,fname2);
    ofstream outfile;
    outfile.open(fname);
    // Сохранен­ие числа вихрей в пелене
    outfile << (size) << endl;
    for (size_t i = 0; i < (size); ++i) {
        outfile<<(int)(i)<<" "<<(double)(Psp.eps)<<" "<<(double)(POS[i].r[0])<<" "<<(double)(POS[i].r[1])<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<(double)(POS[i].g)<<endl;
//      outfile<<(double)(d[i])<<" "<<(double)(POS[i].r[0])<<" "<<(double)(POS[i].r[1])<<" "<<(double)(POS[i].g)<<endl;     
        // нули пишутся для совместимо­сти с трехмерной­ программой­ и обработчик­ами ConMDV, ConMDV-p и Construct
    }//for i
    outfile.close();
} //save_to_file

void save_forces(PVortex F_p, TVars M, int step) {
    using namespace std;
    ofstream outfile;
    if (step == 0) {
        outfile.open("F_p.txt");
    } else {
        outfile.open("F_p.txt", ifstream::app);
    }
    outfile << step << " " << (double)F_p.v[0] << " " << (double)F_p.v[1] << ' ' << M << '\n';
    outfile.close();
}

void load_profile(tPanel *&panels, size_t &p) {
	using namespace std;
	ifstream infile;
	infile.open(PR_FILE);
	char *buff = new char[255];
	infile >> buff;
	infile >> buff;
	infile >> p;
	double rash = (double)(p) / BLOCK_SIZE;
	size_t birth = (size_t)(BLOCK_SIZE * ceil(rash));
	panels = new tPanel[birth];
	for (size_t i = 0; i < p; ++i) {
		infile >> panels[i].n;
		infile >> panels[i].left[0];
		infile >> panels[i].left[1];
		infile >> panels[i].right[0];
		infile >> panels[i].right[1];
		infile >> panels[i].contr[0];
		infile >> panels[i].contr[1];
		infile >> panels[i].birth[0];
		infile >> panels[i].birth[1];
		infile >> panels[i].norm[0];
		infile >> panels[i].norm[1];
		infile >> panels[i].tang[0];
		infile >> panels[i].tang[1];
		infile >> panels[i].length;
		infile >> panels[i].n_of_lpanel;
		infile >> panels[i].n_of_rpanel;
	}
}

