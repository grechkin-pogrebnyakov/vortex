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
#include "unita.h"

int main() {
    using namespace std;
    int cnt = 0;
    size_t birth;                                                   // —Ä–∞—Å—à–∏—Ä–µ–Ω–Ω–æ–µ –∫–æ–ª–∏—á–µ—Å—Ç–≤–æ —Ä–æ–∂–¥–∞–µ–º—ã—Ö –í–≠
    double rash;                                                    // (–∫—Ä–∞—Ç–Ω–æ BLOCK_SIZE)
	size_t p = 0;												// –∫–æ–ª–∏—á–µ—Å—Ç–≤–æ —Ç–æ—á–µ–∫ —Ä–æ–∂–¥–µ–Ω–∏—è –í–≠
    cudaError_t cuerr;												// –æ—à–∏–±–∫–∏ CUDA
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
	        M = matr_creation(panels_host, p);                                       // „ÂÌÂ‡ˆËˇ "Ï‡ÚËˆ˚ ÙÓÏ˚"
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
	n = 0;															// –∫–æ–ª–∏—á–µ—Å—Ç–≤–æ –í–≠
	size = 0;                                                       // —Ä–∞–∑–º–µ—Ä –º–∞—Å—Å–∏–≤–∞ –í–≠
	Psp.eps = 0.008;                                                //
	rash = (double)(p) / BLOCK_SIZE;
	birth = (size_t)(BLOCK_SIZE * ceil(rash));

//-----------------------------------------------------------------------------------------------------------------------------
    // ¬˚˜ËÒÎÂÌËÂ ÒÍÓÓÒÚÂÈ ÔË x = 0.35
    Contr_points_host = new PVortex[500];
    for (int i = 0; i < 500; ++i) {
        Contr_points_host[i].v[1] = 0.01 + 0.002 * i;
        Contr_points_host[i].v[0] = -0.15;
    }

    V_contr_host = new PVortex[500 * SAVING_STEP];
    cuerr=cudaMalloc((void**)&V_contr_device, 500 * SAVING_STEP * sizeof(PVortex));
    cuerr=cudaMalloc((void**)&Contr_points_device, 500 * sizeof(PVortex));
    cuerr=cudaMemcpy(Contr_points_device, Contr_points_host, 500 * sizeof(PVortex), cudaMemcpyHostToDevice);
    int v_n_host = 0;
    int *v_n_device = NULL;
    cuerr=cudaMalloc((void**)&v_n_device, sizeof(int));
    cuerr=cudaMemcpy(v_n_device, &v_n_host, sizeof(int), cudaMemcpyHostToDevice);

//-----------------------------------------------------------------------------------------------------------------------------
/*
	TVars rrr=0;
	TVctr rrr1={0,0};
	TVctr aa={0,0};
	TVctr bb={1,1};
	TVctr rr={0.51,0.5};
	TVctr Nm={1.4121356,-1.4121356};
	I_0(aa, bb,Nm, rr, 0.001, 20, rrr,rrr1);

//	d=new TVars[size];												// —Ö–∞—Ä–∞–∫—Ç–µ—Ä–Ω–æ–µ —Ä–∞—Å—Å—Ç–æ—è–Ω–∏–µ –¥–æ 3-—Ö –±–ª–∏–∂–∞–π—à–∏—Ö –í–≠ (host)
*/
//	save_matr(M, birth + 1, "M.txt");
    F_p_host.v[0] = 0.0;
    F_p_host.v[1] = 0.0;
    // –≤—ã–¥–µ–ª–µ–Ω–∏–µ –ø–∞–º—è—Ç–∏ –∏ –∫–æ–ø–∏—Ä–æ–≤–∞–Ω–∏–µ –Ω–∞ device
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
    // –≤—Å–µ –º–∞—Å—Å–∏–≤—ã –∏–º–µ—é—Ç –ø–µ—Ä–µ–º–µ–Ω–Ω—É—é –¥–ª–∏–Ω—É –∏ –ø–∞–º—è—Ç—å –¥–ª—è –Ω–∏—Ö –≤—ã–¥–µ–ª—è–µ—Ç—Å—è –≤ incr_vort_quont()

	delete[] M;
	cout << "dt = " << dt << '\n';

//	cout << "input saving step" << endl;
//	cin >> sv;

	// —É–≤–µ–ª–∏—á–µ–Ω–∏–µ –º–∞—Å—Å–∏–≤–∞ –í–≠ –Ω–∞ INCR_STEP —ç–ª–µ–º–µ–Ω—Ç–æ–≤
    err = incr_vort_quont(POS_host, POS_device, VEL_host, VEL_device, d_device, size);
	if (err != 0)
	{
		cout << "Increase ERROR!\n";
        mem_clear();
        system("pause");
		return 1;
	}
    float creation_time = 0.0;
    float speed_time = 0.0;
    float step_time = 0.0;
	cudaEvent_t start = 0, stop = 0;
//------------------------------------------------------------------------------------------
    V_inf_host[0] = 0.0;
    cuerr=cudaMemcpy(V_inf_device, &V_inf_host, sizeof(TVctr), cudaMemcpyHostToDevice);
    double d_V_inf = 1.0/100;

    // ˆËÍÎ ¯‡„Ó‚ ‚˚ÔÓÎÌÂÌËˇ ‡Ò˜∏ÚÓ‚
	for (current_step = 0; current_step < st; current_step++) {
        if (current_step < 100) {
            V_inf_host[0] += d_V_inf;
            cuerr=cudaMemcpy(V_inf_device, &V_inf_host, sizeof(TVctr), cudaMemcpyHostToDevice);
        }
        //cout << j << ' ';
        // ÍÓÎË˜ÂÒÚ‚Ó ¬› Ì‡ ÚÂÍÛ˘ÂÏ ¯‡„Â, Û‚ÂÎË˜ÂÌÌÓÂ ‰Ó Í‡ÚÌÓÒÚË BLOCK_SIZE
        size_t s = 0;
        double rashirenie = 0;
        rashirenie = (double)(n+p)/BLOCK_SIZE;
        s = (int)(BLOCK_SIZE*ceil(rashirenie));
        if (s > size) {
            //—É–≤–µ–ª–∏—á–µ–Ω–∏–µ –º–∞—Å—Å–∏–≤–∞ –í–≠ –Ω–∞ INCR_STEP —ç–ª–µ–º–µ–Ω—Ç–æ–≤, –µ—Å–ª–∏ —ç—Ç–æ –Ω–µ–æ–±—Ö–æ–¥–∏–º–æ
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

        // "ÓÊ‰ÂÌËÂ" ¬›
        start = 0; stop = 0;
        start_timer(start, stop);
		err = vort_creation(POS_device, V_inf_device, p, birth, n, M_device, d_g_device, panels_device);

        if (err != 0) {
            cout << "Creation ERROR!" << endl;
            mem_clear();
            cin.get();
            return 1;
        }// if err
		n += p;
        creation_time += stop_timer(start, stop);
//	cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
//	save_to_file_size(2*j+1);

        // ‚˚‚Ó‰ ‰‡ÌÌ˚ı ‚ Ù‡ÈÎ
		if (current_step%sv == 0) {
//			cuerr=cudaMemcpy ( d , d_Dev , size  * sizeof(TVars) , cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
			cuerr=cudaMemcpy(POS_host, POS_device, n  * sizeof(Vortex), cudaMemcpyDeviceToHost);
            if (cuerr != cudaSuccess) {
                cout << cudaGetErrorString(cuerr) << '\n';
                cout << "Saving ERROR\n";
                mem_clear();
                cin.get();
                return 1;
            }// if cuerr
//			cuerr=cudaMemcpy ( POS , posDev , size  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
            cout << "\nOutput " << current_step << '\n';


//////////////////////////////////////////////////////////////////////////


            cuerr=cudaMemcpy(V_contr_host, V_contr_device, 500 * SAVING_STEP * sizeof(PVortex), cudaMemcpyDeviceToHost);
            cuerr=cudaMemcpy(v_n_device, &v_n_host, sizeof(int), cudaMemcpyHostToDevice);
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
            ofstream outfile;
            outfile.open(fname);
            // —Óı‡ÌÂÌ≠ËÂ ˜ËÒÎ‡ ‚ËıÂÈ ‚ ÔÂÎÂÌÂ
//            outfile << (100) << endl;
            for (size_t i = 0; i < (500 * SAVING_STEP); ++i) {
                outfile<<(int)(i)<<" "<<(double)(Contr_points_host[i%500].v[0])<<" "<<(double)(Contr_points_host[i%500].v[1])<<" "<<(double)(V_contr_host[i].v[0])<<" "<<(double)(V_contr_host[i].v[1])<<endl;
                //      outfile<<(double)(d[i])<<" "<<(double)(POS[i].r[0])<<" "<<(double)(POS[i].r[1])<<" "<<(double)(POS[i].g)<<endl;     
                // ÌÛÎË ÔË¯ÛÚÒˇ ‰Îˇ ÒÓ‚ÏÂÒÚËÏÓ≠ÒÚË Ò ÚÂıÏÂÌÓÈ≠ ÔÓ„‡ÏÏÓÈ≠ Ë Ó·‡·ÓÚ˜ËÍ≠‡ÏË ConMDV, ConMDV-p Ë Construct
            }//for i
            outfile.close();


//////////////////////////////////////////////////////////////////////////

			save_to_file(POS_host, n, Psp, current_step);
        }// if sv
        velocity_control(POS_device, V_inf_device, n, Contr_points_device, V_contr_device, v_n_device);
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
			double gamma=0.0;

//			for (int k=(*n)-1.0;k>((*n)-QUANT)-1.0;k--)
			for (int k=0;k<(*n);k++)
				gamma+=POS[k].g;
			cout<<" j= "<<j <<";  gamma= "<<gamma<<endl;
*/

//			cout << j;

//		if ((j%100 == 0) && (j%1000 != 0)) cout<<"j= "<<j<<endl;

        // ‡Ò˜∏Ú ÒÍÓÓÒÚÂÈ
        start = 0; stop = 0;
        start_timer(start, stop);
		err = Speed(POS_device, V_inf_device, s, VEL_device, d_device, nu, panels_device);									
		if (err != 0) {
            cout << "Speed evaluation ERROR!" << endl;
            mem_clear();
			cin.get();
			return 1;
		}
        speed_time += stop_timer(start, stop);
/*
		if (j==0)																		//–≤—ã–≤–æ–¥ —Å–∫–æ—Ä–æ—Å—Ç–µ–π –≤ —Ñ–∞–π–ª
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
        // ÔÂÂÏÂ˘ÂÌËÂ ¬›
        start = 0; stop = 0;
        start_timer(start, stop);
		err = Step(POS_device, VEL_device, n, s, d_g_device, F_p_device , Momentum_device, panels_device);
		if (err != 0) {
            cout << "Moving ERROR!" << endl;
            mem_clear();
			cin.get();
			return 1;
		}
        step_time += stop_timer(start, stop);
//        cout << n << '\n';
/*
		if (j==0)																		//–≤—ã–≤–æ–¥ –≤ —Ñ–∞–π–ª –í–≠ –ø–æ—Å–ª–µ –ø–µ—Ä–µ–º–µ—â–µ–Ω–∏—è
		{

			cuerr=cudaMemcpy ( POS , posDev , (*n)  * sizeof(Vortex) , cudaMemcpyDeviceToHost);

			save_to_file(30);
		}
*/
	}//j
//------------------------------------------------------------------------------------------
//	float time = stop_timer(start, stop);
//	cout << "Computing time = "<< time << " sec\n";
    cout << "Creation time = " << creation_time << " speed time = " << speed_time << " step time = " << step_time << '\n';
	cuerr=cudaMemcpy ( POS_host , POS_device , n  * sizeof(Vortex) , cudaMemcpyDeviceToHost);
    cuerr=cudaMemcpy(&F_p_host, F_p_device, sizeof(PVortex), cudaMemcpyDeviceToHost);
    cuerr=cudaMemcpy(&Momentum_host, Momentum_device, sizeof(PVortex), cudaMemcpyDeviceToHost);
    // –≤—ã–≤–æ–¥ –≤ —Ñ–∞–π–ª –ø–æ—Å–ª–µ–¥–Ω–µ–≥–æ —à–∞–≥–∞
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
    // –°–æ—Ö—Ä–∞–Ω–µ–Ω¬≠–∏–µ —á–∏—Å–ª–∞ –≤–∏—Ö—Ä–µ–π –≤ –ø–µ–ª–µ–Ω–µ
    outfile << (size) << endl;
    for (size_t i = 0; i < (size); ++i) {
        outfile<<(int)(i)<<" "<<(double)(Psp.eps)<<" "<<(double)(POS[i].r[0])<<" "<<(double)(POS[i].r[1])<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<(double)(POS[i].g)<<endl;
//      outfile<<(double)(d[i])<<" "<<(double)(POS[i].r[0])<<" "<<(double)(POS[i].r[1])<<" "<<(double)(POS[i].g)<<endl;
        // –Ω—É–ª–∏ –ø–∏—à—É—Ç—Å—è –¥–ª—è —Å–æ–≤–º–µ—Å—Ç–∏–º–æ¬≠—Å—Ç–∏ —Å —Ç—Ä–µ—Ö–º–µ—Ä–Ω–æ–π¬≠ –ø—Ä–æ–≥—Ä–∞–º–º–æ–π¬≠ –∏ –æ–±—Ä–∞–±–æ—Ç—á–∏–∫¬≠–∞–º–∏ ConMDV, ConMDV-p –∏ Construct
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

