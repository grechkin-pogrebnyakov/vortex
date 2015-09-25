/*
 ============================================================================
 Name        : main.h
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Feb. 22, 2015
 Copyright   : All rights reserved
 Description : main header of vortex project
 ============================================================================
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "incl.h"

void flush_log();

void log_lev( uint8_t lev, char *fmt, ... );

inline uint8_t cusafe( cudaError_t cuerr, const char *file, int line ) {
    if( cuerr != cudaSuccess ) {
        log_lev_file( LEV_ERROR, file, line, "%s", cudaGetErrorString(cuerr) );
        return 1;
    }
    return 0;
}

#endif // MAIN_H_
