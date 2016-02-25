/*
 ============================================================================
 Name        : kernel.cu
 Author      : Sergey Grechkin-Pogrebnyakov
 Version     : Mar. 04, 2015
 Copyright   : All rights reserved
 Description : kernel file of vortex project
 ============================================================================
 */

#include "kernel.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"
#include <assert.h>

extern __constant__ TVars dt;
extern __constant__ size_t quant;
extern __constant__ float ve_size;
extern __constant__ float ve_size2;
extern __constant__ TVars r_col_same_sign2;
extern __constant__ TVars r_col_diff_sign2;
extern __constant__ TVars max_ve_g;
extern __constant__ size_t n_of_points;
extern __constant__ TVars x_max;
extern __constant__ TVars x_min;
extern __constant__ TVars y_max;
extern __constant__ TVars y_min;
extern __constant__ TVars h_col_x;
extern __constant__ TVars h_col_y;
extern __constant__ TVars rho;
extern __constant__ TVars rc_x;
extern __constant__ TVars rc_y;

#define SET_DEFAULT_ARR_VAL( __index ) \
    arr[__index + 0] = x_max; \
    arr[__index + 1] = x_min; \
    arr[__index + 2] = y_max; \
    arr[__index + 3] = y_min

#define LEFT_AND_RIGHT_FUNC( _left_, _right_, _func_ ) \
    _left_ = _func_( _left_, _right_ )

#define LEFT_AND_RIGHT_ARRAY_FUNC( __left__, __right__, __func__ ) \
    LEFT_AND_RIGHT_FUNC( arr[ __left__ ], arr[ __right__ ], __func__ )

#define MAX_AND_MIN_STEP( __left, __right ) \
    LEFT_AND_RIGHT_ARRAY_FUNC( __left + 0, __right + 0, fminf ); \
    LEFT_AND_RIGHT_ARRAY_FUNC( __left + 1, __right + 1, fmaxf ); \
    LEFT_AND_RIGHT_ARRAY_FUNC( __left + 2, __right + 2, fminf ); \
    LEFT_AND_RIGHT_ARRAY_FUNC( __left + 3, __right + 3, fmaxf )

#define FIND_RANGE_REDUCE_STEP(_max_tid_ ) \
    if( block_size >= 2 * _max_tid_ ) { \
        if( tid < _max_tid_ ) { \
            MAX_AND_MIN_STEP( tid * 4, ( tid + _max_tid_ ) * 4 ); \
            __syncthreads(); \
        } \
    }

#define TREE_REDUCE_REDUCE_STEP(_max_tid_ ) \
    if( block_size >= 2 * _max_tid_ ) { \
        if( tid < _max_tid_ ) { \
            for( int i = 0; i < branch_count; ++i ) { \
                MAX_AND_MIN_STEP( tid * size + 4 * i, ( tid + _max_tid_ ) * size + 4 * i ); \
                __syncthreads(); \
            } \
        } \
    }

template <unsigned int block_size>
__global__ void first_find_range_Kernel( Vortex *pos, unsigned int s, node_t *tree ) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * ( block_size * 2 ) + tid;
    unsigned int grid_size = block_size * 2 * gridDim.x;
    __shared__ float arr[block_size * 4];
    SET_DEFAULT_ARR_VAL( 4 * tid );
    while( i < s ) {
        float x_1 = (float)pos[i].r[0];
        float y_1 = (float)pos[i].r[1];
        pos[i].tree_id = 0;
        float x_2 = (i + block_size) < s ? (float)pos[i + block_size].r[0] : x_1;
        float y_2 = (i + block_size) < s ? (float)pos[i + block_size].r[1] : y_1;
        if( i + block_size < s )
            pos[i + block_size].tree_id = 0;
        // x_min
        arr[tid * 4 + 0] = fminf( fminf( x_1, x_2 ), arr[tid * 4 + 0] );
        // x_max
        arr[tid * 4 + 1] = fmaxf( fmaxf( x_1, x_2 ), arr[tid * 4 + 1] );
        // y_min
        arr[tid * 4 + 2] = fminf( fminf( y_1, y_2 ), arr[tid * 4 + 2] );
        // y_max
        arr[tid * 4 + 3] = fmaxf( fmaxf( y_1, y_2 ), arr[tid * 4 + 3] );
        i += grid_size;
    }
    __syncthreads();

    FIND_RANGE_REDUCE_STEP( 256 );
    FIND_RANGE_REDUCE_STEP( 128 );
    FIND_RANGE_REDUCE_STEP(  64 );
    FIND_RANGE_REDUCE_STEP(  32 );
    FIND_RANGE_REDUCE_STEP(  16 );
    FIND_RANGE_REDUCE_STEP(   8 );
    FIND_RANGE_REDUCE_STEP(   4 );
    FIND_RANGE_REDUCE_STEP(   2 );
    FIND_RANGE_REDUCE_STEP(   1 );

    if( 0 == tid ) {
        float xx_min = arr[0], xx_max = arr[1];
        float yy_min = arr[2], yy_max = arr[3];
        tree[blockIdx.x].x_min = xx_min;
        tree[blockIdx.x].x_max = xx_max;
        tree[blockIdx.x].y_min = yy_min;
        tree[blockIdx.x].y_max = yy_max;
        if( xx_max - xx_min > yy_max - yy_min ) {
            tree[blockIdx.x].med = (xx_max + xx_min) / 2.0;
            tree[blockIdx.x].axe = 0;
        } else {
            tree[blockIdx.x].med = (yy_max + yy_min) / 2.0;
            tree[blockIdx.x].axe = 1;
        }
    }
}

template
__global__ void first_find_range_Kernel<BLOCK_SIZE>( Vortex *pos, unsigned int s, node_t *tree );

template <unsigned int block_size>
__global__ void second_find_range_Kernel( node_t *input, unsigned int s, node_t *tree ) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * ( block_size * 2 ) + tid;
    unsigned int grid_size = block_size * 2 * gridDim.x;
    __shared__ float arr[ block_size * 4 ];
    SET_DEFAULT_ARR_VAL(tid * 4);
    while( i < s ) {
        float x_min_1 = input[i].x_min;
        float x_max_1 = input[i].x_max;
        float y_min_1 = input[i].y_min;
        float y_max_1 = input[i].y_max;
        float x_min_2 = input[i + block_size].x_min;
        float x_max_2 = input[i + block_size].x_max;
        float y_min_2 = input[i + block_size].y_min;
        float y_max_2 = input[i + block_size].y_max;
        // x_min
        arr[tid * 4 + 0] = fminf( fminf( x_min_1, x_min_2 ), arr[tid * 4 + 0] );
        // x_max
        arr[tid * 4 + 1] = fmaxf( fmaxf( x_max_1, x_max_2 ), arr[tid * 4 + 1] );
        // y_min
        arr[tid * 4 + 2] = fminf( fminf( y_min_1, y_min_2 ), arr[tid * 4 + 2] );
        // y_max
        arr[tid * 4 + 3] = fmaxf( fmaxf( y_max_1, y_max_2 ), arr[tid * 4 + 3] );
        i += grid_size;
    }
    __syncthreads();

    FIND_RANGE_REDUCE_STEP( 256 );
    FIND_RANGE_REDUCE_STEP( 128 );
    FIND_RANGE_REDUCE_STEP(  64 );
    FIND_RANGE_REDUCE_STEP(  32 );
    FIND_RANGE_REDUCE_STEP(  16 );
    FIND_RANGE_REDUCE_STEP(   8 );
    FIND_RANGE_REDUCE_STEP(   4 );
    FIND_RANGE_REDUCE_STEP(   2 );
    FIND_RANGE_REDUCE_STEP(   1 );

    if( 0 == tid ) {
        float xx_min = arr[0], xx_max = arr[1];
        float yy_min = arr[2], yy_max = arr[3];
        tree[blockIdx.x].x_min = xx_min;
        tree[blockIdx.x].x_max = xx_max;
        tree[blockIdx.x].y_min = yy_min;
        tree[blockIdx.x].y_max = yy_max;
        if( xx_max - xx_min > yy_max - yy_min ) {
            tree[blockIdx.x].med = (xx_max + xx_min) / 2.0;
            tree[blockIdx.x].axe = 0;
        } else {
            tree[blockIdx.x].med = (yy_max + yy_min) / 2.0;
            tree[blockIdx.x].axe = 1;
        }
    }
}

template
__global__ void second_find_range_Kernel<BLOCK_SIZE>( node_t *input, unsigned int s, node_t *tree );

template <size_t block_size, size_t level>
__global__ void first_tree_reduce_Kernel( Vortex *pos, unsigned int s, node_t *tree, node_t *output ) {
    unsigned int tid = threadIdx.x;
    unsigned int ii = blockIdx.x * ( block_size * 2 ) + tid;
    unsigned int grid_size = block_size * 2 * gridDim.x;
    const unsigned int size = 4 << level;
    __shared__ float arr[size * block_size];


    const unsigned int branch_count = 1 << level;

    float medians[ branch_count ];
    uint8_t axe[ branch_count ];

    for( unsigned int j = 0; j < branch_count; ++j ) {
        medians[j] = tree[j].med;
        axe[j] = tree[j].axe;
        SET_DEFAULT_ARR_VAL(size * tid + 4 * j);
    }

    while( ii < s ) {
        float x_1 = (float)pos[ii].r[0];
        float y_1 = (float)pos[ii].r[1];
        unsigned int tree_id_1 = pos[ii].tree_id;
        tree_id_1 = pos[ii].tree_id = ( x_1 > medians[tree_id_1] ) * ( ( axe[tree_id_1] + 1) % 2 ) + ( y_1 > medians[tree_id_1] ) *  axe[tree_id_1] + 2 * tree_id_1;
        // x_min
        LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_1 + 0], x_1, fminf );
        // x_max
        LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_1 + 1], x_1, fmaxf );
        // y_min
        LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_1 + 2], y_1, fminf );
        // y_max
        LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_1 + 3], y_1, fmaxf );

        if( ii + block_size < s ) {
            float x_2 = (float)pos[ii + block_size].r[0];
            float y_2 = (float)pos[ii + block_size].r[1];
            unsigned int tree_id_2 = pos[ii + block_size].tree_id;
            tree_id_2 = pos[ii + block_size].tree_id = ( x_2 > medians[tree_id_2] ) * ( ( axe[tree_id_2] + 1) % 2 ) + ( y_2 > medians[tree_id_2] ) *  axe[tree_id_2] + 2 * tree_id_2;
            // x_min
            LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_2 + 0], x_2, fminf );
            // x_max
            LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_2 + 1], x_2, fmaxf );
            // y_min
            LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_2 + 2], y_2, fminf );
            // y_max
            LEFT_AND_RIGHT_FUNC( arr[size * tid + 4 * tree_id_2 + 3], y_2, fmaxf );
        }
        ii += grid_size;
    }
    __syncthreads();

    TREE_REDUCE_REDUCE_STEP( 256 );
    TREE_REDUCE_REDUCE_STEP( 128 );
    TREE_REDUCE_REDUCE_STEP(  64 );
    TREE_REDUCE_REDUCE_STEP(  32 );
    TREE_REDUCE_REDUCE_STEP(  16 );
    TREE_REDUCE_REDUCE_STEP(   8 );
    TREE_REDUCE_REDUCE_STEP(   4 );
    TREE_REDUCE_REDUCE_STEP(   2 );
    TREE_REDUCE_REDUCE_STEP(   1 );

    if( 0 == tid ) {
        for( int i = 0; i < branch_count; ++i ) {
            float xx_min = arr[0 + 4 * i], xx_max = arr[1 + 4 * i];
            float yy_min = arr[2 + 4 * i], yy_max = arr[3 + 4 * i];
            output[branch_count * blockIdx.x + i].x_min = xx_min;
            output[blockIdx.x * branch_count + i].x_max = xx_max;
            output[blockIdx.x * branch_count + i].y_min = yy_min;
            output[blockIdx.x * branch_count + i].y_max = yy_max;
        }
    }
}

template __global__ void first_tree_reduce_Kernel<BLOCK_SIZE, 1>( Vortex *pos, unsigned int s, node_t *tree, node_t *output );
template __global__ void first_tree_reduce_Kernel<BLOCK_SIZE, 2>( Vortex *pos, unsigned int s, node_t *tree, node_t *output );
template __global__ void first_tree_reduce_Kernel<BLOCK_SIZE, 3>( Vortex *pos, unsigned int s, node_t *tree, node_t *output );
template __global__ void first_tree_reduce_Kernel<BLOCK_SIZE, 4>( Vortex *pos, unsigned int s, node_t *tree, node_t *output );
template __global__ void first_tree_reduce_Kernel<BLOCK_SIZE, 5>( Vortex *pos, unsigned int s, node_t *tree, node_t *output );
template __global__ void first_tree_reduce_Kernel<BLOCK_SIZE, 6>( Vortex *pos, unsigned int s, node_t *tree, node_t *output );

template <size_t block_size, size_t level>
__global__ void second_tree_reduce_Kernel( node_t *input, unsigned int s, node_t *output ) {
    unsigned int tid = threadIdx.x;
    unsigned int ii = blockIdx.x * ( block_size * 2 ) + tid;
    unsigned int grid_size = block_size * 2 * gridDim.x;
    const unsigned int size = 4 << level;

    __shared__ float arr[size * block_size];

    const unsigned int branch_count = 1 << level;

    for( unsigned int i = 0; i < branch_count; ++i ) {
        SET_DEFAULT_ARR_VAL(size * tid + 4 * i);
    }

    while( ii < s ) {
        for( unsigned int j = 0; j < branch_count; ++j ) {
            float x_min_1 = input[ii * branch_count + j].x_min;
            float x_max_1 = input[ii * branch_count + j].x_max;
            float y_min_1 = input[ii * branch_count + j].y_min;
            float y_max_1 = input[ii * branch_count + j].y_max;
            assert((ii + block_size) < s);
            float x_min_2 = input[(ii + block_size) * branch_count + j].x_min;
            float x_max_2 = input[(ii + block_size) * branch_count + j].x_max;
            float y_min_2 = input[(ii + block_size) * branch_count + j].y_min;
            float y_max_2 = input[(ii + block_size) * branch_count + j].y_max;
            // x_min
            arr[size * tid + 4 * j + 0] = fminf( arr[size * tid + 4 * j + 0], fminf( x_min_1, x_min_2 ) );
            // x_max
            arr[size * tid + 4 * j + 1] = fmaxf( arr[size * tid + 4 * j + 1], fmaxf( x_max_1, x_max_2 ) );
            // y_min
            arr[size * tid + 4 * j + 2] = fminf( arr[size + tid + 4 * j + 2], fminf( y_min_1, y_min_2 ) );
            // y_max
            arr[size * tid + 4 * j + 3] = fmaxf( arr[size + tid + 4 * j + 3], fmaxf( y_max_1, y_max_2 ) );
        }
        ii += grid_size;
    }
    __syncthreads();

    TREE_REDUCE_REDUCE_STEP( 256 );
    TREE_REDUCE_REDUCE_STEP( 128 );
    TREE_REDUCE_REDUCE_STEP(  64 );
    TREE_REDUCE_REDUCE_STEP(  32 );
    TREE_REDUCE_REDUCE_STEP(  16 );
    TREE_REDUCE_REDUCE_STEP(   8 );
    TREE_REDUCE_REDUCE_STEP(   4 );
    TREE_REDUCE_REDUCE_STEP(   2 );
    TREE_REDUCE_REDUCE_STEP(   1 );

    if( 0 == tid ) {
        for( int i = 0; i < branch_count; ++i ) {
            float xx_min = arr[0 + 4 * i], xx_max = arr[1 + 4 * i];
            float yy_min = arr[2 + 4 * i], yy_max = arr[3 + 4 * i];
            output[branch_count * blockIdx.x + i].x_min = xx_min;
            output[blockIdx.x * branch_count + i].x_max = xx_max;
            output[blockIdx.x * branch_count + i].y_min = yy_min;
            output[blockIdx.x * branch_count + i].y_max = yy_max;
            if( xx_max - xx_min > yy_max - yy_min ) {
                output[blockIdx.x * branch_count + i].med = (xx_max + xx_min) / 2.0;
                output[blockIdx.x * branch_count + i].axe = 0;
            } else {
                output[blockIdx.x * branch_count + i].med = (yy_max + yy_min) / 2.0;
                output[blockIdx.x * branch_count + i].axe = 1;
            }
        }
    }
}

template __global__ void second_tree_reduce_Kernel<BLOCK_SIZE, 1>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_tree_reduce_Kernel<BLOCK_SIZE, 2>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_tree_reduce_Kernel<BLOCK_SIZE, 3>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_tree_reduce_Kernel<BLOCK_SIZE, 4>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_tree_reduce_Kernel<BLOCK_SIZE, 5>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_tree_reduce_Kernel<BLOCK_SIZE, 6>( node_t *input, unsigned int s, node_t *output );

template <size_t block_size, size_t level>
__global__ void first_find_leaves_params_Kernel( Vortex *pos, unsigned int s, node_t *output ) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * ( block_size * 2 ) + tid;
    unsigned int grid_size = block_size * 2 * gridDim.x;
    const unsigned int size = 6 << level;
    __shared__ float arr[size * block_size];


    const unsigned int branch_count = 1 << level;

    for( unsigned int i = 0; i < branch_count; ++i ) {
        arr[size * tid + 6 * i + 0] = 0;
        arr[size * tid + 6 * i + 1] = 0;
        arr[size * tid + 6 * i + 2] = 0;
        arr[size * tid + 6 * i + 3] = 0;
        arr[size * tid + 6 * i + 4] = 0;
        arr[size * tid + 6 * i + 5] = 0;
    }

    while( i < s ) {
        float x_1 = (float)pos[i].r[0];
        float y_1 = (float)pos[i].r[1];
        float g_1 = (float)pos[i].g;
        unsigned int tree_id_1 = pos[i].tree_id;
        if( g_1 > 0 ) {
            // g_1_above
            arr[size * tid + 6 * tree_id_1 + 0] += g_1;
            // xg_1_above
            arr[size * tid + 6 * tree_id_1 + 1] += x_1 * g_1;
            // yg_1_above
            arr[size * tid + 6 * tree_id_1 + 2] += y_1 * g_1;
        } else {
            // g_1_below
            arr[size * tid + 6 * tree_id_1 + 3] += g_1;
            // xg_1_below
            arr[size * tid + 6 * tree_id_1 + 4] += x_1 * g_1;
            // yg_1_below
            arr[size * tid + 6 * tree_id_1 + 5] += y_1 * g_1;
        }
        if( i + block_size < s ) {
            float x_2 = (float)pos[i + block_size].r[0];
            float y_2 = (float)pos[i + block_size].r[1];
            float g_2 = (float)pos[i + block_size].tree_id;
            unsigned int tree_id_2 = pos[i + block_size].tree_id;
            if( g_2 > 0 ) {
                // g_1_above
                arr[size * tid + 6 * tree_id_2 + 0] += g_2;
                // xg_1_above
                arr[size * tid + 6 * tree_id_2 + 1] += x_2 * g_2;
                // yg_1_above
                arr[size * tid + 6 * tree_id_2 + 2] += y_2 * g_2;
            } else {
                // g_1_below
                arr[size * tid + 6 * tree_id_2 + 3] += g_2;
                // xg_1_below
                arr[size * tid + 6 * tree_id_2 + 4] += x_1 * g_2;
                // yg_1_below
                arr[size * tid + 6 * tree_id_2 + 5] += y_1 * g_2;
            }
        }
        i += grid_size;
    }
    __syncthreads();

    if( block_size >= 512 ) {
        if( tid < 256 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 256 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 256 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 256 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 256 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 256 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 256 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 256 ) {
        if( tid < 128 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 128 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 128 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 128 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 128 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 128 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 128 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 128 ) {
        if( tid < 64 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 64 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 64 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 64 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 64 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 64 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 64 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 64 ) {
        if( tid < 32 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 32 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 32 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 32 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 32 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 32 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 32 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 32 ) {
        if( tid < 16 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 16 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 16 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 16 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 16 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 16 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 16 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 16 ) {
        if( tid < 8 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 8 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 8 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 8 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 8 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 8 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 8 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 8 ) {
        if( tid < 4 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 4 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 4 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 4 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 4 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 4 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 4 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 4 ) {
        if( tid < 2 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 2 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 2 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 2 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 2 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 2 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 2 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 2 ) {
        if( tid < 1 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 1 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 1 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 1 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 1 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 1 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 1 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( 0 == tid ) {
        for( int i = 0; i < branch_count; ++i ) {
            output[branch_count * blockIdx.x + i].g_above = arr[6 * i + 0];
            output[blockIdx.x * branch_count + i].xg_above = arr[6 * i + 1];
            output[blockIdx.x * branch_count + i].yg_above = arr[6 * i + 2];
            output[branch_count * blockIdx.x + i].g_below = arr[6 * i + 3];
            output[blockIdx.x * branch_count + i].xg_below = arr[6 * i + 4];
            output[blockIdx.x * branch_count + i].yg_below = arr[6 * i + 5];
        }
    }
}

template __global__ void first_find_leaves_params_Kernel<BLOCK_SIZE, 1>( Vortex *pos, unsigned int s, node_t *output );
template __global__ void first_find_leaves_params_Kernel<BLOCK_SIZE, 2>( Vortex *pos, unsigned int s, node_t *output );
template __global__ void first_find_leaves_params_Kernel<BLOCK_SIZE, 3>( Vortex *pos, unsigned int s, node_t *output );
template __global__ void first_find_leaves_params_Kernel<BLOCK_SIZE, 4>( Vortex *pos, unsigned int s, node_t *output );
template __global__ void first_find_leaves_params_Kernel<BLOCK_SIZE, 5>( Vortex *pos, unsigned int s, node_t *output );
template __global__ void first_find_leaves_params_Kernel<BLOCK_SIZE, 6>( Vortex *pos, unsigned int s, node_t *output );

template <size_t block_size, size_t level>
__global__ void second_find_leaves_params_Kernel( node_t *input, unsigned int s, node_t *output ) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * ( block_size * 2 ) + tid;
    unsigned int grid_size = block_size * 2 * gridDim.x;
    const unsigned int size = 6 << level;

    __shared__ float arr[size * block_size];

    const unsigned int branch_count = 1 << level;

    for( unsigned int i = 0; i < branch_count; ++i ) {
        arr[size * tid + 6 * i + 0] = 0;
        arr[size * tid + 6 * i + 1] = 0;
        arr[size * tid + 6 * i + 2] = 0;
        arr[size * tid + 6 * i + 3] = 0;
        arr[size * tid + 6 * i + 4] = 0;
        arr[size * tid + 6 * i + 5] = 0;
    }

    while( i < s ) {
        for( unsigned int j = 0; j < branch_count; ++j ) {
            float g_above_1 = input[i * branch_count + j].g_above;
            float xg_above_1 = input[i * branch_count + j].xg_above;
            float yg_above_1 = input[i * branch_count + j].yg_above;
            float g_below_1 = input[i * branch_count + j].g_below;
            float xg_below_1 = input[i * branch_count + j].xg_below;
            float yg_below_1 = input[i * branch_count + j].yg_below;
            float g_above_2 = input[(i + block_size) * branch_count + j].g_above;
            float xg_above_2 = input[(i + block_size) * branch_count + j].xg_above;
            float yg_above_2 = input[(i + block_size) * branch_count + j].yg_above;
            float g_below_2 = input[(i + block_size) * branch_count + j].g_below;
            float xg_below_2 = input[(i + block_size) * branch_count + j].xg_below;
            float yg_below_2 = input[(i + block_size) * branch_count + j].yg_below;
            // g_above
            arr[size * tid + 6 * j + 0] += g_above_1 + g_above_2;
            // xg_above
            arr[size * tid + 6 * j + 1] += xg_above_1 + xg_above_2;
            // yg_above
            arr[size * tid + 6 * j + 2] += yg_above_1 + yg_above_2;
            // g_below
            arr[size * tid + 6 * j + 3] += g_below_1 + g_below_2;
            // xg_below
            arr[size * tid + 6 * j + 4] += xg_below_1 + xg_below_2;
            // yg_above
            arr[size * tid + 6 * j + 5] += yg_below_1 + yg_below_2;
        }
        i += grid_size;
    }
    __syncthreads();

    if( block_size >= 512 ) {
        if( tid < 256 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 256 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 256 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 256 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 256 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 256 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 256 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 256 ) {
        if( tid < 128 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 128 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 128 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 128 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 128 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 128 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 128 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 128 ) {
        if( tid < 64 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 64 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 64 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 64 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 64 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 64 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 64 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 64 ) {
        if( tid < 32 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 32 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 32 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 32 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 32 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 32 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 32 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 32 ) {
        if( tid < 16 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 16 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 16 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 16 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 16 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 16 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 16 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 16 ) {
        if( tid < 8 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 8 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 8 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 8 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 8 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 8 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 8 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 8 ) {
        if( tid < 4 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 4 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 4 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 4 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 4 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 4 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 4 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 4 ) {
        if( tid < 2 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 2 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 2 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 2 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 2 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 2 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 2 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( block_size >= 2 ) {
        if( tid < 1 ) {
            for( int i = 0; i < branch_count; ++i ) {
                arr[tid * size + 6 * i + 0] += arr[( tid + 1 ) * size + 6 * i + 0];
                arr[tid * size + 6 * i + 1] += arr[( tid + 1 ) * size + 6 * i + 1];
                arr[tid * size + 6 * i + 2] += arr[( tid + 1 ) * size + 6 * i + 2];
                arr[tid * size + 6 * i + 3] += arr[( tid + 1 ) * size + 6 * i + 3];
                arr[tid * size + 6 * i + 4] += arr[( tid + 1 ) * size + 6 * i + 4];
                arr[tid * size + 6 * i + 5] += arr[( tid + 1 ) * size + 6 * i + 5];
            }
            __syncthreads();
        }
    }
    if( 0 == tid ) {
        for( int i = 0; i < branch_count; ++i ) {
            output[branch_count * blockIdx.x + i].g_above = arr[6 * i + 0];
            output[blockIdx.x * branch_count + i].xg_above = arr[6 * i + 1];
            output[blockIdx.x * branch_count + i].yg_above = arr[6 * i + 2];
            output[branch_count * blockIdx.x + i].g_below = arr[6 * i + 3];
            output[blockIdx.x * branch_count + i].xg_below = arr[6 * i + 4];
            output[blockIdx.x * branch_count + i].yg_below = arr[6 * i + 5];
        }
    }
}

template __global__ void second_find_leaves_params_Kernel<BLOCK_SIZE, 1>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_find_leaves_params_Kernel<BLOCK_SIZE, 2>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_find_leaves_params_Kernel<BLOCK_SIZE, 3>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_find_leaves_params_Kernel<BLOCK_SIZE, 4>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_find_leaves_params_Kernel<BLOCK_SIZE, 5>( node_t *input, unsigned int s, node_t *output );
template __global__ void second_find_leaves_params_Kernel<BLOCK_SIZE, 6>( node_t *input, unsigned int s, node_t *output );

template <size_t block_size, size_t level>
__global__ void find_tree_params_Kernel( node_t *tree ) {
    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * block_size + tid;

    for( int j = 0; j < level - 1; ++j )
        tree += (1 << j);

    for( int j = level - 1; j >= 0; --j ) {
        unsigned count_on_level = 1 << j;
        for( unsigned jj = 0; jj < count_on_level; jj += block_size ) {
            unsigned k = jj + i;
            if( k < count_on_level ) {
                tree[k].g_above = tree[2 * k + count_on_level].g_above + tree[2 * k + count_on_level + 1].g_above;
                tree[k].xg_above = tree[2 * k + count_on_level].xg_above + tree[2 * k + count_on_level + 1].xg_above;
                tree[k].yg_above = tree[2 * k + count_on_level].yg_above + tree[2 * k + count_on_level + 1].yg_above;
                tree[k].g_below = tree[2 * k + count_on_level].g_below + tree[2 * k + count_on_level + 1].g_below;
                tree[k].xg_below = tree[2 * k + count_on_level].xg_below + tree[2 * k + count_on_level + 1].xg_below;
                tree[k].yg_below = tree[2 * k + count_on_level].yg_below + tree[2 * k + count_on_level + 1].yg_below;
            }
        }
        tree -= (1 << (j - 1));
    }

}

template __global__ void find_tree_params_Kernel<BLOCK_SIZE, 1>( node_t *tree );
template __global__ void find_tree_params_Kernel<BLOCK_SIZE, 2>( node_t *tree );
template __global__ void find_tree_params_Kernel<BLOCK_SIZE, 3>( node_t *tree );
template __global__ void find_tree_params_Kernel<BLOCK_SIZE, 4>( node_t *tree );
template __global__ void find_tree_params_Kernel<BLOCK_SIZE, 5>( node_t *tree );
template __global__ void find_tree_params_Kernel<BLOCK_SIZE, 6>( node_t *tree );

__global__ void zero_Kernel( float *randoms, Vortex *pos, int s ) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    pos[s+ind].r[0]=(2.0e+5)*randoms[ind]+2.0e+5;
    pos[s+ind].r[1]=(2.0e+5)*randoms[ind]+2.0e+5;
    pos[s+ind].g = 0.0;
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
//    d_0 = 1e+5f;
    for (int f = 0 ; f < n ; f += BLOCK_SIZE) {
        b_sh_0[threadIdx.x] = (float)pos[threadIdx.x+f].r[0];
        b_sh_1[threadIdx.x] = (float)pos[threadIdx.x+f].r[1];
        g[threadIdx.x] = (float)pos[threadIdx.x+f].g;
        __syncthreads();
        for (int j = 0 ; j < BLOCK_SIZE ; ++j) {
            if( j + f == i )
                continue;
            dist2 = Ro2f(a0, a1, b_sh_0[j], b_sh_1[j]);
            // dist2 = (a0 - b_sh_0[j]) * (a0 - b_sh_0[j]) + (a1 - b_sh_1[j]) * (a1 - b_sh_1[j]);
            if (d_3 > dist2) {
                    d_3 = dist2;
                    dst = fminf(d_3, d_2);
                    d_3 = fmaxf(d_3, d_2);
                    d_2 = dst;
                    dst = fminf(d_1, d_2);
                    d_2 = fmaxf(d_1, d_2);
                    d_1 = dst;
  //                  dst = fminf(d_1, d_0);
  //                  d_1 = fmaxf(d_1, d_0);
  //                  d_0 = dst;
            }
	//		if (dist2 < ve_size2) dist2=ve_size2;
            dist2 = fmaxf(dist2, ve_size2);
            mnog = g[j] / dist2;
            y1 +=  mnog * (a0 - b_sh_0[j]);
            y0 += -mnog * (a1 - b_sh_1[j]);
        }//j
        __syncthreads();
    }//f
//    d[i] = sqrt(d_1 + d_2 + d_3) / 3;
    d[i] = (sqrtf(d_1) + sqrtf(d_2) + sqrtf(d_3)) / 3;
//    d[i] = max(d[i], 4.0 * ve_size / 3.0);
    V[i].v[0] = (TVars)( y0 / (2 * M_PI) + (*V_inf)[0] );
    V[i].v[1] = (TVars)( y1 / (2 * M_PI) + (*V_inf)[1] );
//    for( int k = 0; k < 2; ++k )
//      V[i].v[k] =  (*V_inf)[k];
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
            if(dist > 0.001 * ve_size) {
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
    for (int f = 0; f < quant; ++f) {

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
            I_0_I_3f(Ra_0, Ra_1, Rb_0, Rb_1, Norm_0, Norm_1, a0, a1, dL, dd, n_of_points, &RES_0, &RES_3_0, &RES_3_1);
            II_0 += (-dd) * RES_0;
            II_3_0 -= RES_3_0;
            II_3_1 -= RES_3_1;
        //  II_0 = 1;  II_3[0] = r ; II_3[1]=2;
        } else if (r <= 0.1 * dL) {
            Norm_0 = N_contr_xf(panels, f);
            Norm_1 = N_contr_yf(panels, f);
            II_0 = M_PI * dd * dd;
            II_3_0 = 2 * Norm_0 * dd * (1 - expf(-dL / (2 * dd)));
            II_3_1 = 2 * Norm_1 * dd * (1 - expf(-dL / (2 * dd)));
          //  II_0=1;II_3[0]=r;II_3[1]=0;
           // f = quant + 5;
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
		d_g_Dev[i] = 0.0;
        TVars d_g = 0.0;
        F_p[i].v[0] = 0.0;
        F_p[i].v[1] = 0.0;
        M[i] = 0.0;
        if (i >= n - quant) {
            F_p[i].v[0] = pos[i].g * (-pos[i].r[1]);
            F_p[i].v[1] = pos[i].g * ( pos[i].r[0]);
            M[i] = pos[i].g * Ro2(pos[i].r[0], pos[i].r[1], rc_x, rc_y);
        }
	    //__syncthreads;

		TVars r_new_0 = pos[i].r[0] + V[i].v[0] * dt;
        TVars r_new_1 = pos[i].r[1] + V[i].v[1] * dt;
//	    TVctr Zero  = {0, 0};
		int hitpan = 0;

	    if ( (pos[i].g != 0) && (hitting(panels, r_new_0, r_new_1, pos[i].r, &hitpan))) {
            F_p[i].v[0] -= pos[i].g * (-panels[hitpan].contr[1]);
            F_p[i].v[1] -= pos[i].g * ( panels[hitpan].contr[0]);
            M[i] -= pos[i].g * Ro2(panels[hitpan].contr[0], panels[hitpan].contr[1], rc_x, rc_y);
		    r_new_0 =  2e+5;
		    r_new_1 =  2e+5;
		    d_g = pos[i].g;
//printf( "d_g[%d] =  %lf \n", i, d_g];
//		    d_g_Dev[i] = pos[i].g;
		    pos[i].g = 0;
		}

		pos[i].r[0] = r_new_0;
		pos[i].r[1] = r_new_1;

	    if ((pos[i].g != 0) && ((pos[i].r[0] > x_max) || (pos[i].r[0] < x_min) || (pos[i].r[1] > y_max) || (pos[i].r[1] < y_min))) {
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
        (*F_p).v[0] *= rho / dt;
        (*F_p).v[1] *= rho / dt;
        (*M) *= rho / (2 * dt);
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
        int Setx_i = floor((r0+2)/h_col_x);
        int Sety_i = floor((r1+10)/h_col_y);
		Setx[i] = Setx_i;
		Sety[i] = Sety_i;
		COL[i] = -1;

		__syncthreads();

		//	for (int j = (i+1); j < n; j++ ) {
		for (int j = 0; j < n; ++j) {
			if ((abs(Setx_i - Setx[j]) < 2) && (abs(Sety_i - Sety[j]) < 2) && (g * pos[j].g > 0) &&
				(Ro2(r0, r1, pos[j].r[0], pos[j].r[1]) < r_col_same_sign2) && (j != i) && (fabs(g + pos[j].g) < max_ve_g)) {
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
        int Setx_i = floor((r0+2)/h_col_x);
        int Sety_i = floor((r1+10)/h_col_y);
        Setx[i] = Setx_i;
        Sety[i] = Sety_i;
        COL[i] = -1;

        __syncthreads();

        //  for (int j = (i+1); j < n; j++ ) {
        for (int j = 0; j < n; ++j) {
            if ((abs(Setx_i - Setx[j]) < 2) && (abs(Sety_i - Sety[j]) < 2) && (g * pos[j].g < 0) &&
                (Ro2(r0, r1, pos[j].r[0], pos[j].r[1]) < r_col_diff_sign2)) {
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
        //    if (fabs(new_g) < 2 * max_ve_g) {
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

inline __device__ void I_0_I_3(TVars Ra_0, TVars Ra_1, TVars Rb_0, TVars Rb_1, TVars Norm_0, TVars Norm_1, TVars Rj_0, TVars Rj_1,
                                TVars dL, TVars d, size_t N,TVars *RES_0, TVars *RES_3_0, TVars *RES_3_1) {
    TVars Rk_0 = 0.0, Rk_1 = 0.0;
    TVars Eta_0 = 0.0, Eta_1 = 0.0;
    TVars dR_0 = 0.0, dR_1 = 0.0;
    *RES_0 = 0.0;
    *RES_3_0 = 0.0;
    *RES_3_1 = 0.0;
    dR_0 = (Rb_0 - Ra_0) / N;
    dR_1 = (Rb_1 - Ra_1) / N;
    TVars delt = dL / N;
    for (size_t k = 0; k < N; ++k) {
        Rk_0 = (Ra_0 + k * dR_0 + Ra_0 + (k + 1) * dR_0) / 2;
        Rk_1 = (Ra_1 + k * dR_1 + Ra_1 + (k + 1) * dR_1) / 2;
        Eta_0 = (Rj_0 - Rk_0) / d;
        Eta_1 = (Rj_1 - Rk_1) / d;
        TVars mod_eta = sqrt(Eta_0 * Eta_0 + Eta_1 * Eta_1);
        *RES_0 += (Eta_0 * Norm_0 + Eta_1 * Norm_1) / (Eta_0 * Eta_0 + Eta_1 * Eta_1) * 
            (mod_eta + 1) * exp(-mod_eta) * delt;
        *RES_3_0 += Norm_0 * exp(-mod_eta) * delt;
        *RES_3_1 += Norm_1 * exp(-mod_eta) * delt;
    }
}

inline __device__ void I_0_I_3f(float Ra_0, float Ra_1, float Rb_0, float Rb_1, float Norm_0, float Norm_1, float Rj_0, float Rj_1,
                                float dL, float d, size_t N, float *RES_0, float *RES_3_0, float *RES_3_1) {
    float Rk_0 = 0.0f, Rk_1 = 0.0f;
    float Eta_0 = 0.0f, Eta_1 = 0.0f;
    float dR_0 = 0.0f, dR_1 = 0.0f;
    *RES_0 = 0.0f;
    *RES_3_0 = 0.0f;
    *RES_3_1 = 0.0f;
    dR_0 = (Rb_0 - Ra_0) / N;
    dR_1 = (Rb_1 - Ra_1) / N;
    float delt = dL / N;
    for (size_t k = 0; k < N; ++k) {
        Rk_0 = (Ra_0 + k * dR_0 + Ra_0 + (k + 1) * dR_0) / 2;
        Rk_1 = (Ra_1 + k * dR_1 + Ra_1 + (k + 1) * dR_1) / 2;
        Eta_0 = (Rj_0 - Rk_0) / d;
        Eta_1 = (Rj_1 - Rk_1) / d;
        TVars mod_eta = sqrtf(Eta_0 * Eta_0 + Eta_1 * Eta_1);
        *RES_0 += (Eta_0 * Norm_0 + Eta_1 * Norm_1) / (Eta_0 * Eta_0 + Eta_1 * Eta_1) * 
            (mod_eta + 1) * expf(-mod_eta) * delt;
        *RES_3_0 += Norm_0 * expf(-mod_eta) * delt;
        *RES_3_1 += Norm_1 * expf(-mod_eta) * delt;
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
        for(int i=0; i<quant; ++i)
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
        float y0 = 0.0f, y1 = 0.0f;
//        TVars dist2;
        float mnog = 0.0f;
        float dist2 = 0.0f;
// координаты расчётной точки
        float a0 = 0.0f, a1 = 0.0f;
// координаты воздействующей точки
        __shared__ float b_sh_0 [BLOCK_SIZE];
        __shared__ float b_sh_1 [BLOCK_SIZE];
// интенсивность воздействующей точки
        __shared__ float g [BLOCK_SIZE];
        a0 = (float)Contr_points[i].v[0];
        a1 = (float)Contr_points[i].v[1];
        for (int f = 0 ; f < n ; f += BLOCK_SIZE) {
            b_sh_0[threadIdx.x] = (float)pos[threadIdx.x+f].r[0];
            b_sh_1[threadIdx.x] = (float)pos[threadIdx.x+f].r[1];
            g[threadIdx.x] = (float)pos[threadIdx.x+f].g;

            __syncthreads();

            for (int j = 0 ; j < BLOCK_SIZE ; ++j) {
                dist2 = Ro2f(a0, a1, b_sh_0[j], b_sh_1[j]);
//                if (dist2 < ve_size2) dist2=ve_size2;
                dist2 = fmaxf(dist2, ve_size2);
                mnog = g[j] / dist2;
                y1 +=  mnog * (a0 - b_sh_0[j]);
                y0 += -mnog * (a1 - b_sh_1[j]);
            }//j
            __syncthreads();
        }//f
        V[i + (*n_v)].v[0] = ( (TVars)y0 )/(2*M_PI) + (*V_inf)[0];
        V[i + (*n_v)].v[1] = ( (TVars)y1 )/(2*M_PI) + (*V_inf)[1];
//        V[i].v[k] =  (*V_inf)[k];
        __syncthreads();
        if (i == 1) {
            (*n_v) += 500;
        }
    }
}
