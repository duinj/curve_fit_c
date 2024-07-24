/*
 * ©JD July 24
 */
#define PI 3.141592654
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include <float.h>
#include <stdio.h>

#define MAX_ITER 100
#define EPSILON 5e-2 //convergence criterium is pretty moderate, for 3 points not too bad 
#define PARAM_A_ESTIMATE_MULTIPLIER 3
#define PARAM_gamma_ESTIMATE        1

typedef struct {
    size_t n;
    double *y;
    double *x_values;
} data_to_fit;

typedef struct {
    double A;
    double x0;
    double gamma;
} CauchyParams;

//laurentzian function, gaussian could also be implemented
void cauchy_function_own(const CauchyParams *params, const data_to_fit *d, double *f) {
    double A = params->A;
    double x0 = params->x0;
    double gamma = params->gamma;

    for (size_t i = 0; i < d->n; i++) {
        double term = d->x_values[i] - x0;
        double Yi = A / (PI * gamma * (1 + term * term / (gamma * gamma)));
        f[i] = Yi - d->y[i];
    }
}

void jacobian(const CauchyParams *params, const data_to_fit *d, double J[3][3]) {
    double A = params->A;
    double x0 = params->x0;
    double gamma = params->gamma;

    memset(J, 0, sizeof(double) * 3 * 3);
    //calculate partial derivates
    for (size_t i = 0; i < d->n; i++) {
        double term = d->x_values[i] - x0;
        double denom = PI * gamma * (1 + term * term / (gamma * gamma));
        double denom_sq = denom * denom;

        J[0][0] += 1 / (denom * denom);
        J[0][1] += 2 * A * term / (gamma * gamma * denom_sq);
        J[0][2] += A * (term * term - gamma * gamma) / (gamma * gamma * gamma * denom_sq);
        J[1][1] += 4 * A * A * term * term / (gamma * gamma * gamma * gamma * denom_sq * denom);
        J[2][2] += A * A * (term * term - gamma * gamma) * (term * term - gamma * gamma) / (gamma * gamma * gamma * gamma * gamma * gamma * denom_sq * denom);
    }
    //fill redundant matrix slots
    J[1][0] = J[0][1];
    J[2][0] = J[0][2];
    J[2][1] = J[1][2] = 2 * A * (J[0][2] / A);
}

void levenberg_marquardt(data_to_fit *d, CauchyParams *params) {
    double lambda = 0.1; 
    double *f = malloc(d->n * sizeof(double));
    double h[3];
    double J[3][3];
    double prev_chi_sq = DBL_MAX; 

    for (int iter = 0; iter < MAX_ITER; iter++) {
        cauchy_function_own(params, d, f);
        jacobian(params, d, J);

        double chi_sq = 0;
        //calculate chi_sq
        for (size_t i = 0; i < d->n; i++) {
            chi_sq += f[i] * f[i];
        }
        // Add lambda to diagonal
        for (int i = 0; i < 3; i++) {
            J[i][i] *= (1 + lambda); 
        }
        // Solve J * h = -f
        for (int i = 0; i < 3; i++) {
            h[i] = 0;
            for (size_t j = 0; j < d->n; j++) {
                h[i] -= f[j] * (i == 0 ? 1 / J[0][0] : (i == 1 ? (d->x_values[j] - params->x0) / (params->gamma * params->gamma) : (d->x_values[j] - params->x0) * (d->x_values[j] - params->x0) / (params->gamma * params->gamma * params->gamma) - 1 / params->gamma));
            }
        }

        CauchyParams new_params = *params;
        new_params.A += h[0];
        new_params.x0 += h[1];
        new_params.gamma += h[2];
        // Check for parameter validity, vanish 
        if (new_params.A <= 0 || new_params.gamma <= 0 || isnan(new_params.A) || isnan(new_params.x0) || isnan(new_params.gamma)) {
            lambda *= 10;
            continue;
        }

        cauchy_function_own(&new_params, d, f);
        double new_chi_sq = 0;
        for (size_t i = 0; i < d->n; i++) {
            new_chi_sq += f[i] * f[i];
        }
        if (new_chi_sq < chi_sq) {
            *params = new_params;
            //some literature I found uses absolute and not relative criterium but this is the mosr robust afaik
            if (fabs(chi_sq - new_chi_sq) / chi_sq < EPSILON){
                break;
            }
            lambda *= 0.1;
        } else {
            lambda *= 10;
        }

        if (lambda > 1e10) {
            break; // Algorithm is not converging, exit
        }

        prev_chi_sq = chi_sq;
    }

    free(f);
}

static double estimate_x0_param_own(double *data, size_t len) {
    // Use only the middle three points
    size_t mid = len / 2;
    double left = data[mid - 1];
    double center = data[mid];
    double right = data[mid + 1];
    
    // This is the original basic computation of the x0 param
    // imprecise but still pretty good.
    return -((left - right) / center);
}


double cauchy_position(int *data_t, size_t len) {
    double *x = malloc(len * sizeof(double));
    double *y = malloc(len * sizeof(double));

    for (size_t i = 0; i < len; i++) {
        x[i] = 2.0 * i / (len - 1) - 1;  // Map to [-1, 1]
        y[i] = (double)data_t[i];
    }

    data_to_fit d = {len, y, x};

    CauchyParams params;
    //make guess
    //A guess = mid point (M) * a multiplier, expecting the peak point of the laurentzian to be much higher. This is only inefficient if the mid point is very close to the peak, otherwise this saves lots of iterations
    //x0 guess = relation between outer points and M point 
    //gamma is just 1
    params.A = y[len/2] * PARAM_A_ESTIMATE_MULTIPLIER;
    params.x0 = estimate_x0_param_own(y, len);
    params.gamma = PARAM_gamma_ESTIMATE;

    levenberg_marquardt(&d, &params);

    if (isnan(params.x0)) {
        params.x0 = estimate_x0_param_own(y, len);
    }

    free(x);
    free(y);

    return params.x0;
}

int main(){
  //dummy data, this is the structure that should be used, 3 points where the B the highest
  //structure could be thought of like this: -1 : A, 0 : B, 1 : C
  /*
                      .-.
                     /   \
                    /     \
                  B/       \
                  /         \
                 /           \C
                /             \
               /               \
              /                 \
             /                   \
            /                     \
          A/                       \
          /                         \
         /                           \
        /                             \
      ./               |               \.
    .´                 |                `.
..´                    |                  `..
calculates            x0
in this example the x0 should be above 0 (below if data[0] was higher than data[2])
if B and C are basically mirror points the value is expected to be close to 0.5 (right in the middle between B and C)
  */
    uint8_t size = 3;
    int data[3] = {119, 220, 170};

    //size = 5;
    //int data[5] = {70, 119, 220, 170, 112};


    printf("x0 result from fit: %.2f\n", cauchy_position(data, size));

    return 0;
}

