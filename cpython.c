#include <stdlib.h>
#include <math.h>
#include <stdio.h>

const double EULER_E = 2.718281828459045;
const double LEARNING_RATE = 0.1;
const double SIGMA = 1;
const double PI = 3.14159265358979323846;

double logistic_activation(double* tokens, double* weights, int n) {
    double linear_term = 0;
    // calculate the linear term
    for (int i = 0; i < n; i++) {
        //printf("token: %f ",tokens[i]);
        linear_term += (tokens[i]) * weights[i];
        //printf("weight_term: %f, linear_term: %f\n", weights[i], linear_term);
    }

    

    double negative_lin_term = (-1) * linear_term;
    
    // calculate and return logistic 
    return (1)/(1 + pow(EULER_E, negative_lin_term));

}


double logistic_derivative(double activated) {
    // return derivative calculation
    return (activated) * (1 - activated);
}



// A LOT OF THESE NEED TO BE FIXED
double residual_error(int y, double activated) {
    // return residual error of the system
    return y - activated;
}



double back_propagation_term(double residual_error, double activated_derivative) {
    // return the back propogation term of this observation
    return residual_error * activated_derivative;
}



double get_gaussian_num() {
    double u1 = (double)rand()/(double)RAND_MAX;
    double u2 = (double)rand()/(double)RAND_MAX;

    while ( u1 == 0.0) {
        u1 = (double)rand()/(double)RAND_MAX;
    }
    while ( u2 == 0.0) {
        u2 = (double)rand()/(double)RAND_MAX;
    }


    // create the single standard muller distribution stuff, divide by sigma to get the number
    return (sqrt((-2) * log(u1)) * cos(2 * PI * u2))*SIGMA;
}


double* get_gaussian_weight_vector(double* tokens, int dimensions) {
    // for our purposes, int d will be desired number of dimensions here
    // dimensions will be the size of tokens
    // create the weight vector
    double* g_weight_vec = malloc(dimensions * sizeof(double));
    if (!g_weight_vec) {
        return NULL;
    }

    for (int i = 0; i < dimensions; i++) {
        // fill the temporary weight vector
        g_weight_vec[i] = (get_gaussian_num()) * tokens[i];
    }

    return g_weight_vec;
}

// exactly as it sounds
double vector_mult(double* w, double* x, int n) {
    // simple vector multiplication
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += w[i] * x[i];
    }

    return result;
}


// simple normalization function
void normalize_vector(double* vector, int n) {
    double sum_squares = 0;
    for (int i = 0; i < n; i++) {
        sum_squares += vector[i] * vector[i];
    }

    double magnitude = sqrt(sum_squares);

    for (int i = 0; i < n; i ++) {
        vector[i] = vector[i]/magnitude;
    }
}



// the function below is too be utilized for the initial training stages, up until enough data is 
// availible for a proper RFF training function
double* token_early_RFF_embedding(double* tokens, int current_dims, int desired_dims) {
    normalize_vector(tokens, current_dims);
    if (current_dims > desired_dims) {
        return NULL;
    }
    //desired_dims is the size of the existing/permenant weight vectors - 1, ex, w= 500, desired_dims = 499
    // +1 is left over to be added to augment the token vector to 500 dimensions for the bias scalar
    double* embedded_tokens = malloc((desired_dims) * sizeof(double));
    if (!embedded_tokens) {
        return NULL;
    }    

    // create a shallow copy of tokens
    for (int i = 0; i < current_dims; i++) {
        embedded_tokens[i] = tokens[i];
    }


    // if current == 400. current dims is length, so tokens[399] is the last one
    // desired == 500, so last index is 499. current_dims + i goes too index 498
    // i = 0,1,2...,98. 
    for (int i = current_dims; i < desired_dims - 1; i++) {
        // create weight vector for RFF associated with gaussian distribution
        // this is on a single dimension, and is a simplified RFF embedding on a vector.
        double* g_weight_vec = get_gaussian_weight_vector(tokens, current_dims);
        embedded_tokens[i] = vector_mult(g_weight_vec, tokens, current_dims);
        free(g_weight_vec);
    }

    embedded_tokens[desired_dims - 1] = 1;
    return embedded_tokens;
}



double* predict_weight_vector(double* tokens, double* weights, int token_length, int w_length, int y, int currrent_emotion) {
    // tokens and weights will be arrays of the same size. n is size of weights, y value 0 or 1


    // get the embedded tokens 
    double* embedded_tokens = token_early_RFF_embedding(tokens, token_length, w_length);

    // get activate value
    double activated = logistic_activation(embedded_tokens, weights, w_length);

    // get derivative of activated observation
    double derivative = logistic_derivative(activated);

    // get residual error
    double error = residual_error(y, activated);

    // get back_propogation factor
    double back_prop = back_propagation_term(error, derivative);



    double* new_weight_vec = malloc(sizeof(double) * w_length);
    if (!new_weight_vec) {
        return NULL;
    }

    for (int i = 0; i < w_length; i++) {
        new_weight_vec[i] = weights[i] + LEARNING_RATE * embedded_tokens[i] * back_prop;
    }

    if (y == 1) {
        // this data is being saved so that it may be used on a better RFF embedding later
        FILE *fptr;
        fptr = fopen("token_data.txt", "a");
        if (fptr != NULL) {
            // print tokenized data too a txt file to be used later in a better embedding
            for (int i = 0; i < w_length - 1; i++) {
                fprintf(fptr, "%f,", embedded_tokens[i]);
            }

            fprintf(fptr, "%f\n", embedded_tokens[w_length - 1]);
            fprintf(fptr, "%d", currrent_emotion);
            fclose(fptr);
        }
    }

    free(embedded_tokens);
    return new_weight_vec;

}


// meant to be used on the python side to free new_weight_vec and embedded_vec
void free_memory(double* vector) {
    free(vector);
}


double get_probability(double* tokens, double* weights, int token_length, int w_length) {
    // tokens and weights will be arrays of the same size. n is size of weights, y value 0 or 1

    // get the embedded tokens 
    double* embedded_tokens = token_early_RFF_embedding(tokens, token_length, w_length);

    // get activate value, or in this case, the probability of the text being a specific emotion
    double probability = logistic_activation(embedded_tokens, weights, w_length);


    free(embedded_tokens);

    return probability;


}

// FOR MACOS (mac operating system) use the command:
// gcc -fPIC -shared -o cpython.so cpython.c
// to compile the file

// FOR WINDOWS, USE THE COMMAND:
// gcc -fPIC -shared -o cpython.dll cpython.c