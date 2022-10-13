using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace Linear_Regression_With_Regularization
{

    class Linear_Regression
    {
        //Fields
        protected double[,] x_data;
        protected double[] features_mean;
        protected double[] features_std;
        protected double[] y_actual;
        protected double[] weights = null;
        protected double bias = 0; //also called intercept or constant //default value = 0
        protected bool fit_intercept = true; //add a bias to the model, set false if not
        protected double learning_rate = 1;
        protected int max_n_iter = 1000;


        //Properties
        protected double[] Weights_to_use //Standardized weights
        {
            get 
            {
                double[] weights_to_use;
                if(fit_intercept)
                {
                    weights_to_use = new double[this.weights.Length + 1];
                    weights_to_use [0] = bias;
                    Array.Copy(this.weights, 0, weights_to_use, 1, this.weights.Length);
                }
                else
                {
                    weights_to_use = this.weights;
                }

                return weights_to_use;
            }

            set
            {
                if (fit_intercept)
                {
                    this.bias = value[0];
                    Array.Copy(value, 1, this.weights, 0, this.weights.Length);
                }
                else
                {
                    this.weights = value;
                }
            }
        }
        protected double[] Weights_unstandardized
        { //https://stackoverflow.com/questions/54067474/comparing-results-from-standardscaler-vs-normalizer-in-linear-regression
            get
            {
                double[] weights_unstandardized = Weights_to_use;
                if (fit_intercept) //if intercept fitted during training
                {
                    double standardized_bias_penalty = 0;
                    for(int i=1; i < Weights_to_use.Length; i++)
                    {
                        standardized_bias_penalty -= (Weights_to_use[i] * features_mean[i]) / features_std[i];
                        weights_unstandardized[i] = Weights_to_use[i] / features_std[i];
                    }
                    weights_unstandardized[0] = Weights_to_use[0] + standardized_bias_penalty;
                }
                else
                {
                    for (int i = 0; i < Weights_to_use.Length; i++)
                    {
                        weights_unstandardized[i] = Weights_to_use[i] / features_std[i];
                    }

                }
                return weights_unstandardized;
            }
        }
        protected double[,] X_data_to_use //we are appending ones to the first column of the x_data and standardizing x values
        {
            get
            {
                //if user wants to fit an intercept or bias term, X_data_to_use must have ones on the first column else use this.x_data only.
                return (this.fit_intercept)?LinearAlgebra_tools.insert_a_val(this.x_data, 1, 1):this.x_data;
            }
        }
        public (double[], bool) Get_weights
        {
            get { return (Weights_unstandardized, fit_intercept); }
        } //return value weights and fit_intercept
        protected int Total_rows
        {
            get { return x_data.GetLength(0); }
        }
        protected int Total_cols //also total weights (with out bias)
        {
            get { return x_data.GetLength(1); }
        }
        
        //Constructor
        public Linear_Regression(double learning_rate = 1, bool fit_intercept = true)
        {
            this.fit_intercept = fit_intercept;
            this.learning_rate = learning_rate;
        }

        //Methods
        public void fit(double[,] x_data, double[] y_actual, int max_n_iter = 1000, bool re_fit = false, bool verbose = true, int report_index_partition = 50)
        {
            //lets get the mean and standard deviation of each column from our x_data so that we could standardize it in our training
            this.features_mean = LinearAlgebra_tools.matrix_get_col_mean(x_data);
            this.features_std = LinearAlgebra_tools.matrix_get_col_std(x_data);

            //standaridze our x_data
            this.x_data = LinearAlgebra_tools.standardize_matrix(x_data, this.features_mean, this.features_std);
            this.y_actual = y_actual;
            this.max_n_iter = max_n_iter;

            //we should append a mean value and standard deviation value of each columns in our matrix if user set intercept to true...
            //this will be useful for unstandardizing our weights and for predition on test & validation data
            this.features_mean = LinearAlgebra_tools.insert_a_val(this.features_mean, 1, 1); //the mean of a vector of ones is 1, thus we need to insert it in the first index for bias term
            this.features_std = LinearAlgebra_tools.insert_a_val(this.features_std, 0, 1); // the standard deviation of a vector of ones is 0, thus we ne to insert it in the first index for bias term

            if (this.weights == null || re_fit == false) //we should initialize weights to zero if user doesn't want to re_fit the model or weights are null
            {
                weights = Enumerable.Repeat((double)0, this.Total_cols).ToArray(); //initialize all weights to zero
                //type casting + enumerable object to arrray for repeated zeros initialization
            }

            Console.WriteLine("Training Linear Model with Parameters: ");
            Console.WriteLine($"(Max Iterations: {max_n_iter}, Learning_rate: {this.learning_rate}, Refit: {re_fit})");
            double current_cost = 0, prev_cost = 0, tolerance = 0.0001; //use this variables to stop searching for global minimum if |current_cost - prev_cost| < tolerance
            for(int i = 0; i < max_n_iter; i++)
            {
                Weights_to_use = gradient_descent();
                prev_cost = current_cost;
                current_cost = cost_function();
                if(Math.Abs(prev_cost - current_cost) < tolerance)
                {
                    break;
                }
                if (verbose)
                {
                    if((i+1)%report_index_partition == 0)
                    {
                        Console.WriteLine($"Iteration [{i + 1}] Cost: {current_cost:f4}");
                    }
                }
                if(i == max_n_iter - 1 && Math.Abs(prev_cost - current_cost) > tolerance)
                {
                    Console.WriteLine("CONVERGENCE WARNING: The model failed failed to converge. Please increase the iteration or adjust the learning rate.");
                }
                
            }

        }

        public double[] predict(double[,] x_data, double[] weights = null)
        {
            //for testing/validation dataset
            if(weights == null)
            {
                weights = this.Weights_unstandardized;
                //if user trained the model with a bias term or has set the fit_intercept to true, then...
                //we need to predict values with a bias term thus we need to append ones to x_data's first column
                if (fit_intercept)
                {
                    x_data = LinearAlgebra_tools.insert_a_val(x_data, 1, 1);
                }
            }
            return LinearAlgebra_tools.dot(x_data, ref weights); //get the double[] y_predicitons
        }

        protected virtual double[] gradient_descent()
        {
            //LinearAlgebra_tools.print_matrix(X_data_to_use); //uncomment for debugging purposes
            //update the weights of our model together, not one by one.
            double[] new_weights = new double[this.Weights_to_use.Length];

            //computing for each y_pred in y_prediction
            double[] y_predict = this.predict(X_data_to_use, Weights_to_use);
            double[] error = LinearAlgebra_tools.subtract(ref y_predict, ref this.y_actual);

            for (int i = 0; i < Weights_to_use.Length; i++)
            {
                double[] x_i = LinearAlgebra_tools.get_vector_column_from_matrix(X_data_to_use, i);
                double[] error_x_i = LinearAlgebra_tools.multiply(ref error, ref x_i);
                double error_x_i_average = error_x_i.Average();
                double weight_penalty = this.learning_rate * error_x_i_average;
                new_weights[i] = Weights_to_use[i] - weight_penalty;
            }
            return new_weights;
        }

        protected virtual double cost_function()
        {   
            double[] y_predict = this.predict(X_data_to_use, Weights_to_use); //predict y values
            double[] squared_error = LinearAlgebra_tools.subtract(ref y_predict, ref this.y_actual).Select(x=>x*x).ToArray();
            //We are using lambda expressions to square each elements in the returned array of the error
            double average_squared_error = squared_error.Average();
            double total_cost = average_squared_error / 2;
            return total_cost;
        }

        //Update values
        public (double,double,double,double) score(double[,] x_data, double[] y_actual)
        {
            //This computes for R2_score or coefficient of determination
            double SSR = 0, SST = 0, R2_score, Adj_R2_score; //sum of squared residuals, sum of squares total, coefficient of determination, adjusted coefficient of determination
            double[] y_pred;
            int tot_samples, tot_features;

            y_pred = predict(x_data);
            //Compute for SSR
            SSR = LinearAlgebra_tools.subtract(ref y_actual, ref y_pred).Select(x => x * x).Sum();
            //Compute for SST
            double y_actual_mean = y_actual.Average();
            SST = y_actual.Select(y_actual_i => Math.Pow(y_actual_i - y_actual_mean, 2)).Sum();

            R2_score = 1 - (SSR / SST); //compute for R2 score
            tot_samples = x_data.GetLength(0); //get total samples
            tot_features = x_data.GetLength(1); //get total predictors
            Adj_R2_score = 1 - (((1 - R2_score) * (tot_samples - 1)) / (tot_samples - tot_features - 1));
            return (SSR,SST,R2_score,Adj_R2_score);
        }

    }

    class Ridge_Regression : Linear_Regression
    {
        //Fields
        double lambda;

        //Constructor
        public Ridge_Regression(double lambda = 1, double learning_rate = 1, bool fit_intercept = true ) : base(learning_rate, fit_intercept)
        {
            this.lambda = lambda;
        }

        //Methods
        protected override double[] gradient_descent()
        {
            //LinearAlgebra_tools.print_matrix(X_data_to_use); //uncomment for debugging purposes
            //update the weights of our model together, not one by one.
            double[] new_weights = new double[this.Weights_to_use.Length];

            //computing for each y_pred in y_prediction
            double[] y_predict = this.predict(X_data_to_use, Weights_to_use);
            double[] error = LinearAlgebra_tools.subtract(ref y_predict, ref this.y_actual);

            for (int i = 0; i < Weights_to_use.Length; i++)
            {
                double[] x_i = LinearAlgebra_tools.get_vector_column_from_matrix(X_data_to_use, i);
                double[] error_x_i = LinearAlgebra_tools.multiply(ref error, ref x_i);
                double error_x_i_average = error_x_i.Average();
                double weight_penalty;
                if(fit_intercept && i == 0) //if fit_intercept and i==0 (bias term) then no need regularization term for the bias term
                {
                    weight_penalty = this.learning_rate * error_x_i_average; //for bias term
                }
                else
                {
                    weight_penalty = this.learning_rate * (error_x_i_average + ((lambda * Weights_to_use[i]) / this.Total_rows));
                }
                new_weights[i] = Weights_to_use[i] - weight_penalty;
            }
            return new_weights;
        }

        protected override double cost_function()
        {
            double[] y_predict = this.predict(X_data_to_use, Weights_to_use); //predict y values
            double[] squared_error = LinearAlgebra_tools.subtract(ref y_predict, ref this.y_actual).Select(x => x * x).ToArray();
            //We are using lambda expressions to square each elements in the returned array of the error
            double average_squared_error = squared_error.Average();
            double sum_of_square_of_weights = this.weights.Select(x => x*x).Sum();
            double total_cost = (average_squared_error / 2) + (lambda*sum_of_square_of_weights/2*Total_rows);
            //Console.WriteLine($"OLS cost_function value: {base.cost_function()}"); //Uncomment to report the corresponding cost function of OLS
            return total_cost;
        }

        
    }

    class Lasso_Regression : Linear_Regression
    {
        //Fields
        double lambda;

        //Constructors
        public Lasso_Regression (double lambda = 1, double learning_rate = 1, bool fit_intercept = true):base(learning_rate, fit_intercept)
        {
            this.lambda = lambda;
        }

        //Methosd
        protected override double[] gradient_descent()
        {
            //There are two ways/methods for the gradient descent of Lasso: 1)Subgradient Descent and 2) Coordinate Descent
            //We will use coordinate descent because it converges faster than subgradient descent and it is also used by sci-kit learn
            //LinearAlgebra_tools.print_matrix(X_data_to_use); //uncomment for debugging purposes
            //update the weights of our model together, not one by one.
            double[] new_weights = new double[this.Weights_to_use.Length];

            for (int i = 0; i < Weights_to_use.Length; i++)
            {
                //create temp weights_array where weight_i is set to zero to compute for Pj (OLS term computation excluding (or set to 0) the weight instance that we are trying to update)
                double[] temp_weights = Weights_to_use;
                temp_weights[i] = 0;
                //computing for Pj
                double[] y_Pj_predict = this.predict(X_data_to_use, temp_weights);
                double[] y_Pj_diff = LinearAlgebra_tools.subtract(ref this.y_actual, ref y_Pj_predict);

                double[] x_i = LinearAlgebra_tools.get_vector_column_from_matrix(X_data_to_use, i);
                double[] y_Pj_array = LinearAlgebra_tools.multiply(ref y_Pj_diff, ref x_i);
                double y_Pj = y_Pj_array.Average();
                if (fit_intercept && i == 0) //if fit_intercept and i==0 (bias term) then no need regularization term for the bias term
                {
                    new_weights[i] = this.learning_rate * y_Pj; //for bias term
                }
                else
                {
                    new_weights[i] = this.learning_rate * soft_threshold(y_Pj, lambda / this.Total_rows);
                    //set learning rate to 1 to apply the normal/regular soft threshold function: https://xavierbourretsicotte.github.io/lasso_implementation.html
                    //we are dividing lambda by m or Total rows of training data such that it will be automatically adjust on the batch size of training data...
                    //this will be usefule for stochastic gradient descent, such that users can use the same lambda value regardless of the batch size
                }
            }
            return new_weights;
        }

        private double soft_threshold(double rho, double lambda) //rho == pJ and lambda == regularization penalty or l1-term value
        {
            double penalty = 0;
            if(rho < -lambda)
            {
                penalty = rho + lambda;
            }
            else if (rho > lambda) {
                penalty = rho - lambda;
            }
            else
            {
                penalty = 0;
            }
            return penalty;
        }


        protected override double cost_function()
        {
            double[] y_predict = this.predict(X_data_to_use, Weights_to_use); //predict y values
            double[] squared_error = LinearAlgebra_tools.subtract(ref y_predict, ref this.y_actual).Select(x => x * x).ToArray();
            //We are using lambda expressions to square each elements in the returned array of the error
            double average_squared_error = squared_error.Average();
            double sum_of_absolute_of_weights = this.weights.Select(x=>Math.Abs(x)).Sum();
            double total_cost = (average_squared_error / 2) + (lambda * sum_of_absolute_of_weights / 2 * Total_rows);
            //Console.WriteLine($"OLS cost_function value: {base.cost_function()}"); //Uncomment to report the corresponding cost function of OLS
            return total_cost;
        }
    }

    static class LinearAlgebra_tools {

        static public double[] dot(double[,] matrix, ref double[] vector) //dot(x_data, weights);
        {
            int total_rows = matrix.GetLength(0);
            int total_cols = matrix.GetLength(1);
            double[] dot_result = new double[total_rows];

            for (int i = 0; i < total_rows; i++)
            {
                double dot_y = 0; //continuosly add values to dot_y for each matrix_i_j*vector_j then insert to dot_result
                for (int j = 0; j < total_cols; j++)
                {
                    dot_y += matrix[i, j] * vector[j];
                }
                dot_result[i] = dot_y; //insert dot_y to dot_result array
            }

            return dot_result;
        }

        static public double[] subtract(ref double[] vector_minuend, ref double[] vector_subtrahend) //minuend-subtrahed = difference
        {
            int total_rows = vector_minuend.Length;
            double[] difference = new double[total_rows];

            for (int i = 0; i < total_rows; i++)
            {
                difference[i] = vector_minuend[i] - vector_subtrahend[i];
            }
            return difference;
        }

        static public double[] multiply(ref double[] vector1, ref double[] vector2)
        {
            int total_rows = vector1.Length;
            double[] product = new double[total_rows];

            for (int i = 0; i < total_rows; i++)
            {
                product[i] = vector1[i] * vector2[i];
            }
            return product;
        }

        static public double[] get_vector_column_from_matrix(double[,] matrix, int col)
        {
            int total_rows = matrix.GetLength(0);
            double[] vector_extracted = new double[total_rows];

            for (int i = 0; i < total_rows; i++)
            {
                vector_extracted[i] = matrix[i, col];
            }

            return vector_extracted;
        }

        static public double[,] standardize_matrix(double[,] matrix, double[] mean_values, double[] std_values, double null_replace = 1)
        {
            int total_rows = matrix.GetLength(0);
            int total_cols = matrix.GetLength(1);
            double[,] matrix_standardize = new double[total_rows, total_cols];

            for (int i = 0; i < total_cols; i++)
            {
                double[] vector_col = get_vector_column_from_matrix(matrix, i);

                //standardize each values of the column
                int j = 0; //index for each rows
                foreach (var x in vector_col)
                {
                    double val = (x - mean_values[i]) / std_values[i]; //get the standardized value
                    matrix_standardize[j, i] = (std_values[i] == 0) ? null_replace : val; //for bias terms, std will be zero, thus will create error
                    j++;
                }
            }
            return matrix_standardize;
        }

        static public double[,] standardize_matrix(double[,] matrix, double null_replace = 1) //null replace is the value set if std is 0, applicable especially if fit_intercept = true
        {
            int total_rows = matrix.GetLength(0);
            int total_cols = matrix.GetLength(1);
            double[,] matrix_standardize = new double[total_rows, total_cols];
            double[] mean_values = LinearAlgebra_tools.matrix_get_col_mean(matrix), std_values = LinearAlgebra_tools.matrix_get_col_std(matrix);

            for (int i = 0; i < total_cols; i++)
            {
                double[] vector_col = get_vector_column_from_matrix(matrix, i);

                //standardize each values of the column
                int j = 0; //index for each rows
                foreach (var x in vector_col)
                {
                    double val = (x - mean_values[i]) / std_values[i]; //get the standardized value
                    matrix_standardize[j, i] = (std_values[i] == 0) ? null_replace : val; //for bias terms, std will be zero, thus will create error
                    j++;
                }
            }
            return matrix_standardize;
        }

        static public double[] matrix_get_col_mean(double[,] matrix)
        {
            int total_cols = matrix.GetLength(1);
            double[] mean_values = new double[total_cols];
            for (int i = 0; i < total_cols; i++)
            {
                double[] vector_col = get_vector_column_from_matrix(matrix, i);
                double average = vector_col.Average();
                mean_values[i] = average;
            }

            return mean_values;

        }

        static public double[] matrix_get_col_std(double[,] matrix)
        {
            int total_rows = matrix.GetLength(0);
            int total_cols = matrix.GetLength(1);
            double[] std_values = new double[total_cols];
            for (int i = 0; i < total_cols; i++)
            {
                double[] vector_col = get_vector_column_from_matrix(matrix, i);
                double average = vector_col.Average();
                double sumOfSquaresOfDifferences = vector_col.Select(val => Math.Pow((val - average), 2)).Sum();
                double std = Math.Sqrt(sumOfSquaresOfDifferences / total_rows);
                std_values[i] = std;
            }

            return std_values;
        }

        static public void print_matrix(double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                Console.Write($"Row {i}: ");
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    Console.Write($"{matrix[i, j]} ");
                }
                Console.WriteLine();
            }
        }

        static public double[] insert_a_val(double[] vector_to_insert, double value_to_insert, int position_to_insert)
        {
            int vector_to_insert_len = vector_to_insert.Length;
            double[] vector_inserted = new double[vector_to_insert_len+1];
            for (int i = 0; i < vector_to_insert_len + 1; i++)
            {
                if (i < position_to_insert - 1)
                    vector_inserted[i] = vector_to_insert[i];
                else if (i == position_to_insert - 1)
                    vector_inserted[i] = value_to_insert;
                else
                    vector_inserted[i] = vector_to_insert[i - 1];
            }
            return vector_inserted;
        }

        static public double[,] insert_a_val(double[,] matrix_to_insert, double value_to_insert, int col_position_to_insert)
        {
            int total_rows = matrix_to_insert.GetLength(0);
            int total_cols = matrix_to_insert.GetLength(1);
            double[,] matrix_inserted = new double[total_rows, total_cols + 1]; //for col_position_to_insert
            for (int i = 0; i < total_rows; i++)
            {
                for (int j = 0; j < total_cols+1; j++)
                {
                    //if current col position is below/less than the col position where to insert the value_to_insert, then just insert the value of
                    //matrix_to_insert[i, j] in matrix_inserted[i, j]
                    if (j < col_position_to_insert - 1)
                    {
                        matrix_inserted[i, j] = matrix_to_insert[i, j];
                    }
                    //if current col position is equal to the col position where to insert the value_to_insert,
                    //then insert the value_to_insert in matrix_inserted[i, j]
                    else if  (j == col_position_to_insert - 1)
                    {
                        matrix_inserted[i, j] = value_to_insert;
                    }
                    //if current col position is greater than the col position where to insert the value_to_insert, then just insert the value of
                    //matrix_to_insert[i, j-1] in matrix_inserted[i, j]
                    else
                    {
                        matrix_inserted[i, j] = matrix_to_insert[i, j - 1];
                    }
                }
            }
            return matrix_inserted;
        }
    }

    internal class Program
    {
        static void Main(string[] args)
        {

            Linear_Regression linear_model = null; //learning rate = 0.5, fit_intercept = true
            double[,] x_data;
            double[] y_data;
            double learning_rate = 0.5, lambda = 0;
            int total_rows = 0, total_cols = 0, iter = 1, report_index_partition = 10;
            int linear_model_type = 0;
            bool verbose = true, refit = false, fit_intercept = true;
            char[] delimiters = { ' ', ',' };

            print_main_header();
            Exception_check_int(ref total_rows, "> How many samples/rows in your training data?: ");
            Exception_check_int(ref total_cols, "> How many variables/features in your training data?: ");
            //Get type of linear regression model
            Console.WriteLine("> Please choose type of your linear regression model: ");
            Console.WriteLine("> [1] (Non-Regularized) Linear Regression");
            Console.WriteLine("> [2] Ridge Regression");
            Console.WriteLine("> [3] Lasso Regression");
            Exception_check_int(ref linear_model_type, "> Selection: ");
            if (linear_model_type == 2 || linear_model_type == 3)
            {
                Exception_check_double(ref lambda, "> Please enter a lambda value to train your regularized model: ");
            }
            Console.WriteLine("-------------------------------------------");
            x_data = new double[total_rows, total_cols];
            y_data = new double[total_rows];

            Console.WriteLine("> Please input each of your (x values/predictor variables) separated either by comma or space.");
            for (int i = 0; i < total_rows; i++)
            {
                Console.Write($"Row x[{i + 1}]: ");
                double[] row = Array.ConvertAll(Console.ReadLine().Split(delimiters), Double.Parse); //Console read each row of data and convert to double array separated by comma or space
                for (int j = 0; j < total_cols; j++) //insert elements of each row to x_data
                {
                    x_data[i, j] = row[j];
                }
            }
            Console.WriteLine("-------------------------------------------");

            Console.WriteLine("> Please input the corresponding (y value/response variable) of each of your (x values/predictor variables).");
            {
                for (int i = 0; i < total_rows; i++)
                {
                    Console.Write($"Row y[{i + 1}]: ");
                    y_data[i] = Convert.ToDouble(Console.ReadLine());

                }
            }
            Console.WriteLine("-------------------------------------------");

            Console.Write("> Would you like to turn on verbosity? Type y/Y if \"YES\" and any key if \"NO\": ");
            verbose = char.ToLower(Console.ReadKey().KeyChar) == 'y';
            Console.Write("\n> Would you like to refit your model? Type y/Y if \"YES\" and any key if \"NO\": ");
            refit = char.ToLower(Console.ReadKey().KeyChar) == 'y';
            Console.Write("\n> Would you like to add a bias term to your model? Type y/Y if \"YES\" and any key if \"NO\": ");
            fit_intercept = char.ToLower(Console.ReadKey().KeyChar) == 'y';
            Exception_check_double(ref learning_rate,"\n> Please enter a learning rate to train your model: ");
            Exception_check_int(ref iter, "> Please enter max iterations to train your model: ");

            if (verbose)
            {
                Exception_check_int(ref report_index_partition, "> Please input report index partition for training iterations (Default = 50): ");
            }
            Console.WriteLine("-------------------------------------------");

            //Instantiate and fit/train model
            if(linear_model_type == 1)
            {
                linear_model = new Linear_Regression(learning_rate, fit_intercept);
            }
            else if (linear_model_type == 2)
            {
                linear_model = new Ridge_Regression(lambda, learning_rate, fit_intercept);
            }
            else if (linear_model_type == 3)
            {
                linear_model = new Lasso_Regression(lambda, learning_rate, fit_intercept);
            }

            //Train model and get weights & bias learned 
            linear_model.fit(x_data, y_data, iter, refit, verbose, report_index_partition);
            (double[] weights, bool fit_intercept_status ) = linear_model.Get_weights;
            Console.WriteLine("-------------------------------------------");
            Console.WriteLine("> Printing coefficents of your linear regression model");
            if (fit_intercept_status)
            {
                int weight_index = 0;
                foreach (var weight in weights)
                {   //print bias and weight, the index of bias in weights array is in 0;
                    Console.WriteLine($"{((weight_index == 0) ? "Bias:" : $"Weight [{weight_index}]:")} {weight:f4}");
                    weight_index++;
                }
            }
            else
            {
                int weight_index = 1;
                foreach (var weight in weights)
                {   //print bias and weight, the index of bias in weights array is in 0;
                    Console.WriteLine($"Weight [{weight_index}]: {weight:f4}");
                    weight_index++;
                }
            }
            Console.WriteLine("-------------------------------------------");

            //Computer for R2 score of your model
            (double,double,double,double) metrics = linear_model.score(x_data, y_data);
            Console.WriteLine("> Linear Model Metrics Report: ");
            Console.WriteLine($"SSR (Sum of Squared Residuals): {metrics.Item1:f4}");
            Console.WriteLine($"SST (Sum of Squares Total): {metrics.Item2:f4}");
            Console.WriteLine($"R^2 score: {metrics.Item3:f4}");
            Console.WriteLine($"Adjusted-R^2 score: {metrics.Item4:f4}");
            Console.WriteLine("-------------------------------------------");

            //Use the trained linear model to predict x data
            Console.WriteLine("Predicting x data values from linear model:");
            int index = 1;
            foreach(var prediction in linear_model.predict(x_data))
            {
                Console.WriteLine($"Prediction [{index}]: {prediction}");
                index++;
            }


            Console.ReadLine();
            //Runtime ends here

            //Functions
            void Exception_check_double(ref double val, string msg)
            {
                bool error = false;
                do
                {
                    Console.Write(msg);
                    try
                    {
                        val.GetType();
                        val = Convert.ToDouble(Console.ReadLine());
                        error = false;
                    }
                    catch (FormatException)
                    {
                        exception_clear();
                        error = true;
                    }
                } while (error);
            }

            void Exception_check_int(ref int val, string msg)
            {
                bool error = false;
                do
                {
                    Console.Write(msg);
                    try
                    {
                        val = Convert.ToInt32(Console.ReadLine());
                        error = false;
                    }
                    catch (FormatException)
                    {
                        exception_clear();
                        error = true;
                    }
                } while (error);
            }

            void exception_clear()
            {
                Console.Write("WARNING: Format Exception error raised. Press enter to try again.");
                Console.ReadLine();
                Console.SetCursorPosition(Console.CursorLeft, Console.CursorTop - 2);
                for (int i = 0; i < 2; i++)
                {
                    char[] blankline = new char[80];
                    Console.Write(blankline, 0, 80);
                    Console.WriteLine();
                }
                Console.SetCursorPosition(Console.CursorLeft, Console.CursorTop - 2);
            }

            void print_main_header()
            {
                Console.WriteLine("\n\n");
                Console.WriteLine("\t\t██╗     ██╗███╗   ██╗███████╗ █████╗ ██████╗         ██████╗ ███████╗ ██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗");
                Console.WriteLine("\t\t██║     ██║████╗  ██║██╔════╝██╔══██╗██╔══██╗        ██╔══██╗██╔════╝██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║");
                Console.WriteLine("\t\t██║     ██║██╔██╗ ██║█████╗  ███████║██████╔╝        ██████╔╝█████╗  ██║  ███╗██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║");
                Console.WriteLine("\t\t██║     ██║██║╚██╗██║██╔══╝  ██╔══██║██╔══██╗        ██╔══██╗██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║");
                Console.WriteLine("\t\t███████╗██║██║ ╚████║███████╗██║  ██║██║  ██║        ██║  ██║███████╗╚██████╔╝██║  ██║███████╗███████║███████║██║╚██████╔╝██║ ╚████║");
                Console.WriteLine("\t\t╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝        ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝");
                Console.WriteLine("\n\n");
            }
        }


    }
}
 