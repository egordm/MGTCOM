"""Parsing the hyperparameters."""

import argparse


def random_string():
    """Generate a random string."""
    import random
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(10))

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Facebook politicians network.
    """
    parser = argparse.ArgumentParser(description="Run GEMSEC.")

    parser.add_argument("--dataset",
        nargs="?",
        default="StarWars",
        help="Input graph path.")

    parser.add_argument("--dataset_version",
        nargs="?",
        default="base",
        help="Input graph path.")

    parser.add_argument("--run_name",
        nargs="?",
        default=random_string(),
        help="Input graph path.")

    parser.add_argument("--dump-matrices",
                        type=bool,
                        default=True,
	                help="Save the embeddings to disk or not. Default is not.")

    parser.add_argument("--model",
                        nargs="?",
                        default="GEMSECWithRegularization",
	                help="The model type.")

    parser.add_argument("--P",
                        type=float,
                        default=1,
	                help="Return hyperparameter. Default is 1.")

    parser.add_argument("--Q",
                        type=float,
                        default=1,
	                help="In-out hyperparameter. Default is 1.")

    parser.add_argument("--walker",
                        nargs="?",
                        default="first",
	                help="Random walker order. Default is first.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=16,
	                help="Number of dimensions. Default is 16.")

    parser.add_argument("--random-walk-length",
                        type=int,
                        default=80,
	                help="Length of random walk per source. Default is 80.")

    parser.add_argument("--num-of-walks",
                        type=int,
                        default=5,
	                help="Number of random walks per source. Default is 5.")

    parser.add_argument("--window-size",
                        type=int,
                        default=5,
                    	help="Window size for proximity statistic extraction. Default is 5.")

    parser.add_argument("--distortion",
                        type=float,
                        default=0.75,
	                help="Downsampling distortion. Default is 0.75.")

    parser.add_argument("--negative-sample-number",
                        type=int,
                        default=10,
	                help="Number of negative samples to draw. Default is 10.")

    parser.add_argument("--initial-learning-rate",
                        type=float,
                        default=0.01,
	                help="Initial learning rate. Default is 0.01.")

    parser.add_argument("--minimal-learning-rate",
                        type=float,
                        default=0.001,
	                help="Minimal learning rate. Default is 0.001.")

    parser.add_argument("--annealing-factor",
                        type=float,
                        default=1,
	                help="Annealing factor. Default is 1.0.")

    parser.add_argument("--initial-gamma",
                        type=float,
                        default=0.1,
	                help="Initial clustering weight. Default is 0.1.")

    parser.add_argument("--final-gamma",
                        type=float,
                        default=0.5,
	                help="Final clustering weight. Default is 0.5.")

    parser.add_argument("--lambd",
                        type=float,
                        default=2.0**-4,
	                help="Smoothness regularization penalty. Default is 0.0625.")

    parser.add_argument("--cluster-number",
                        type=int,
                        default=20,
	                help="Number of clusters. Default is 20.")

    parser.add_argument("--overlap-weighting",
                        nargs="?",
                        default="normalized_overlap",
	                help="Weight construction technique for regularization.")

    parser.add_argument("--regularization-noise",
                        type=float,
                        default=10**-8,
	                help="Uniform noise max and min on the feature vector distance.")

    return parser.parse_args()
