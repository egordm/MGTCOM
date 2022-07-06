import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from texttable import Texttable

root_path = Path(__file__).parent.parent.parent.parent
exports_path = root_path / "storage" / "exports"
outputs_path = root_path / "storage" / "outputs"

def graph_reader(input_path):
    """
    Function to read a csv edge list and transform it to a networkx graph object.
    """    
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def log_setup(args_in):
    """
    Function to setup the logging hash table.
    """    
    log = dict()
    log["times"] = []
    log["losses"] = []
    log["cluster_quality"] = []
    log["params"] = vars(args_in)
    return log

def json_dumper(data, path):
    """
    Function to dump the logs and assignments.
    """    
    with open(path, "w") as outfile:
        json.dump(data, outfile)

def initiate_dump_gemsec(log, assignments, args, final_embeddings, c_means):
    """
    Function to dump the logs and assignments for GEMSEC. If the matrix saving boolean is true the embedding is also saved.
    """
    output_dir = outputs_path / 'GEMSEC' / args.dataset / args.dataset_version / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    json_dumper(log, output_dir / 'log.json')
    # json_dumper(assignments, output_dir / 'assignments.json')
    if args.dump_matrices:
        np.save(output_dir / 'assignments.npy', assignments)
        np.save(output_dir / 'embeddings.npy', final_embeddings)
        np.save(output_dir / 'means.npy', c_means)
        # final_embeddings = pd.DataFrame(final_embeddings)
        # final_embeddings.to_csv(output_dir / 'embeddings.csv', index=None)
        # c_means = pd.DataFrame(c_means)
        # c_means.to_csv(output_dir / 'means.csv', index=None)

def initiate_dump_dw(log, assignments, args, final_embeddings):
    """
    Function to dump the logs and assignments for DeepWalk. If the matrix saving boolean is true the embedding is also saved.
    """
    output_dir = outputs_path / 'GEMSEC' / args.dataset / args.dataset_version / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    json_dumper(log, output_dir / 'log.json')
    # json_dumper(assignments, output_dir / 'assignments.json')
    if args.dump_matrices:
        np.save(output_dir / 'assignments.npy', assignments)
        np.save(output_dir / 'embeddings.npy', final_embeddings)
        # final_embeddings = pd.DataFrame(final_embeddings)
        # final_embeddings.to_csv(output_dir / 'embeddings.csv', index=None)

def tab_printer(log):
    """
    Function to print the logs in a nice tabular format.
    """    
    t = Texttable() 
    t.add_rows([["Epoch", log["losses"][-1][0]]])
    print(t.draw())

    t = Texttable()
    t.add_rows([["Loss", round(log["losses"][-1][1],3)]])
    print(t.draw()) 

    t = Texttable()
    t.add_rows([["Modularity", round(log["cluster_quality"][-1][1],3)]])
    print(t.draw()) 

def epoch_printer(repetition):
    """
    Function to print the epoch number.
    """    
    print("")
    print("Epoch " + str(repetition+1) + ". initiated.")
    print("")

def log_updater(log, repetition, average_loss, optimization_time, modularity_score):
    """ 
    Function to update the log object.
    """    
    index = repetition + 1
    log["losses"] = log["losses"] + [[int(index), float(average_loss)]]
    log["times"] = log["times"] + [[int(index), float(optimization_time)]]
    log["cluster_quality"] = log["cluster_quality"] + [[int(index), float(modularity_score)]]
    return log
