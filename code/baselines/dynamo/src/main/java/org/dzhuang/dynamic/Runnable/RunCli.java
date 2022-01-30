package org.dzhuang.dynamic.Runnable;

import org.dzhuang.dynamic.DynaMo.ModularityOptimizer_DynaMo;
import org.dzhuang.dynamic.DynaMo.ModularityOptimizer_Louvain;
import org.dzhuang.dynamic.preprocessing.toComparison;
import org.dzhuang.dynamic.preprocessing.toDynaMo;

import static org.dzhuang.dynamic.Runnable.RunAlgorithm.runIncremental;
import static org.dzhuang.dynamic.Runnable.RunAlgorithm.runLBTR;

public class RunCli {

    public static void main(String args[]) throws Exception{
        String dataset = args[1];
        int size = Integer.parseInt(args[2]);

        if(args.length == 3) {
            toDynaMo.run(dataset, size);
            toComparison.trans2Comparisons(dataset, size);
        }

//        ModularityOptimizer_Louvain.runLouvain(dataset, size);
        ModularityOptimizer_DynaMo.runDynamicModularity(dataset, size);
//        runIncremental(dataset);
//        runLBTR(dataset, size);
    }
}
