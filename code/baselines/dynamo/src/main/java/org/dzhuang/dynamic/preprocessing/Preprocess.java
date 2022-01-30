package org.dzhuang.dynamic.preprocessing;


public class Preprocess {
    public static void main(String[] args) throws Exception{
        toDynaMo.run(args[0], Integer.parseInt(args[1]));
        toComparison.trans2Comparisons(args[0], Integer.parseInt(args[1]));
    }
}
