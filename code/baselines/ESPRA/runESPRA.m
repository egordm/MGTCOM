function runESPRA(snapshotFiles, outDir, alpha, beta)

T = length(snapshotFiles);

% Determine the node count in the dataset
N = 0;
for i = 1:T
    xx=load(snapshotFiles{i});
    N = max([N, max(max(xx))]);
end

% Load the network snapshots in memory
adjMatrix=cell(1,T);
for i = 1:T
    adjMatrix{i}=edgelistFile2adjMatrix(snapshotFiles{i}, N);
end

% evolutionary clustering
[result] = ESPRA( adjMatrix, alpha, beta);

% save to files
for t=1:T
    fp = fopen([outDir, 'dynamic.', num2str(t,'%02.f'), '.communities.txt'],'wt');
    r = result{t};
    for i=1:size(r, 1)
        fprintf(fp, '%d\t%d\n', r(i, 1), r(i, 2));
    end
end
fclose(fp); 
end