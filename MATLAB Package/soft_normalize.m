
function [B] = soft_normalize(A,dim, cost)
% Normalizza con la formula 
% norm_response = response / ( range(response)+cost )
% dim indica lungo quale direzione (1, lungo le righe; 2, lungo le colonne)

if dim == 2
    A = A';
end

for neu = 1:size(A,1)
    resp = A(neu,:);
    B(neu,:) = resp ./ (range(resp)+cost);
end

if dim == 2
    B = B';
end

end

