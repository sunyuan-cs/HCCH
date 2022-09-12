function result = hash_const(C,b,n)

 tmpcjc = C*C' - 1/n*C*ones(n,1)*ones(n,1)'*C';
        [V,D] = eig(tmpcjc);
        [D,index_D] = sort(diag(D),'descend');
        D = diag(D);
        V = V(:,index_D);
        for r=1:b
            if D(r,r)<1e-5
                r = r-1;
                break;
            end
        end
        D = D(1:r,1:r);
        D = D.^0.5;
        U = (C'*V(:,1:r)-1/n*ones(n,1)*(ones(n,1)'*C'*V(:,1:r)))/D;
        b_ = b-r;
        if(b_>0)
            UY = rand(n,b_);
            UY = UY - repmat(mean(UY),n,1);
            U = [U,UY];
            U = Schmidt(U);
        end
        result = sqrt(n)*U*V';
        result=sgn(result');
end