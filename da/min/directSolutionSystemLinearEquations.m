function x = directSolutionSystemLinearEquations(A, b)

    x = (A'*A) \ A' * sparse(double(b));

end

