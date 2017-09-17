function score = scoreSVM(w_pos, w_neg, b, x)

    % Score computation: w·x - b = ?
    score = x' * (w_pos - w_neg) + b;
    
end