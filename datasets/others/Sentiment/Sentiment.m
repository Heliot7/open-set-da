classdef Sentiment
    properties
        path = 'Real\DomainAdaptation\Sentiment\';
        % Included Datasets: books, dvd, electronics, kitchen
        source = 'kitchen';
        target = 'dvd';
        % All classes: {'positive' 'negative'}
        classes = {'pos', 'neg'};
    end
end
