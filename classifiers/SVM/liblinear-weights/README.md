liblinear-mmdt
==============

Fast Maximum Margin Domain Transform

This code is a modification of liblinear for large-scale domain adaptation and free for research purposes. It is written in C but also comes with a MATLAB mex interface:

	[ target_model, W ] = train_linear_mmdt_fast( target_weights, target_labels, target_data, source_weights, source_labels, source_data )

Where the target model refers to a struct that can be used with the classify function of liblinear (also contained in the package) to perform classification in the target domain. The training label vectors and the data matrices are the supervised training data of the target and the source domain. The function performs block-wise coordinate descent optimization like described in the paper and the regularization parameters have to be given as instance weights to the function.

If you use this code, please cite the following technical report:

@techreport{Rodner:EECS-2013-154,
    Author = {Rodner, Erik and Hoffman, Judith and Donahue, Jeffrey and Darrell, Trevor and Saenko, Kate},
    Title = {Towards Adapting ImageNet to Reality: Scalable Domain Adaptation with Implicit Low-rank Transformations},
    Institution = {EECS Department, University of California, Berkeley},
    Year = {2013},
    Month = {Aug},
    URL = {http://www.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-154.html},
    Number = {UCB/EECS-2013-154}
}
