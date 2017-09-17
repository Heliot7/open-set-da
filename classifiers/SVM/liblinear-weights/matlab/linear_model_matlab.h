const char *model_to_matlab_structure(mxArray *plhs[], const struct model *model_);
const char *matlab_matrix_to_model(struct model *model_, const mxArray *matlab_struct);
